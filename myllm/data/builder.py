import uuid
from typing import List, Dict, Any, Iterator, Optional
from myllm.data.storage import StorageBackend, calculate_hash
from myllm.data.provenance import ShardProvenance, ProvenanceTracker


class DataBuilder:
    """
    Local engine to download samples, run filters, support stratified 
    manual audits, and emit immutable snapshots.
    """
    def __init__(
        self, 
        storage: StorageBackend, 
        tracker: ProvenanceTracker, 
        max_shard_bytes: int = 100 * 1024 * 1024,
        build_date: Optional[str] = None
    ):
        self.storage = storage
        self.tracker = tracker
        self.max_shard_bytes = max_shard_bytes
        self.build_date = build_date
        self._current_shard_buffer: List[bytes] = []
        self._current_shard_size = 0
        self._current_dataset_ids: set = set()
        
    def _flush_shard(self):
        """Writes the current buffer to an immutable shard in storage."""
        if not self._current_shard_buffer:
            return
            
        # Compile content
        content = b"\n".join(self._current_shard_buffer)
        content_hash = calculate_hash(content)
        # Deterministic shard ID based purely on content
        shard_id = f"shard_{content_hash[:16]}"
        
        # Write to storage
        path = f"shards/{shard_id}.jsonl"
        # Since shards are strictly content-addressed, if it exists, it's already identical.
        try:
            self.storage.write(path, content)
        except FileExistsError:
            pass # Immutable storage prevents overwrite, which is fine since content_hash is identical
        
        # Record provenance
        shard_prov = ShardProvenance(
            shard_id=shard_id,
            content_hash=content_hash,
            dataset_ids=list(self._current_dataset_ids)
        )
        if self.build_date:
            shard_prov.creation_date = self.build_date
        self.tracker.add_shard(shard_prov)
        
        # Reset buffer
        self._current_shard_buffer = []
        self._current_shard_size = 0
        self._current_dataset_ids = set()

    def process_stream(self, dataset_id: str, stream: Iterator[bytes]):
        """
        Processes a stream of raw bytes, applies filters, and packs them into buffers.
        Flushes automatically only when max_shard_bytes is reached.
        """
        # Ensure dataset is registered
        if dataset_id not in self.tracker.datasets:
            raise ValueError(f"Dataset {dataset_id} must be registered in provenance tracker.")
            
        for document in stream:
            doc_size = len(document)
            if self._current_shard_size + doc_size > self.max_shard_bytes and self._current_shard_buffer:
                self._flush_shard()
                
            self._current_shard_buffer.append(document)
            self._current_shard_size += doc_size
            self._current_dataset_ids.add(dataset_id)

    def freeze_corpus(self) -> str:
        """
        Finalizes the data build process, flushes remaining buffers, 
        and writes the master manifest to storage.
        """
        # Flush any remaining data before freezing
        if self._current_shard_buffer:
            self._flush_shard()
            
        manifest_content = self.tracker.export_manifest().encode('utf-8')
        manifest_hash = calculate_hash(manifest_content)
        manifest_path = f"manifests/manifest_{manifest_hash[:16]}.json"
        
        uri = self.storage.write(manifest_path, manifest_content)
        return uri
