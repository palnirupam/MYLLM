from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
from datetime import datetime


@dataclass
class SourceProvenance:
    """Level 1: The overarching source or provider of data."""
    source_id: str
    name: str
    url: str
    license_type: str
    is_audited: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "url": self.url,
            "license_type": self.license_type,
            "is_audited": self.is_audited,
            "metadata": self.metadata
        }


@dataclass
class DatasetProvenance:
    """Level 2: A specific dataset or subset from a source."""
    dataset_id: str
    source_id: str
    version_or_revision: str
    language: str
    domain: str
    retrieval_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_id": self.source_id,
            "version_or_revision": self.version_or_revision,
            "language": self.language,
            "domain": self.domain,
            "retrieval_date": self.retrieval_date,
            "metadata": self.metadata
        }


@dataclass
class DocumentProvenance:
    """Level 3: An individual document/file within a dataset. 
    Only used sparingly for sensitive/code domains where file-level tracking is required."""
    document_id: str
    dataset_id: str
    original_url_or_path: Optional[str] = None
    specific_license: Optional[str] = None # e.g. for individual git repos
    content_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "original_url_or_path": self.original_url_or_path,
            "specific_license": self.specific_license,
            "content_hash": self.content_hash
        }


@dataclass
class ShardProvenance:
    """Level 4: The frozen, immutable shard ready for training."""
    shard_id: str
    content_hash: str
    dataset_ids: List[str]
    creation_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    token_count: int = 0
    language_distribution: Dict[str, float] = field(default_factory=dict)
    domain_distribution: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "content_hash": self.content_hash,
            "dataset_ids": self.dataset_ids,
            "creation_date": self.creation_date,
            "token_count": self.token_count,
            "language_distribution": self.language_distribution,
            "domain_distribution": self.domain_distribution
        }


class ProvenanceTracker:
    """Tracks hierarchical provenance throughout the data pipeline."""
    
    def __init__(self):
        self.sources: Dict[str, SourceProvenance] = {}
        self.datasets: Dict[str, DatasetProvenance] = {}
        self.documents: Dict[str, DocumentProvenance] = {}
        self.shards: Dict[str, ShardProvenance] = {}

    def add_source(self, source: SourceProvenance):
        self.sources[source.source_id] = source

    def add_dataset(self, dataset: DatasetProvenance):
        if dataset.source_id not in self.sources:
            raise ValueError(f"Unknown source_id: {dataset.source_id}")
        self.datasets[dataset.dataset_id] = dataset
        
    def add_document(self, document: DocumentProvenance):
        if document.dataset_id not in self.datasets:
            raise ValueError(f"Unknown dataset_id: {document.dataset_id}")
        self.documents[document.document_id] = document
        
    def add_shard(self, shard: ShardProvenance):
        for ds_id in shard.dataset_ids:
            if ds_id not in self.datasets:
                raise ValueError(f"Unknown dataset_id in shard: {ds_id}")
        self.shards[shard.shard_id] = shard

    def export_manifest(self) -> str:
        """Exports the entire provenance tree to a JSON manifest."""
        manifest = {
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "shards": {k: v.to_dict() for k, v in self.shards.items()},
            # Note: We do not serialize documents here to avoid massive manifests.
            # Document provenance is logged separately if needed.
        }
        return json.dumps(manifest, indent=2)
