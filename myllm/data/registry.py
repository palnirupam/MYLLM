from typing import Dict, List, Optional
from myllm.data.provenance import SourceProvenance, DatasetProvenance, ProvenanceTracker


class DatasetRegistry:
    """Central registry cataloging approved candidate sources and datasets."""
    
    def __init__(self, tracker: ProvenanceTracker):
        self.tracker = tracker
        self._register_default_sources()
        
    def _register_default_sources(self):
        """Registers the initial candidate sources for Dhruva V0."""
        
        # 1. FineWeb-Edu (ODC-BY 1.0)
        self.tracker.add_source(SourceProvenance(
            source_id="src_fineweb_edu",
            name="HuggingFace FineWeb-Edu",
            url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            license_type="ODC-BY-1.0",
            metadata={"domain": "educational", "tier": "A"}
        ))
        
        # 2. AI4Bharat Sangraha (CC BY 4.0)
        self.tracker.add_source(SourceProvenance(
            source_id="src_sangraha",
            name="AI4Bharat Sangraha",
            url="https://huggingface.co/datasets/ai4bharat/sangraha",
            license_type="CC-BY-4.0",
            metadata={"domain": "general", "tier": "A"}
        ))
        
        # 3. SmolLM-Corpus
        self.tracker.add_source(SourceProvenance(
            source_id="src_smollm",
            name="HuggingFace SmolLM-Corpus",
            url="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
            license_type="Apache-2.0",
            metadata={"domain": "mixed", "tier": "A"}
        ))
        
        # 4. The Stack v2 (Code)
        self.tracker.add_source(SourceProvenance(
            source_id="src_the_stack_v2",
            name="BigCode The Stack v2",
            url="https://huggingface.co/datasets/bigcode/the-stack-v2",
            license_type="Various", # Requires strict document-level provenance
            metadata={"domain": "code", "tier": "A"}
        ))
        
        # 5. OpenWebMath
        self.tracker.add_source(SourceProvenance(
            source_id="src_openwebmath",
            name="OpenWebMath",
            url="https://huggingface.co/datasets/open-web-math/open-web-math",
            license_type="ODC-BY-1.0",
            metadata={"domain": "math", "tier": "B"}
        ))

    def register_dataset(
        self, 
        dataset_id: str, 
        source_id: str, 
        version: str, 
        language: str, 
        domain: str,
        retrieval_date: Optional[str] = None
    ) -> DatasetProvenance:
        """Registers a specific dataset slice from a source."""
        ds = DatasetProvenance(
            dataset_id=dataset_id,
            source_id=source_id,
            version_or_revision=version,
            language=language,
            domain=domain
        )
        if retrieval_date:
            ds.retrieval_date = retrieval_date
        self.tracker.add_dataset(ds)
        return ds
        
    def get_source(self, source_id: str) -> Optional[SourceProvenance]:
        return self.tracker.sources.get(source_id)
        
    def get_dataset(self, dataset_id: str) -> Optional[DatasetProvenance]:
        return self.tracker.datasets.get(dataset_id)
