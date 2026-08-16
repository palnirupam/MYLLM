# MyLLM Specification: Dataset Manifest & Compliance Lineage

- **Document ID**: `SPEC-0008`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Training Data Provenance, Governance & Compliance
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Architectural Principles

The `DatasetManifest` establishes an immutable, cryptographically verifiable record of training, evaluation, and fine-tuning datasets.

### Core Requirements:
1. **End-to-End Lineage**: Every dataset tracks the precise code commit of its builder/curation script, upstream data sources, license compatibility analysis, and preprocessing pipeline.
2. **Deterministic Merkle Root**: The dataset's `content_hash` represents a cryptographic Merkle root across all constituent shard files and token matrices.
3. **Audit-Preserving Tombstoning**: In compliance with global data privacy regulations (GDPR "Right to be Forgotten", CCPA, copyright DMCA take-downs), datasets must support immutable tombstoning. If specific documents are purged, a cryptographically linked `TombstoneRecord` documents *why*, *when*, and *what hash was excised*, without destroying historical audit integrity.

---

## 2. Protobuf Specification (`myllm/dataset/v1/dataset_manifest.proto`)

```protobuf
syntax = "proto3";

package myllm.dataset.v1;

import "google/protobuf/timestamp.proto";

option go_package = "github.com/myllm/core/gen/go/dataset/v1;datasetv1";
option java_package = "ai.myllm.dataset.v1";

// Upstream data source attribution
message DataSourceInfo {
  string source_name = 1;
  string source_uri = 2;
  string original_license = 3;
  string extraction_method = 4; // "crawl", "licensed_corpus", "synthetic_curated"
  uint64 raw_document_count = 5;
  uint64 raw_byte_size = 6;
}

// License compliance and lineage verification
message LicenseLineage {
  repeated string declared_licenses = 1;
  bool permits_commercial_training = 2;
  bool requires_share_alike = 3;
  string legal_review_reference_id = 4;
  string compliance_signoff_identity = 5;
}

// Quality, filtering, and perplexity statistics
message QualityMetrics {
  double mean_heuristic_quality_score = 1;
  double toxicity_flagged_ratio = 2;
  double pii_scrubbed_ratio = 3;
  double mean_doc_length_tokens = 4;
  map<string, double> filter_rejection_rates = 5;
}

// Deduplication metrics across dataset
message DeduplicationStats {
  uint64 exact_hash_duplicates_removed = 1;
  uint64 minhash_near_duplicates_removed = 2;
  double near_duplicate_similarity_threshold = 3; // e.g. 0.85 Jaccard
  uint64 unique_documents_retained = 4;
}

// Legal / PII Tombstone Record
message TombstoneRecord {
  // Unique tombstone event ID
  string tombstone_id = 1;

  // Reason: "GDPR_ERASURE_REQUEST", "DMCA_TAKEDOWN", "DATA_POISONING_PURGE"
  string reason = 2;

  // List of SHA-256 document content hashes excised
  repeated string excised_document_hashes = 3;

  // Legal reference / Ticket ID authorizing removal
  string legal_authorization_id = 4;

  // Timestamp of excision
  google.protobuf.Timestamp excised_at = 5;

  // Hash of the previous DatasetManifest prior to excision
  string parent_dataset_manifest_hash = 6;

  // Authorized officer identity
  string signed_by = 7;
}

// Canonical Dataset Manifest
message DatasetManifest {
  // Canonical Dataset Identifier (e.g., "myllm.datasets.bengali-pretrain-v2:1.0.0")
  string dataset_id = 1;

  // SemVer or content version
  string version = 2;

  // Git commit SHA of the builder/curation codebase
  string builder_script_commit = 3;

  // List of upstream source corpuses
  repeated DataSourceInfo source_descriptions = 4;

  // Complete licensing analysis and signoff
  LicenseLineage license_lineage = 5;

  // Tokenization recipe and pipeline version (e.g. "bengali-byte-bpe-v2")
  string preprocessing_version = 6;

  // Total token count in dataset (as tokenized by preprocessing_version)
  uint64 token_count = 7;

  // Language distribution map (ISO 639-1 / 639-3 code -> token percentage)
  map<string, double> language_distribution = 8;

  // Quantitative quality indicators
  QualityMetrics quality_metrics = 9;

  // Deduplication breakdown
  DeduplicationStats deduplication_stats = 10;

  // SHA-256 Merkle root hash of all dataset shard files
  string content_hash = 11;

  // URI to directory of raw/tokenized shard files (Safetensors / Arrow / Parquet / Megatron Bin)
  string storage_uri = 12;

  // Optional reference to a Tombstone record if this dataset derived from an excision
  TombstoneRecord tombstone_reference = 13;

  // Manifest creation timestamp
  google.protobuf.Timestamp created_at = 14;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/dataset-manifest.json",
  "title": "DatasetManifest",
  "type": "object",
  "required": [
    "dataset_id",
    "version",
    "builder_script_commit",
    "source_descriptions",
    "license_lineage",
    "preprocessing_version",
    "token_count",
    "language_distribution",
    "quality_metrics",
    "deduplication_stats",
    "content_hash",
    "storage_uri",
    "created_at"
  ],
  "properties": {
    "dataset_id": { "type": "string" },
    "version": { "type": "string" },
    "builder_script_commit": { "type": "string", "pattern": "^[a-f0-9]{40}$" },
    "source_descriptions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["source_name", "source_uri", "original_license"],
        "properties": {
          "source_name": { "type": "string" },
          "source_uri": { "type": "string" },
          "original_license": { "type": "string" },
          "extraction_method": { "type": "string" },
          "raw_document_count": { "type": "integer" },
          "raw_byte_size": { "type": "integer" }
        }
      }
    },
    "license_lineage": {
      "type": "object",
      "required": ["declared_licenses", "permits_commercial_training"],
      "properties": {
        "declared_licenses": { "type": "array", "items": { "type": "string" } },
        "permits_commercial_training": { "type": "boolean" },
        "requires_share_alike": { "type": "boolean" },
        "legal_review_reference_id": { "type": "string" },
        "compliance_signoff_identity": { "type": "string" }
      }
    },
    "preprocessing_version": { "type": "string" },
    "token_count": { "type": "integer" },
    "language_distribution": {
      "type": "object",
      "additionalProperties": { "type": "number" }
    },
    "quality_metrics": { "type": "object" },
    "deduplication_stats": { "type": "object" },
    "content_hash": { "type": "string", "pattern": "^sha256:[a-f0-9]{64}$" },
    "storage_uri": { "type": "string" },
    "tombstone_reference": { "type": "object" },
    "created_at": { "type": "string", "format": "date-time" }
  }
}
```

---

## 4. Complete JSON Example

```json
{
  "dataset_id": "myllm.datasets.bengali-english-pretrain:2.0.0",
  "version": "2.0.0",
  "builder_script_commit": "e8d3b9c01f2a3456789abcdef0123456789abcde",
  "source_descriptions": [
    {
      "source_name": "Bengali Wikipedia & Wikibooks Dump 2026-06",
      "source_uri": "https://dumps.wikimedia.org/bnwiki/",
      "original_license": "CC-BY-SA-4.0",
      "extraction_method": "licensed_corpus",
      "raw_document_count": 284000,
      "raw_byte_size": 2147483648
    },
    {
      "source_name": "National Curated Bengali Open Literature & News",
      "source_uri": "s3://myllm-internal-raw/bengali-curated-2026/",
      "original_license": "Public Domain / CC0",
      "extraction_method": "crawl",
      "raw_document_count": 14200000,
      "raw_byte_size": 85899345920
    }
  ],
  "license_lineage": {
    "declared_licenses": ["CC-BY-SA-4.0", "CC0", "Apache-2.0"],
    "permits_commercial_training": true,
    "requires_share_alike": false,
    "legal_review_reference_id": "LEGAL-2026-AUG-882",
    "compliance_signoff_identity": "legal-compliance-officer@myllm.internal"
  },
  "preprocessing_version": "bengali-bpe-vocab128k-v2",
  "token_count": 25000000000,
  "language_distribution": {
    "ben": 0.65,
    "eng": 0.30,
    "code": 0.05
  },
  "quality_metrics": {
    "mean_heuristic_quality_score": 0.942,
    "toxicity_flagged_ratio": 0.0004,
    "pii_scrubbed_ratio": 0.0182,
    "mean_doc_length_tokens": 1480.5,
    "filter_rejection_rates": {
      "short_doc": 0.12,
      "low_perplexity": 0.04,
      "toxicity": 0.001
    }
  },
  "deduplication_stats": {
    "exact_hash_duplicates_removed": 4201900,
    "minhash_near_duplicates_removed": 1890200,
    "near_duplicate_similarity_threshold": 0.85,
    "unique_documents_retained": 12800000
  },
  "content_hash": "sha256:112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
  "storage_uri": "s3://myllm-datasets/tokenized/bengali-english-pretrain-v2/",
  "tombstone_reference": null,
  "created_at": "2026-08-16T04:00:00Z"
}
```
