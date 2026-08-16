# MyLLM Specification: Persistent Memory Schema

- **Document ID**: `SPEC-0011`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Long-Term Knowledge, Episodic Memory & Vector Indexing
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Architectural Principles

`PersistentMemory` represents durable knowledge, user preferences, enterprise facts, and project context that must survive indefinitely across the 15-year platform lifecycle.

### Invariants & Survival Guarantees:
1. **Decoupled from Ephemeral Sessions**: Deleting, resetting, or timing out a `Session` has zero impact on `PersistentMemory`.
2. **Survives Model Architecture Migrations**: When MyLLM transitions between base models (e.g. from Transformer 7B to MoE 100B or State Space Models), persistent memories remain untouched in canonical text and structured form.
3. **Re-Embeddable Vectors**: Embedding vectors are stored as decoupled references (`embedding_vector_ref`). When embedding models are upgraded, a background batch job re-indexes existing memories without mutating the underlying knowledge content.

---

## 2. Protobuf Specification (`myllm/memory/v1/memory.proto`)

```protobuf
syntax = "proto3";

package myllm.memory.v1;

import "google/protobuf/timestamp.proto";

option go_package = "github.com/myllm/core/gen/go/memory/v1;memoryv1";
option java_package = "ai.myllm.memory.v1";

enum MemoryType {
  MEMORY_TYPE_UNSPECIFIED = 0;
  MEMORY_TYPE_USER = 1;      // User preferences, style, personal profile
  MEMORY_TYPE_PROJECT = 2;   // Codebase conventions, project architecture, team rules
  MEMORY_TYPE_DOCUMENT = 3;  // Ingested documentation, PDFs, manual knowledge bases
  MEMORY_TYPE_RETRIEVAL = 4; // Distilled episodic facts from past agent interactions
}

enum PrivacyScope {
  PRIVACY_SCOPE_UNSPECIFIED = 0;
  PRIVACY_SCOPE_PRIVATE_USER = 1;
  PRIVACY_SCOPE_TEAM_SHARED = 2;
  PRIVACY_SCOPE_ORGANIZATION_PUBLIC = 3;
  PRIVACY_SCOPE_SYSTEM_GLOBAL = 4;
}

// Vector store reference linking memory to semantic search indexes
message EmbeddingVectorRef {
  // Vector identifier in the underlying Vector Database / HNSW index
  string vector_id = 1;
  // Dimensionality (e.g., 768, 1536, 3072)
  uint32 dimension = 2;
  // Canonical identifier of the embedding model used
  string embedding_model_id = 3;
  // Embedding model version / contract hash
  string embedding_version = 4;
  // Distance metric used: "COSINE", "DOT_PRODUCT", "EUCLIDEAN"
  string metric = 5;
}

// Persistent Memory Entity
message PersistentMemory {
  // Canonical Memory UUID
  string memory_id = 1;

  // Owner identifier (user_id, project_id, org_id)
  string owner_id = 2;

  // Category of memory
  MemoryType memory_type = 3;

  // Access control and sharing boundary
  PrivacyScope privacy_scope = 4;

  // Distilled factual content / natural language summary
  string content = 5;

  // Reference to semantic search vector representation
  EmbeddingVectorRef embedding_vector_ref = 6;

  // Confidence score of memory extraction (0.0 to 1.0)
  double confidence_score = 7;

  // Session ID from which this memory was distilled (optional provenance)
  string source_session_id = 8;

  // Custom categorization tags and domain filters
  map<string, string> tags = 9;

  // Creation and modification timestamps
  google.protobuf.Timestamp created_at = 10;
  google.protobuf.Timestamp updated_at = 11;
  // Optional TTL / expiration timestamp (empty for permanent memories)
  google.protobuf.Timestamp expires_at = 12;
}

// Query parameters for semantic memory retrieval
message MemoryQuery {
  string owner_id = 1;
  repeated MemoryType allowed_types = 2;
  string query_text = 3;
  uint32 top_k = 4;
  double min_similarity = 5;
  map<string, string> tag_filters = 6;
}

message MemoryQueryResult {
  PersistentMemory memory = 1;
  double similarity_score = 2;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/memory-schema.json",
  "title": "PersistentMemory",
  "type": "object",
  "required": [
    "memory_id",
    "owner_id",
    "memory_type",
    "privacy_scope",
    "content",
    "created_at",
    "updated_at"
  ],
  "properties": {
    "memory_id": { "type": "string", "format": "uuid" },
    "owner_id": { "type": "string" },
    "memory_type": {
      "type": "string",
      "enum": [
        "MEMORY_TYPE_USER",
        "MEMORY_TYPE_PROJECT",
        "MEMORY_TYPE_DOCUMENT",
        "MEMORY_TYPE_RETRIEVAL"
      ]
    },
    "privacy_scope": {
      "type": "string",
      "enum": [
        "PRIVACY_SCOPE_PRIVATE_USER",
        "PRIVACY_SCOPE_TEAM_SHARED",
        "PRIVACY_SCOPE_ORGANIZATION_PUBLIC",
        "PRIVACY_SCOPE_SYSTEM_GLOBAL"
      ]
    },
    "content": { "type": "string" },
    "embedding_vector_ref": {
      "type": "object",
      "required": ["vector_id", "dimension", "embedding_model_id", "embedding_version"],
      "properties": {
        "vector_id": { "type": "string" },
        "dimension": { "type": "integer" },
        "embedding_model_id": { "type": "string" },
        "embedding_version": { "type": "string" },
        "metric": { "type": "string" }
      }
    },
    "confidence_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "source_session_id": { "type": "string" },
    "tags": {
      "type": "object",
      "additionalProperties": { "type": "string" }
    },
    "created_at": { "type": "string", "format": "date-time" },
    "updated_at": { "type": "string", "format": "date-time" },
    "expires_at": { "type": "string", "format": "date-time" }
  }
}
```

---

## 4. Complete JSON Example

```json
{
  "memory_id": "a4d3e8f1-2c5b-4a78-9012-3456789abcde",
  "owner_id": "usr_99214890",
  "memory_type": "MEMORY_TYPE_PROJECT",
  "privacy_scope": "PRIVACY_SCOPE_TEAM_SHARED",
  "content": "In project 'core-billing', all financial calculations must use decimal floating point or integer cents. Never use IEEE-754 binary floats for money.",
  "embedding_vector_ref": {
    "vector_id": "vec_proj_billing_rule_089",
    "dimension": 1536,
    "embedding_model_id": "myllm/bengali-english-embed-v1",
    "embedding_version": "1.0.0",
    "metric": "COSINE"
  },
  "confidence_score": 0.985,
  "source_session_id": "c8f5e1b2-7a39-4d2b-9e41-0f81a7b45620",
  "tags": {
    "project": "core-billing",
    "topic": "coding_standards",
    "language": "python,rust"
  },
  "created_at": "2026-08-16T10:05:00Z",
  "updated_at": "2026-08-16T10:05:00Z",
  "expires_at": null
}
```
