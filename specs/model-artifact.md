# MyLLM Specification: Model Artifact

- **Document ID**: `SPEC-0002`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Packaging, Distribution & Cryptographic Provenance
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Core Philosophy

A `ModelArtifact` represents a sealed, immutable, content-addressed distribution package for a model. It bundles all components necessary to reconstruct, verify, and execute a model instance.

### Invariants:
1. **Content-Addressable Identity**: The primary key `artifact_id` is computed strictly as the SHA-256 digest of the canonical serialization of the package metadata and its referenced contents.
2. **Cryptographic Provenance**: Every artifact MUST contain a non-repudiable signature and a deterministic lineage linking it to the training run, code commit, dataset manifest, and tokenizer artifact.
3. **OCI Distribution Independence**: While MyLLM defines mappings to OCI (Open Container Initiative) artifact registries for ecosystem distribution, the `ModelArtifact` contract is self-contained and operates independently of any specific container registry or transport protocol.

---

## 2. Protobuf Specification (`myllm/artifact/v1/model_artifact.proto`)

```protobuf
syntax = "proto3";

package myllm.artifact.v1;

import "google/protobuf/timestamp.proto";

option go_package = "github.com/myllm/core/gen/go/artifact/v1;artifactv1";
option java_package = "ai.myllm.artifact.v1";

// Individual weight shard or supplementary file entry
message ArtifactFileRef {
  // Relative path or shard identifier within the artifact package
  string path = 1;
  // Exact byte length
  uint64 size_bytes = 2;
  // Hex-encoded SHA-256 checksum of the uncompressed file
  string sha256_checksum = 3;
  // Storage URI (e.g., s3://..., file://..., oci://...)
  string uri = 4;
  // MIME/Content type (e.g., "application/x-safetensors", "application/json")
  string mime_type = 5;
}

// OCI Registry Distribution Mapping
message OCIDistribution {
  // OCI Manifest Media Type (e.g., "application/vnd.myllm.model.v1+json")
  string media_type = 1;
  // Registry endpoint (e.g., "registry.myllm.internal:5000" or "ghcr.io")
  string registry = 2;
  // Repository path (e.g., "myllm-artifacts/bengali-reasoning-7b")
  string repository = 3;
  // Distribution tag (e.g., "1.0.0", "latest")
  string tag = 4;
  // Exact OCI Manifest Digest (e.g., "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
  string digest = 5;
}

// Training & Build Lineage
message ProvenanceChain {
  // Unique training run identifier
  string training_run_id = 1;
  // Checkpoint iteration/step number from which this artifact was sealed
  uint64 checkpoint_step = 2;
  // Full Git commit SHA of the training and modeling codebase
  string code_commit_hash = 3;
  // Content hash of the DatasetManifest used for final training/alignment
  string dataset_manifest_hash = 4;
  // Builder/orchestrator identity (CI/CD pipeline ID, worker node, human sign-off)
  string builder_identity = 5;
  // Build environment metadata (compiler versions, CUDA driver, PyTorch version)
  map<string, string> build_environment = 6;
  // UTC timestamp when build was completed
  google.protobuf.Timestamp build_timestamp = 7;
}

// Digital Signature and Trust Attestation
message CryptographicSignature {
  // Algorithm used: "ED25519", "ECDSA_SHA256", "COSIGN_SIGSTORE"
  string signature_algorithm = 1;
  // Public Key ID or Key URI used for verification
  string public_key_id = 2;
  // Base64-encoded digital signature over canonical artifact metadata
  string signature_base64 = 3;
  // X.509 certificate chain or Rekor transparency log bundle (PEM format)
  repeated string certificate_chain = 4;
  // Timestamp when signature was affixed
  google.protobuf.Timestamp signed_at = 5;
}

// Top-level Model Artifact definition
message ModelArtifact {
  // Canonical Content-Addressable Hash (e.g., "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069")
  string artifact_id = 1;

  // Exact digest string (algorithm:hex)
  string artifact_digest = 2;

  // Logical Model Contract Identifier (e.g., "myllm/bengali-reasoning-7b:1.0.0")
  string model_id = 3;

  // Reference to Architecture Config artifact
  ArtifactFileRef architecture_config_ref = 4;

  // Reference to Tensor Manifest artifact
  ArtifactFileRef tensor_manifest_ref = 5;

  // Reference to Tokenizer artifact (vocab, merges, tokenizer config)
  ArtifactFileRef tokenizer_ref = 6;

  // List of weight shard files (Safetensors format)
  repeated ArtifactFileRef weight_shards = 7;

  // Supplementary artifacts (eval results, system prompt templates, license text)
  repeated ArtifactFileRef supplementary_files = 8;

  // OCI Distribution coordinates (optional, present when distributed via OCI)
  OCIDistribution oci_distribution = 9;

  // Complete end-to-end lineage and build provenance
  ProvenanceChain provenance = 10;

  // Cryptographic attestations and signatures
  repeated CryptographicSignature signatures = 11;

  // SPDX License and copyright notices
  string license_spdx = 12;
  string license_text_ref = 13;

  // Metadata creation timestamp
  google.protobuf.Timestamp created_at = 14;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/model-artifact.json",
  "title": "ModelArtifact",
  "type": "object",
  "required": [
    "artifact_id",
    "artifact_digest",
    "model_id",
    "architecture_config_ref",
    "tensor_manifest_ref",
    "tokenizer_ref",
    "weight_shards",
    "provenance",
    "signatures",
    "license_spdx",
    "created_at"
  ],
  "properties": {
    "artifact_id": { "type": "string", "pattern": "^sha256:[a-f0-9]{64}$" },
    "artifact_digest": { "type": "string" },
    "model_id": { "type": "string" },
    "architecture_config_ref": { "$ref": "#/$defs/ArtifactFileRef" },
    "tensor_manifest_ref": { "$ref": "#/$defs/ArtifactFileRef" },
    "tokenizer_ref": { "$ref": "#/$defs/ArtifactFileRef" },
    "weight_shards": {
      "type": "array",
      "items": { "$ref": "#/$defs/ArtifactFileRef" },
      "minItems": 1
    },
    "supplementary_files": {
      "type": "array",
      "items": { "$ref": "#/$defs/ArtifactFileRef" }
    },
    "oci_distribution": {
      "type": "object",
      "properties": {
        "media_type": { "type": "string" },
        "registry": { "type": "string" },
        "repository": { "type": "string" },
        "tag": { "type": "string" },
        "digest": { "type": "string" }
      }
    },
    "provenance": {
      "type": "object",
      "required": ["training_run_id", "checkpoint_step", "code_commit_hash", "dataset_manifest_hash", "builder_identity"],
      "properties": {
        "training_run_id": { "type": "string" },
        "checkpoint_step": { "type": "integer" },
        "code_commit_hash": { "type": "string" },
        "dataset_manifest_hash": { "type": "string" },
        "builder_identity": { "type": "string" },
        "build_environment": { "type": "object", "additionalProperties": { "type": "string" } },
        "build_timestamp": { "type": "string", "format": "date-time" }
      }
    },
    "signatures": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["signature_algorithm", "public_key_id", "signature_base64", "signed_at"],
        "properties": {
          "signature_algorithm": { "type": "string" },
          "public_key_id": { "type": "string" },
          "signature_base64": { "type": "string" },
          "certificate_chain": { "type": "array", "items": { "type": "string" } },
          "signed_at": { "type": "string", "format": "date-time" }
        }
      },
      "minItems": 1
    },
    "license_spdx": { "type": "string" },
    "license_text_ref": { "type": "string" },
    "created_at": { "type": "string", "format": "date-time" }
  },
  "$defs": {
    "ArtifactFileRef": {
      "type": "object",
      "required": ["path", "size_bytes", "sha256_checksum", "uri", "mime_type"],
      "properties": {
        "path": { "type": "string" },
        "size_bytes": { "type": "integer" },
        "sha256_checksum": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
        "uri": { "type": "string" },
        "mime_type": { "type": "string" }
      }
    }
  }
}
```

---

## 4. Complete JSON Example

```json
{
  "artifact_id": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "artifact_digest": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "model_id": "myllm/bengali-reasoning-7b:1.0.0",
  "architecture_config_ref": {
    "path": "config/architecture_config.json",
    "size_bytes": 1420,
    "sha256_checksum": "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
    "uri": "s3://myllm-artifacts/models/bengali-reasoning-7b/1.0.0/architecture_config.json",
    "mime_type": "application/json"
  },
  "tensor_manifest_ref": {
    "path": "weights/tensor_manifest.json",
    "size_bytes": 48290,
    "sha256_checksum": "3a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b",
    "uri": "s3://myllm-artifacts/models/bengali-reasoning-7b/1.0.0/tensor_manifest.json",
    "mime_type": "application/json"
  },
  "tokenizer_ref": {
    "path": "tokenizer/tokenizer.json",
    "size_bytes": 2450128,
    "sha256_checksum": "8f4c2b9a7812de4f9011ba2134567890abcdef1234567890abcdef1234567890",
    "uri": "s3://myllm-artifacts/models/bengali-reasoning-7b/1.0.0/tokenizer.json",
    "mime_type": "application/json"
  },
  "weight_shards": [
    {
      "path": "weights/model-00001-of-00002.safetensors",
      "size_bytes": 7450201920,
      "sha256_checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "uri": "s3://myllm-artifacts/models/bengali-reasoning-7b/1.0.0/model-00001-of-00002.safetensors",
      "mime_type": "application/x-safetensors"
    },
    {
      "path": "weights/model-00002-of-00002.safetensors",
      "size_bytes": 7450198400,
      "sha256_checksum": "ca978112ca1bbdcafac231b39a23dc4da786081cd1e14dd6da60ef4e98b50e67",
      "uri": "s3://myllm-artifacts/models/bengali-reasoning-7b/1.0.0/model-00002-of-00002.safetensors",
      "mime_type": "application/x-safetensors"
    }
  ],
  "supplementary_files": [
    {
      "path": "LICENSE",
      "size_bytes": 11358,
      "sha256_checksum": "2b8b815229aa8a61e483fb4ba0588b8b6c491890278c440db8d505010e6a83b4",
      "uri": "s3://myllm-artifacts/models/bengali-reasoning-7b/1.0.0/LICENSE",
      "mime_type": "text/plain"
    }
  ],
  "oci_distribution": {
    "media_type": "application/vnd.myllm.model.artifact.v1+json",
    "registry": "registry.myllm.internal",
    "repository": "models/bengali-reasoning-7b",
    "tag": "1.0.0",
    "digest": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
  },
  "provenance": {
    "training_run_id": "run-2026-08-14-bengali-7b-v1-stage3",
    "checkpoint_step": 125000,
    "code_commit_hash": "4a5e3d7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e",
    "dataset_manifest_hash": "sha256:112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
    "builder_identity": "ci-release-pipeline@myllm.internal",
    "build_environment": {
      "cuda_version": "12.6",
      "driver_version": "560.35.03",
      "pytorch_version": "2.5.0a0+git",
      "python_version": "3.12.4"
    },
    "build_timestamp": "2026-08-16T08:30:00Z"
  },
  "signatures": [
    {
      "signature_algorithm": "ED25519",
      "public_key_id": "keys/prod-model-signing-2026.pub",
      "signature_base64": "G5e3u9fQW...====",
      "certificate_chain": [],
      "signed_at": "2026-08-16T08:35:00Z"
    }
  ],
  "license_spdx": "Apache-2.0",
  "license_text_ref": "LICENSE",
  "created_at": "2026-08-16T08:30:00Z"
}
```

---

## 5. Ingestion & Cryptographic Verification Workflow

```
       +---------------------------------------------+
       |   Fetch ModelArtifact JSON / OCI Manifest   |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |  Verify Signature with Trusted Public Key   |
       |  (Reject immediately if invalid or forged) |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |  Stream / Fetch Each Weight Shard & Config  |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Compute Incremental SHA-256 for Each File |
       |   Match against ArtifactFileRef.sha256      |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       | Pass Validated Shards to Storage / Runtime  |
       +---------------------------------------------+
```

1. **Signature Verification**: Before streaming gigabytes of weights, the signature over the manifest is validated against the root-of-trust keyring.
2. **File Checksums**: During download, streaming SHA-256 calculators ensure byte-level integrity against corrupted or substituted shards.
3. **Zero-Pickle Enforcement**: Any file lacking MIME type `application/x-safetensors` or `application/json` is blocked from weight execution pathways.
