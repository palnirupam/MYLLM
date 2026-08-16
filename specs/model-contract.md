# MyLLM Specification: Model Contract

- **Document ID**: `SPEC-0001`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Model Registry & Lifecycle
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Core Philosophy

The `ModelContract` establishes an immutable, architecture-agnostic contract between a trained model and the MyLLM runtime ecosystem. Under the 15-year platform horizon, model architectures (e.g., dense transformers, Mixture-of-Experts, State Space Models, recurrent-attention hybrids) will change frequently. The `ModelContract` ensures that the runtime, serving layer, evaluation pipelines, and orchestration agents interact with a stable schema, regardless of internal model topology.

### Invariants:
1. **Contract Stability**: The `ModelContract` is fully decoupled from tensor storage layouts and internal hardware execution kernels.
2. **Immutability**: Once a `ModelContract` instance is published with a given `model_id` and content digest, its properties can never be mutated.
3. **Explicit Capabilities**: Models declare capabilities via an extensible name-version-property registry, eliminating rigid boolean feature flags.

---

## 2. Protobuf Specification (`myllm/model/v1/model_contract.proto`)

```protobuf
syntax = "proto3";

package myllm.model.v1;

import "google/protobuf/timestamp.proto";
import "myllm/capability/v1/capability.proto";

option go_package = "github.com/myllm/core/gen/go/model/v1;modelv1";
option java_package = "ai.myllm.model.v1";

// High-level architecture category
enum ArchitectureFamily {
  ARCHITECTURE_FAMILY_UNSPECIFIED = 0;
  ARCHITECTURE_FAMILY_TRANSFORMER_DECODER = 1;
  ARCHITECTURE_FAMILY_TRANSFORMER_ENCODER = 2;
  ARCHITECTURE_FAMILY_TRANSFORMER_ENCODER_DECODER = 3;
  ARCHITECTURE_FAMILY_MIXTURE_OF_EXPERTS = 4;
  ARCHITECTURE_FAMILY_STATE_SPACE_MODEL = 5;
  ARCHITECTURE_FAMILY_HYBRID_SSM_TRANSFORMER = 6;
  ARCHITECTURE_FAMILY_RECURRENT = 7;
}

// Precision & quantization formats supported natively
enum ModelPrecision {
  PRECISION_UNSPECIFIED = 0;
  PRECISION_FP32 = 1;
  PRECISION_FP16 = 2;
  PRECISION_BF16 = 3;
  PRECISION_FP8_E4M3 = 4;
  PRECISION_FP8_E5M2 = 5;
  PRECISION_INT8 = 6;
  PRECISION_INT4 = 7;
}

// License policy and commercial constraint metadata
message LicensePolicy {
  // SPDX identifier (e.g., "Apache-2.0", "MIT", "Llama-3.1", "Proprietary")
  string spdx_identifier = 1;
  // Permitted usage contexts: ["commercial", "research", "internal-only"]
  repeated string permitted_contexts = 2;
  // Specific attribution or custom legal notice
  string custom_notice = 3;
  // Explicit flag if weights have non-compete or user-count caps
  bool has_commercial_restrictions = 4;
}

// Minimum and recommended execution constraints
message HardwareRequirement {
  // Minimum accelerator VRAM in megabytes for base FP/BF precision
  uint64 min_vram_mb = 1;
  // Recommended accelerator VRAM in megabytes (including KV-cache headroom)
  uint64 recommended_vram_mb = 2;
  // Supported precision formats
  repeated ModelPrecision supported_precisions = 3;
  // Minimum CPU RAM in megabytes for host staging
  uint64 min_host_ram_mb = 4;
  // Minimum compute capability (e.g., "sm_80", "sm_90", "rocm_gfx90a", "metal_3")
  repeated string required_compute_capabilities = 5;
}

// Compatibility constraints for runtimes and tooling
message CompatibilityMetadata {
  // Minimum MyLLM runtime engine version required (SemVer)
  string min_runtime_version = 1;
  // Maximum tested runtime engine version (optional)
  string max_tested_runtime_version = 2;
  // Hardware and precision profile
  HardwareRequirement hardware_requirements = 3;
  // Environment variable overrides or kernel optimization flags required
  map<string, string> runtime_flags = 4;
}

// Canonical Model Contract
message ModelContract {
  // Canonical Model Identifier (e.g., "myllm/bengali-reasoning-7b:1.2.0")
  string model_id = 1;

  // Semantic architecture classification
  ArchitectureFamily architecture_family = 2;

  // Architecture contract specification version (e.g., "transformer-decoder-v1")
  string architecture_version = 3;

  // Extensible capability declarations (coding, reasoning, long_context, etc.)
  myllm.capability.v1.Capabilities capabilities = 4;

  // Content-addressable identifier / URI of the Tokenizer Contract artifact
  string tokenizer_version = 5;

  // Content-addressable identifier / URI of the Tensor Manifest artifact
  string tensor_manifest_version = 6;

  // Content-addressable identifier / URI of the Architecture Config artifact
  string architecture_config_version = 7;

  // Licensing and legal policy
  LicensePolicy license_policy = 8;

  // Optional parent model ID for derived, fine-tuned, or quantized models
  string parent_model = 9;

  // Execution and runtime compatibility constraints
  CompatibilityMetadata compatibility = 10;

  // Creation timestamp (UTC)
  google.protobuf.Timestamp created_at = 11;

  // Cryptographic author/publisher identity (e.g., key ID, identity URI)
  string publisher_identity = 12;

  // Human-readable description and release notes
  string description = 13;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/model-contract.json",
  "title": "ModelContract",
  "type": "object",
  "required": [
    "model_id",
    "architecture_family",
    "architecture_version",
    "capabilities",
    "tokenizer_version",
    "tensor_manifest_version",
    "architecture_config_version",
    "license_policy",
    "compatibility"
  ],
  "properties": {
    "model_id": {
      "type": "string",
      "pattern": "^[a-z0-9_-]+/[a-z0-9_.-]+:[a-z0-9_.-]+$"
    },
    "architecture_family": {
      "type": "string",
      "enum": [
        "ARCHITECTURE_FAMILY_UNSPECIFIED",
        "ARCHITECTURE_FAMILY_TRANSFORMER_DECODER",
        "ARCHITECTURE_FAMILY_TRANSFORMER_ENCODER",
        "ARCHITECTURE_FAMILY_TRANSFORMER_ENCODER_DECODER",
        "ARCHITECTURE_FAMILY_MIXTURE_OF_EXPERTS",
        "ARCHITECTURE_FAMILY_STATE_SPACE_MODEL",
        "ARCHITECTURE_FAMILY_HYBRID_SSM_TRANSFORMER",
        "ARCHITECTURE_FAMILY_RECURRENT"
      ]
    },
    "architecture_version": { "type": "string" },
    "capabilities": {
      "type": "object",
      "required": ["capabilities"],
      "properties": {
        "capabilities": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
              "name": { "type": "string" },
              "version": { "type": "string" },
              "properties": {
                "type": "object",
                "additionalProperties": { "type": "string" }
              }
            }
          }
        }
      }
    },
    "tokenizer_version": { "type": "string" },
    "tensor_manifest_version": { "type": "string" },
    "architecture_config_version": { "type": "string" },
    "license_policy": {
      "type": "object",
      "required": ["spdx_identifier", "permitted_contexts"],
      "properties": {
        "spdx_identifier": { "type": "string" },
        "permitted_contexts": {
          "type": "array",
          "items": { "type": "string" }
        },
        "custom_notice": { "type": "string" },
        "has_commercial_restrictions": { "type": "boolean" }
      }
    },
    "parent_model": { "type": "string" },
    "compatibility": {
      "type": "object",
      "required": ["min_runtime_version", "hardware_requirements"],
      "properties": {
        "min_runtime_version": { "type": "string" },
        "max_tested_runtime_version": { "type": "string" },
        "hardware_requirements": {
          "type": "object",
          "required": ["min_vram_mb", "supported_precisions"],
          "properties": {
            "min_vram_mb": { "type": "integer" },
            "recommended_vram_mb": { "type": "integer" },
            "supported_precisions": {
              "type": "array",
              "items": { "type": "string" }
            },
            "min_host_ram_mb": { "type": "integer" },
            "required_compute_capabilities": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        },
        "runtime_flags": {
          "type": "object",
          "additionalProperties": { "type": "string" }
        }
      }
    },
    "created_at": { "type": "string", "format": "date-time" },
    "publisher_identity": { "type": "string" },
    "description": { "type": "string" }
  }
}
```

---

## 4. Example Instance

```json
{
  "model_id": "myllm/bengali-reasoning-7b:1.0.0",
  "architecture_family": "ARCHITECTURE_FAMILY_TRANSFORMER_DECODER",
  "architecture_version": "transformer-decoder-v1",
  "capabilities": {
    "capabilities": [
      {
        "name": "reasoning",
        "version": "1.0.0",
        "properties": {
          "cot_prompt_template": "deepseek_r1_style",
          "supports_backtracking": "true"
        }
      },
      {
        "name": "bengali",
        "version": "2.1.0",
        "properties": {
          "script_support": "Bengali,Latin",
          "dialect_coverage": "Standard,Dhaka,Chittagong,Sylhet"
        }
      },
      {
        "name": "long_context",
        "version": "1.0.0",
        "properties": {
          "max_tokens": "131072",
          "rope_scaling": "yarn"
        }
      },
      {
        "name": "coding",
        "version": "1.0.0",
        "properties": {
          "languages": "python,rust,go,typescript,sql"
        }
      }
    ]
  },
  "tokenizer_version": "sha256:8f4c2b9a7812de4f9011ba2134567890abcdef1234567890abcdef1234567890",
  "tensor_manifest_version": "sha256:3a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b",
  "architecture_config_version": "sha256:d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
  "license_policy": {
    "spdx_identifier": "Apache-2.0",
    "permitted_contexts": ["commercial", "research", "internal-only"],
    "custom_notice": "Copyright 2026 MyLLM Contributors. No proprietary data used.",
    "has_commercial_restrictions": false
  },
  "parent_model": "",
  "compatibility": {
    "min_runtime_version": "1.0.0",
    "max_tested_runtime_version": "1.99.0",
    "hardware_requirements": {
      "min_vram_mb": 16384,
      "recommended_vram_mb": 24576,
      "supported_precisions": ["PRECISION_BF16", "PRECISION_FP8_E4M3", "PRECISION_INT4"],
      "min_host_ram_mb": 32768,
      "required_compute_capabilities": ["sm_80", "sm_89", "sm_90", "rocm_gfx90a"]
    },
    "runtime_flags": {
      "ENABLE_FLASH_ATTN_3": "1",
      "KV_CACHE_DTYPE": "fp8"
    }
  },
  "created_at": "2026-08-16T10:00:00Z",
  "publisher_identity": "did:key:z6MkuT...myllm-core-signer",
  "description": "Production 7B bilingual (Bengali-English) reasoning and coding model."
}
```

---

## 5. Lifecycle & Verification Rules

1. **Validation Pipeline**:
   - The runtime MUST resolve `tokenizer_version`, `tensor_manifest_version`, and `architecture_config_version` against the content-addressable storage.
   - The runtime MUST verify that the declared capabilities match the actual evaluated capabilities before serving traffic.
2. **Deprecation**:
   - When a model contract is deprecated, it is marked with a tombstone event in the registry. Existing inferences remain functional; new fine-tuning runs must target a newer contract.
