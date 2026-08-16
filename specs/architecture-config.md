# MyLLM Specification: Architecture Configuration

- **Document ID**: `SPEC-0003`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Model Architecture Topology
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Separation Rationale

The `ArchitectureConfig` defines the structural, mathematical, and topological hyperparameters required by an inference engine or training framework to construct a neural network computational graph.

### Architectural Decoupling:
- **`ArchitectureConfig` (This Spec)**: Understands layers, hidden dimensions (`d_model`), attention heads (`n_heads`, `n_kv_heads`), activation functions, RoPE theta, normalization epsilon, and MoE routing logic.
- **`TensorManifest` (`SPEC-0004`)**: Completely generic. It treats tensors as raw named multi-dimensional arrays stored in byte slices. It has zero knowledge of transformer heads, KV projections, or expert dispatch.

This strict separation ensures that future architectures (e.g., State Space Models like Mamba, linear attention, sparse Mixture-of-Experts, or neuromorphic graphs) can introduce new architecture configuration schemas without modifying the storage, sharding, or serialization mechanics of tensor storage.

---

## 2. Protobuf Specification (`myllm/architecture/v1/architecture_config.proto`)

```protobuf
syntax = "proto3";

package myllm.architecture.v1;

option go_package = "github.com/myllm/core/gen/go/architecture/v1;architecturev1";
option java_package = "ai.myllm.architecture.v1";

enum NormalizationType {
  NORM_TYPE_UNSPECIFIED = 0;
  NORM_TYPE_RMS_NORM = 1;
  NORM_TYPE_LAYER_NORM = 2;
  NORM_TYPE_GROUP_NORM = 3;
  NORM_TYPE_RMS_NORM_ZERO_CENTERED = 4;
}

enum ActivationFunction {
  ACTIVATION_UNSPECIFIED = 0;
  ACTIVATION_SWIGLU = 1;
  ACTIVATION_GEGLU = 2;
  ACTIVATION_GELU = 3;
  ACTIVATION_GELU_PYTORCH_TANH = 4;
  ACTIVATION_SILU = 5;
  ACTIVATION_RELU = 6;
}

enum PositionalEncodingType {
  POS_ENCODING_UNSPECIFIED = 0;
  POS_ENCODING_ROPE = 1;
  POS_ENCODING_ALIBI = 2;
  POS_ENCODING_LEARNED_ABSOLUTE = 3;
  POS_ENCODING_NOPE = 4; // No positional embeddings
}

message RoPEScalingConfig {
  // Scaling type: "linear", "dynamic", "yarn", "llama3"
  string scaling_type = 1;
  double factor = 2;
  double low_freq_factor = 3;
  double high_freq_factor = 4;
  uint32 original_max_position_embeddings = 5;
  double beta_fast = 6;
  double beta_slow = 7;
  double mscale = 8;
  double mscale_all_dim = 9;
}

// Configuration for standard Dense Decoder-Only Transformer (e.g., Llama, Mistral, Qwen)
message DecoderOnlyTransformerConfig {
  uint32 vocab_size = 1;
  uint32 d_model = 2;
  uint32 n_layers = 3;
  uint32 n_heads = 4;
  uint32 n_kv_heads = 5; // Equals n_heads for MHA, < n_heads for GQA, 1 for MQA
  uint32 intermediate_size = 6; // FFN hidden dimension
  uint32 max_seq_len = 7;
  NormalizationType norm_type = 8;
  double norm_eps = 9;
  ActivationFunction activation = 10;
  PositionalEncodingType positional_encoding = 11;
  double rope_theta = 12;
  RoPEScalingConfig rope_scaling = 13;
  bool tie_word_embeddings = 14;
  bool use_bias = 15;
  uint32 head_dim = 16; // If 0, computed as d_model / n_heads
}

// Configuration for Mixture-of-Experts Transformer (e.g., Mixtral, DeepSeek-V2/V3)
message MoETransformerConfig {
  DecoderOnlyTransformerConfig base_transformer = 1;
  uint32 num_experts = 2;
  uint32 num_experts_per_tok = 3;
  uint32 num_shared_experts = 4;
  uint32 moe_intermediate_size = 5;
  double router_aux_loss_coef = 6;
  bool norm_topk_prob = 7;
  string scoring_func = 8; // "softmax", "sigmoid"
}

// Configuration for State Space Models (e.g., Mamba, Mamba-2)
message StateSpaceModelConfig {
  uint32 vocab_size = 1;
  uint32 d_model = 2;
  uint32 n_layers = 3;
  uint32 d_state = 4; // SSM state expansion factor (e.g., 16, 64, 128)
  uint32 d_conv = 5;  // 1D convolution kernel width (e.g., 4)
  uint32 expand = 6;  // Block expansion factor (e.g., 2)
  uint32 dt_rank = 7; // Rank of delta projection
  NormalizationType norm_type = 8;
  double norm_eps = 9;
  bool tie_word_embeddings = 10;
}

// Top-level Architecture Configuration container
message ArchitectureConfig {
  // Schema version (SemVer, e.g., "1.0.0")
  string schema_version = 1;

  // Family identifier matching ModelContract
  string architecture_family = 2;

  // Architecture subtype name (e.g., "decoder_only_transformer", "moe_transformer", "mamba")
  string architecture_name = 3;

  // Family-specific payload
  oneof config_payload {
    DecoderOnlyTransformerConfig decoder_only_transformer = 10;
    MoETransformerConfig moe_transformer = 11;
    StateSpaceModelConfig state_space_model = 12;
  }

  // Custom extension parameters for experimental kernel configurations
  map<string, string> custom_parameters = 20;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/architecture-config.json",
  "title": "ArchitectureConfig",
  "type": "object",
  "required": ["schema_version", "architecture_family", "architecture_name"],
  "properties": {
    "schema_version": { "type": "string" },
    "architecture_family": { "type": "string" },
    "architecture_name": { "type": "string" },
    "decoder_only_transformer": {
      "type": "object",
      "required": [
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "intermediate_size",
        "max_seq_len",
        "norm_type",
        "norm_eps",
        "activation",
        "positional_encoding"
      ],
      "properties": {
        "vocab_size": { "type": "integer" },
        "d_model": { "type": "integer" },
        "n_layers": { "type": "integer" },
        "n_heads": { "type": "integer" },
        "n_kv_heads": { "type": "integer" },
        "intermediate_size": { "type": "integer" },
        "max_seq_len": { "type": "integer" },
        "norm_type": { "type": "string" },
        "norm_eps": { "type": "number" },
        "activation": { "type": "string" },
        "positional_encoding": { "type": "string" },
        "rope_theta": { "type": "number" },
        "rope_scaling": { "type": "object" },
        "tie_word_embeddings": { "type": "boolean" },
        "use_bias": { "type": "boolean" },
        "head_dim": { "type": "integer" }
      }
    },
    "moe_transformer": { "type": "object" },
    "state_space_model": { "type": "object" },
    "custom_parameters": {
      "type": "object",
      "additionalProperties": { "type": "string" }
    }
  }
}
```

---

## 4. Complete JSON Example (Decoder-Only Transformer)

```json
{
  "schema_version": "1.0.0",
  "architecture_family": "ARCHITECTURE_FAMILY_TRANSFORMER_DECODER",
  "architecture_name": "decoder_only_transformer",
  "decoder_only_transformer": {
    "vocab_size": 128256,
    "d_model": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "intermediate_size": 14336,
    "max_seq_len": 131072,
    "norm_type": "NORM_TYPE_RMS_NORM",
    "norm_eps": 1e-5,
    "activation": "ACTIVATION_SWIGLU",
    "positional_encoding": "POS_ENCODING_ROPE",
    "rope_theta": 500000.0,
    "rope_scaling": {
      "scaling_type": "llama3",
      "factor": 8.0,
      "low_freq_factor": 1.0,
      "high_freq_factor": 4.0,
      "original_max_position_embeddings": 8192
    },
    "tie_word_embeddings": false,
    "use_bias": false,
    "head_dim": 128
  },
  "custom_parameters": {
    "attention_dropout": "0.0",
    "sliding_window": "0"
  }
}
```
