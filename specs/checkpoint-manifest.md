# MyLLM Specification: Checkpoint Manifest

- **Document ID**: `SPEC-0009`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Training State Checkpointing & Fault Tolerance
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Dual Loading Philosophy

The `CheckpointManifest` captures the complete, atomic snapshot of an active training run.

### Dual-Use Loading Contract:
1. **`resume_exactly` (Training Fault Recovery)**: Restores the full mathematical state of the cluster, including weight tensors, distributed optimizer states (e.g. AdamW first/second moments, ZeRO/FSDP shards), learning rate scheduler steps, per-rank random number generator (RNG) seeds, and exact dataset stream position (epoch, shard, sample, token index).
2. **`weights_only_load` (Inference & Evaluation Staging)**: Allows inference engines and automated evaluation harnesses to extract purely the model weights (in zero-copy Safetensors format) and tokenizer configuration, completely ignoring gigabytes of optimizer and RNG overhead.

---

## 2. Protobuf Specification (`myllm/checkpoint/v1/checkpoint_manifest.proto`)

```protobuf
syntax = "proto3";

package myllm.checkpoint.v1;

import "google/protobuf/timestamp.proto";

option go_package = "github.com/myllm/core/gen/go/checkpoint/v1;checkpointv1";
option java_package = "ai.myllm.checkpoint.v1";

// Dataset data loader stream cursor
message DataStreamPosition {
  uint32 epoch = 1;
  uint32 shard_index = 2;
  uint64 sample_index = 3;
  uint64 token_index = 4;
  string dataset_manifest_hash = 5;
}

// Per-rank PRNG state capture
message RNGStateSnapshot {
  uint32 global_seed = 1;
  // Per-node/per-GPU serialized RNG byte states (CUDA, PyTorch, Python random)
  map<string, string> rank_rng_states_base64 = 2;
}

// Cluster and compute hardware execution topology
message HardwareTopology {
  string accelerator_model = 1; // e.g. "NVIDIA_H100_SXM5_80GB"
  uint32 total_node_count = 2;
  uint32 accelerators_per_node = 3;
  uint32 world_size = 4;
  string interconnect_type = 5; // e.g. "NVLink-4_InfiniBand-NDR400"
  string driver_version = 6;
  string cuda_version = 7;
}

// Checkpoint Manifest
message CheckpointManifest {
  // Unique training run identifier
  string run_id = 1;

  // Exact step / iteration count
  uint64 checkpoint_step = 2;

  // Exact epoch (float representation)
  double epoch = 3;

  // Draft model version (e.g., "myllm/bengali-7b:step-125000-draft")
  string model_version_draft = 4;

  // Dataset manifest content hash
  string dataset_version = 5;

  // Tokenizer contract / artifact content hash
  string tokenizer_version = 6;

  // Architecture config artifact content hash
  string architecture_version = 7;

  // SHA-256 hash of training hyperparameter configuration JSON
  string training_config_hash = 8;

  // Git commit SHA of the training codebase
  string code_commit = 9;

  // Physical hardware and cluster configuration
  HardwareTopology hardware_description = 10;

  // Primary model weights storage URI (Safetensors shards)
  string weights_uri = 11;

  // Distributed optimizer state URI (AdamW m/v buffers, FSDP states)
  string optimizer_state_uri = 12;

  // Learning rate and warm-up scheduler state URI
  string scheduler_state_uri = 13;

  // PRNG state container
  RNGStateSnapshot rng_state = 14;

  // Exact data cursor for seamless streaming resume
  DataStreamPosition data_position = 15;

  // Checkpoint loss and validation metrics at this step
  map<string, double> step_metrics = 16;

  // Timestamp when checkpoint was completed on disk/object storage
  google.protobuf.Timestamp created_at = 17;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/checkpoint-manifest.json",
  "title": "CheckpointManifest",
  "type": "object",
  "required": [
    "run_id",
    "checkpoint_step",
    "epoch",
    "dataset_version",
    "tokenizer_version",
    "architecture_version",
    "training_config_hash",
    "code_commit",
    "hardware_description",
    "weights_uri",
    "optimizer_state_uri",
    "scheduler_state_uri",
    "rng_state",
    "data_position",
    "created_at"
  ],
  "properties": {
    "run_id": { "type": "string" },
    "checkpoint_step": { "type": "integer" },
    "epoch": { "type": "number" },
    "model_version_draft": { "type": "string" },
    "dataset_version": { "type": "string" },
    "tokenizer_version": { "type": "string" },
    "architecture_version": { "type": "string" },
    "training_config_hash": { "type": "string" },
    "code_commit": { "type": "string", "pattern": "^[a-f0-9]{40}$" },
    "hardware_description": {
      "type": "object",
      "required": ["accelerator_model", "world_size"],
      "properties": {
        "accelerator_model": { "type": "string" },
        "total_node_count": { "type": "integer" },
        "accelerators_per_node": { "type": "integer" },
        "world_size": { "type": "integer" },
        "interconnect_type": { "type": "string" },
        "driver_version": { "type": "string" },
        "cuda_version": { "type": "string" }
      }
    },
    "weights_uri": { "type": "string" },
    "optimizer_state_uri": { "type": "string" },
    "scheduler_state_uri": { "type": "string" },
    "rng_state": {
      "type": "object",
      "required": ["global_seed"],
      "properties": {
        "global_seed": { "type": "integer" },
        "rank_rng_states_base64": { "type": "object" }
      }
    },
    "data_position": {
      "type": "object",
      "required": ["epoch", "shard_index", "sample_index", "token_index"],
      "properties": {
        "epoch": { "type": "integer" },
        "shard_index": { "type": "integer" },
        "sample_index": { "type": "integer" },
        "token_index": { "type": "integer" },
        "dataset_manifest_hash": { "type": "string" }
      }
    },
    "step_metrics": {
      "type": "object",
      "additionalProperties": { "type": "number" }
    },
    "created_at": { "type": "string", "format": "date-time" }
  }
}
```

---

## 4. Complete JSON Example

```json
{
  "run_id": "run-2026-08-bengali-7b-stage3",
  "checkpoint_step": 125000,
  "epoch": 2.45,
  "model_version_draft": "myllm/bengali-7b:step-125000-draft",
  "dataset_version": "sha256:112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
  "tokenizer_version": "sha256:8f4c2b9a7812de4f9011ba2134567890abcdef1234567890abcdef1234567890",
  "architecture_version": "sha256:d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
  "training_config_hash": "sha256:99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa",
  "code_commit": "4a5e3d7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e",
  "hardware_description": {
    "accelerator_model": "NVIDIA_H100_SXM5_80GB",
    "total_node_count": 8,
    "accelerators_per_node": 8,
    "world_size": 64,
    "interconnect_type": "NVLink-4_InfiniBand-NDR400",
    "driver_version": "560.35.03",
    "cuda_version": "12.6"
  },
  "weights_uri": "s3://myllm-checkpoints/run-2026-08-bengali-7b-stage3/step-125000/weights/",
  "optimizer_state_uri": "s3://myllm-checkpoints/run-2026-08-bengali-7b-stage3/step-125000/optimizer.pt",
  "scheduler_state_uri": "s3://myllm-checkpoints/run-2026-08-bengali-7b-stage3/step-125000/scheduler.json",
  "rng_state": {
    "global_seed": 4242,
    "rank_rng_states_base64": {
      "rank_0": "gASV...==",
      "rank_63": "gASV...=="
    }
  },
  "data_position": {
    "epoch": 2,
    "shard_index": 412,
    "sample_index": 981200,
    "token_index": 15320000000,
    "dataset_manifest_hash": "sha256:112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00"
  },
  "step_metrics": {
    "train_loss": 1.412,
    "grad_norm": 0.84,
    "learning_rate": 0.00015,
    "tokens_per_sec_per_gpu": 4820.0
  },
  "created_at": "2026-08-16T07:15:00Z"
}
```
