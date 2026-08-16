# MyLLM Specification: Tensor Manifest

- **Document ID**: `SPEC-0004`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Tensor Storage, Zero-Copy I/O & Sharding
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Core Principles

The `TensorManifest` provides a deterministic, architecture-agnostic index of all tensor arrays comprising a model's weights. It decouples high-level model semantics from low-level byte layout, memory alignment, and multi-file sharding.

### Invariants:
1. **Zero Architecture Coupling**: The `TensorManifest` contains NO neural network topology concepts (e.g., `d_model`, `n_heads`, `n_layers`). It knows only names, data types, shapes, byte offsets, and shard filenames.
2. **Safe Serialization Only**: Only secure, header-indexed, zero-copy binary formats are permitted (`safetensors`). Python `pickle`, `torch.save`, or unauthenticated object streams are strictly forbidden across all tiers.
3. **Zero-Copy Memory-Map Alignment**: Tensor byte offsets within shards MUST align to 8-byte boundaries (64-bit alignment) or 64-byte boundaries for direct Direct Memory Access (DMA) and GPUDirect Storage (GDS).

---

## 2. Protobuf Specification (`myllm/tensor/v1/tensor_manifest.proto`)

```protobuf
syntax = "proto3";

package myllm.tensor.v1;

option go_package = "github.com/myllm/core/gen/go/tensor/v1;tensorv1";
option java_package = "ai.myllm.tensor.v1";

enum TensorDataType {
  DTYPE_UNSPECIFIED = 0;
  DTYPE_FLOAT32 = 1;
  DTYPE_FLOAT16 = 2;
  DTYPE_BFLOAT16 = 3;
  DTYPE_FP8_E4M3 = 4;
  DTYPE_FP8_E5M2 = 5;
  DTYPE_INT32 = 6;
  DTYPE_INT16 = 7;
  DTYPE_INT8 = 8;
  DTYPE_INT4 = 9;
  DTYPE_UINT8 = 10;
  DTYPE_BOOL = 11;
}

enum TensorContainerFormat {
  FORMAT_UNSPECIFIED = 0;
  FORMAT_SAFETENSORS = 1;
  FORMAT_RAW_ALIGNED_BINARY = 2;
}

message QuantizationParameter {
  // Quantization scheme: "blockwise_fp8", "awq", "gptq", "marlin", "squeezellm"
  string scheme = 1;
  // Block or group size (e.g., 32, 64, 128)
  uint32 group_size = 2;
  // Name of the scale tensor within the manifest
  string scale_tensor_name = 3;
  // Name of the zero-point tensor within the manifest (optional)
  string zero_point_tensor_name = 4;
}

message TensorEntry {
  // Canonical tensor parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
  string name = 1;

  // Tensor shape dimensions [dim0, dim1, ...]
  repeated uint64 shape = 2;

  // Primitive data type
  TensorDataType dtype = 3;

  // Relative shard filename containing this tensor's raw bytes
  string shard_file = 4;

  // Byte offset from start of the raw data payload in shard_file
  uint64 byte_offset = 5;

  // Exact byte length of the tensor buffer
  uint64 byte_length = 6;

  // SHA-256 hash of the uncompressed raw tensor byte slice
  string content_hash = 7;

  // Optional quantization parameters
  QuantizationParameter quantization = 8;
}

message ShardFileInfo {
  // Relative filename within the artifact bundle (e.g., "model-00001-of-00002.safetensors")
  string filename = 1;
  // Total byte size of the shard file
  uint64 size_bytes = 2;
  // SHA-256 checksum of the entire shard file
  string sha256_checksum = 3;
  // Count of individual tensors contained in this shard
  uint32 tensor_count = 4;
}

message TensorManifest {
  // Schema version (SemVer, e.g., "1.0.0")
  string tensor_schema_version = 1;

  // Storage serialization format
  TensorContainerFormat format = 2;

  // Total byte size of all weight shards combined
  uint64 total_bytes = 3;

  // Total count of tensors in this manifest
  uint32 total_tensors = 4;

  // List of shard files
  repeated ShardFileInfo shards = 5;

  // Individual tensor entries indexed by canonical name
  repeated TensorEntry tensors = 6;

  // Content-addressable manifest digest (SHA-256 of canonical manifest JSON)
  string manifest_digest = 7;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/tensor-manifest.json",
  "title": "TensorManifest",
  "type": "object",
  "required": [
    "tensor_schema_version",
    "format",
    "total_bytes",
    "total_tensors",
    "shards",
    "tensors"
  ],
  "properties": {
    "tensor_schema_version": { "type": "string" },
    "format": { "type": "string", "enum": ["FORMAT_SAFETENSORS", "FORMAT_RAW_ALIGNED_BINARY"] },
    "total_bytes": { "type": "integer" },
    "total_tensors": { "type": "integer" },
    "shards": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["filename", "size_bytes", "sha256_checksum", "tensor_count"],
        "properties": {
          "filename": { "type": "string" },
          "size_bytes": { "type": "integer" },
          "sha256_checksum": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
          "tensor_count": { "type": "integer" }
        }
      }
    },
    "tensors": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "shape", "dtype", "shard_file", "byte_offset", "byte_length", "content_hash"],
        "properties": {
          "name": { "type": "string" },
          "shape": { "type": "array", "items": { "type": "integer" } },
          "dtype": { "type": "string" },
          "shard_file": { "type": "string" },
          "byte_offset": { "type": "integer" },
          "byte_length": { "type": "integer" },
          "content_hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
          "quantization": {
            "type": "object",
            "required": ["scheme", "group_size", "scale_tensor_name"],
            "properties": {
              "scheme": { "type": "string" },
              "group_size": { "type": "integer" },
              "scale_tensor_name": { "type": "string" },
              "zero_point_tensor_name": { "type": "string" }
            }
          }
        }
      }
    },
    "manifest_digest": { "type": "string" }
  }
}
```

---

## 4. Complete JSON Example

```json
{
  "tensor_schema_version": "1.0.0",
  "format": "FORMAT_SAFETENSORS",
  "total_bytes": 14900398080,
  "total_tensors": 291,
  "shards": [
    {
      "filename": "weights/model-00001-of-00002.safetensors",
      "size_bytes": 7450201920,
      "sha256_checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "tensor_count": 145
    },
    {
      "filename": "weights/model-00002-of-00002.safetensors",
      "size_bytes": 7450196160,
      "sha256_checksum": "ca978112ca1bbdcafac231b39a23dc4da786081cd1e14dd6da60ef4e98b50e67",
      "tensor_count": 146
    }
  ],
  "tensors": [
    {
      "name": "model.embed_tokens.weight",
      "shape": [128256, 4096],
      "dtype": "DTYPE_BFLOAT16",
      "shard_file": "weights/model-00001-of-00002.safetensors",
      "byte_offset": 64,
      "byte_length": 1050673152,
      "content_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0"
    },
    {
      "name": "model.layers.0.self_attn.q_proj.weight",
      "shape": [4096, 4096],
      "dtype": "DTYPE_BFLOAT16",
      "shard_file": "weights/model-00001-of-00002.safetensors",
      "byte_offset": 1050673216,
      "byte_length": 33554432,
      "content_hash": "b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef01"
    },
    {
      "name": "model.layers.0.self_attn.k_proj.weight",
      "shape": [1024, 4096],
      "dtype": "DTYPE_BFLOAT16",
      "shard_file": "weights/model-00001-of-00002.safetensors",
      "byte_offset": 1084227648,
      "byte_length": 8388608,
      "content_hash": "c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef012"
    },
    {
      "name": "model.layers.0.input_layernorm.weight",
      "shape": [4096],
      "dtype": "DTYPE_BFLOAT16",
      "shard_file": "weights/model-00001-of-00002.safetensors",
      "byte_offset": 1092616256,
      "byte_length": 8192,
      "content_hash": "d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0123"
    }
  ],
  "manifest_digest": "sha256:3a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b"
}
```

---

## 5. Memory-Mapping & Direct I/O Rules

1. **Alignment Verification**:
   - `byte_offset % 8 == 0` is strictly enforced for all CPU/POSIX `mmap` calls.
   - `byte_offset % 64 == 0` is required for direct GPUDirect Storage / cuFile bypass routines.
2. **Deterministic Sharding**: Shard boundaries are determined strictly by byte count limits (default 5.0 GB per shard) to ensure predictable network transmission and concurrent multi-threaded loading.
