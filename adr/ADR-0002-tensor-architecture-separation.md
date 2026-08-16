# ADR-0002: Separation of Tensor Manifest from Architecture Configuration

## Status: Accepted
## Date: 2026-08-16

---

## Context
In early deep learning frameworks and simple model packaging formats (e.g. legacy PyTorch `.pt` files, monolithic Hugging Face `config.json` files), neural network architectural parameters (such as `n_heads`, `d_model`, `intermediate_size`, `rope_theta`) and physical tensor storage metadata (tensor names, data types, sharding boundaries, byte offsets) were frequently coupled into a single schema.

Over a 15-year platform horizon, neural network topologies are guaranteed to diversify dramatically:
1. **Mixture-of-Experts (MoE)** models introduce routing gates, shared experts, and dynamic dispatch parameters.
2. **State Space Models (SSMs / Mamba)** eliminate attention heads, multi-head KV caches, and RoPE entirely in favor of recurrent state matrices ($A, B, C$), 1D convolution kernels, and $\Delta$ projections.
3. **Hybrid Architectures** combine attention blocks, convolution layers, and recurrence in alternating stages.
4. **Graph / Neuromorphic Models** use adjacency matrices and continuous-time differential operators.

Coupling tensor layout with architectural topology forces the tensor storage layer to change whenever a novel neural architecture emerges, breaking zero-copy storage loaders and sharding tools.

---

## Decision
We will strictly separate the **`TensorManifest`** (`SPEC-0004`) from the **`ArchitectureConfig`** (`SPEC-0003`) into two independent, decoupled contracts.

1. **`TensorManifest`**:
   - Represents the physical, byte-level layout of weights.
   - Contains *only* tensor names, shape dimensions, data types (`BF16`, `FP8`, `INT4`), shard filenames, byte offsets, byte lengths, and SHA-256 slice hashes.
   - Has zero knowledge of layers, attention heads, activation functions, or execution mechanics.
2. **`ArchitectureConfig`**:
   - Represents the mathematical and graph structure of the neural network.
   - Contains architecture family definitions (`TRANSFORMER_DECODER`, `MIXTURE_OF_EXPERTS`, `STATE_SPACE_MODEL`), hyperparameters (`d_model`, `n_heads`, `d_state`), and layer connectivity.
   - Has zero knowledge of file paths, byte offsets, or physical sharding boundaries.

---

## Alternatives Considered
1. **Single Monolithic Config (Hugging Face style `config.json`)**:
   - *Pros*: Single file to inspect and parse; familiar to existing ecosystem tooling.
   - *Cons*: Schema becomes a polluted union of optional fields; adding an SSM or MoE requires adding hundreds of non-applicable fields to transformer configs; breaks type safety and strict schema validation.
2. **Embedding Architecture Metadata inside Tensor Manifest**:
   - *Pros*: Everything needed to construct and load the model is in one JSON file.
   - *Cons*: Storage and memory-mapping engines must parse high-level architectural hyperparameters; changes to network design invalidate the tensor storage schema.

---

## Trade-offs
- **Indirection**: Model loading requires reading and validating two distinct metadata files (`architecture_config.json` and `tensor_manifest.json`) instead of one.
- **Coordination**: The runtime engine must verify that the tensor names listed in `TensorManifest` satisfy the weight requirements declared in `ArchitectureConfig`.

---

## Consequences
- **Positive**:
  - The tensor loading engine, memory-mapping subsystem, and GPUDirect Storage I/O paths remain 100% frozen and stable, regardless of what new neural architectures are invented over the next 15 years.
  - Adding a completely new model family (e.g. Mamba-3, Graph Neural Networks) requires only introducing a new `ArchitectureConfig` schema without altering tensor storage logic.
  - Zero-copy weight sharding and quantization tools can operate purely on `TensorManifest` without importing deep learning frameworks or model definitions.
- **Negative**:
  - Tooling must handle dual schema validation when publishing and verifying new model artifacts.
