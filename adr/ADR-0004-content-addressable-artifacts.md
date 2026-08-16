# ADR-0004: Content-Addressable Storage and Cryptographic Artifact Identity

## Status: Accepted
## Date: 2026-08-16

---

## Context
In distributed AI platforms, artifacts (model weights, tokenizer configs, dataset shards, execution logs, system prompts) are continually replicated across training clusters, object stores, local NVMe caches, and inference serving nodes.

Traditional systems using mutable identifiers (such as database auto-increment IDs, semantic tag strings like `latest` or `v1.0`, or random UUIDs without content binding) suffer from severe failure modes:
1. **Silent Mutation**: A file in S3 or local disk can be overwritten or corrupted without changing its identifier, causing silent training divergence or inference corruption.
2. **Cache Incoherency**: Serving nodes caching `model-v1.0` cannot determine if the underlying weights were updated without polling centralized databases.
3. **Storage Waste**: Re-uploaded or shared weight shards across fine-tuned models are duplicated repeatedly because random UUIDs mask identical byte content.
4. **Untrusted Supply Chains**: Lacking cryptographic binding, downstream nodes cannot verify if an artifact was tampered with in transit.

---

## Decision
We will enforce **Content-Addressable Identity** across all MyLLM artifacts, datasets, checkpoints, and files.

1. **Deterministic Primary Key**: Every artifact's primary key (`artifact_id`) is strictly computed as the hexadecimal SHA-256 hash of its canonical byte stream:
   $$\text{artifact\_id} = \text{sha256:}\langle\text{64-hex-characters}\rangle$$
2. **Metadata Canonicalization**: Structured metadata manifests (such as `ModelContract`, `ArchitectureConfig`, `DatasetManifest`) are normalized using deterministic JSON canonicalization (RFC 8785) before hashing.
3. **Write-Once Immutability**: All storage keys in `StorageProvider` under the `/artifacts/` namespace are immutable. Writes to existing hashes are treated as idempotent no-ops.
4. **Logical Pointers as Aliases**: Human-readable names, tags, and versions (e.g. `bengali-reasoning-7b:1.0.0`) are purely mutable alias pointers that resolve to immutable content-addressable SHA-256 digests.

---

## Alternatives Considered
1. **UUIDv4 / Random Unique Identifiers**:
   - *Pros*: Fast generation with zero I/O; no hashing required.
   - *Cons*: Two identical 5 GB weight shards receive two different UUIDs, causing 100% duplicate storage; cannot detect silent file corruption or tampering.
2. **Database Sequence IDs / Monotonic Counters**:
   - *Pros*: Compact integer keys; chronological ordering.
   - *Cons*: Single point of failure (central database required for ID generation); distributed training nodes cannot independently mint IDs; zero integrity guarantees.
3. **Mutable Version Strings (e.g., `model-v1.0.0.bin`)**:
   - *Pros*: Simple human readability.
   - *Cons*: Developers and pipelines frequently re-tag or overwrite files with the same version name, breaking reproducibility and distributed cache consistency.

---

## Trade-offs
- **Hashing Computational Cost**: Computing SHA-256 across large multi-gigabyte weight shards consumes CPU cycles.
  - *Mitigation*: Modern CPUs and accelerators feature hardware-accelerated SHA-NI (Intel/AMD SHA Extensions) capable of hashing at $>3.5\text{ GB/s}$ per core, making hash computation a tiny fraction of network/disk I/O transfer time.
- **Human Readability**: SHA-256 hex strings (e.g. `sha256:7f83b165...`) are not friendly for human communication.
  - *Mitigation*: The Model Registry maintains human-friendly aliases and tags mapping to immutable SHA-256 digests.

---

## Consequences
- **Positive**:
  - **Zero Silent Corruption**: Any bit rot, disk failure, or network transmission error is instantly caught by SHA-256 validation before weights hit GPU memory.
  - **Global Deduplication**: Identical weight layers (e.g. frozen base model layers across multiple fine-tuned variants) are stored exactly once, saving terabytes of storage.
  - **Immutable Caching**: Distributed inference nodes can cache artifacts locally forever with zero cache-invalidation logic.
  - **Audit Provenance**: Cryptographic proofs link every model artifact deterministically to the dataset and code commit that produced it.
- **Negative**:
  - All tools and publishing pipelines must include deterministic hashing and canonical serialization libraries.
