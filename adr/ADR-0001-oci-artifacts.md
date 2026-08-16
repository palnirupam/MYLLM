# ADR-0001: Use of OCI Artifacts for Model Distribution

## Status: Accepted
## Date: 2026-08-16

---

## Context
Model weights, tokenizer vocabularies, configuration manifests, and evaluation artifacts in modern LLM ecosystems require a scalable, vendor-neutral, and versioned distribution mechanism. Traditionally, machine learning teams have relied on custom HTTP endpoints, direct S3 bucket syncing, or Git LFS.

However, these approaches present significant drawbacks over a 15-year horizon:
1. **Git LFS**: Struggles with multi-gigabyte files, lacks native chunked parallel distribution, has high server overhead, and provides poor support for content verification during streaming.
2. **Raw Object Storage (S3-only)**: Lacks a standardized, standardized registry discovery protocol, role-based access control federation across hybrid clouds, and standard image-tagging mechanisms.
3. **Custom Model Registries**: Create vendor lock-in, require maintaining custom edge caching daemons, and fail to leverage enterprise container infrastructure already deployed in Kubernetes clusters.

The Open Container Initiative (OCI) Artifact Specification provides an established standard for storing, signing, versioning, and distributing non-container payloads (Wasm modules, Helm charts, AI models) across standard container registries (e.g., Harbor, GitHub Packages, AWS ECR, Google Artifact Registry).

---

## Decision
We will use **OCI Artifacts** as the standard distribution packaging format for MyLLM model releases, while keeping internal MyLLM contract definitions completely independent of OCI runtime or transport internals.

Specifically:
1. Every published `ModelArtifact` will define a standard OCI media type (`application/vnd.myllm.model.artifact.v1+json`) and map weight shards and config files as OCI image layers (`application/vnd.myllm.model.shard.v1+safetensors`).
2. Standard OCI tooling (e.g., `oras`, `skopeo`, `cosign`) can be used to push, pull, inspect, and cryptographically sign MyLLM models across enterprise registries.
3. Core MyLLM runtime contracts (`ModelContract`, `TensorManifest`, `StorageProvider`) will remain decoupled from OCI libraries, interacting only with abstract content-addressed streams and URIs (`oci://`, `s3://`, `file://`).

---

## Alternatives Considered
1. **Git LFS (Hugging Face style)**:
   - *Pros*: Familiar to data scientists; native integration with Git.
   - *Cons*: High latency on large scale pulls; poor deduplication; git history bloat; high operational overhead for self-hosted large file storage.
2. **Direct Cloud Object Storage (S3 / GCS buckets with custom metadata)**:
   - *Pros*: Simple to implement initially.
   - *Cons*: No standardized multi-tenant registry protocol; custom authorization layer required; lacks interoperability with enterprise artifact scanners and Sigstore tooling.
3. **Proprietary Custom Binary Registry Protocol**:
   - *Pros*: Can be optimized specifically for GPU Direct I/O.
   - *Cons*: Massive development and maintenance burden over a 15-year horizon; zero third-party ecosystem integration.

---

## Trade-offs
- **Complexity**: Integrating OCI registry protocols adds a layer of abstraction compared to simple S3 file downloads.
- **Layer Size Limits**: Standard container registries may have layer size limitations (e.g., 10 GB per layer), requiring deterministic sharding of large 70B+ model weights into 5 GB shard files.
- **Tooling Dependency**: Build and CI/CD pipelines must include OCI client tools (`oras` or native OCI SDKs).

---

## Consequences
- **Positive**:
  - Models can be pushed to, pulled from, and cached by any standard enterprise OCI registry (Harbor, ECR, GCR, Docker Hub).
  - Native integration with modern supply-chain security tools (e.g., Cosign, Sigstore, Notary v2) for cryptographic signing and vulnerability scanning.
  - Transparent edge caching in Kubernetes environments via standard registry mirrors.
- **Negative / Operational Requirements**:
  - Model weights MUST be deterministically partitioned into shards $\le 5\text{ GB}$ to prevent registry payload rejections.
  - The deployment pipeline must maintain OCI credentials and registry synchronizers.
