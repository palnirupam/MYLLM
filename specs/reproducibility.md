# MyLLM Specification: Training & Inference Reproducibility

- **Document ID**: `SPEC-0018`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Scientific Integrity, Deterministic Training & Verification
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Core Philosophy & The Reproducibility Frontier

Reproducibility in deep learning systems operating on distributed GPUs is fundamentally constrained by floating-point non-associativity:
$$(a + b) + c \neq a + (b + c) \quad \text{in IEEE 754 floating-point arithmetic}$$

When collective reduction operations (`AllReduce`, `ReduceScatter`) aggregate gradients across changing cluster topologies or varying thread scheduling, rounding differences accumulate, inevitably causing divergence in bit-level weights after thousands of optimization steps.

To provide honest, scientifically rigorous guarantees, MyLLM formalizes two distinct tiers of reproducibility:

```
+-----------------------------------------------------------------------------------+
|                            REPRODUCIBILITY TIERS                                  |
+-------------------------------------------------+---------------------------------+
|              EXACT REPRODUCIBILITY              |    STATISTICAL REPRODUCIBILITY  |
|            (Bit-for-Bit Determinism)            |       (Distributional Bounds)   |
+-------------------------------------------------+---------------------------------+
| - Identical hardware topology (e.g. 64x H100)  | - Heterogeneous / Rescaled HW   |
| - Identical software stack (CUDA/PyTorch)       | - Different cluster sizes       |
| - Deterministic kernel flags enabled            | - Asynchronous data loaders     |
| - Guarantee: EXACT bitwise loss and weights     | - Guarantee: Loss delta <= 1e-3 |
+-------------------------------------------------+---------------------------------+
```

---

## 2. Tier 1: Exact Reproducibility Contract

Exact reproducibility guarantees that re-running a training or evaluation job produces **100% bitwise identical output weights and loss logs**.

### Mandatory Preconditions:
1. **Identical Code Base**: Exact Git commit SHA matching `CheckpointManifest.code_commit`.
2. **Identical Data Stream**: Exact `DatasetManifest.content_hash` and identical data loading cursor (`epoch`, `shard_index`, `sample_index`, `token_index`).
3. **Identical Tokenizer**: Exact `tokenizer_version` hash.
4. **Identical Software Container**: Locked base image with pinned compiler and kernel versions:
   - Linux Kernel & NVIDIA Driver version (e.g., Driver 560.35.03)
   - CUDA Toolkit (e.g., CUDA 12.6.1)
   - cuDNN & NCCL versions (e.g., NCCL 2.22.3)
   - PyTorch, Triton, FlashAttention exact commit hashes
5. **Identical Physical Topology**:
   - Exact accelerator model (e.g., 8x NVIDIA H100 SXM5 80GB per node, 8 nodes = 64 GPUs).
   - Exact interconnect topology (NVLink 4 + InfiniBand NDR400).
   - Identical tensor parallel ($TP$), pipeline parallel ($PP$), and data parallel ($DP$) ranks.
6. **Deterministic Execution Flags**:
   ```python
   import torch
   import os

   torch.manual_seed(4242)
   torch.cuda.manual_seed_all(4242)
   os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
   torch.use_deterministic_algorithms(True, warn_only=False)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   torch.backends.cuda.matmul.allow_tf32 = False # Force exact IEEE-754 precision where required
   ```

---

## 3. Tier 2: Statistical Reproducibility Contract

When training across different hardware architectures (e.g., migrating from A100 to H100, or scaling from 32 to 128 GPUs), exact bitwise identity is mathematically impossible without severe performance degradation.

Statistical reproducibility guarantees that two independent training runs converge to equivalent representations within strict $\epsilon$-bounds.

### Mathematical Acceptance Criteria:
1. **Loss Trajectory Convergence**:
   $$\forall t \in [0.1 \cdot T, T], \quad \left| \frac{\mathcal{L}_{\text{run1}}(t) - \mathcal{L}_{\text{run2}}(t)}{\mathcal{L}_{\text{run1}}(t)} \right| \le \epsilon_{\text{loss}} \quad (\epsilon_{\text{loss}} = 1.0 \times 10^{-3})$$
2. **Downstream Benchmark Equivalence**:
   $$\forall B \in \text{RegressionBenchmarks}, \quad |\text{Score}_{\text{run1}}(B) - \text{Score}_{\text{run2}}(B)| \le 2 \cdot \sigma_{\text{benchmark}}$$
3. **Gradient Norm Alignment**:
   $$\mathbb{E}[\|\mathbf{g}_{\text{run1}}\|_2] \approx \mathbb{E}[\|\mathbf{g}_{\text{run2}}\|_2] \quad (\text{within } 2.5\% \text{ relative delta})$$

---

## 4. Reproducibility Verification Checklist

Before certifying a training run as reproducible:
- [ ] Record all PRNG seeds across global, worker, and data loader threads.
- [ ] Seal `training_config.json` and compute its canonical SHA-256 hash.
- [ ] Freeze Python `requirements.lock` and container image digest.
- [ ] Export NCCL topology graph (`NCCL_TOPO_DUMP_FILE`).
- [ ] Execute 1,000-step twin-run deterministic smoke test (asserting `diff loss_run1.log loss_run2.log` is empty).
