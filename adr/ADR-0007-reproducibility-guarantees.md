# ADR-0007: Realistic Reproducibility Guarantees (Exact vs. Statistical)

## Status: Accepted
## Date: 2026-08-16

---

## Context
Scientific integrity and engineering verification in AI systems require that training runs, fine-tuning jobs, and evaluations be reproducible. However, deep learning systems running on modern distributed accelerator clusters face fundamental physical and mathematical limitations:

1. **Floating-Point Non-Associativity**: In IEEE-754 arithmetic, $(a + b) + c \neq a + (b + c)$. Distributed collective operations (`AllReduce`, `ReduceScatter`) aggregate floating-point tensors across worker nodes in non-deterministic order depending on network jitter, packet routing, and thread scheduling.
2. **GPU Kernel Non-Determinism**: High-throughput atomic operations in GPU kernels (e.g. atomic adds in scatter/gather or flash-attention backward passes) introduce nondeterminism unless strict deterministic modes are enabled at significant throughput cost (15% - 40% slowdown).
3. **Hardware & Topology Heterogeneity**: Running identical training code on 64x NVIDIA H100 SXM5 vs 128x NVIDIA A100 PCIe vs AMD MI300X produces different floating-point rounding accumulations at every layer.

Many AI platforms make sweeping, unscientific claims of "100% deterministic training everywhere", which inevitably fail under real-world cluster conditions and lead to debugging confusion.

---

## Decision
We establish a two-tiered, scientifically grounded **Reproducibility Contract** (`SPEC-0018`):

1. **Tier 1: Exact Reproducibility (Bit-for-Bit Determinism)**:
   - **Scope**: Guaranteed *only* under identical hardware topology, pinned software containers (CUDA/PyTorch/NCCL), identical PRNG seeds, and deterministic algorithmic flags (`torch.use_deterministic_algorithms(True)`).
   - **Verification**: Evaluated via bitwise equality of saved weight shards and loss logs over twin runs.
2. **Tier 2: Statistical Reproducibility (Distributional Bounds)**:
   - **Scope**: Applied when training across different hardware topologies, varying cluster scales, or non-deterministic high-throughput production runs.
   - **Verification**: Guaranteed convergence where the loss trajectory delta $|\Delta \mathcal{L}| \le 1.0 \times 10^{-3}$ and downstream benchmark scores match within $2\sigma$ of the reference distribution.

---

## Alternatives Considered
1. **Unconditional Claim of 100% Exact Bitwise Determinism Across All Hardware**:
   - *Pros*: Simple marketing message.
   - *Cons*: Mathematically impossible in distributed floating-point systems; creates false expectations; fails immediately when clusters scale or switch GPU hardware generations.
2. **Abandon Reproducibility Guarantees Entirely**:
   - *Pros*: Zero performance overhead; no need to track hardware topologies or lock seeds.
   - *Cons*: Eliminates scientific verification; makes root-cause analysis of silent model regressions, data poisoning, or training instability impossible.

---

## Trade-offs
- **Clarity over Slogans**: Requires educating engineering and research teams on the mathematical boundaries of floating-point reduction vs exact topology matching.
- **Harness Complexity**: The CI/CD and evaluation framework must support two different verification modes (bitwise hash comparison for regression suites vs statistical tolerance testing for scaled runs).

---

## Consequences
- **Positive**:
  - **Honest Engineering Foundation**: Eliminates chasing ghost bugs caused by expected floating-point reduction drift across different cluster sizes.
  - **Rigorous Golden Testing**: CI/CD regression suites running on dedicated, fixed hardware nodes can rely on 100% bitwise determinism to catch real algorithmic bugs.
  - **High Performance in Production**: Production pre-training runs can utilize maximum-speed non-deterministic kernels while remaining protected by statistical convergence bounds.
- **Negative**:
  - Requires recording full hardware topology metadata (`HardwareTopology`) in every `CheckpointManifest`.
