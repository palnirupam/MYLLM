# ADR-0003: Abstract Sandbox Runtime Interface

## Status: Accepted
## Date: 2026-08-16

---

## Context
Agentic LLM platforms frequently execute untrusted code, run dynamic scripts, parse untrusted file formats, and interact with external APIs via tools. Sandboxing is essential to prevent host compromise, data exfiltration, lateral network movement, and denial-of-service attacks.

However, sandboxing technologies are constantly evolving:
- **Docker / OCI Containers**: Mature and feature-rich, but have slow startup times (100ms - 2s), high memory overhead, large security attack surfaces (shared Linux kernel), and container-escape vulnerabilities.
- **WebAssembly (Wasm / WASI)**: Near-instant startup (<5ms), minimal memory footprint (<10MB), mathematically provable memory isolation, but currently has incomplete POSIX socket/threading support for legacy languages.
- **MicroVMs (Firecracker / gVisor / Cloud Hypervisor)**: Strong hardware-level virtualization with kernel isolation and fast boot (5ms - 50ms), but require KVM hardware virtualization support not available on all cloud/edge instances.
- **Confidential Computing (Intel SGX / AMD SEV-SNP / NVIDIA H100 CC)**: Hardware-encrypted enclaves emerging for sensitive tenant isolation.

Committing MyLLM directly to Docker or any single container runtime creates a massive technical debt liability and limits deployment to environments with Docker daemon privileges.

---

## Decision
We will define an abstract, technology-neutral **`SandboxRuntime`** interface for all tool execution and untrusted computation.

1. **Abstract Interface**:
   ```python
   class SandboxRuntime(ABC):
       @abstractmethod
       def create_environment(self, config: SandboxConfig) -> SandboxHandle: ...
       @abstractmethod
       def execute(self, handle: SandboxHandle, command: List[str], stdin: bytes, timeout_ms: int) -> ExecutionResult: ...
       @abstractmethod
       def destroy_environment(self, handle: SandboxHandle) -> None: ...
   ```
2. **Multi-Backend Runtime Strategy**:
   - **Wasm Runtime (Wasmtime / Wasmer)**: The preferred default backend for lightweight tools, pure-logic computation, data parsers, and language interpreters compiled to Wasm/WASI.
   - **MicroVM / Container Runtime (gVisor / Firecracker / Docker)**: The fallback backend for complex environments requiring full Linux kernels, GPU acceleration, or legacy binary dependencies.
3. **Strict Contract Independence**: Tool contracts (`SPEC-0007`) declare *capabilities* and *resource limits*, never container engine specifics.

---

## Alternatives Considered
1. **Commit Exclusively to Docker Engine**:
   - *Pros*: Vast existing ecosystem of prebuilt Docker images.
   - *Cons*: Requires root/privileged daemon access; vulnerable to kernel privilege escalations; startup latency is too high for high-frequency agent tool calls (500ms vs 2ms in Wasm).
2. **Direct Host Process Execution with Seccomp / Cgroups**:
   - *Pros*: Zero virtualization overhead; maximum raw execution speed.
   - *Cons*: Unacceptable security risk in multi-tenant cloud environments; platform-dependent (Linux-only, fails on macOS/Windows); prone to accidental privilege leaks.
3. **Pure Wasm-Only Strategy**:
   - *Pros*: Maximum security, portability, and instant startup everywhere.
   - *Cons*: Cannot run complex scientific Python packages (e.g. legacy C/Fortran extensions, full PyTorch/CUDA) inside Wasm today.

---

## Trade-offs
- **Engineering Overhead**: Developing and maintaining multiple runtime backends (Wasm + MicroVM) requires additional engineering investment.
- **Feature Parity**: Some features (e.g. raw networking or multi-threading) behave differently in Wasm vs MicroVMs, requiring runtime capability negotiation.

---

## Consequences
- **Positive**:
  - Future-proof: MyLLM can seamlessly transition to new sandboxing technologies (e.g., next-generation micro-hypervisors, Wasm Component Model, confidential hardware enclaves) without changing a single line of agent or tool contract code.
  - High Performance: 90% of lightweight tool executions (calculators, JSON formatters, text parsers) run in sub-millisecond Wasm sandboxes, dramatically reducing agent response latency.
  - Portability: MyLLM can execute securely in serverless environments, Kubernetes pods without Docker-in-Docker permissions, and edge devices.
- **Negative**:
  - The runtime orchestrator must maintain routing logic to select the appropriate backend based on tool manifest requirements.
