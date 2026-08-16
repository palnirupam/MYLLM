# ADR-0005: Extensible Capability Registry over Fixed Boolean Flags

## Status: Accepted
## Date: 2026-08-16

---

## Context
AI platforms must communicate the specialized skills, modalities, and execution features of diverse models to routing gateways, orchestration agents, and client applications.

Traditional AI system designs model capabilities as a fixed collection of boolean fields (e.g., `supports_vision: true`, `supports_function_calling: true`, `supports_json_mode: true`).

Over a 15-year platform horizon, this fixed-boolean pattern breaks down:
1. **Unpredictable Future Modalities**: Capabilities like formal verification, multi-agent consensus protocols, robotics trajectory generation, and specialized dialect reasoning cannot be predicted in advance.
2. **Lack of Parametric Nuance**: A boolean `supports_tools: true` cannot indicate whether a model supports parallel tool calls, strict JSON Schema grammar generation, or execution timeouts.
3. **Continuous Schema Churn**: Every new capability discovery requires modifying, re-compiling, and re-deploying core Protobuf and OpenAPI schemas across the entire platform.

---

## Decision
We will model all model capabilities using an **Extensible Capability Contract** (`SPEC-0006`) based on a tuple of:
$$\text{Capability} = (\text{name: string}, \, \text{version: SemVer}, \, \text{properties: map}\langle\text{string, string}\rangle)$$

1. **Self-Describing Models**: Models declare a list of `Capability` objects in their `ModelContract`.
2. **Dynamic Routing**: Orchestrators and agent frameworks match client requests against advertised model capabilities using SemVer constraints and property matching.
3. **No Schema Mutations for New Features**: Introducing a new domain capability (e.g. `bengali`, `formal_math_lean4`, `realtime_audio`) requires only registering the capability name and properties in the capability catalog—the underlying Protobuf and REST schemas remain unchanged.

---

## Alternatives Considered
1. **Fixed Boolean Flags in Model Manifest (`supports_vision: bool`, `supports_code: bool`)**:
   - *Pros*: Trivial to parse; simple if-statements in routing code.
   - *Cons*: Rigid; breaks every time a new modality is created; lacks versioning and configuration parameters.
2. **Integer Bitmasks**:
   - *Pros*: Extremely compact binary representation.
   - *Cons*: Limited to 64 flags; zero parameterization; obscure debugging; impossible to coordinate across decentralized teams without centralized bit allocation.
3. **Static Enumerations (`enum CapabilityType`)**:
   - *Pros*: Strong compile-time type checking.
   - *Cons*: Still requires schema updates and recompilation for every new feature.

---

## Trade-offs
- **Query Complexity**: Filtering models requires matching string keys and parsing SemVer ranges rather than checking simple boolean flags.
- **Typing Safety**: Property values are passed as key-value string maps, requiring capability-specific validation logic at the orchestrator layer.

---

## Consequences
- **Positive**:
  - **15-Year Extensibility**: The platform can support completely unprecedented AI modalities and skills decades into the future without a single breaking change to the core `ModelContract`.
  - **Granular Negotiation**: Clients and agents can specify exact requirements (e.g., `coding` with `languages: "rust,python"` and `version: ">=2.0.0"`).
  - **Decentralized Innovation**: Specialized research teams (e.g. Bengali NLU, Medical Reasoning) can define and advertise domain capabilities independently without waiting for core platform schema approvals.
- **Negative**:
  - Requires maintaining a capability validation library to prevent typos in capability names and properties.
