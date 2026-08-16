# MyLLM Specification: Capability Contract

- **Document ID**: `SPEC-0006`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Capability Registry & Dynamic Feature Negotiation
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Architectural Principles

The `CapabilityContract` provides an extensible, polymorphic registry for advertising, discovering, and validating model skills, modalities, and specializations.

### Why Not Fixed Booleans?
Traditional AI platforms use rigid boolean flags (e.g., `supports_vision: bool`, `supports_tools: bool`, `supports_json: bool`). Over a 15-year horizon, this design fails catastrophically:
1. **Schema Bloat**: Every new frontier skill (e.g., formal theorem proving, audio synthesis, multi-agent debate, Bengali dialect translation) requires mutating the core schema.
2. **Lack of Granularity**: A boolean `supports_tools: true` cannot express timeout limits, Wasm sandbox compatibility, schema dialects, or parallel execution caps.
3. **No Versioning**: As capability mechanics evolve (e.g., Tool Calling v1 vs v2 with grammar constraints), booleans cannot communicate backward-incompatible feature shifts.

In MyLLM, capabilities are defined as a structured tuple: `name` (string) + `version` (SemVer) + `properties` (key-value string map).

---

## 2. Protobuf Specification (`myllm/capability/v1/capability.proto`)

```protobuf
syntax = "proto3";

package myllm.capability.v1;

option go_package = "github.com/myllm/core/gen/go/capability/v1;capabilityv1";
option java_package = "ai.myllm.capability.v1";

// Atomic Capability Declaration
message Capability {
  // Canonical capability name (e.g., "coding", "reasoning", "long_context", "bengali", "function_calling")
  string name = 1;

  // Semantic version of the capability contract (e.g., "1.0.0", "2.1.0")
  string version = 2;

  // Dynamic, capability-specific configuration and constraints
  map<string, string> properties = 3;
}

// Container for a collection of declared model capabilities
message Capabilities {
  repeated Capability capabilities = 1;
}

// Query requirement used by orchestrators to match models
message CapabilityRequirement {
  // Required capability name
  string name = 1;

  // SemVer range requirement (e.g., ">=1.2.0", "^2.0.0")
  string version_constraint = 2;

  // Mandatory property key-value constraints
  map<string, string> required_properties = 3;
}

// Model filter request
message CapabilityQuery {
  repeated CapabilityRequirement requirements = 1;
}
```

---

## 3. Standard Capability Taxonomy & Property Conventions

| Capability Name | Version | Example Key Properties | Description |
| :--- | :--- | :--- | :--- |
| `coding` | `1.0.0` | `languages`: "python,rust,go,typescript", `fill_in_middle`: "true" | Code generation, analysis, and execution capabilities. |
| `reasoning` | `1.0.0` | `cot_style`: "deepseek_r1", `supports_reflection`: "true", `max_reasoning_tokens`: "32768" | Multi-step mathematical and logical chain-of-thought. |
| `long_context` | `1.0.0` | `max_tokens`: "131072", `effective_needle_depth`: "0.99" | Extended context window support with retrieval fidelity. |
| `function_calling` | `2.0.0` | `supports_parallel`: "true", `strict_json_schema`: "true", `grammar_guided`: "true" | Tool calling, schema adherence, and API orchestration. |
| `bengali` | `2.1.0` | `script_support`: "Bengali,Latin", `dialects`: "Standard,Dhaka,Sylhet,Chittagong" | High-fidelity Bengali natural language understanding & generation. |
| `vision_understanding`| `1.0.0` | `max_resolution`: "4096x4096", `supported_formats`: "png,jpeg,webp" | Visual OCR, document layout analysis, and diagram reasoning. |
| `structured_output` | `1.0.0` | `engine`: "outlines_regex", `formats`: "json,yaml,regex" | Guaranteed schema and grammar-constrained token generation. |

---

## 4. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/capability-contract.json",
  "title": "Capabilities",
  "type": "object",
  "required": ["capabilities"],
  "properties": {
    "capabilities": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "version"],
        "properties": {
          "name": { "type": "string", "pattern": "^[a-z0-9_.-]+$" },
          "version": { "type": "string" },
          "properties": {
            "type": "object",
            "additionalProperties": { "type": "string" }
          }
        }
      }
    }
  }
}
```

---

## 5. Complete JSON Example

```json
{
  "capabilities": [
    {
      "name": "coding",
      "version": "1.0.0",
      "properties": {
        "languages": "python,rust,go,typescript,c,cpp,sql",
        "fill_in_middle": "true",
        "test_generation": "true"
      }
    },
    {
      "name": "reasoning",
      "version": "1.0.0",
      "properties": {
        "cot_prompt_template": "deepseek_r1_style",
        "supports_backtracking": "true",
        "max_reasoning_tokens": "32768"
      }
    },
    {
      "name": "long_context",
      "version": "1.0.0",
      "properties": {
        "max_tokens": "131072",
        "rope_scaling": "yarn",
        "needle_in_haystack_accuracy": "0.998"
      }
    },
    {
      "name": "function_calling",
      "version": "2.0.0",
      "properties": {
        "supports_parallel_calls": "true",
        "strict_json_schema_enforced": "true",
        "supports_streaming_tool_calls": "true"
      }
    },
    {
      "name": "bengali",
      "version": "2.1.0",
      "properties": {
        "script_support": "Bengali,Latin",
        "dialect_coverage": "Standard,Dhaka,Chittagong,Sylhet",
        "indic_glue_benchmark_score": "88.4"
      }
    },
    {
      "name": "structured_output",
      "version": "1.0.0",
      "properties": {
        "grammar_engine": "regex_finite_automata",
        "supported_schemas": "json_schema_draft_2020_12"
      }
    }
  ]
}
```

---

## 6. Capability Matching Engine Contract

When an incoming prompt or orchestrator requests a model with specific requirements:
1. **Name Equivalence**: The candidate model must declare the capability `name`.
2. **SemVer Compatibility**: The advertised `version` must satisfy the requester's SemVer expression (e.g. `^1.0.0` matches `1.2.0` but rejects `2.0.0`).
3. **Property Subsetting**: For each key-value pair in `required_properties`, the model's capability MUST contain the key and satisfy the value constraint (e.g. `languages` list inclusion, numeric inequality for `max_tokens`).
