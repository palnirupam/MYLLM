# MyLLM Specification: Tool Contract

- **Document ID**: `SPEC-0007`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Tool & Plugin Interface, Security Sandboxing & Auditing
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Core Principles

The `ToolContract` provides an immutable, technology-neutral definition for external tools, functions, APIs, and computational modules executable by MyLLM agentic pipelines.

### Key Architectural Invariants:
1. **Technology Neutrality**: The tool contract defines *what* the tool does, *how* its inputs/outputs are structured (JSON Schema), and *what resource limits* it operates under. It does NOT bind to Docker, Kubernetes, Linux cgroups, or Wasm runtimes. Execution backends are abstracted via `SandboxRuntime`.
2. **Explicit Least-Privilege Permissions**: Tools cannot execute with default ambient authority. Every network endpoint, filesystem path, memory threshold, and CPU budget must be declared explicitly.
3. **Structured Schemas**: Inputs and outputs must adhere strictly to JSON Schema (Draft 2020-12), enabling constrained grammar generation, runtime validation, and automated mock generation during testing.

---

## 2. Protobuf Specification (`myllm/tool/v1/tool_contract.proto`)

```protobuf
syntax = "proto3";

package myllm.tool.v1;

import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";

option go_package = "github.com/myllm/core/gen/go/tool/v1;toolv1";
option java_package = "ai.myllm.tool.v1";

enum AuthenticationType {
  AUTH_TYPE_NONE = 0;
  AUTH_TYPE_API_KEY = 1;
  AUTH_TYPE_OAUTH2_BEARER = 2;
  AUTH_TYPE_MUTUAL_TLS = 3;
  AUTH_TYPE_IAM_ROLE = 4;
}

// Fine-grained security permissions
message ToolPermissions {
  // Whether outbound network calls are allowed
  bool allow_network = 1;

  // Allowed domain whitelists (e.g. ["api.weather.gov", "*.internal.org"])
  repeated string allowed_domains = 2;

  // Explicit read-only path prefixes
  repeated string filesystem_read_paths = 3;

  // Explicit writable path prefixes
  repeated string filesystem_write_paths = 4;

  // When true, all writes are restricted to an ephemeral scratch directory wiped on exit
  bool temp_filesystem_only = 5;

  // Direct Host OS execution access (MUST ALWAYS BE FALSE for untrusted tools)
  bool allow_host_os_access = 6;

  // Maximum memory quota in megabytes
  uint32 max_memory_mb = 7;

  // Maximum CPU quota in millicores (e.g. 1000 = 1 full CPU core)
  uint32 max_cpu_millicores = 8;

  // Execution timeout in milliseconds (hard kill threshold)
  uint32 timeout_ms = 9;
}

// Authentication configuration for secure tool invocation
message AuthenticationRequirements {
  AuthenticationType auth_type = 1;
  // Vault secret path or environment key reference (e.g., "vault://secrets/my-api-key")
  string secret_reference = 2;
  // Header name (e.g., "Authorization", "X-API-Key")
  string target_header = 3;
}

// Security and audit classification
message AuditMetadata {
  // Whether input and output payloads should be written to tamper-evident audit logs
  bool execution_logging_enabled = 1;
  // Whether sensitive input properties should be redacted in logs
  repeated string redact_property_keys = 2;
  // Data classification tier: "PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"
  string data_classification = 3;
}

// Canonical Tool Contract
message ToolContract {
  // Canonical Tool Identifier (e.g., "myllm.tools.python_repl:1.0.0")
  string tool_id = 1;

  // Human- and LLM-readable tool name
  string name = 2;

  // Semantic version
  string version = 3;

  // Prompt-injected description explaining usage, edge cases, and semantics
  string description = 4;

  // JSON Schema (Draft 2020-12) describing expected input parameters
  string input_schema_json = 5;

  // JSON Schema (Draft 2020-12) describing return values
  string output_schema_json = 6;

  // Declared sandbox permissions and resource boundaries
  ToolPermissions permissions = 7;

  // Credential and authorization requirements
  AuthenticationRequirements authentication = 8;

  // Governance and compliance audit configuration
  AuditMetadata audit_metadata = 9;

  // Creation timestamp
  google.protobuf.Timestamp created_at = 10;
}

// Tool Execution Invocation
message ToolExecutionRequest {
  string execution_id = 1;
  string tool_id = 2;
  // Raw JSON arguments string submitted by LLM tool call
  string arguments_json = 3;
  // Session / Tenant context
  string session_id = 4;
  string tenant_id = 5;
  // Ephemeral environment variables injected by orchestrator
  map<string, string> environment_variables = 6;
}

// Tool Execution Response
message ToolExecutionResponse {
  string execution_id = 1;
  bool is_error = 2;
  // Structured JSON result or error message
  string output_json = 3;
  // Execution time in milliseconds
  uint32 duration_ms = 4;
  // Peak memory consumed in megabytes
  uint32 peak_memory_mb = 5;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/tool-contract.json",
  "title": "ToolContract",
  "type": "object",
  "required": [
    "tool_id",
    "name",
    "version",
    "description",
    "input_schema_json",
    "output_schema_json",
    "permissions"
  ],
  "properties": {
    "tool_id": { "type": "string" },
    "name": { "type": "string", "pattern": "^[a-zA-Z0-9_]+$" },
    "version": { "type": "string" },
    "description": { "type": "string" },
    "input_schema_json": { "type": "string" },
    "output_schema_json": { "type": "string" },
    "permissions": {
      "type": "object",
      "required": ["allow_network", "temp_filesystem_only", "allow_host_os_access", "max_memory_mb", "timeout_ms"],
      "properties": {
        "allow_network": { "type": "boolean" },
        "allowed_domains": { "type": "array", "items": { "type": "string" } },
        "filesystem_read_paths": { "type": "array", "items": { "type": "string" } },
        "filesystem_write_paths": { "type": "array", "items": { "type": "string" } },
        "temp_filesystem_only": { "type": "boolean" },
        "allow_host_os_access": { "type": "boolean", "const": false },
        "max_memory_mb": { "type": "integer", "maximum": 4096 },
        "max_cpu_millicores": { "type": "integer" },
        "timeout_ms": { "type": "integer", "maximum": 60000 }
      }
    },
    "authentication": {
      "type": "object",
      "properties": {
        "auth_type": { "type": "string" },
        "secret_reference": { "type": "string" },
        "target_header": { "type": "string" }
      }
    },
    "audit_metadata": {
      "type": "object",
      "properties": {
        "execution_logging_enabled": { "type": "boolean" },
        "redact_property_keys": { "type": "array", "items": { "type": "string" } },
        "data_classification": { "type": "string" }
      }
    }
  }
}
```

---

## 4. Complete JSON Example (Code Interpreter Tool)

```json
{
  "tool_id": "myllm.tools.python_interpreter:1.0.0",
  "name": "python_interpreter",
  "version": "1.0.0",
  "description": "Executes isolated Python 3 code in a secure sandboxed environment. Returns stdout, stderr, and generated charts/files. Use for data processing, symbolic math, and algorithm verification.",
  "input_schema_json": "{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"type\":\"object\",\"required\":[\"code\"],\"properties\":{\"code\":{\"type\":\"string\",\"description\":\"The Python 3 source code to execute.\"},\"timeout_seconds\":{\"type\":\"integer\",\"default\":10,\"maximum\":30}}}",
  "output_schema_json": "{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"type\":\"object\",\"required\":[\"exit_code\",\"stdout\",\"stderr\"],\"properties\":{\"exit_code\":{\"type\":\"integer\"},\"stdout\":{\"type\":\"string\"},\"stderr\":{\"type\":\"string\"},\"generated_files\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}}}",
  "permissions": {
    "allow_network": false,
    "allowed_domains": [],
    "filesystem_read_paths": [],
    "filesystem_write_paths": ["/tmp/workspace"],
    "temp_filesystem_only": true,
    "allow_host_os_access": false,
    "max_memory_mb": 512,
    "max_cpu_millicores": 2000,
    "timeout_ms": 15000
  },
  "authentication": {
    "auth_type": "AUTH_TYPE_NONE",
    "secret_reference": "",
    "target_header": ""
  },
  "audit_metadata": {
    "execution_logging_enabled": true,
    "redact_property_keys": [],
    "data_classification": "INTERNAL"
  }
}
```
