# MyLLM Specification: Ephemeral Session Schema

- **Document ID**: `SPEC-0010`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Conversation State, Ephemeral Working Memory & Turn Orchestration
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Architectural Principles

A `Session` encapsulates the ephemeral state of an active multi-turn interaction between a user (or agent) and MyLLM.

### Fundamental Separation of Concerns:
- **`Session` (This Spec)**: Strictly ephemeral, conversational, and transient. It tracks active conversation history, transient tool working directories, intermediate variables, and temporary file handles. Sessions expire and can be safely deleted or flushed to cold storage at any time.
- **`PersistentMemory` (`SPEC-0011`)**: Long-term, distilled knowledge that survives session termination, tenant rotation, model architecture migrations, and tokenization changes. A `Session` DOES NOT own or mutate `PersistentMemory` directly without explicit distillation actions.

---

## 2. Protobuf Specification (`myllm/session/v1/session.proto`)

```protobuf
syntax = "proto3";

package myllm.session.v1;

import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";
import "myllm/input/v1/input_contract.proto";

option go_package = "github.com/myllm/core/gen/go/session/v1;sessionv1";
option java_package = "ai.myllm.session.v1";

enum SessionStatus {
  SESSION_STATUS_UNSPECIFIED = 0;
  SESSION_STATUS_ACTIVE = 1;
  SESSION_STATUS_IDLE = 2;
  SESSION_STATUS_SUSPENDED = 3;
  SESSION_STATUS_ARCHIVED = 4;
  SESSION_STATUS_EXPIRED = 5;
}

// Session configuration overrides
message SessionConfig {
  // Override default model ID for this session
  string model_id = 1;
  // System prompt override
  string system_prompt = 2;
  // Maximum context window tokens to retain in active memory
  uint32 max_context_tokens = 3;
  // Temperature override (optional)
  optional float temperature = 4;
  // Top-p sampling override (optional)
  optional float top_p = 5;
  // Tool IDs enabled in this session
  repeated string enabled_tool_ids = 6;
}

// Ephemeral Session Entity
message Session {
  // Canonical Session UUID
  string session_id = 1;

  // Tenant / User owner identifier
  string user_id = 2;

  // Organization or Project namespace
  string organization_id = 3;

  // Lifecycle status
  SessionStatus status = 4;

  // Session-level execution parameters
  SessionConfig config = 5;

  // Full chronological conversation turn history
  repeated myllm.input.v1.Message message_history = 6;

  // Key-value scratchpad for agent reasoning, scratch variables, and tool state
  google.protobuf.Struct temporary_state = 7;

  // Ephemeral file/resource attachments associated with this active session
  repeated myllm.input.v1.ResourceReference resource_references = 8;

  // Token usage telemetry for current session
  uint64 total_prompt_tokens = 9;
  uint64 total_completion_tokens = 10;

  // Timestamps
  google.protobuf.Timestamp created_at = 11;
  google.protobuf.Timestamp updated_at = 12;
  google.protobuf.Timestamp expires_at = 13;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/session-schema.json",
  "title": "Session",
  "type": "object",
  "required": [
    "session_id",
    "user_id",
    "status",
    "message_history",
    "temporary_state",
    "created_at",
    "updated_at",
    "expires_at"
  ],
  "properties": {
    "session_id": { "type": "string", "format": "uuid" },
    "user_id": { "type": "string" },
    "organization_id": { "type": "string" },
    "status": {
      "type": "string",
      "enum": [
        "SESSION_STATUS_ACTIVE",
        "SESSION_STATUS_IDLE",
        "SESSION_STATUS_SUSPENDED",
        "SESSION_STATUS_ARCHIVED",
        "SESSION_STATUS_EXPIRED"
      ]
    },
    "config": {
      "type": "object",
      "properties": {
        "model_id": { "type": "string" },
        "system_prompt": { "type": "string" },
        "max_context_tokens": { "type": "integer" },
        "temperature": { "type": "number" },
        "top_p": { "type": "number" },
        "enabled_tool_ids": { "type": "array", "items": { "type": "string" } }
      }
    },
    "message_history": {
      "type": "array",
      "items": { "$ref": "https://specs.myllm.ai/v1/input-contract.json" }
    },
    "temporary_state": { "type": "object" },
    "resource_references": {
      "type": "array",
      "items": { "type": "object" }
    },
    "total_prompt_tokens": { "type": "integer" },
    "total_completion_tokens": { "type": "integer" },
    "created_at": { "type": "string", "format": "date-time" },
    "updated_at": { "type": "string", "format": "date-time" },
    "expires_at": { "type": "string", "format": "date-time" }
  }
}
```

---

## 4. Complete JSON Example

```json
{
  "session_id": "c8f5e1b2-7a39-4d2b-9e41-0f81a7b45620",
  "user_id": "usr_99214890",
  "organization_id": "org_enterprise_corp",
  "status": "SESSION_STATUS_ACTIVE",
  "config": {
    "model_id": "myllm/bengali-reasoning-7b:1.0.0",
    "system_prompt": "You are MyLLM, an expert AI programming and reasoning assistant.",
    "max_context_tokens": 32768,
    "temperature": 0.2,
    "top_p": 0.95,
    "enabled_tool_ids": ["myllm.tools.python_interpreter:1.0.0"]
  },
  "message_history": [
    {
      "role": "ROLE_USER",
      "parts": [
        {
          "modality": "MODALITY_TEXT",
          "text_content": "Can you compute the sum of prime numbers under 1000 in Python?"
        }
      ],
      "timestamp": "2026-08-16T10:00:00Z"
    },
    {
      "role": "ROLE_ASSISTANT",
      "parts": [
        {
          "modality": "MODALITY_TEXT",
          "text_content": "I will execute a Python script to compute the sum of all primes below 1,000."
        }
      ],
      "tool_calls": [
        {
          "id": "call_prime_calc_01",
          "name": "python_interpreter",
          "arguments_json": "{\"code\":\"def is_prime(n):\\n    if n < 2: return False\\n    for i in range(2, int(n**0.5)+1):\\n        if n % i == 0: return False\\n    return True\\nprint(sum(x for x in range(1000) if is_prime(x)))\"}"
        }
      ],
      "timestamp": "2026-08-16T10:00:02Z"
    }
  ],
  "temporary_state": {
    "sandbox_container_id": "wasm_box_882910",
    "active_turn_count": 2,
    "last_tool_invoked": "myllm.tools.python_interpreter:1.0.0"
  },
  "resource_references": [],
  "total_prompt_tokens": 128,
  "total_completion_tokens": 64,
  "created_at": "2026-08-16T09:59:58Z",
  "updated_at": "2026-08-16T10:00:02Z",
  "expires_at": "2026-08-16T11:59:58Z"
}
```
