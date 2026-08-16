# MyLLM Specification: Input & Multimodal Message Contract

- **Document ID**: `SPEC-0005`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Multimodal Messaging, Prompting & Offloaded Resource Handling
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Architectural Principles

The `InputContract` defines how users, agents, and external systems construct multimodal prompts and conversation turns for MyLLM.

### Fundamental Rules:
1. **Zero Large Inline Binaries**: Binary assets (images, audio waveforms, video frames, PDFs, datasets) MUST NEVER be passed as inline base64 strings inside API payloads. All media and files are uploaded to a `StorageProvider` first and passed by `ResourceReference` (URI + Content Hash + Size).
2. **Modality Polymorphism**: Every multimodal atom is modeled as an `InputPart` containing a typed `Modality` enum and a `oneof payload`.
3. **Deterministic Content-Addressing**: Every external resource is pinned by its cryptographic SHA-256 content hash, preventing cache poisoning and silent resource mutation during asynchronous inference pipelines.

---

## 2. Protobuf Specification (`myllm/input/v1/input_contract.proto`)

```protobuf
syntax = "proto3";

package myllm.input.v1;

import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";

option go_package = "github.com/myllm/core/gen/go/input/v1;inputv1";
option java_package = "ai.myllm.input.v1";

enum Modality {
  MODALITY_UNSPECIFIED = 0;
  MODALITY_TEXT = 1;
  MODALITY_IMAGE = 2;
  MODALITY_AUDIO = 3;
  MODALITY_VIDEO = 4;
  MODALITY_FILE = 5;
  MODALITY_STRUCTURED_DATA = 6;
}

enum Role {
  ROLE_UNSPECIFIED = 0;
  ROLE_SYSTEM = 1;
  ROLE_USER = 2;
  ROLE_ASSISTANT = 3;
  ROLE_TOOL = 4;
}

// Offloaded external or storage-backed resource
message ResourceReference {
  // Storage URI (e.g., "s3://myllm-uploads/2026/08/doc123.pdf", "file:///data/cache/...")
  string uri = 1;

  // IANA Media / MIME type (e.g., "image/png", "audio/opus", "application/pdf")
  string mime_type = 2;

  // Exact SHA-256 hash of the remote resource payload
  string content_hash = 3;

  // Exact size in bytes
  uint64 size_bytes = 4;

  // Optional StorageProvider identifier (e.g., "s3-primary", "local-nvme")
  string storage_provider_id = 5;

  // Expiration timestamp for temporary presigned or ephemeral session artifacts
  google.protobuf.Timestamp expires_at = 6;
}

// Multimodal atomic input element
message InputPart {
  // Explicit Modality
  Modality modality = 1;

  // Oneof payload container: Only small text or structured schemas are inline.
  // Any binary payload MUST use resource_ref.
  oneof payload {
    // UTF-8 text content (Max inline limit: 1,048,576 bytes / 1MB)
    string text_content = 2;

    // Content-addressed reference to external binary / large document
    ResourceReference resource_ref = 3;

    // Structured JSON / Protobuf Struct for schema inputs or tool outputs
    google.protobuf.Struct structured_data = 4;
  }

  // Modality-specific layout/bounding-box/temporal metadata (optional)
  map<string, string> metadata = 5;
}

// Invocation of a tool requested by the model
message ToolCall {
  // Unique tool call invocation ID
  string id = 1;
  // Function / Tool name
  string name = 2;
  // JSON-encoded arguments string matching ToolContract.input_schema
  string arguments_json = 3;
}

// Complete conversation message turn
message Message {
  // Participant role in conversation
  Role role = 1;

  // Ordered multimodal parts making up this turn
  repeated InputPart parts = 2;

  // Optional author name (useful for multi-agent conversations)
  string name = 3;

  // Associated tool call ID when role == ROLE_TOOL
  string tool_call_id = 4;

  // Model-generated tool call requests when role == ROLE_ASSISTANT
  repeated ToolCall tool_calls = 5;

  // Message generation / submission timestamp
  google.protobuf.Timestamp timestamp = 6;
}
```

---

## 3. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/input-contract.json",
  "title": "Message",
  "type": "object",
  "required": ["role", "parts"],
  "properties": {
    "role": {
      "type": "string",
      "enum": ["ROLE_SYSTEM", "ROLE_USER", "ROLE_ASSISTANT", "ROLE_TOOL"]
    },
    "name": { "type": "string" },
    "tool_call_id": { "type": "string" },
    "tool_calls": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "name", "arguments_json"],
        "properties": {
          "id": { "type": "string" },
          "name": { "type": "string" },
          "arguments_json": { "type": "string" }
        }
      }
    },
    "parts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["modality"],
        "properties": {
          "modality": {
            "type": "string",
            "enum": [
              "MODALITY_TEXT",
              "MODALITY_IMAGE",
              "MODALITY_AUDIO",
              "MODALITY_VIDEO",
              "MODALITY_FILE",
              "MODALITY_STRUCTURED_DATA"
            ]
          },
          "text_content": { "type": "string", "maxLength": 1048576 },
          "resource_ref": {
            "type": "object",
            "required": ["uri", "mime_type", "content_hash", "size_bytes"],
            "properties": {
              "uri": { "type": "string" },
              "mime_type": { "type": "string" },
              "content_hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
              "size_bytes": { "type": "integer" },
              "storage_provider_id": { "type": "string" },
              "expires_at": { "type": "string", "format": "date-time" }
            }
          },
          "structured_data": { "type": "object" },
          "metadata": {
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

## 4. Multimodal Message Example

```json
{
  "role": "ROLE_USER",
  "parts": [
    {
      "modality": "MODALITY_TEXT",
      "text_content": "Please analyze this architectural diagram and the attached financial spreadsheet. Compare the projected cloud infrastructure costs."
    },
    {
      "modality": "MODALITY_IMAGE",
      "resource_ref": {
        "uri": "s3://myllm-uploads/tenant-918/images/arch-diagram-v3.png",
        "mime_type": "image/png",
        "content_hash": "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78",
        "size_bytes": 1845200,
        "storage_provider_id": "s3-us-east-1"
      },
      "metadata": {
        "width": "2048",
        "height": "1536",
        "color_space": "sRGB"
      }
    },
    {
      "modality": "MODALITY_FILE",
      "resource_ref": {
        "uri": "s3://myllm-uploads/tenant-918/docs/q3-infra-forecast.xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "content_hash": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
        "size_bytes": 489201,
        "storage_provider_id": "s3-us-east-1"
      }
    }
  ]
}
```
