# MyLLM Specification: REST/JSON API v1

- **Document ID**: `SPEC-0014`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Public & Internal HTTP Endpoints, Streaming SSE & Wire Protocols
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & API Classification

The MyLLM v1 API provides a RESTful JSON interface conforming to OpenAPI 3.1 specifications. All public endpoints maintain high backward compatibility with standard LLM client ecosystems while supporting MyLLM's advanced multimodal `InputPart` offloading, extensible capabilities, and sandbox tool execution.

### Endpoint Roadmap Classification

| Classification | Endpoint | Method | Purpose |
| :--- | :--- | :--- | :--- |
| **MVP** | `/v1/chat/completions` | `POST` | Multi-turn conversational completion & streaming SSE. |
| **MVP** | `/v1/models` | `GET` | List available models, contract digests, and capabilities. |
| **MVP** | `/v1/models/{model_id}` | `GET` | Retrieve detailed model contract and hardware requirements. |
| **MVP** | `/v1/files` | `POST` | Upload multimodal binary files to StorageProvider; returns `ResourceReference`. |
| **MVP** | `/v1/files/{file_id}` | `GET` | Retrieve uploaded file metadata or signed download stream. |
| **Later** | `/v1/sessions` | `POST`/`GET`/`DEL`| Ephemeral multi-turn session lifecycle management. |
| **Later** | `/v1/tools` | `POST`/`GET` | Dynamic tool registration, schema inspection, and verification. |
| **Later** | `/v1/batches` | `POST`/`GET` | Asynchronous high-throughput offline batch processing. |
| **Later** | `/v1/jobs` | `GET`/`DEL` | General long-running job monitoring and cancellation. |
| **Later** | `/v1/evaluations` | `POST`/`GET` | Trigger automated benchmark evaluation runs. |
| **Internal**| `/v1/internal/datasets` | `POST`/`GET` | Ingest and validate DatasetManifest instances. |
| **Internal**| `/v1/internal/runs` | `POST`/`GET` | Register training runs, capture CheckpointManifest snapshots. |

---

## 2. MVP Endpoints

### 2.1 `POST /v1/chat/completions`

Executes a chat completion request with optional Server-Sent Events (`SSE`) streaming.

#### Request Headers:
- `Content-Type: application/json`
- `Authorization: Bearer <api_key>`
- `Accept: text/event-stream` *(when `stream: true`)*

#### Request Body Schema:
```json
{
  "model": "myllm/bengali-reasoning-7b:1.0.0",
  "messages": [
    {
      "role": "user",
      "content": "Explain the concept of quantum superposition in Bengali."
    }
  ],
  "temperature": 0.3,
  "top_p": 0.95,
  "max_tokens": 2048,
  "stream": true,
  "tools": [],
  "response_format": { "type": "text" }
}
```

#### Non-Streaming Response (`200 OK`):
```json
{
  "id": "chatcmpl-9f8a2b3c4d5e",
  "object": "chat.completion",
  "created": 1723800000,
  "model": "myllm/bengali-reasoning-7b:1.0.0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "কোয়ান্টাম সুপারপজিশন (Quantum Superposition) হলো কোয়ান্টাম বলবিদ্যার একটি মৌলিক নীতি..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 142,
    "total_tokens": 160
  }
}
```

#### Streaming SSE Wire Protocol (`text/event-stream`):
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"id":"chatcmpl-9f8a2b3c4d5e","object":"chat.completion.chunk","created":1723800000,"model":"myllm/bengali-reasoning-7b:1.0.0","choices":[{"index":0,"delta":{"role":"assistant","content":"কোয়ান্টাম"},"finish_reason":null}]}

data: {"id":"chatcmpl-9f8a2b3c4d5e","object":"chat.completion.chunk","created":1723800000,"model":"myllm/bengali-reasoning-7b:1.0.0","choices":[{"index":0,"delta":{"content":" সুপারপজিশন"},"finish_reason":null}]}

data: {"id":"chatcmpl-9f8a2b3c4d5e","object":"chat.completion.chunk","created":1723800000,"model":"myllm/bengali-reasoning-7b:1.0.0","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

### 2.2 `GET /v1/models`

Lists all registered models, their capabilities, and contract summaries.

#### Response (`200 OK`):
```json
{
  "object": "list",
  "data": [
    {
      "id": "myllm/bengali-reasoning-7b:1.0.0",
      "object": "model",
      "created": 1723800000,
      "owned_by": "myllm-core",
      "architecture_family": "ARCHITECTURE_FAMILY_TRANSFORMER_DECODER",
      "capabilities": [
        { "name": "reasoning", "version": "1.0.0" },
        { "name": "bengali", "version": "2.1.0" },
        { "name": "coding", "version": "1.0.0" }
      ],
      "max_context_length": 131072
    }
  ]
}
```

---

### 2.3 `POST /v1/files`

Uploads an image, document, audio clip, or dataset to `StorageProvider`, generating an immutable `ResourceReference` for use in multimodal prompts.

#### Request:
- `Content-Type: multipart/form-data`
- Body fields:
  - `file`: binary stream
  - `purpose`: "multimodal_prompt" | "fine_tuning" | "batch_input"

#### Response (`201 Created`):
```json
{
  "id": "file_89ab34cd56ef",
  "object": "file",
  "bytes": 2048576,
  "created_at": 1723800100,
  "filename": "financial_chart.png",
  "mime_type": "image/png",
  "content_hash": "sha256:b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78",
  "resource_reference": {
    "uri": "s3://myllm-uploads/tenant-1/images/b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78.png",
    "mime_type": "image/png",
    "content_hash": "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78",
    "size_bytes": 2048576,
    "storage_provider_id": "s3-primary"
  }
}
```

---

## 3. Standard Error Format (RFC 7807 Problem Details)

All API errors return a structured JSON response conforming to RFC 7807:

```json
{
  "type": "https://api.myllm.ai/errors/invalid-payload",
  "title": "Invalid Input Part Payload",
  "status": 400,
  "detail": "Large binary data cannot be passed inline. Please upload via POST /v1/files and pass a resource_reference.",
  "instance": "/v1/chat/completions/req_99214a",
  "error_code": "ERR_INLINE_BINARY_FORBIDDEN",
  "invalid_params": [
    {
      "name": "messages[0].parts[1]",
      "reason": "Inline binary payload exceeds maximum permitted inline limit of 1MB."
    }
  ]
}
```
