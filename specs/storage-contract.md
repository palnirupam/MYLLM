# MyLLM Specification: Storage Provider Contract & Content Addressability

- **Document ID**: `SPEC-0012`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: I/O Abstraction, Artifact Storage & Content Addressability
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Core Philosophy

The `StorageProvider` interface abstracts all persistent binary and structured I/O operations across local development environments, on-premises NVMe storage, and distributed cloud object stores (S3, GCS, Ceph).

### Key Architectural Invariants:
1. **Content-Addressable Artifact Identity**: Artifacts are stored under keys derived directly from their cryptographic SHA-256 digest:
   $$\text{artifact\_id} = \text{SHA256}(\text{canonical\_bytes}(\text{content}) \parallel \text{canonical\_json}(\text{metadata}))$$
2. **Write-Once, Read-Many (WORM)**: Once an artifact is written to a key, its content cannot be overwritten. Updates produce new content-addressed keys.
3. **Range-Read & Zero-Copy Support**: The storage interface MUST support byte-range slicing to enable direct memory-mapping (`mmap`) of `safetensors` weight shards without loading whole multi-gigabyte files into memory.

---

## 2. Protobuf Service & Message Specification (`myllm/storage/v1/storage.proto`)

```protobuf
syntax = "proto3";

package myllm.storage.v1;

import "google/protobuf/timestamp.proto";

option go_package = "github.com/myllm/core/gen/go/storage/v1;storagev1";
option java_package = "ai.myllm.storage.v1";

message StorageMetadata {
  string key = 1;
  uint64 size_bytes = 2;
  string sha256_hash = 3;
  string content_type = 4;
  google.protobuf.Timestamp created_at = 5;
  google.protobuf.Timestamp last_modified = 6;
  map<string, string> custom_tags = 7;
  bool is_tombstoned = 8;
  string tombstone_reason = 9;
}

message ReadRequest {
  string key = 1;
  // Optional byte range for zero-copy slicing
  optional uint64 offset_bytes = 2;
  optional uint64 length_bytes = 3;
}

message ReadResponse {
  bytes data = 1;
  StorageMetadata metadata = 2;
}

message WriteRequest {
  string key = 1;
  bytes data = 2;
  string content_type = 3;
  map<string, string> custom_tags = 4;
  // Expected SHA-256 for server-side integrity validation
  string expected_sha256 = 5;
}

message WriteResponse {
  StorageMetadata metadata = 1;
  bool already_existed = 2; // True if content-addressed deduplication occurred
}

message StatRequest {
  string key = 1;
}

message ExistsRequest {
  string key = 1;
}

message ExistsResponse {
  bool exists = 1;
}

message HashRequest {
  string key = 1;
}

message HashResponse {
  string sha256_hash = 1;
}

message ListRequest {
  string prefix = 1;
  uint32 page_size = 2;
  string page_token = 3;
}

message ListResponse {
  repeated StorageMetadata items = 1;
  string next_page_token = 2;
}

message DeleteRequest {
  string key = 1;
}

message DeleteResponse {
  bool success = 1;
}

message TombstoneStorageRequest {
  string key = 1;
  string reason = 2;
  string authorization_id = 3;
}

message TombstoneStorageResponse {
  bool success = 1;
  StorageMetadata metadata = 2;
}

// StorageProvider RPC Service Interface
service StorageService {
  rpc Read(ReadRequest) returns (ReadResponse);
  rpc Write(WriteRequest) returns (WriteResponse);
  rpc Stat(StatRequest) returns (StorageMetadata);
  rpc Exists(ExistsRequest) returns (ExistsResponse);
  rpc Hash(HashRequest) returns (HashResponse);
  rpc List(ListRequest) returns (ListResponse);
  rpc Delete(DeleteRequest) returns (DeleteResponse);
  rpc Tombstone(TombstoneStorageRequest) returns (TombstoneStorageResponse);
}
```

---

## 3. Abstract Interface Definitions

### Python Abstract Interface
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional, Dict, List

@dataclass(frozen=True)
class StorageMetadata:
    key: str
    size_bytes: int
    sha256_hash: str
    content_type: str
    created_at: datetime
    last_modified: datetime
    custom_tags: Dict[str, str]
    is_tombstoned: bool = False
    tombstone_reason: Optional[str] = None

class StorageProvider(ABC):
    @abstractmethod
    def read(self, key: str, offset: int = 0, length: Optional[int] = None) -> bytes:
        """Reads full byte slice or range from the storage key."""
        pass

    @abstractmethod
    def write(self, key: str, data: bytes, content_type: str = "application/octet-stream",
              custom_tags: Optional[Dict[str, str]] = None, expected_sha256: Optional[str] = None) -> StorageMetadata:
        """Writes data under key. Enforces content verification and immutability."""
        pass

    @abstractmethod
    def stat(self, key: str) -> StorageMetadata:
        """Retrieves object metadata without downloading payload bytes."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Returns True if the object exists and is not tombstoned."""
        pass

    @abstractmethod
    def hash(self, key: str) -> str:
        """Returns the hex SHA-256 digest of the object."""
        pass

    @abstractmethod
    def list(self, prefix: str = "", page_size: int = 1000) -> Iterator[StorageMetadata]:
        """Iterates over object metadata with matching key prefix."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Deletes key. Only permitted in development or temporary scratch buckets."""
        pass

    @abstractmethod
    def tombstone(self, key: str, reason: str, authorization_id: str) -> StorageMetadata:
        """Zeroes/excises data bytes while preserving signed metadata audit records."""
        pass
```

---

## 4. Standard Implementation Matrix

| Provider Name | Target Environment | Byte Range `mmap` | Zero-Copy DMA | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `LocalFilesystemStorage` | Local Workstation / CI / Single Node | Supported (`os.open` / `mmap`) | Supported via O_DIRECT | Development, Unit Testing, Edge Serving |
| `ObjectStorage` (S3/GCS) | Cloud Distributed Infrastructure | Supported via `Range:` HTTP header | Not direct (staged to local NVMe) | Model Registry, Global Checkpoints, Datasets |
| `GPUDirectStorage` (cuFile) | High-Performance DGX/H100 Clusters | Supported (GPU NVMe Direct DMA) | Supported (cuFileRead) | Ultra-fast multi-GPU model weight loading |
| `InMemoryStorage` | Unit Testing Suites | Simulated memory slices | N/A | Mock testing, isolation harnesses |
