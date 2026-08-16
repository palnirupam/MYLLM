# MyLLM Specification: Semantic Versioning & Evolution Policy

- **Document ID**: `SPEC-0015`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Contract Evolution, Protocol Buffers Hygiene & Deprecation Lifecycle
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Core Principles

All MyLLM contracts, schemas, artifacts, and public APIs adhere strictly to **Semantic Versioning 2.0.0** (`MAJOR.MINOR.PATCH`):
- **`MAJOR`**: Incompatible breaking changes to schemas, wire protocols, or behavioral semantics.
- **`MINOR`**: Backward-compatible feature additions, optional fields, new capabilities, or new endpoints.
- **`PATCH`**: Backward-compatible bug fixes, performance optimizations, or documentation corrections.

---

## 2. Protocol Buffers Evolution Rules

Protobuf definitions (`.proto`) across all MyLLM packages MUST adhere to the following backward- and forward-compatibility rules:

### Allowed Changes (Minor / Patch):
1. **Adding Fields**: New fields may be added with new, unused field tag numbers. All new fields MUST be optional.
2. **Adding Enum Constants**: New enum constants may be added at the end of the enum definition. Code MUST handle unknown enum values gracefully.
3. **Adding RPC Methods**: New RPC methods may be added to existing services.

### Forbidden Breaking Changes (Requires Major Version Bump):
1. **Never Change Field Numbers**: A field number can NEVER be re-assigned to a different field.
2. **Never Change Field Types**: Changing `int32` to `string` or `message` to `repeated` breaks wire decoding.
3. **Never Remove Fields Directly**: When removing a deprecated field, the field number and name MUST be marked as `reserved`:

```protobuf
message ModelContract {
  reserved 14, 15, 22 to 25;
  reserved "legacy_pickle_path", "obsolete_flag";

  // Active fields
  string model_id = 1;
  // ...
}
```

---

## 3. Artifact Immutability Invariant

1. **Write-Once Integrity**: Once an artifact (Model Artifact, Tensor Manifest, Dataset Manifest, Checkpoint) is published and assigned a SHA-256 digest, it is **permanently immutable**.
2. **Zero In-Place Edits**: If a typo or metadata fix is required, a new artifact MUST be published with a distinct content-addressable hash.
3. **Registry Pointers**: Human-readable tags (e.g. `latest`, `prod`) are mutable aliases pointing to immutable content-addressed digests.

---

## 4. Deprecation & Sunset Lifecycle

```
    Active Contract
          |
          v
  [Deprecation Announced] ---> HTTP 200 + 'Deprecation: @timestamp' + 'Sunset: @timestamp'
          |                    Minimum 180-day grace period
          v
   [Sunset Reached]       ---> HTTP 410 Gone / Migration Gateway Required
          |
          v
   [Archived Storage]     ---> Read-only historical storage for legal/audit compliance
```

1. **Minimum Grace Period**: Any public API or contract marked for deprecation MUST provide a minimum of **180 calendar days** before decommissioning.
2. **Standard Headers**: Deprecated endpoints MUST return RFC 8594 compliant HTTP headers:
   - `Deprecation: @<unix_timestamp>`
   - `Sunset: <HTTP-date>` (e.g. `Sunset: Wed, 16 Feb 2027 00:00:00 GMT`)
   - `Link: <https://specs.myllm.ai/migrations/v1-to-v2>; rel="sunset"`
3. **Archival & Auditing**: Deprecated artifacts and contracts are never permanently deleted from registry catalogs; they transition to archived status to preserve reproducible execution for historical runs.
