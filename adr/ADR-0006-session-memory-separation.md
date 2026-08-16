# ADR-0006: Strict Architectural Separation of Ephemeral Session from Persistent Memory

## Status: Accepted
## Date: 2026-08-16

---

## Context
Conversational AI systems must manage two fundamentally different categories of state:
1. **Conversational Working State**: Ephemeral turn history, intermediate reasoning scratchpads, uncommitted tool outputs, and temporary uploaded file attachments during an active interaction.
2. **Long-Term Knowledge & Facts**: Enduring user preferences, enterprise rules, project architecture invariants, and distilled knowledge meant to persist for months or years.

Many existing chatbot frameworks couple these concepts together by either treating the raw conversation log as the sole memory store or appending persistent memories directly into the session document.

This architectural coupling leads to catastrophic issues over a long horizon:
- **Session Bloat & Eviction**: Large conversation histories must eventually be truncated or garbage-collected due to context window and storage limits, inadvertently wiping permanent user preferences.
- **Model Upgrade Breakage**: When migrating across model generations with different tokenizers or context formats, re-parsing raw multi-year session logs becomes brittle and computationally expensive.
- **Privacy & Compliance Collisions**: Data privacy mandates (e.g. "Clear chat history") should not necessarily wipe distilled organization-wide project knowledge, and vice versa.

---

## Decision
We will enforce a **Strict Architectural Separation** between **`Session`** (`SPEC-0010`) and **`PersistentMemory`** (`SPEC-0011`).

1. **`Session` is Ephemeral and Disposable**:
   - Manages immediate turn-by-turn context, token counts, and scratch variables.
   - Has a defined Time-to-Live (TTL) and can be purged, archived, or reset without data loss to long-term memory.
2. **`PersistentMemory` is Durable and Model-Agnostic**:
   - Represents distilled, factual knowledge, user preferences, and project guidelines.
   - Stored independently in canonical text/structured schemas with decoupled vector embeddings (`EmbeddingVectorRef`).
   - Survives indefinitely across model replacements, tokenizer upgrades, and session resets.
3. **Explicit Distillation Pathway**:
   - Moving information from a `Session` into `PersistentMemory` occurs via an explicit, auditable distillation action (either user-directed or via an automated memory-extraction pipeline).

---

## Alternatives Considered
1. **Monolithic Unified Conversation State (Single Session History as Memory)**:
   - *Pros*: Simple architecture; single database collection.
   - *Cons*: High context token consumption on every turn; no semantic indexing; deleting a conversation accidentally destroys permanent knowledge; unmanageable across 15 years.
2. **In-Context System Prompt Injection Only**:
   - *Pros*: Memory logic handled entirely outside database schema by frontend prompt builders.
   - *Cons*: Lacks standardized access control, sharing scopes (team vs private), and vector similarity search integration.

---

## Trade-offs
- **Explicit Synchronization**: Requires maintaining a dedicated memory retrieval and injection pipeline that searches `PersistentMemory` and injects relevant facts into active `Session` prompts.
- **Memory Distillation Overhead**: Requires background worker jobs or agent turns to distill raw conversations into structured facts.

---

## Consequences
- **Positive**:
  - **Longevity Across Model Upgrades**: Models and tokenizers can be completely swapped without losing user preferences or project facts.
  - **Zero Unintended Memory Loss**: Users and tenants can safely wipe active session histories for privacy or performance without corrupting their persistent knowledge base.
  - **Granular Access Control**: Permissions (Private User vs Team Shared vs Org Public) can be managed cleanly on `PersistentMemory` records.
  - **Cost Efficiency**: Active session prompts only load semantically relevant memory fragments via vector search rather than replaying massive historical chat logs.
- **Negative**:
  - Requires developers to understand and interact with two distinct API entities (`/v1/sessions` and `/v1/memories`).
