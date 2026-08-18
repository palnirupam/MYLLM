# Dhruva V2 Design Specification

## Scope

Dhruva V2 is a new, dense multilingual decoder-only research model designed
for distributed training on Kaggle with 2 x 16 GB Tesla T4 GPUs. The V1
checkpoint is an immutable reference and is never modified by V2 tooling.

The target is better data efficiency and more stable instruction behavior,
not a claim of parity with large proprietary models.

## Primary Candidate Architecture

| Component | Choice | Rationale |
| --- | --- | --- |
| Parameters | 112,382,208 | Better token/parameter balance for the available compute |
| Vocabulary | 48,000 | Reduces embedding cost while retaining multilingual capacity |
| Hidden size | 768 | Preserves efficient 64-dimensional attention heads |
| Layers | 12 | More compositional depth than V1 without excessive activation memory |
| Query heads | 12 | Head dimension is exactly 64 |
| KV heads | 4 | 3:1 GQA reduces KV-cache and attention memory |
| FFN | SwiGLU, 2,048 intermediate | Proven compute-efficient V1 width with greater depth |
| Position encoding | RoPE | Efficient extrapolatable positional representation |
| Normalization | RMSNorm, eps=1e-5 | Stable and inexpensive pre-norm blocks |
| QK normalization | Per-head RMSNorm | Reduces attention-logit instability |
| Embeddings | tied input/output | Saves parameters and regularizes the output space |
| Context | 1,024 target; 512 smoke tests | Longer context without requiring a large first run |
| Dropout | 0.0 | Token-scale training and deterministic evaluation |

The architecture formula gives 112,382,208 trainable parameters with tied
embeddings. Compared with V1, embedding parameters fall from 49.15M to 36.86M
while the transformer body grows from 50.34M to about 75.52M. The instantiated
count must match before training.

This remains a candidate until a real 2 x T4 throughput and peak-VRAM smoke
test passes. A larger model is not accepted if it would receive too few tokens
per parameter or make resumable Kaggle training impractical.

The exact unique trainable parameter count is 112,382,208 for the primary
configuration: projection biases are disabled, QK RMSNorm has two independent
per-head norms, and input/output embeddings are tied. The scientific training
budget is cumulative non-ignored next-token labels (`loss-bearing tokens`).
Source, packed, EOS, discarded-tail, and padding counters remain separate.
The formula is `V*d + L*(4*d*d + 2*d*kv_heads*head_dim +
3*d*ffn + 2*d + 2*head_dim) + d`, with the tied output projection adding zero
unique parameters.

## Reliability Design

Confidence is not taken from raw softmax probability. V2 will expose three
separate signals:

1. token likelihood for generation diagnostics;
2. an answerability/abstention decision trained on supported and unsupported
   examples;
3. deterministic verification for math, code, and retrieved evidence.

The model must be allowed to answer that evidence is insufficient. Retrieval
and tools remain external execution paths and are not hidden inside the base
language model.

## Training Direction

1. Build and audit the 48K tokenizer on the actual multilingual mixture.
2. Run a short architecture smoke test before any long Kaggle job.
3. Pretrain with packed sequences, gradient checkpointing, FP16 AMP, and
   resumable checkpoints.
4. Use clean mixed-task SFT with assistant-only loss masking.
5. Evaluate factuality, language adherence, repetition, calibration, and
   abstention separately. A heuristic fluency score is not factual accuracy.

## Data Contract

The V2 base corpus is token-budgeted, not document-count-balanced. The initial
mixture is a starting allocation and must be rebalanced after tokenizer audit:

| Slice | Target token share | Rule |
| --- | ---: | --- |
| Bengali | 12% | Clean Bengali sources; no transliteration-only copies |
| Hindi | 12% | Clean Devanagari sources; no duplicate translations |
| Other Indic languages | 20% | Tamil, Telugu, Sanskrit, Urdu and other audited languages |
| English educational/encyclopedic | 36% | High-quality explanatory prose |
| Mathematics | 12% | Deduplicated expository and verified problem text |
| Code and technical text | 8% | License-traceable code with comments and documentation |

No generated text is allowed in base pretraining. Synthetic examples are
restricted to later SFT and must pass an executable verifier: `ast.parse` for
code and exact-answer checks for mathematics.

Every accepted document must have `source`, `source_revision`, `language`,
`domain`, `license`, `document_id`, and `preprocessing_version` metadata, a
pinned source revision, NFC-normalized text, and a content hash.
Exact normalized duplicates are removed globally before train/validation split;
held-out benchmark fragments are rejected from the training corpus. The
quality gate fails closed when rejection, duplicate, script-mismatch, or
contamination thresholds are exceeded.

## T4 Constraints

V2 training scripts are Kaggle-only and must use one DDP process per T4 GPU.
Local execution is limited to static validation, CPU shape tests, tokenizer
audits, and checkpoint manifest checks. No script should silently start a
local CUDA training run.

The initial effective batch is 32 packed sequences: 2 GPUs x micro-batch 1 x
16 accumulation steps. At context 1,024 this is at most 32,768 tokens per
optimizer step, preserving the original batch-token target while using both
GPUs.

The first real run must print and verify: model parameter count, source data
hashes, tokenizer hash, base/reference paths, target tokens, effective batch,
and checkpoint output path.

Checkpoints are schema-versioned complete artifacts. They include model,
optimizer, scheduler, GradScaler, token counters/cursor, rank-specific RNG
states, artifact identities, and runtime metadata. Resume rejects missing state
or architecture/tokenizer/data/config/world-size identity mismatches. The
default retention policy keeps only three optimizer-heavy checkpoints. Exact
bitwise reproducibility is promised only within matching hardware, software,
world-size, sampler, and artifact identities.

## Non-goals

- no MoE in the first V2 run;
- no hidden chain-of-thought training;
- no unsupported universal-language claim;
- no factuality claim from perplexity or heuristic similarity alone.
