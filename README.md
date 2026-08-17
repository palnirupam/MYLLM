<div align="center">

# 🌟 Dhruva V1 (~100M)
### *A Lightweight Multilingual Foundation Model & Compound AI Architecture*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-GQA__SwiGLU-7B1FA2?style=for-the-badge)](https://github.com/palnirupam/MYLLM)
[![Parameters](https://img.shields.io/badge/Parameters-99.5M-00897B?style=for-the-badge)](https://github.com/palnirupam/MYLLM)
[![Vocabulary](https://img.shields.io/badge/Vocab-64K%20BPE-1E88E5?style=for-the-badge)](https://github.com/palnirupam/MYLLM)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](LICENSE)

<br/>

[**Key Features**](#-key-features) • [**Architecture**](#-architecture-specifications) • [**Compound AI**](#-compound-ai-system) • [**Quickstart**](#-quickstart-guide) • [**Evaluation**](#-benchmarks--performance) • [**Roadmap**](#-development-roadmap)

</div>

---

> [!NOTE]
> **Stage 1A Base Foundation Release**:  
> The current released weights (`releases/dhruva-v1-100m/inference_model/`) represent an **experimental base pre-trained foundation model** trained on $100\text{M}$ multilingual tokens (Wikipedia EN/BN/HI, FineWeb-Edu, and OpenWebMath).  
> Like raw foundation base LLMs (e.g. GPT-2 or LLaMA-Base), it is optimized for **next-token continuation and language modeling**. Supervised Instruction Tuning (SFT) and Chat Alignment are scheduled for Stage 2.

---

## 🚀 Key Features

- ⚡ **Ultra-Lightweight & Efficient**: ~99.5 Million parameters engineered to run in low-compute and edge environments ($< 500\text{ MB}$ peak GPU VRAM).
- 🌐 **Native Multilingual Representation**: Trained from scratch with a custom **64,000-vocabulary Byte-Level BPE** tokenizer covering **English**, **Bengali (বাংলা)**, and **Hindi (हिंदी)** with strict **Unicode NFC** normalization.
- 🧠 **Compound AI Intelligence Layer**: Augments raw Transformer generation with a modular execution pipeline: **Rule Routing**, **Deterministic Tools** (Safe Calculator & Sandboxed Python), **BM25 Evidence Retrieval**, and **Multi-Stage Verifiers**.
- 🏎️ **Fast Local Inference**: Achieves **$110 - 230\text{ tokens/second}$** generation throughput on consumer GPUs (e.g. RTX 3050) and dual Tesla T4s.
- 🔬 **100% Reproducible & Audit-Verified**: Fully deterministic sampling pipeline (`seed=20260817`), exact SHA256 checksum tracking, and zero data contamination between splits.

---

## 📐 Architecture Specifications

Dhruva V1 implements a modern, memory-efficient decoder-only Transformer backbone:

```
                  ┌───────────────────────────────────────────┐
                  │           Dhruva V1 Transformer           │
                  │      [ 8 Layers | d_model = 768 ]         │
                  └─────────────────────┬─────────────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
 ┌───────────────────┐                                     ┌───────────────────┐
 │ Grouped-Query Attn│                                     │    SwiGLU FFN     │
 │ 12 Q / 4 KV Heads │                                     │  dim = 2048 (FFN) │
 │  RoPE (θ = 10000) │                                     │      RMSNorm      │
 └───────────────────┘                                     └───────────────────┘
```

| Hyperparameter | Value | Description / Engineering Rationale |
| :--- | :---: | :--- |
| **Total Parameters** | **`99,496,704`** | Exactly calculated from weights (~99.5M) |
| **Hidden Dimension ($d_{\text{model}}$)** | **`768`** | Optimal latent capacity for sub-100M models |
| **Transformer Layers ($n_{\text{layers}}$)** | **`8`** | Balanced depth-to-latency trade-off |
| **Query Attention Heads ($n_{\text{heads}}$)** | **`12`** | Head dimension $d_k = 64$ |
| **KV Attention Heads ($n_{\text{kv\_heads}}$)** | **`4`** | Grouped-Query Attention (3:1 query-to-KV compression) |
| **Intermediate Size (FFN)** | **`2048`** | SwiGLU projection ($8/3 \times d_{\text{model}}$) |
| **Vocabulary Size** | **`64,000`** | `Dhruva-BPE-64K` Byte-Level BPE |
| **Context Window Length** | **`512 tokens`** | Stage 1A pre-training sequence context |
| **Positional Embeddings** | **RoPE** | Rotary Position Embedding ($\text{base} = 10,000.0$) |
| **Layer Normalization** | **RMSNorm** | Root Mean Square Normalization ($\epsilon = 10^{-5}$) |
| **Tied Word Embeddings** | **True** | Input embedding and output projection matrices shared |
| **Training Precision** | **FP16 AMP** | Automatic Mixed Precision with PyTorch `GradScaler` |

---

## 🛡️ Compound AI System

Dhruva combines neural language generation with deterministic symbolic reasoning and retrieval to prevent hallucinations on factual and numerical tasks:

```
                            USER QUERY
                                │
                                ▼
                       ┌─────────────────┐
                       │   Rule Router   │
                       └────────┬────────┘
                                │
        ┌───────────────────────┼───────────────────────┬───────────────────────┐
        ▼                       ▼                       ▼                       ▼
  ┌───────────┐           ┌───────────┐           ┌───────────┐           ┌───────────┐
  │ FastPath  │           │ ToolPath  │           │ Retrieve  │           │ ThinkPath │
  │  (Direct  │           │(Calculator│           │  (BM25 /  │           │ (Bounded  │
  │Generation)│           │  / REPL)  │           │ Grounding)│           │Reasoning) │
  └─────┬─────┘           └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
        │                       │                       │                       │
        └───────────────────────┼───────────────────────┼───────────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    Verifiers    │
                       │(Math/Fact/Struct│
                       └────────┬────────┘
                                │
                                ▼
                      PASS / REVISE / ABSTAIN
```

- **Router**: Classifies intent and dispatches queries to the most resource-efficient path.
- **Tools**: Sandboxed Python REPL and precision math calculator for zero-error calculations.
- **Evidence Retriever**: Sparse BM25 engine providing factual grounded context.
- **Verifiers**: Formal checks for mathematical precision, factual grounding, and structure.

---

## ⚡ Quickstart Guide

### 1. Installation & Environment Setup

```bash
# Clone repository
git clone https://github.com/palnirupam/MYLLM.git
cd MYLLM

# Install required dependencies
pip install torch safetensors tokenizers

# Initialize Git LFS and pull model weights
git lfs install
git lfs pull
```

### 2. Local Text Generation (CLI)

```bash
# English Text Continuation
python scripts/generate.py \
  --model releases/dhruva-v1-100m/inference_model \
  --prompt "Photosynthesis is the process where" \
  --max-tokens 50 \
  --temperature 0.7

# Bengali (বাংলা) Text Continuation
python scripts/generate.py \
  --model releases/dhruva-v1-100m/inference_model \
  --prompt "বাংলা সাহিত্যের উল্লেখযোগ্য বৈশিষ্ট্য হলো" \
  --max-tokens 40 \
  --temperature 0.7
```

### 3. Python API Integration

```python
from myllm.runtime.local.inference import LocalInferenceRuntime

# Load model onto available device (CUDA / CPU automatically detected)
runtime = LocalInferenceRuntime(model_path="releases/dhruva-v1-100m/inference_model")

# Generate completion
prompt = "Artificial intelligence and machine learning"
output = runtime.generate(
    prompt=prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_k=40,
    top_p=0.9
)

print(f"\nGenerated Output:\n{output}")
```

---

## 📊 Benchmarks & Performance

### 1. Hardware & Memory Footprint
*Measured locally on an **NVIDIA GeForce RTX 3050 Laptop GPU (6GB)**:*

| Metric | Measured Value | Analysis |
| :--- | :---: | :--- |
| **Model Weights File** | **`379.56 MB`** | `model.safetensors` ($397,994,400$ bytes) |
| **Tokenizer Table** | **`4.72 MB`** | Full $64,000$-entry BPE JSON table |
| **Peak GPU VRAM (Inference)** | **`476.5 MB`** | Extremely lightweight; runs under $1\text{ GB}$ VRAM |
| **Generation Speed** | **`110 – 230 tok/s`** | Real-time high-throughput generation |
| **Model Load Time** | **`~1.88 s`** | Instantaneous cold start |
| **Validation Loss / Perplexity** | **`4.238 / 69.29 PPL`** | Solid statistical language modeling on held-out data |

### 2. Full 17-Suite Test Battery
Verify all unit, integration, tokenizer, and architecture invariance tests ($130$ individual checks):

```bash
python -c "
import subprocess, sys
test_files = [
    'tests/unit/test_model_audit.py', 'tests/unit/test_tokenizer_audit.py',
    'tests/unit/test_checkpoint_audit.py', 'tests/unit/test_schemas_policy.py',
    'tests/unit/test_router.py', 'tests/unit/test_fast_path.py',
    'tests/unit/test_sandbox.py', 'tests/unit/test_sandbox_security.py',
    'tests/unit/test_verifier_specialized.py', 'tests/unit/test_policy_hardening.py',
    'tests/unit/test_orchestrator.py', 'tests/unit/test_phase2_tools_verifiers.py',
    'tests/unit/test_phase3_retrieval.py', 'tests/unit/test_phase4_reasoning.py',
    'tests/unit/test_real_checkpoint_eval.py', 'tests/integration/test_fp16_training_smoke.py',
    'tests/unit/test_stage1a_sampler.py'
]
for f in test_files:
    subprocess.run([sys.executable, f], check=True)
print('>>> ALL 17 TEST SUITES PASSED (130/130 GREEN)!')
"
```

---

## 🛠️ Pre-Training & Asset Pipeline

The training pipeline is fully configured for distributed multi-GPU environments (e.g. **Kaggle 2x Tesla T4 DDP**):

```bash
# 1. Hardware & NCCL Preflight Check
torchrun --nproc_per_node=2 scripts/kaggle_ddp_preflight.py

# 2. Deterministic Stratified 100M Corpus Sampling (seed=20260817)
python scripts/sample_stage1a_corpus.py \
  --input-corpus /kaggle/working/dhruva-v1-assets/corpus/stage1a_train_master.jsonl \
  --output-corpus /kaggle/working/dhruva-v1-assets/corpus/stage1a_train.jsonl \
  --tokenizer-dir /kaggle/working/dhruva-v1-assets/tokenizer \
  --target-tokens 100000000 \
  --seed 20260817

# 3. Stage 1A Pre-Training Master Execution
torchrun --nproc_per_node=2 scripts/run_kaggle_stage1a.py \
  --assets-dir /kaggle/working/dhruva-v1-assets \
  --config configs/dhruva_v1_production.yaml \
  --execute-stage1a
```

---

## 🗺️ Development Roadmap

- [x] **Phase 0**: Architecture design, 64K Byte-Level BPE tokenizer, and streaming data pipeline.
- [x] **Phase 1**: Core Transformer backbone with GQA, SwiGLU, and FP16 AMP DDP training.
- [x] **Phase 2**: Deterministic Tool Path (Safe Calculator & Sandboxed Python REPL).
- [x] **Phase 3**: Evidence-grounded Retrieval (BM25 & Factual Verifier).
- [x] **Phase 4**: Bounded Adaptive Reasoning (ThinkPath & Telemetry Engine).
- [x] **Stage 1A Base Model**: Pre-training on $100\text{M}$ multilingual tokens & release (`releases/dhruva-v1-100m`).
- [ ] **Stage 1B Extended Pre-training**: Scale corpus to $500\text{M}+$ tokens with extended sequence length ($1024 / 2048$).
- [ ] **Stage 2 SFT (Supervised Fine-Tuning)**: Multi-turn instruction following, structured reasoning, and dialogue tuning in English, Bengali, and Hindi.
- [ ] **Stage 3 Alignment**: Direct Preference Optimization (DPO) and safety hardening.

---

## 📜 License & Provenance

- **Codebase & Architecture**: Licensed under the **[Apache-2.0 License](LICENSE)**.
- **Corpus Data Provenance**:
  - *Wikimedia Wikipedia* (`20231101.en`, `20231101.bn`, `20231101.hi`) — [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
  - *FineWeb-Edu* (`sample-100BT`) — [ODC-By-1.0](https://opendatacommons.org/licenses/by/1-0/)
  - *OpenWebMath* — [Open-Web-Math Permissive](https://huggingface.co/datasets/open-web-math/open-web-math)

<div align="center">
<b>MYLLM / Dhruva V1</b> • Developed by Nirupam Pal • Built from scratch with PyTorch
</div>