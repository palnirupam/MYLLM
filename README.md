# MYLLM — Dhruva V1 (~100M)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Architecture: Transformer-GQA](https://img.shields.io/badge/Architecture-GQA_SwiGLU-purple.svg)]()
[![Model Status: Stage_1A_Base](https://img.shields.io/badge/Status-Stage_1A_Base_Pretrained-orange.svg)]()

**MYLLM (Dhruva V1)** is a research-oriented, lightweight multilingual (~100M parameter) foundation model and Compound AI System engineered from scratch. It is designed specifically for low-compute environments (such as dual Tesla T4s on Kaggle or entry-level consumer GPUs like the RTX 3050).

> **Important Note on Current Status (Stage 1A Base Model)**:  
> The current released weights (`releases/dhruva-v1-100m/inference_model/`) represent an **experimental base pre-trained foundation model** trained on initial multilingual web text (Wikipedia EN/BN/HI, FineWeb-Edu, and OpenWebMath).  
> Like any raw base autoregressive LLM (e.g. raw GPT-2 or LLaMA-Base), it performs **next-token autocompletion and continuation** rather than conversational question-answering. Chat alignment, supervised instruction tuning (SFT), and RLHF/DPO are part of upcoming development stages.

---

## Architecture Specification

Dhruva V1 utilizes a modern decoder-only Transformer backbone with Grouped-Query Attention (GQA) and SwiGLU activations:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | **~100 Million** | Lightweight, fast inference on commodity hardware |
| **Hidden Dimension ($d_{\text{model}}$)** | **768** | Latent representation size |
| **Transformer Layers ($n_{\text{layers}}$)** | **8** | Sequential attention & feed-forward blocks |
| **Attention Heads ($n_{\text{heads}}$)** | **12** | Query attention heads |
| **KV Heads ($n_{\text{kv\_heads}}$)** | **4** | Grouped-Query Attention (3:1 query-to-KV ratio) |
| **Intermediate FFN Size** | **2048** | SwiGLU non-linear projection |
| **Vocabulary Size** | **64,000** | Byte-Level BPE (`Dhruva-BPE-64K`) |
| **Max Sequence Length** | **512 tokens** | Stage 1A context window |
| **Positional Embeddings** | **RoPE** | Rotary Position Embedding ($\theta = 10,000$) |
| **Normalization** | **RMSNorm** | Root Mean Square layer normalization ($\epsilon = 10^{-5}$) |
| **Tied Word Embeddings** | **True** | Shared input embedding and output LM head weights |
| **Training Precision** | **FP16 AMP** | Automatic Mixed Precision with `GradScaler` |

---

## Multilingual Tokenizer (`Dhruva-BPE-64K`)

Dhruva uses a custom-trained **64,000-vocabulary Byte-Level BPE** tokenizer:
- **Normalization**: Strict **Unicode NFC** (preserves Indic ligatures and mathematical formatting).
- **Linguistic Coverage**: English, Bengali (বাংলা), Hindi (हिंदी), and Python/LaTeX symbols.
- **Special Tokens**: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<tool_call>`, `<tool_result>`, `<scratchpad>`, `</scratchpad>`, `<evidence>`, `</evidence>`.
- **Implementation**: Native HuggingFace Rust backend supporting high-throughput multi-threaded batch tokenization (`encode_batch`).

---

## Dhruva Compound AI System

Rather than relying solely on next-token generation for factual and numerical tasks, Dhruva incorporates a modular Compound AI architecture:

```
                      USER QUERY
                          │
                          ▼
                 ┌─────────────────┐
                 │   Rule Router   │
                 └────────┬────────┘
                          │
       ┌──────────────────┼──────────────────┬─────────────────┐
       ▼                  ▼                  ▼                 ▼
 ┌───────────┐      ┌───────────┐      ┌───────────┐     ┌───────────┐
 │ FastPath  │      │ ToolPath  │      │ Retrieve  │     │ ThinkPath │
 │  (Direct  │      │(Calculator│      │  (BM25 /  │     │ (Bounded  │
 │Inference) │      │  / REPL)  │      │ Grounding)│     │Reasoning) │
 └─────┬─────┘      └─────┬─────┘      └─────┬─────┘     └─────┬─────┘
       │                  │                  │                 │
       └──────────────────┼──────────────────┼─────────────────┘
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

- **Router**: Dispatches queries based on intent (arithmetic, code execution, factual inquiry, or general text).
- **Tools**: Sandboxed Python REPL and precision math calculator.
- **Retriever**: Evidence-grounded BM25 knowledge retriever.
- **Verifiers**: Multi-stage verification to filter out false premises and ungrounded hallucinations.

---

## Local Quickstart

### 1. Prerequisites & Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/palnirupam/MYLLM.git
cd MYLLM
pip install -r requirements.txt
```

Ensure Git LFS is installed to pull model weights:
```bash
git lfs install
git lfs pull
```

### 2. Run Direct Model Generation

```bash
# English Text Continuation
python scripts/generate.py \
  --model releases/dhruva-v1-100m/inference_model \
  --prompt "Photosynthesis is the process where" \
  --max-tokens 50 \
  --temperature 0.7

# Bengali Text Continuation
python scripts/generate.py \
  --model releases/dhruva-v1-100m/inference_model \
  --prompt "বাংলা সাহিত্যের উল্লেখযোগ্য বৈশিষ্ট্য হলো" \
  --max-tokens 40 \
  --temperature 0.7
```

### 3. Python API Usage

```python
from myllm.runtime.local.inference import LocalInferenceRuntime

# Load model weights onto available device (CUDA / CPU)
runtime = LocalInferenceRuntime(model_path="releases/dhruva-v1-100m/inference_model")

# Generate text
output = runtime.generate(
    prompt="In computer science, an algorithm is",
    max_new_tokens=40,
    temperature=0.7,
    top_k=40,
    top_p=0.9
)
print(output)
```

---

## Full Evaluation & Test Suite

### Run Comprehensive Benchmark Battery
Execute the 12-domain diagnostic evaluation harness (profiles throughput, latency, repetition score, and peak VRAM):
```bash
python scripts/evaluate_released_model.py
```

### Run Unit & Integration Tests (17 Suites, 130 Checks)
```bash
python -c "
import subprocess, sys
test_files = [
    'tests/unit/test_model_audit.py',
    'tests/unit/test_tokenizer_audit.py',
    'tests/unit/test_checkpoint_audit.py',
    'tests/unit/test_schemas_policy.py',
    'tests/unit/test_router.py',
    'tests/unit/test_fast_path.py',
    'tests/unit/test_sandbox.py',
    'tests/unit/test_sandbox_security.py',
    'tests/unit/test_verifier_specialized.py',
    'tests/unit/test_policy_hardening.py',
    'tests/unit/test_orchestrator.py',
    'tests/unit/test_phase2_tools_verifiers.py',
    'tests/unit/test_phase3_retrieval.py',
    'tests/unit/test_phase4_reasoning.py',
    'tests/unit/test_real_checkpoint_eval.py',
    'tests/integration/test_fp16_training_smoke.py',
    'tests/unit/test_stage1a_sampler.py'
]
for f in test_files:
    res = subprocess.run([sys.executable, f], check=True)
print('ALL 17 TEST SUITES PASSED!')
"
```

---

## Measured Hardware & Resource Profile

Tested locally on an entry-level **NVIDIA GeForce RTX 3050 Laptop GPU (6GB)**:

- **Peak GPU VRAM**: `476.5 MB` (Inference requires $< 1\text{GB}$ VRAM).
- **Generation Throughput**: `110 - 230 tokens/sec` (depending on batch/sequence length).
- **Model Weight Size**: `379.56 MB` (`model.safetensors`).
- **Tokenizer Size**: `4.72 MB` (`tokenizer.json`).

---

## Kaggle Pre-Training Pipeline

The training infrastructure is optimized for **Kaggle 2x Tesla T4 (DDP)** environments:

1. **Hardware Pre-flight**:
   ```bash
   torchrun --nproc_per_node=2 scripts/kaggle_ddp_preflight.py
   ```
2. **Deterministic Corpus Sampling (100M Tokens)**:
   ```bash
   python scripts/sample_stage1a_corpus.py \
     --input-corpus /kaggle/working/dhruva-v1-assets/corpus/stage1a_train_master.jsonl \
     --output-corpus /kaggle/working/dhruva-v1-assets/corpus/stage1a_train.jsonl \
     --tokenizer-dir /kaggle/working/dhruva-v1-assets/tokenizer \
     --target-tokens 100000000 \
     --seed 20260817
   ```
3. **Stage 1A Pre-Training Execution**:
   ```bash
   torchrun --nproc_per_node=2 scripts/run_kaggle_stage1a.py \
     --assets-dir /kaggle/working/dhruva-v1-assets \
     --config configs/dhruva_v1_production.yaml \
     --execute-stage1a
   ```

---

## Development Roadmap

- [x] **Phase 0**: Architecture design, Byte-level BPE tokenizer (64K), and multi-lingual ingestion pipeline.
- [x] **Phase 1**: Core Transformer backbone with GQA, SwiGLU, and FP16 AMP training loop.
- [x] **Phase 2**: Deterministic Tool Path (Calculator & Sandboxed Python REPL).
- [x] **Phase 3**: Evidence-grounded Retrieval (BM25 & Factual Verifier).
- [x] **Phase 4**: Bounded Adaptive Reasoning (ThinkPath & Telemetry).
- [x] **Stage 1A Base Model**: Pre-training on 100M tokens & model release (`releases/dhruva-v1-100m`).
- [ ] **Stage 1B Extended Pre-training**: Scale pre-training corpus to 500M+ tokens with longer sequence length (1024 / 2048).
- [ ] **Stage 2 SFT (Supervised Fine-Tuning)**: Multi-turn instruction following, tool calling, and dialogue tuning in English, Bengali, and Hindi.
- [ ] **Stage 3 Alignment**: Direct Preference Optimization (DPO) and safety hardening.

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
Corpus sources preserve their respective licenses: Wikimedia (CC-BY-SA-4.0), FineWeb-Edu (ODC-By-1.0), and OpenWebMath.