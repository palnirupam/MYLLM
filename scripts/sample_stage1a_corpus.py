"""
scripts/sample_stage1a_corpus.py — CLI Entrypoint for Deterministic Stage 1A Sampling.

Usage:
  python scripts/sample_stage1a_corpus.py \\
    --input-corpus /kaggle/working/dhruva-v1-assets/corpus/stage1a_train_master.jsonl \\
    --output-corpus /kaggle/working/dhruva-v1-assets/corpus/stage1a_train.jsonl \\
    --tokenizer-dir /kaggle/working/dhruva-v1-assets/tokenizer \\
    --target-tokens 100000000 \\
    --seed 20260817
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.training.data.sampler import sample_stage1a_corpus_cli

if __name__ == "__main__":
    sample_stage1a_corpus_cli()
