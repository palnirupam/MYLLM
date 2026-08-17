"""
tests/unit/test_stage1a_sampler.py — Deterministic Sampler Invariance & Reproducibility Tests.
Proves that: same seed + same master corpus = 100% bit-for-bit identical selected sequence.
"""

import sys
import json
import hashlib
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.training.data.sampler import DeterministicStage1ASampler


def calculate_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_deterministic_sampler_reproducibility():
    temp_dir = tempfile.mkdtemp()
    try:
        root = Path(temp_dir)
        corpus_path = root / "master_corpus.jsonl"

        # 1. Create realistic multi-lingual, multi-domain master corpus
        sample_docs = [
            {"text": "Linear algebra and calculus underpin modern deep neural network optimizations.", "language": "English", "domain": "math", "source": "open-web-math/open-web-math"},
            {"text": "বাংলা সাহিত্য ও ব্যাকরণের সমৃদ্ধ ইতিহাস হাজার বছরের প্রাচীন ঐতিহ্যের ধারক।", "language": "Bengali", "domain": "indic_humanities", "source": "wikimedia/wikipedia"},
            {"text": "हिंदी भारत की प्रमुख भाषा है जो देवनागरी लिपि में अत्यंत वैज्ञानिक तरीके से लिखी जाती है।", "language": "Hindi", "domain": "indic_humanities", "source": "wikimedia/wikipedia"},
            {"text": "The educational benefits of active recall and spaced repetition in cognitive science.", "language": "English", "domain": "educational_web", "source": "HuggingFaceFW/fineweb-edu"},
            {"text": "Photosynthesis is the biochemical mechanism whereby radiant energy converts into chemical energy.", "language": "English", "domain": "encyclopedia", "source": "wikimedia/wikipedia"},
            {"text": "Euler's identity e^(i*pi) + 1 = 0 connects fundamental mathematical constants.", "language": "English", "domain": "math", "source": "open-web-math/open-web-math"},
            {"text": "রবীন্দ্রনাথ ঠাকুরের রচিত সাহিত্য সমগ্র বিশ্বে উচ্চ প্রশংসিত ও সমাদৃত।", "language": "Bengali", "domain": "indic_humanities", "source": "wikimedia/wikipedia"},
            {"text": "कंप्यूटर विज्ञान में एल्गोरिदम का अध्ययन समस्याओं को हल करने की कुशलता निर्धारित करता है।", "language": "Hindi", "domain": "indic_humanities", "source": "wikimedia/wikipedia"},
        ]

        with open(corpus_path, "w", encoding="utf-8") as f:
            for i in range(100):
                doc = sample_docs[i % len(sample_docs)].copy()
                doc["doc_id"] = f"master-{i:05d}"
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # Load test tokenizer
        tok_dir = Path("dhruva-v1-assets/tokenizer")
        if not (tok_dir / "tokenizer.json").exists():
            tok_dir = Path("tokenizer")
        tok = BPETokenizer.load(str(tok_dir))

        # 2. Run Sampler Twice with identical seed (20260817)
        out1_path = root / "sampled_run1.jsonl"
        out2_path = root / "sampled_run2.jsonl"

        sampler1 = DeterministicStage1ASampler(seed=20260817, target_tokens=500)
        res1 = sampler1.sample_corpus(str(corpus_path), str(out1_path), tok)

        sampler2 = DeterministicStage1ASampler(seed=20260817, target_tokens=500)
        res2 = sampler2.sample_corpus(str(corpus_path), str(out2_path), tok)

        # 3. Assert Bit-for-Bit Identical Hash & Content
        hash1 = calculate_sha256(out1_path)
        hash2 = calculate_sha256(out2_path)

        assert hash1 == hash2, f"Determinism failure: SHA256 mismatch! {hash1} != {hash2}"
        assert res1["actual_selected_train_tokens"] == res2["actual_selected_train_tokens"]
        assert res1["total_documents_selected"] == res2["total_documents_selected"]
        assert res1["per_language_tokens"] == res2["per_language_tokens"]
        assert res1["per_domain_tokens"] == res2["per_domain_tokens"]

        # Verify line-by-line exact sequence
        lines1 = out1_path.read_text(encoding="utf-8").strip().split("\n")
        lines2 = out2_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines1) == len(lines2)
        for idx, (l1, l2) in enumerate(zip(lines1, lines2)):
            assert l1 == l2, f"Line {idx} mismatch between deterministic runs!"

        print(f"  [PASS] Deterministic reproducibility verified: 2 runs produced 100% identical SHA256 ({hash1})")

        # 4. Assert Different Seed Produces Different Ordering
        out3_path = root / "sampled_run3_different_seed.jsonl"
        sampler3 = DeterministicStage1ASampler(seed=9999999, target_tokens=500)
        res3 = sampler3.sample_corpus(str(corpus_path), str(out3_path), tok)
        hash3 = calculate_sha256(out3_path)

        assert hash1 != hash3, "Different seeds should produce different document sequences!"
        print(f"  [PASS] Seed sensitivity verified: seed=9999999 produced distinct SHA256 ({hash3})")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_diversity_and_boundary_accounting():
    temp_dir = tempfile.mkdtemp()
    try:
        root = Path(temp_dir)
        corpus_path = root / "master_corpus.jsonl"

        sample_docs = [
            {"text": "Linear algebra and calculus underpin modern deep neural network optimizations.", "language": "English", "domain": "math", "source": "open-web-math/open-web-math"},
            {"text": "বাংলা সাহিত্য ও ব্যাকরণের সমৃদ্ধ ইতিহাস হাজার বছরের প্রাচীন ঐতিহ্যের ধারক।", "language": "Bengali", "domain": "indic_humanities", "source": "wikimedia/wikipedia"},
            {"text": "हिंदी भारत की प्रमुख भाषा है जो देवनागरी लिपि में अत्यंत वैज्ञानिक तरीके से लिखी जाती है।", "language": "Hindi", "domain": "indic_humanities", "source": "wikimedia/wikipedia"},
            {"text": "The educational benefits of active recall and spaced repetition in cognitive science.", "language": "English", "domain": "educational_web", "source": "HuggingFaceFW/fineweb-edu"},
        ]

        with open(corpus_path, "w", encoding="utf-8") as f:
            for i in range(200):
                doc = sample_docs[i % len(sample_docs)].copy()
                doc["doc_id"] = f"master-{i:05d}"
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        tok_dir = Path("dhruva-v1-assets/tokenizer")
        if not (tok_dir / "tokenizer.json").exists():
            tok_dir = Path("tokenizer")
        tok = BPETokenizer.load(str(tok_dir))

        out_path = root / "sampled_diversity.jsonl"
        sampler = DeterministicStage1ASampler(seed=20260817, target_tokens=800)
        res = sampler.sample_corpus(str(corpus_path), str(out_path), tok)

        # Assert all requested languages are present
        assert "Bengali" in res["per_language_tokens"] and res["per_language_tokens"]["Bengali"] > 0
        assert "Hindi" in res["per_language_tokens"] and res["per_language_tokens"]["Hindi"] > 0
        assert "English" in res["per_language_tokens"] and res["per_language_tokens"]["English"] > 0

        # Assert document boundary delta is recorded and bounded
        delta = res["document_boundary_delta_tokens"]
        assert abs(delta) < 150, f"Boundary delta {delta} exceeds single document length!"
        print(f"  [PASS] Linguistic diversity & boundary tracking verified (Delta: {delta:+d} tokens)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_deterministic_sampler_reproducibility()
    test_diversity_and_boundary_accounting()
    print("\nALL STAGE 1A SAMPLER TESTS PASSED")
