import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from myllm.core.tokenizer.bpe import BPETokenizer

def build_mock_multilingual_corpus(output_path: Path):
    """
    Builds a small, representative multilingual corpus for fast tokenizer testing.
    In a real scenario, this would stream from HuggingFace datasets.
    """
    print("Building representative multilingual sample...")
    
    # Using mock text with different language characteristics
    english = "The quick brown fox jumps over the lazy dog. Machine learning is the study of computer algorithms that can improve automatically through experience and by the use of data. " * 500
    bengali = "মেশিন লার্নিং বা যন্ত্র শিক্ষণ হলো কৃত্রিম বুদ্ধিমত্তার একটি শাখা। এটি কম্পিউটারকে মানুষের মত শিখতে ও সিদ্ধান্ত নিতে সাহায্য করে। " * 500
    hindi = "मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक शाखा है। यह कंप्यूटर को डेटा से सीखने और बिना स्पष्ट प्रोग्रामिंग के कार्य करने में सक्षम बनाती है। " * 500
    code = "def calculate_hash(content: bytes) -> str:\n    import hashlib\n    return hashlib.sha256(content).hexdigest()\n" * 500
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(english + "\n" + bengali + "\n" + hindi + "\n" + code)
    
    print(f"Sample corpus saved to {output_path} ({os.path.getsize(output_path) / 1024:.2f} KB)")


import os
import sys
import json
import time
import glob
from pathlib import Path
from transformers import AutoTokenizer
from myllm.core.tokenizer.bpe import BPETokenizer

def load_corpora(directory: Path):
    corpora = {}
    total_text = ""
    for file in directory.glob("*_sample.txt"):
        lang = file.stem.split('_')[0].capitalize()
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            corpora[lang] = text
            total_text += text + "\n"
    return corpora, total_text

def measure_speed(tokenizer, text, is_hf=False):
    # Encoding speed
    start = time.time()
    if is_hf:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    enc_time = time.time() - start
    
    # Decoding speed
    start = time.time()
    if is_hf:
        tokenizer.decode(encoded)
    else:
        tokenizer.decode(encoded)
    dec_time = time.time() - start
    
    return enc_time, dec_time, len(encoded)

def run_tokenizer_experiment(corpora: dict, full_text: str, candidates: list, baselines: list):
    d_model = 768
    base_params = 100_000_000 # 100M base
    
    results = []
    
    # 1. Train and Evaluate Candidates
    for vocab_size in candidates:
        print(f"\n--- Training Candidate: {vocab_size} Vocab ---")
        start_t = time.time()
        tokenizer = BPETokenizer.train_from_texts([full_text], vocab_size=vocab_size)
        train_time = time.time() - start_t
        print(f"Trained in {train_time:.2f} seconds.")
        
        # Calculate parameters
        embedding_params = vocab_size * d_model
        output_params = vocab_size * d_model # logically same, but in total model they are shared
        overhead = embedding_params # Since tied, only embedding adds to total
        
        res = {
            "name": f"Dhruva-BPE-{vocab_size//1000}k",
            "vocab_size": vocab_size,
            "train_time": train_time,
            "embedding_params_M": embedding_params / 1e6,
            "output_params_M": embedding_params / 1e6, # same
            "total_params_M": (base_params + overhead) / 1e6,
            "overhead_pct": (overhead / (base_params + overhead)) * 100,
            "lang_stats": {}
        }
        
        for lang, text in corpora.items():
            enc_time, dec_time, num_tokens = measure_speed(tokenizer, text, is_hf=False)
            num_bytes = len(text.encode('utf-8'))
            num_chars = len(text)
            num_words = len(text.split())
            
            res["lang_stats"][lang] = {
                "tokens": num_tokens,
                "bytes_per_token": num_bytes / num_tokens if num_tokens else 0,
                "tokens_per_byte": num_tokens / num_bytes if num_bytes else 0,
                "tokens_per_char": num_tokens / num_chars if num_chars else 0,
                "tokens_per_word": num_tokens / num_words if num_words else 0,
                "enc_speed_tok_sec": num_tokens / enc_time if enc_time else 0,
                "dec_speed_tok_sec": num_tokens / dec_time if dec_time else 0
            }
        results.append(res)
        
    # 2. Evaluate Baselines
    for baseline_name in baselines:
        print(f"\n--- Evaluating Baseline: {baseline_name} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(baseline_name)
            vocab_size = tokenizer.vocab_size
        except Exception as e:
            print(f"Failed to load {baseline_name}: {e}")
            continue
            
        embedding_params = vocab_size * d_model
        output_params = vocab_size * d_model
        overhead = embedding_params # Assuming tied for fair comparison
        
        res = {
            "name": baseline_name,
            "vocab_size": vocab_size,
            "train_time": 0.0,
            "embedding_params_M": embedding_params / 1e6,
            "output_params_M": output_params / 1e6,
            "total_params_M": (base_params + overhead) / 1e6,
            "overhead_pct": (overhead / (base_params + overhead)) * 100,
            "lang_stats": {}
        }
        
        for lang, text in corpora.items():
            enc_time, dec_time, num_tokens = measure_speed(tokenizer, text, is_hf=True)
            num_bytes = len(text.encode('utf-8'))
            num_chars = len(text)
            num_words = len(text.split())
            
            res["lang_stats"][lang] = {
                "tokens": num_tokens,
                "bytes_per_token": num_bytes / num_tokens if num_tokens else 0,
                "tokens_per_byte": num_tokens / num_bytes if num_bytes else 0,
                "tokens_per_char": num_tokens / num_chars if num_chars else 0,
                "tokens_per_word": num_tokens / num_words if num_words else 0,
                "enc_speed_tok_sec": num_tokens / enc_time if enc_time else 0,
                "dec_speed_tok_sec": num_tokens / dec_time if dec_time else 0
            }
        results.append(res)
        
    return results

def main():
    from myllm.utils.env import get_project_root
    workspace = get_project_root() / "artifacts/stage2_tokenizer_experiments"
    
    print("Loading corpora...")
    corpora, full_text = load_corpora(workspace)
    if not corpora:
        print("No sample text found. Run build_tokenizer_sample.py first.")
        return
        
    candidate_sizes = [32000, 48000, 64000]
    baselines = ["google-bert/bert-base-multilingual-cased"]
    
    results = run_tokenizer_experiment(corpora, full_text, candidate_sizes, baselines)
    
    # Generate Markdown Report
    report_path = workspace / "tokenizer_selection_report.md"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Tokenizer Selection Report for Dhruva 100M\n\n")
        
        # Summary Table
        f.write("## 1. Parameter Cost & Overhead\n")
        f.write("| Tokenizer | Vocab Size | Embedding (M) | Output (M) | Total Model (M) | Overhead % |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['vocab_size']:,} | {r['embedding_params_M']:.2f}M | {r['output_params_M']:.2f}M | {r['total_params_M']:.2f}M | {r['overhead_pct']:.1f}% |\n")
            
        f.write("\n## 2. Multilingual Efficiency (Bytes per Token)\n")
        f.write("*(Higher is better. Indicates stronger compression.)*\n\n")
        
        langs = list(corpora.keys())
        header = "| Tokenizer | " + " | ".join(langs) + " |"
        f.write(header + "\n")
        f.write("|---" + "|---" * len(langs) + "|\n")
        
        for r in results:
            row = f"| {r['name']} | "
            for lang in langs:
                bpt = r['lang_stats'][lang]['bytes_per_token']
                row += f"{bpt:.2f} | "
            f.write(row + "\n")
            
        f.write("\n## 3. Detailed Speed & Tokenization Metrics\n")
        for r in results:
            f.write(f"\n### {r['name']}\n")
            for lang, s in r['lang_stats'].items():
                f.write(f"- **{lang}**: {s['tokens_per_char']:.2f} toks/char, {s['tokens_per_word']:.2f} toks/word. Speed: {s['enc_speed_tok_sec']:.0f} tok/s (enc)\n")
        
        f.write("\n## 4. Final Recommendation & Trade-offs\n")
        f.write("*(To be filled by analyzing the above metrics before freezing.)*\n")
        f.write("\n**Which tokenizer is best for Dhruva 100M?**\n")
        f.write("\n**Why?**\n")
        f.write("\n**What is the parameter cost vs compression trade-off?**\n")
        
    print(f"\nExperiment complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
