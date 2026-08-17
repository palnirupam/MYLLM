"""
scripts/evaluate_released_model.py — Full Production Capability Evaluation of Released Dhruva V1.
Evaluates English QA, Bengali QA, Hindi QA, Mathematics, Python generation, Reasoning,
Hallucination, Repetition rate, Throughput (tokens/sec), and VRAM utilization.
"""

import sys
import json
import time
from pathlib import Path
import torch

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.runtime.local.inference import LocalInferenceRuntime
from myllm.evaluation.eval_harness import ProductionEvaluationHarness
from myllm.evaluation.datasets.benchmark_v1 import get_benchmark_dataset
from myllm.evaluation.metrics import analyze_repetition


def main():
    model_path = "releases/dhruva-v1-100m/inference_model"
    print("================================================================================")
    print(f" DHRUVA V1 FULL MODEL CAPABILITY & MULTI-DOMAIN EVALUATION")
    print(f" Target Model Path: {Path(model_path).resolve()}")
    print("================================================================================\n")

    # 1. Initialize Runtime
    t0 = time.time()
    runtime = LocalInferenceRuntime(model_path=model_path)
    init_time = time.time() - t0
    device = runtime.device
    print(f"[*] Runtime initialized in {init_time:.2f}s on device: {device}")
    if torch.cuda.is_available() and "cuda" in str(device):
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[*] GPU: {gpu_name}")
    print(f"[*] Vocab Size: {runtime.tokenizer.vocab_size:,} | Max Seq Len: {runtime.config.max_seq_len}\n")

    # 2. Multi-Language & Multi-Domain Prompt Battery
    test_prompts = [
        # English QA & Reasoning
        ("English Science", "Explain how photosynthesis works in plants:"),
        ("English Reasoning", "If all mammals breathe air and whales are mammals, do whales breathe air? Explain:"),
        ("English Fact", "What is the capital city of France and what is its famous landmark?"),
        
        # Bengali (বাংলা)
        ("Bengali General", "বাংলা সাহিত্যের ইতিহাসে রবীন্দ্রনাথ ঠাকুরের অবদান সম্পর্কে সংক্ষেপে বলো:"),
        ("Bengali Science", "সূর্য থেকে পৃথিবী আলো ও শক্তি কীভাবে পায়?"),
        ("Bengali Grammar", "বাংলা ভাষায় স্বরবর্ণ এবং ব্যঞ্জনবর্ণের পার্থক্য কী?"),

        # Hindi (हिंदी)
        ("Hindi Science", "प्रकाश संश्लेषण क्या है और पौधों के लिए यह क्यों आवश्यक है?"),
        ("Hindi General", "भारत की राजधानी क्या है और इसके प्रमुख ऐतिहासिक स्थल कौन से हैं?"),

        # Python Code Generation
        ("Python Code", "Write a Python function to calculate the factorial of a positive integer n:"),
        ("Python Algorithm", "Write a Python function to check if a string is a palindrome:"),

        # Mathematics
        ("Math Arithmetic", "Calculate: (25 * 4) + 50 - 15 ="),
        ("Math Word Problem", "If a car travels at 60 km/h for 2.5 hours, what is the total distance traveled?"),
    ]

    print("================================================================================")
    print(" 1. DIRECT GENERATION BATTERY (Sample Qualitative Outputs)")
    print("================================================================================")

    results_table = []

    for category, prompt in test_prompts:
        print(f"\n--- [{category}] ---")
        print(f"Prompt: {prompt}")

        t_start = time.time()
        output = runtime.generate(prompt, max_new_tokens=40, temperature=0.7, top_k=40, top_p=0.9)
        elapsed_ms = (time.time() - t_start) * 1000.0

        gen_tokens = len(runtime.tokenizer.encode(output, add_special_tokens=False))
        tok_per_sec = (gen_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
        rep_score = analyze_repetition(output)

        print(f"Response: {output.strip()}")
        print(f"[Latency: {elapsed_ms:.1f}ms | Throughput: {tok_per_sec:.1f} tok/s | Repetition: {rep_score:.2f}]")

        results_table.append({
            "category": category,
            "prompt": prompt,
            "response": output.strip(),
            "latency_ms": elapsed_ms,
            "tok_per_sec": tok_per_sec,
            "rep_score": rep_score,
        })

    # 3. Standard Gold Benchmark Battery (12 Domains)
    print("\n================================================================================")
    print(" 2. STANDARDIZED GOLD BENCHMARK SUITE (Production Evaluation Harness)")
    print("================================================================================")

    dataset = get_benchmark_dataset()
    harness = ProductionEvaluationHarness(runtime=runtime, model_name="Dhruva-V1-100M")
    summary = harness.evaluate_battery(dataset, mode="adaptive")

    print("\n================================================================================")
    print(" 3. BENCHMARK SUMMARY & METRICS MATRIX")
    print("================================================================================")
    print(f" Total Evaluation Samples  : {summary.total_samples}")
    print(f" Verification / Pass Rate  : {summary.overall_accuracy_or_pass_rate * 100:.1f}%")
    print(f" Hallucination Rate        : {summary.overall_hallucination_rate * 100:.1f}%")
    print(f" Mean Latency per Query    : {summary.average_latency_ms:.1f} ms")
    print(f" Mean Generated Tokens     : {summary.average_tokens_per_sample:.1f} tokens/query")
    print(f" Mean Generation Speed     : {summary.overall_throughput_tokens_per_sec:.1f} tokens/sec")
    print(f" Mean Repetition Loop Score: {summary.overall_repetition_score:.3f}")
    print(f" Peak GPU VRAM Allocated   : {summary.gpu_peak_memory_mb:.1f} MB")

    print("\n Category Breakdown:")
    for cat, data in summary.category_metrics.items():
        pass_pct = data.get("accuracy_or_pass_rate", 0.0) * 100
        count = data.get("count", 0)
        lat = data.get("average_latency_ms", 0.0)
        print(f"  - {cat:25s}: {pass_pct:5.1f}% pass (N={count}) | Latency: {lat:.1f}ms")

    # Export structured evaluation report
    out_dir = Path("artifacts/evaluation_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "dhruva_v1_full_evaluation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)
    print(f"\n[*] Full report persisted to: {report_file}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
