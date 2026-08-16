import json
import time
from pathlib import Path
from typing import List, Dict, Any

from myllm.runtime.local.inference import LocalInferenceRuntime

# Fixed evaluation suite version
EVALUATION_VERSION = "myllm-eval-v1"

PROMPTS = {
    "general_qa": [
        "Explain recursion simply.",
        "Give three benefits of exercise.",
        "Summarize this paragraph."
    ],
    "factual_qa": [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What is the boiling point of water at sea level?"
    ],
    "reasoning": [
        "If Alice has 3 apples and gets 2 more, how many does she have?",
        "What comes next: 2, 4, 6, 8, ?"
    ],
    "coding": [
        "Write a Python function that adds two numbers.",
        "Find the bug in a small Python snippet."
    ],
    "bengali": [
        "বাংলায় নিজের পরিচয় দাও।",
        "পাইথন কী?",
        "৫ + ৭ কত?"
    ],
    "english": [
        "Introduce yourself.",
        "Explain what a variable is."
    ],
    "repetition_check": [
        "The quick brown fox jumps over the lazy dog.",
        "Tell me a story about a repeating echo."
    ]
}

def analyze_repetition(text: str) -> float:
    """Calculate a simple repetition score (0.0 to 1.0) based on repeating n-grams."""
    words = text.split()
    if len(words) < 3:
        return 0.0
    
    # Check 3-gram repetition
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0.0
    
    unique_trigrams = set(trigrams)
    repetition_ratio = 1.0 - (len(unique_trigrams) / len(trigrams))
    return repetition_ratio

def run_evaluation_suite(model_path: str, output_file: str, is_sft: bool = False, chat_template_version: str = "none") -> None:
    print(f"Loading model from {model_path} for evaluation...")
    runtime = LocalInferenceRuntime(model_path=model_path)
    
    generation_config = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
    }
    
    results = {
        "metadata": {
            "model_path": model_path,
            "evaluation_version": EVALUATION_VERSION,
            "is_sft": is_sft,
            "chat_template_version": chat_template_version,
            "generation_config": generation_config,
            "timestamp": time.time()
        },
        "outputs": {},
        "metrics": {
            "average_length": 0.0,
            "average_repetition_score": 0.0,
            "category_scores": {}
        }
    }
    
    total_length = 0
    total_rep_score = 0.0
    num_prompts = 0
    
    for category, prompts in PROMPTS.items():
        print(f"\n--- Evaluating Category: {category} ---")
        category_results = []
        cat_length = 0
        cat_rep = 0.0
        
        for prompt in prompts:
            # If SFT, we format the prompt. Otherwise, we pass it raw.
            # In V1 SFT, we will use chat_template_version="myllm-alpaca-v1"
            if is_sft and chat_template_version == "myllm-alpaca-v1":
                formatted_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
            else:
                formatted_prompt = prompt
                
            print(f"\nPrompt: {prompt}")
            
            # Generate
            output = runtime.generate(
                formatted_prompt, 
                max_new_tokens=generation_config["max_new_tokens"],
                temperature=generation_config["temperature"],
                top_k=generation_config["top_k"],
                top_p=generation_config["top_p"]
            )
            
            # Extract only the newly generated text (remove prompt)
            if output.startswith(formatted_prompt):
                generated_text = output[len(formatted_prompt):].strip()
            else:
                generated_text = output.strip()
                
            print(f"Response: {generated_text}")
            
            rep_score = analyze_repetition(generated_text)
            length = len(generated_text.split())
            
            cat_length += length
            cat_rep += rep_score
            total_length += length
            total_rep_score += rep_score
            num_prompts += 1
            
            category_results.append({
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "generated_text": generated_text,
                "repetition_score": rep_score,
                "length": length
            })
            
        results["outputs"][category] = category_results
        results["metrics"]["category_scores"][category] = {
            "average_length": cat_length / max(len(prompts), 1),
            "average_repetition_score": cat_rep / max(len(prompts), 1)
        }
        
    results["metrics"]["average_length"] = total_length / max(num_prompts, 1)
    results["metrics"]["average_repetition_score"] = total_rep_score / max(num_prompts, 1)
    
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nEvaluation saved to {output_file}")
    print(f"Overall Repetition Score: {results['metrics']['average_repetition_score']:.4f}")
    print(f"Overall Average Length (words): {results['metrics']['average_length']:.2f}")

if __name__ == "__main__":
    import argparse
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--template", type=str, default="none")
    args = parser.parse_args()
    
    run_evaluation_suite(args.model, args.output, args.sft, args.template)
