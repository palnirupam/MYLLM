#!/usr/bin/env python3
"""MyLLM V0 Generation Script."""

import argparse
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from myllm.runtime.local.inference import LocalInferenceRuntime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    runtime = LocalInferenceRuntime(model_path=args.model)
    
    print("\nGenerating...")
    output = runtime.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print("\nOutput:")
    print("-" * 40)
    print(output)
    print("-" * 40)

if __name__ == "__main__":
    main()
