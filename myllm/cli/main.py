import argparse
import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from myllm.runtime.local.inference import LocalInferenceRuntime
from myllm.evaluation.evaluator import evaluate_model
from myllm.training.checkpoint.manager import CheckpointManager

def main():
    parser = argparse.ArgumentParser(prog="myllm", description="MyLLM V0 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    parser_train = subparsers.add_parser("train", help="Run training pipeline")
    parser_train.add_argument("--config", type=str, required=True, help="Path to YAML config")

    # Chat
    parser_chat = subparsers.add_parser("chat", help="Interactive chat loop")
    parser_chat.add_argument("--model", type=str, required=True, help="Path to model directory")

    # Generate
    parser_generate = subparsers.add_parser("generate", help="Single generation")
    parser_generate.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser_generate.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser_generate.add_argument("--max-tokens", type=int, default=100)
    parser_generate.add_argument("--temperature", type=float, default=0.8)

    # Models
    parser_models = subparsers.add_parser("models", help="Model management")
    models_sub = parser_models.add_subparsers(dest="models_command", required=True)
    
    parser_list = models_sub.add_parser("list", help="List models (checkpoints)")
    parser_list.add_argument("--dir", type=str, required=True, help="Directory containing checkpoints")

    # Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Run perplexity evaluation")
    parser_eval.add_argument("--model", type=str, required=True, help="Path to model directory")

    args = parser.parse_args()

    if args.command == "train":
        # B9 FIX: Use subprocess.run instead of os.system to capture exit code.
        # os.system() silently ignores non-zero exit codes (failed training looks like success).
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", args.config]
        )
        if result.returncode != 0:
            print(f"ERROR: Training failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

    elif args.command == "chat":
        runtime = LocalInferenceRuntime(model_path=args.model)
        print("========================================")
        print("        🌟 Dhruva AI Chat 🌟            ")
        print("       (Type 'exit' to quit)            ")
        print("========================================")
        print("Model loaded. Ready for conversation.\n")
        while True:
            try:
                user_input = input("User> ")
                if user_input.strip().lower() == "exit":
                    break
                    
                if getattr(args, 'sft', False) or "sft" in args.model.lower() or "dhruva" in args.model.lower():
                    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{user_input}\n\n### Response:\n"
                else:
                    prompt = user_input
                    
                print("Dhruva> ", end="", flush=True)
                for token in runtime.generate_stream(prompt, temperature=0.7):
                    print(token, end="", flush=True)
                print()
            except (KeyboardInterrupt, EOFError):
                break
                
    elif args.command == "generate":
        runtime = LocalInferenceRuntime(model_path=args.model)
        
        is_sft_model = getattr(args, 'sft', False) or "dhruva" in args.model.lower() or "sft" in args.model.lower()
        if is_sft_model:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{args.prompt}\n\n### Response:\n"
        else:
            prompt = args.prompt
            
        output = runtime.generate(
            prompt, 
            max_new_tokens=args.max_tokens, 
            temperature=args.temperature
        )
        
        if is_sft_model:
            # Extract just the response part
            if "### Response:\n" in output:
                output = output.split("### Response:\n")[-1].strip()
                
        print(output)
        
    elif args.command == "models" and args.models_command == "list":
        manager = CheckpointManager(args.dir)
        checkpoints = manager.list_checkpoints()
        for ckpt in checkpoints:
            print(f"Step: {ckpt.get('step')} | Loss: {ckpt.get('loss')} | Path: {ckpt.get('path')}")
            
    elif args.command == "evaluate":
        results = evaluate_model(args.model)
        print(f"Evaluation Results:")
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"Parameters: {results['model_params']}")
        print(f"Tokens: {results['num_tokens']}")

if __name__ == "__main__":
    main()
