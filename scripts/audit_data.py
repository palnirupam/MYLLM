import argparse
import random
import json
from pathlib import Path


def stratified_sample(data_path: Path, num_samples: int = 10, strata_key: str = "language"):
    """
    Reads a JSONL file and extracts a stratified random sample.
    (Mock implementation for scaffolding the CLI)
    """
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return
        
    print(f"--- Stratified Human Audit Tool ---")
    print(f"Loading data from: {data_path}")
    print(f"Target samples: {num_samples} (Stratified by: {strata_key})\n")
    
    # In a real implementation, we would stream the file, bucket by strata_key,
    # and use reservoir sampling. For this mock, we just demonstrate the UI.
    
    mock_samples = [
        {"language": "bn", "text": "Dhruva is a new AI model."},
        {"language": "hi", "text": "This is a boilerplate header menu."},
        {"language": "en", "text": "def calculate_hash(content):\n    pass"}
    ]
    
    for i, sample in enumerate(mock_samples):
        print(f"--- Sample {i+1} ---")
        print(f"Strata [{strata_key}]: {sample.get(strata_key, 'unknown')}")
        print("-" * 20)
        print(sample.get("text", ""))
        print("-" * 20)
        
        while True:
            score = input("Rate quality (1-5), or type 'drop' for PII/Boilerplate: ")
            if score.lower() == 'drop' or score in ['1', '2', '3', '4', '5']:
                break
            print("Invalid input. Please enter 1-5 or 'drop'.")
            
        print(f"Recorded: {score}\n")

    print("Audit session complete. Results saved to audit_log.json")


def main():
    parser = argparse.ArgumentParser(description="Stratified Human Audit Tool for Dhruva datasets.")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL dataset shard.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to audit.")
    parser.add_argument("--strata", type=str, default="language", help="Key to stratify by (e.g. language, domain).")
    
    args = parser.parse_args()
    
    stratified_sample(Path(args.data), args.samples, args.strata)


if __name__ == "__main__":
    main()
