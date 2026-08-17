import os
import argparse
from pathlib import Path
from datasets import load_dataset
import tqdm

def fetch_sample(dataset_name, name_or_subset, split, lang_key, text_column, target_chars, output_file):
    print(f"Fetching {dataset_name} ({name_or_subset}) - {lang_key}...")
    try:
        # Load in streaming mode to avoid huge downloads
        ds = load_dataset(dataset_name, name_or_subset, split=split, streaming=True)
        
        collected_chars = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for row in ds:
                text = row.get(text_column, "")
                if len(text.strip()) > 50: # basic quality filter
                    # Write with a delimiter
                    f.write(text.replace("\n", " ") + "\n")
                    collected_chars += len(text)
                    if collected_chars >= target_chars:
                        break
        print(f"  -> Saved {collected_chars} characters for {lang_key}.")
        return collected_chars
    except Exception as e:
        print(f"  -> Error fetching {dataset_name}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Build representative tokenizer corpus.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--size_mb", type=float, default=5.0, help="Approx size per language in MB")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_chars = int(args.size_mb * 1024 * 1024)

    # We use Wikipedia as a reliable, fast-to-download multilingual source for this sample.
    sources = [
        {"ds": "wikimedia/wikipedia", "subset": "20231101.en", "split": "train", "lang": "English", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.bn", "split": "train", "lang": "Bengali", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.hi", "split": "train", "lang": "Hindi", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.ta", "split": "train", "lang": "Tamil", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.te", "split": "train", "lang": "Telugu", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.mr", "split": "train", "lang": "Marathi", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.gu", "split": "train", "lang": "Gujarati", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.kn", "split": "train", "lang": "Kannada", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.ml", "split": "train", "lang": "Malayalam", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.pa", "split": "train", "lang": "Punjabi", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.ur", "split": "train", "lang": "Urdu", "col": "text"},
        
        # Global
        {"ds": "wikimedia/wikipedia", "subset": "20231101.es", "split": "train", "lang": "Spanish", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.fr", "split": "train", "lang": "French", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.ar", "split": "train", "lang": "Arabic", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.pt", "split": "train", "lang": "Portuguese", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.de", "split": "train", "lang": "German", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.ru", "split": "train", "lang": "Russian", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.ja", "split": "train", "lang": "Japanese", "col": "text"},
        {"ds": "wikimedia/wikipedia", "subset": "20231101.ko", "split": "train", "lang": "Korean", "col": "text"},
    ]
    
    total_chars = 0
    for s in sources:
        out_file = out_dir / f"{s['lang'].lower()}_sample.txt"
        if not out_file.exists():
            total_chars += fetch_sample(s['ds'], s['subset'], s['split'], s['lang'], s['col'], target_chars, out_file)
        else:
            print(f"{s['lang']} already fetched.")
            
    code_file = out_dir / "code_sample.txt"
    manifest_file = out_dir / "code_sample_manifest.json"
    if not code_file.exists():
        import hashlib
        print("Fetching Python Code sample...")
        try:
            ds_code = load_dataset(
                "iamtarun/python_code_instructions_18k_alpaca", 
                split="train", 
                streaming=True,
                revision="7cae181e29701a8663a07a3ea43c8e105b663ba1"
            )
            collected_chars = 0
            examples = 0
            hasher = hashlib.sha256()
            
            with open(code_file, "w", encoding="utf-8") as f:
                for row in ds_code:
                    code_text = row.get("output", "")
                    if len(code_text.strip()) > 50:
                        content = code_text.strip() + "\n\n"
                        f.write(content)
                        collected_chars += len(content)
                        examples += 1
                        hasher.update(content.encode('utf-8'))
                        if collected_chars >= target_chars:
                            break
            
            if collected_chars == 0:
                raise ValueError("Zero code examples collected. Validation failed.")
                
            file_hash = hasher.hexdigest()
            manifest = {
                "language": "Python",
                "repository": "iamtarun/python_code_instructions_18k_alpaca",
                "commit_SHA": "7cae181e29701a8663a07a3ea43c8e105b663ba1",
                "dataset_file_hash": file_hash,
                "examples_count": examples,
                "character_count": collected_chars,
                "SHA-256": file_hash
            }
            
            import json
            with open(manifest_file, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)
                
            print(f"Code repository: {manifest['repository']}")
            print(f"Code commit SHA: {manifest['commit_SHA']}")
            print(f"Code examples: {manifest['examples_count']}")
            print(f"Code characters: {manifest['character_count']}")
            print(f"SHA-256: {manifest['SHA-256']}")
            
        except Exception as e:
            print(f"  -> Error fetching code sample: {e}")
            raise
    else:
        print("Code already fetched.")

    print(f"Done. Total representative characters collected.")

if __name__ == "__main__":
    main()
