import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from myllm.training.sft.dataset import SFTDataset
from myllm.core.tokenizer.bpe import BPETokenizer

def test_sft_masking():
    # Load tokenizer from the baseline model
    tokenizer_path = "output/v0_100m/final_model/tokenizer"
    if not os.path.exists(tokenizer_path):
        print(f"Skipping test: tokenizer not found at {tokenizer_path}")
        return
        
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    mock_data = [{
        "instruction": "What is 2+2?",
        "input": "",
        "output": "2+2 is 4."
    }]
    
    dataset = SFTDataset(mock_data, tokenizer, max_seq_len=64)
    item = dataset[0]
    
    input_ids = item["input_ids"]
    labels = item["labels"]
    
    # Verify sequence lengths match
    assert len(input_ids) == len(labels), "Input IDs and labels length mismatch."
    
    # Count masked tokens and supervised tokens
    masked_count = (labels == -100).sum().item()
    supervised_count = (labels != -100).sum().item()
    
    # 1. Verify prompt tokens are masked
    assert masked_count > 0, "No tokens were masked!"
    
    # 2. Verify response tokens are NOT masked
    assert supervised_count > 0, "No supervised response tokens found!"
    
    # 3. Verify padding tokens are masked
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # Any input_id == pad_id should have label == -100
    for i in range(len(input_ids)):
        if input_ids[i].item() == pad_id:
            assert labels[i].item() == -100, f"Padding token at {i} was not masked in labels!"
            
    print("SFT Masking Test Passed!")
    print(f"Total tokens: {len(input_ids)}")
    print(f"Masked tokens (prompt + pad): {masked_count}")
    print(f"Supervised tokens (response): {supervised_count}")
    
    # Print decoded prompt to verify logic
    # Find start of supervised tokens
    supervised_start = -1
    for i in range(len(labels)):
        if labels[i].item() != -100:
            supervised_start = i
            break
            
    print("\n--- Decoded Prompt (Masked) ---")
    print(tokenizer.decode(input_ids[:supervised_start].tolist()))
    
    print("\n--- Decoded Response (Supervised) ---")
    # End of supervised is where padding starts (if any padding)
    # We can just filter out -100
    supervised_ids = [idx.item() for idx in input_ids[supervised_start:] if idx.item() != pad_id]
    print(tokenizer.decode(supervised_ids))

if __name__ == "__main__":
    test_sft_masking()
