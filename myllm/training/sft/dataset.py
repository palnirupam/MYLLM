import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, Any, List

class SFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_seq_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        self.instruction_prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
        self.response_prefix = "\n\n### Response:\n"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        example = self.data[idx]
        
        # Build prompt
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        response = example.get("output", "")
        
        if input_text:
            prompt_text = self.instruction_prefix + f"{instruction}\n{input_text}" + self.response_prefix
        else:
            prompt_text = self.instruction_prefix + f"{instruction}" + self.response_prefix
            
        if not response.strip():
            raise ValueError(f"SFT example {idx} has an empty assistant response")

        # Encode the two regions independently so the supervised boundary cannot
        # shift because of a BPE merge spanning prompt and assistant text.
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        bos_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        response_budget = self.max_seq_len - len(bos_ids) - len(eos_ids)
        response_ids = response_ids[:response_budget]
        prompt_budget = self.max_seq_len - len(bos_ids) - len(response_ids) - len(eos_ids)
        if prompt_budget < len(prompt_ids):
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

        full_ids = bos_ids + prompt_ids + response_ids + eos_ids
        labels = [-100] * (len(bos_ids) + len(prompt_ids)) + response_ids + eos_ids
            
        # Padding
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        input_ids = full_ids
        attention_mask = [1] * len(input_ids)
        
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

def get_sft_dataloaders(tokenizer, max_seq_len: int = 512, batch_size: int = 2):
    """
    Loads yahma/alpaca-cleaned with pinned revision.
    Splits into 95/5 train/val.
    """
    # Pinning dataset and revision
    dataset_name = "yahma/alpaca-cleaned"
    revision = "12567cabf869d7c92e573c7c783905fc160e9639" # Pinned commit
    
    print(f"Loading dataset {dataset_name} at revision {revision}...")
    dataset = load_dataset(dataset_name, revision=revision, split="train")
    
    # Deterministic split
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    
    train_data = [item for item in dataset["train"]]
    val_data = [item for item in dataset["test"]]
    
    print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples.")
    
    train_dataset = SFTDataset(train_data, tokenizer, max_seq_len)
    val_dataset = SFTDataset(val_data, tokenizer, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
