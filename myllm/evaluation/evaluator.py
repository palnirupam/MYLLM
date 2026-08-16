import torch
import torch.nn.functional as F
from pathlib import Path
import math

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel
from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.training.data.dataset import load_and_tokenize_dataset, create_dataloader
from safetensors.torch import load_model

def evaluate_perplexity(model, dataloader, device='cuda') -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids)
            
            # Dataset already provides shifted labels
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100,
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()
            
    if total_tokens == 0:
        return float('inf')
    return math.exp(total_loss / total_tokens)

def evaluate_model(model_path: str, dataset_name: str = 'Salesforce/wikitext', dataset_config: str = 'wikitext-2-raw-v1', device='cuda') -> dict:
    p = Path(model_path)
    config = ModelConfig.load(str(p / "config.json"))
    tokenizer = BPETokenizer.load(str(p / "tokenizer"))
    
    model = MyLLMModel(config)
    load_model(model, str(p / "model.safetensors"))
    model.to(device)
    
    dataset = load_and_tokenize_dataset(tokenizer, config.max_seq_len, dataset_name=dataset_name, dataset_config=dataset_config, split='test')
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
    
    ppl = evaluate_perplexity(model, dataloader, device=device)
    num_tokens = len(dataset) * config.max_seq_len
    
    return {
        "perplexity": ppl,
        "num_tokens": num_tokens,
        "model_params": model.count_parameters()
    }
