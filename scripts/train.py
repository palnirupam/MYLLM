#!/usr/bin/env python3
"""MyLLM V0 Training Script — End-to-end training pipeline."""

import argparse
import yaml
import torch
from myllm.utils.runtime_guard import assert_training_environment
import random
import numpy as np
from pathlib import Path
import shutil

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel
from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.training.data.dataset import load_and_tokenize_dataset, create_dataloader
from myllm.training.checkpoint.manager import CheckpointManager
from myllm.training.pretraining.trainer import Trainer

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    assert_training_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/v0_100m.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    set_seed(config_dict.get("training", {}).get("seed", 42))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_cfg = config_dict.get("data", {})
    ds_name = data_cfg.get("dataset_name", "wikitext")
    ds_config = data_cfg.get("dataset_config", "wikitext-2-raw-v1")
    max_seq_len = data_cfg.get("max_seq_len", 512)

    output_dir = Path(config_dict.get("output", {}).get("dir", "./output/v0_100m"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer_dir = output_dir / "tokenizer"

    print("Training tokenizer...")
    # Mock text loading for tokenizer training for simplicity
    from datasets import load_dataset
    raw_ds = load_dataset(ds_name, ds_config, split="train")
    texts = (x['text'] for x in raw_ds if x['text'].strip())
    
    tokenizer = BPETokenizer.train_from_texts(texts, vocab_size=config_dict["model"]["vocab_size"])
    tokenizer.save(str(tokenizer_dir))

    print("Preparing dataset...")
    train_dataset = load_and_tokenize_dataset(
        tokenizer, 
        max_seq_len, 
        dataset_name=ds_name, 
        dataset_config=ds_config, 
        split='train'
    )
    
    train_dataloader = create_dataloader(
        train_dataset, 
        batch_size=config_dict["training"]["batch_size"], 
        shuffle=True
    )

    print("Initializing model...")
    model_cfg = ModelConfig(**config_dict.get("model", {}))
    model = MyLLMModel(model_cfg)
    
    print(f"Model parameters: {model.count_parameters():,}")

    checkpoint_manager = CheckpointManager(str(output_dir / "checkpoints"))

    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config_dict.get("training", {}),
        checkpoint_manager=checkpoint_manager,
        device=device
    )

    start_step = 0
    samples_consumed = 0
    checkpoints = checkpoint_manager.list_checkpoints()
    if checkpoints:
        latest = checkpoints[-1]["path"]
        print(f"Resuming from checkpoint: {latest}")
        metadata = checkpoint_manager.load(
            str(latest),
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler
        )
        start_step = metadata.get("step", 0)
        samples_consumed = metadata.get("dataset_position", 0)
        print(f"  | Checkpoint dataset position: {samples_consumed}")
        print(f"  | Optimizer state restored")
        print(f"  | Scheduler state restored")
        print(f"  | RNG states restored")

    results = trainer.train(
        num_steps=config_dict["training"]["max_steps"],
        start_step=start_step,
        initial_samples_consumed=samples_consumed
    )

    # Save final model
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    import safetensors.torch
    safetensors.torch.save_model(model, final_dir / "model.safetensors")
    
    model_cfg.save(str(final_dir / "config.json"))
    shutil.copytree(tokenizer_dir, final_dir / "tokenizer", dirs_exist_ok=True)
    
    print("\n--- Training Complete ---")
    print(f"Final Loss: {results['final_loss']:.4f}")
    print(f"Total Steps: {results['steps_completed']}")
    print(f"Total Tokens: {results['total_tokens']:,}")
    print(f"Model saved to: {final_dir}")

if __name__ == "__main__":
    main()
