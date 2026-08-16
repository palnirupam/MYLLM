import math
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ── Pinned dataset revisions ───────────────────────────────────────────────────
# B6 FIX: Every public HuggingFace dataset is mutable — load_dataset() without
# revision= will silently download whatever is current on the Hub, making
# training non-reproducible between runs. All datasets must be pinned.
#
# How to find a revision hash:
#   from datasets import load_dataset_builder
#   info = load_dataset_builder("Salesforce/wikitext", "wikitext-2-raw-v1").info
#   print(info.download_checksums)
#
# Or check the Hub commit hash at:
#   https://huggingface.co/datasets/Salesforce/wikitext/tree/main
#
# Pinning to a known stable commit ensures identical data across:
#   - Local runs
#   - Cloud runs
#   - Reproduced experiments
PINNED_DATASET_REVISIONS = {}


class TextDataset(Dataset):
    def __init__(self, all_ids: list[int], max_seq_len: int, eos_token_id: int):
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        
        chunk_size = max_seq_len + 1
        self.chunks = [all_ids[i:i + chunk_size]
                       for i in range(0, len(all_ids) - chunk_size + 1, chunk_size)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels    = torch.tensor(chunk[1:],  dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels':    labels
        }


def load_and_tokenize_dataset(
    tokenizer,
    max_seq_len: int,
    dataset_name: str = 'Salesforce/wikitext',
    dataset_config: str = 'wikitext-2-raw-v1',
    split: str = 'train',
    revision: str = None,   # B6: Explicit revision parameter
) -> TextDataset:
    """
    Load, tokenize, and chunk a dataset.

    Args:
        tokenizer: BPETokenizer instance (must have .encode() and .eos_token_id)
        max_seq_len: Maximum sequence length (chunk size = max_seq_len + 1)
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration name
        split: 'train', 'validation', 'test'
        revision: B6 FIX — Git commit hash to pin the dataset to.
                  If None, uses the pinned revision from PINNED_DATASET_REVISIONS,
                  or falls back to latest (with a warning).

    Returns:
        TextDataset ready for DataLoader
    """
    # B6 FIX: Resolve revision
    if revision is None:
        revision = PINNED_DATASET_REVISIONS.get(dataset_name)
        if revision is None:
            import warnings
            warnings.warn(
                f"No pinned revision for dataset '{dataset_name}'. "
                "Using latest version from Hub — this makes training non-reproducible. "
                "Add a revision hash to PINNED_DATASET_REVISIONS in dataset.py.",
                UserWarning,
                stacklevel=2,
            )

    load_kwargs = {
        "name": dataset_config,
        "split": split,
    }
    if revision is not None:
        load_kwargs["revision"] = revision

    dataset = load_dataset(dataset_name, **load_kwargs)

    all_ids = []
    for item in dataset:
        text = item['text']
        if text.strip():
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                all_ids.extend(ids)
                all_ids.append(tokenizer.eos_token_id)

    return TextDataset(all_ids, max_seq_len, tokenizer.eos_token_id)


def create_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader with deterministic worker seeding.

    B7 FIX: DataLoader workers must be seeded per-worker to ensure reproducible
    shuffle order. Without this, two runs with the same global seed will see
    different batch orderings when num_workers > 0.
    """
    def worker_init_fn(worker_id: int) -> None:
        """Seed each DataLoader worker deterministically."""
        import numpy as np
        import random
        worker_seed = seed + worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
