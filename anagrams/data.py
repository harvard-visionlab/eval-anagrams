"""Load the visual-anagrams dataset from the HuggingFace Hub and wrap it for PyTorch."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import torch
from datasets import ClassLabel, Dataset, load_dataset

REPO_ID = "visionlab/visual-anagrams"
CONFIGS = ("pairs-72", "pairs-1440")
DEFAULT_CONFIG = "pairs-72"


def load_anagrams(config: str = DEFAULT_CONFIG, revision: str | None = None) -> Dataset:
    """HF Dataset (split 'test') for one config. Cached by `datasets` after first download."""
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}, got {config!r}")
    return load_dataset(REPO_ID, config, split="test", revision=revision)


def metadata_frame(ds: Dataset) -> pd.DataFrame:
    """All non-image columns as a DataFrame, ClassLabel ints decoded to category names."""
    columns = [c for c in ds.column_names if c != "image"]
    df = ds.select_columns(columns).to_pandas()
    for col in columns:
        if isinstance(ds.features[col], ClassLabel):
            df[col] = df[col].map(ds.features[col].int2str)
    return df


class AnagramDataset(torch.utils.data.Dataset):
    """(transform(image), index) pairs; index joins back to `metadata_frame(ds)` rows."""

    def __init__(self, ds: Dataset, transform: Callable):
        self.ds = ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        image = self.ds[index]["image"].convert("RGB")
        return self.transform(image), index
