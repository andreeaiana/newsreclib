# Utilities for caching and reading files

import os
from typing import Dict

import pandas as pd


def to_tsv(df: pd.DataFrame, fpath: str) -> None:
    """Stores a dataframe in `.tsv` format."""
    df.to_csv(fpath, sep="\t", index=False)


def load_idx_map_as_dict(fpath: str) -> Dict[str, int]:
    """Loads a table as dictionary."""
    idx_map_dict = dict(pd.read_table(fpath).values.tolist())
    return idx_map_dict


def check_integrity(fpath: str) -> bool:
    """Checks whether a file exists."""
    if not os.path.isfile(fpath):
        return False
    return True
