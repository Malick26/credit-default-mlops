"""Dataset versions: loading, data contract checks and DVC identifiers."""
import re
from pathlib import Path

import pandas as pd

TARGET = "default"
ID_COLUMN = "source_row"
FEATURES = [
    "limit_bal", "sex", "education", "marriage", "age",
    "pay_1", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
]
VERSIONS = ("v1", "v2", "v3")
DATA_DIR = Path("data")


def load_version(version: str, data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, evaluation) for a dataset version, after checking the data contract."""
    if version not in VERSIONS:
        raise ValueError(f"Unknown dataset version '{version}', expected one of {VERSIONS}")
    folder = data_dir / version
    train = pd.read_csv(folder / "train.csv")
    evaluation = pd.read_csv(folder / "evaluation.csv")
    validate(train, evaluation)
    # Float features give a model schema that accepts both 50000 and 50000.0 from the app.
    train[FEATURES] = train[FEATURES].astype("float64")
    evaluation[FEATURES] = evaluation[FEATURES].astype("float64")
    return train, evaluation


def validate(train: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Data contract: columns, no missing values, valid labels, train/evaluation separation."""
    required = [ID_COLUMN, *FEATURES, TARGET]
    for name, df in (("train", train), ("evaluation", evaluation)):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        if df.empty:
            raise ValueError(f"{name}: no rows")
        if df[required].isna().any().any():
            raise ValueError(f"{name}: missing values")
        if not df[TARGET].isin([0, 1]).all():
            raise ValueError(f"{name}: label '{TARGET}' must be 0 or 1")
        if df[TARGET].nunique() < 2:
            raise ValueError(f"{name}: label has a single class")
        if (df["age"] < 18).any() or (df["limit_bal"] <= 0).any():
            raise ValueError(f"{name}: out-of-range age or credit limit")
    overlap = set(train[ID_COLUMN]) & set(evaluation[ID_COLUMN])
    if overlap:
        raise ValueError(f"{len(overlap)} rows are in both train and evaluation")


def dvc_hash(version: str, data_dir: Path = DATA_DIR) -> str:
    """The DVC content hash recorded in data/<version>.dvc."""
    text = (data_dir / f"{version}.dvc").read_text()
    match = re.search(r"md5:\s*([0-9a-f]+(?:\.dir)?)", text)
    if not match:
        raise ValueError(f"No md5 found in data/{version}.dvc")
    return match.group(1)
