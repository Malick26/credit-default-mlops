"""Build the reproducible dataset versions (v1, v2, v3) from the UCI source.

Source: "Default of Credit Card Clients" (I-Cheng Yeh, 2009), UCI Machine Learning
Repository, dataset id 350, licence CC BY 4.0.

The source has no date per client, so new data arriving is simulated with fixed,
reproducible batches:
  - evaluation: 20% of the clients, stratified on the label, held out once and never
    used for training. Every version shares it, so champion and candidate are always
    compared on the same observations.
  - v1: the first 3,000 training clients (small first delivery -> baseline model)
  - v2: v1 + 18,000 new labelled clients (a big delivery -> should trigger retraining)
  - v3: v2 + 3,000 new labelled clients (a small delivery -> champion may be retained)

Usage:
    python ml/prepare_data.py            # downloads the source if needed
    python ml/prepare_data.py --raw path/to/file.xls
"""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET = "default"
ID_COLUMN = "source_row"
SEED = 42
EVAL_FRACTION = 0.2
# Cumulative number of training rows in each version.
VERSION_SIZES = {"v1": 3_000, "v2": 21_000, "v3": 24_000}


def download(raw_dir: Path) -> Path:
    """Download the UCI zip once and keep the .xls in data/raw (ignored by Git)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    xls = raw_dir / "default_of_credit_card_clients.xls"
    if xls.exists():
        return xls
    print(f"Downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".xls"))
        xls.write_bytes(zf.read(name))
    return xls


def load(xls: Path) -> pd.DataFrame:
    """Read the source file and give the columns clean, lowercase names."""
    df = pd.read_excel(xls, header=1)  # the first row of the sheet is a group header
    df = df.rename(columns={"ID": ID_COLUMN, "PAY_0": "PAY_1", "default payment next month": TARGET})
    df.columns = [c.lower() for c in df.columns]
    if len(df) != 30_000 or TARGET not in df.columns:
        raise ValueError(f"Unexpected source file: {len(df)} rows, columns={list(df.columns)}")
    return df


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(df: pd.DataFrame, out_dir: Path) -> None:
    pool, evaluation = train_test_split(
        df, test_size=EVAL_FRACTION, stratify=df[TARGET], random_state=SEED
    )
    if len(pool) < max(VERSION_SIZES.values()):
        raise ValueError("Not enough training rows for the configured versions")

    for version, size in VERSION_SIZES.items():
        folder = out_dir / version
        folder.mkdir(parents=True, exist_ok=True)
        train = pool.iloc[:size]
        train.to_csv(folder / "train.csv", index=False)
        evaluation.to_csv(folder / "evaluation.csv", index=False)

        overlap = set(train[ID_COLUMN]) & set(evaluation[ID_COLUMN])
        if overlap:
            raise ValueError(f"{version}: {len(overlap)} rows are in both train and evaluation")

        metadata = {
            "version": version,
            "source": SOURCE_URL,
            "seed": SEED,
            "train_rows": len(train),
            "evaluation_rows": len(evaluation),
            "train_positive_rate": round(float(train[TARGET].mean()), 4),
            "evaluation_positive_rate": round(float(evaluation[TARGET].mean()), 4),
            "sha256": {
                "train.csv": sha256(folder / "train.csv"),
                "evaluation.csv": sha256(folder / "evaluation.csv"),
            },
        }
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{version}: {len(train)} train rows, {len(evaluation)} evaluation rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, help="Local copy of the UCI .xls file")
    parser.add_argument("--out", type=Path, default=Path("data"), help="Output folder (default: data)")
    args = parser.parse_args()

    xls = args.raw or download(args.out / "raw")
    build(load(xls), args.out)


if __name__ == "__main__":
    main()
