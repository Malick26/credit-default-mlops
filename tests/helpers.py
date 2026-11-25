"""Builds a throwaway MLflow registry with small models, so tests never touch DagsHub."""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from ml.data import FEATURES, TARGET
from ml.model import train
from ml.registry import MODEL_NAME, register_approved, set_champion


def synthetic_data(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.integers(0, 5, size=(rows, len(FEATURES))), columns=FEATURES).astype("float64")
    df["age"] = rng.integers(21, 70, rows)
    df["limit_bal"] = rng.integers(10, 500, rows) * 1000
    df[TARGET] = (df["pay_1"] + rng.normal(0, 1, rows) > 2.5).astype(int)
    return df


def register_model(seed: int, dataset_version: str = "v1") -> str:
    """Train a tiny model, log it and register it as an approved version."""
    data = synthetic_data(seed=seed)
    model = train(data, {"max_iter": 20})
    mlflow.set_experiment("tests")
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="model")
    return register_approved(MlflowClient(), run.info.run_id, dataset_version, 0.8)


def build_registry(champion: int = 1, versions: int = 2) -> None:
    """Register `versions` models and put the champion alias on one of them."""
    for seed in range(versions):
        register_model(seed)
    if champion:
        set_champion(MlflowClient(), str(champion))


__all__ = ["MODEL_NAME", "build_registry", "register_model", "synthetic_data"]
