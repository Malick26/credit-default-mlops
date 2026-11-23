"""Train a model on one dataset version and log everything to MLflow.

    python -m ml.train --dataset-version v1
    python -m ml.train --dataset-version v1 --param learning_rate=0.1 --register

Settings come from the environment: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME,
MLFLOW_TRACKING_PASSWORD (DagsHub token). --register adds the model to the registry
without moving the champion alias (the CT workflow decides promotions).
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from ml.data import FEATURES, VERSIONS, dvc_hash, load_version
from ml.model import DEFAULT_PARAMS, evaluate, train

EXPERIMENT = "credit-default"
MODEL_NAME = "credit-default"


def git_sha() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_and_log(version: str, params: dict | None = None, register: bool = False) -> dict:
    """Train on <version>/train.csv, evaluate on <version>/evaluation.csv, log the run."""
    train_df, evaluation = load_version(version)
    all_params = {**DEFAULT_PARAMS, **(params or {})}

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"train-{version}") as run:
        mlflow.set_tags({
            "git_sha": git_sha(),
            "dataset_version": version,
            "dvc_hash": dvc_hash(version),
            "stage": "candidate",
        })
        mlflow.log_params({**all_params, "dataset_version": version,
                           "train_rows": len(train_df), "evaluation_rows": len(evaluation)})

        model = train(train_df, all_params)
        metrics = evaluate(model, evaluation)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "evaluation.json"
            report.write_text(json.dumps({"dataset_version": version, **metrics}, indent=2))
            mlflow.log_artifact(str(report))

        example = train_df[FEATURES].head(3)
        info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(example, model.predict_proba(example)),
            input_example=example,
            registered_model_name=MODEL_NAME if register else None,
        )
        result = {
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "dataset_version": version,
            "metrics": metrics,
        }
        if register:
            result["registered_version"] = str(info.registered_model_version)
    return result


def parse_param(text: str) -> tuple[str, float | int]:
    key, value = text.split("=", 1)
    number = float(value)
    return key, int(number) if number.is_integer() else number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-version", choices=VERSIONS, required=True)
    parser.add_argument("--param", action="append", default=[], type=parse_param,
                        help="Override a hyperparameter, e.g. --param learning_rate=0.1")
    parser.add_argument("--register", action="store_true", help="Register the model (no alias change)")
    args = parser.parse_args()
    print(json.dumps(train_and_log(args.dataset_version, dict(args.param), args.register), indent=2))


if __name__ == "__main__":
    main()
