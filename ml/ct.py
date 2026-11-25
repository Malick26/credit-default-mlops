"""Continuous training: evaluate the champion on a new delivery, retrain, promote, roll back.

    python -m ml.ct pipeline --dataset-version v2
    python -m ml.ct rollback --version 1

The decision is printed and written to results/decision.json (uploaded by the workflow).
Exit code 1 means nothing in production changed because of an error or failed validation.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from ml import policy, registry
from ml.data import FEATURES, VERSIONS, dvc_hash, load_version
from ml.model import DEFAULT_PARAMS, evaluate, train
from ml.train import EXPERIMENT, git_sha

RESULTS = Path(os.environ.get("RUN_OUTPUT_DIR", "results"))


def write_decision(decision: dict) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    if mlflow.active_run():
        mlflow.log_dict(decision, "decision.json")
        mlflow.set_tag("decision", decision["decision"])
    print(json.dumps(decision, indent=2))
    return decision


def verify_champion(expected_version: str, evaluation) -> dict:
    """Load the model the alias now points to, like the app does, and predict 3 rows."""
    client = MlflowClient()
    champion = registry.get_champion(client)
    if champion is None or str(champion.version) != expected_version:
        raise RuntimeError(f"champion alias does not point to version {expected_version}")
    model = mlflow.sklearn.load_model(f"models:/{registry.MODEL_NAME}@{registry.CHAMPION}")
    probabilities = model.predict_proba(evaluation[FEATURES].head(3))[:, 1]
    return {"model_version": str(champion.version),
            "probabilities": [round(float(p), 4) for p in probabilities]}


def run_pipeline(version: str) -> dict:
    client = MlflowClient()
    decision = {"operation": "pipeline", "dataset_version": version, "model_name": registry.MODEL_NAME}

    # 1. Fetch and validate the release. A failure stops before any production change.
    try:
        train_df, evaluation = load_version(version)
    except (OSError, ValueError) as exc:
        write_decision({**decision, "validation": "failed", "decision": "validation_failed",
                        "reason": str(exc)})
        raise SystemExit(1) from exc
    decision.update({"validation": "passed", "dvc_hash": dvc_hash(version),
                     "train_rows": len(train_df), "evaluation_rows": len(evaluation)})

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"ct-{version}") as run:
        mlflow.set_tags({"git_sha": git_sha(), "dataset_version": version,
                         "dvc_hash": decision["dvc_hash"], "operation": "pipeline"})
        decision["run_id"] = run.info.run_id

        # 2. Evaluate the current champion on the held-out rows of this delivery.
        champion = registry.get_champion(client)
        champion_auc = None
        decision["champion_before"] = str(champion.version) if champion else None
        if champion is None:
            reason = "no champion yet: train the baseline (bootstrap)"
        else:
            champion_model = mlflow.sklearn.load_model(f"models:/{registry.MODEL_NAME}/{champion.version}")
            champion_metrics = evaluate(champion_model, evaluation)
            champion_auc = champion_metrics["roc_auc"]
            approved_auc = float(champion.tags.get("approved_roc_auc", champion_auc))
            decision.update({"champion_metrics": champion_metrics,
                             "champion_approved_roc_auc": approved_auc})
            mlflow.log_metrics({"champion_roc_auc": champion_auc,
                                "champion_approved_roc_auc": approved_auc})
            reason = policy.retrain_reason(champion_auc, approved_auc,
                                           champion.tags.get("dataset_version", ""), version)

        if reason is None:
            decision.update({"retrained": False, "decision": "retained_no_retraining",
                             "reason": "champion meets every gate on this delivery",
                             "champion_after": decision["champion_before"]})
            return write_decision(decision)

        # 3. Train a candidate and compare it with the champion on the same rows.
        decision.update({"retrained": True, "retrain_reason": reason})
        mlflow.log_params({**DEFAULT_PARAMS, "train_rows": len(train_df)})
        candidate = train(train_df)
        candidate_metrics = evaluate(candidate, evaluation)
        mlflow.log_metrics({"candidate_roc_auc": candidate_metrics["roc_auc"],
                            "candidate_average_precision": candidate_metrics["average_precision"]})
        example = train_df[FEATURES].head(3)
        mlflow.sklearn.log_model(candidate, artifact_path="model", input_example=example,
                                 signature=infer_signature(example, candidate.predict_proba(example)))
        decision["candidate_metrics"] = candidate_metrics

        promote, why = policy.promotion_reason(candidate_metrics["roc_auc"], champion_auc)
        decision["gate"] = {"min_roc_auc": policy.MIN_ROC_AUC,
                            "min_improvement": policy.MIN_IMPROVEMENT, "result": why}
        if not promote:
            outcome = "bootstrap_failed" if champion is None else "retained_candidate_rejected"
            decision.update({"decision": outcome, "champion_after": decision["champion_before"]})
            write_decision(decision)
            if champion is None:
                raise SystemExit(1)
            return decision

        # 4. Promote: register an approved version, move the alias, verify it serves.
        new_version = registry.register_approved(client, run.info.run_id, version,
                                                 candidate_metrics["roc_auc"])
        registry.set_champion(client, new_version)
        decision.update({"decision": "bootstrapped" if champion is None else "promoted",
                         "candidate_version": new_version, "champion_after": new_version,
                         "verification": verify_champion(new_version, evaluation)})
        return write_decision(decision)


def run_rollback(target: str) -> dict:
    client = MlflowClient()
    champion = registry.get_champion(client)
    decision = {"operation": "rollback", "model_name": registry.MODEL_NAME,
                "rollback_version": target,
                "champion_before": str(champion.version) if champion else None}
    try:
        restored = registry.check_rollback_target(client, target)
    except ValueError as exc:
        write_decision({**decision, "decision": "rollback_refused", "reason": str(exc),
                        "champion_after": decision["champion_before"]})
        raise SystemExit(1) from exc

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"rollback-to-v{target}"):
        mlflow.set_tags({"git_sha": git_sha(), "operation": "rollback"})
        registry.set_champion(client, target)
        dataset_version = restored.tags.get("dataset_version", "v1")
        _, evaluation = load_version(dataset_version) if Path(f"data/{dataset_version}").exists() else (None, None)
        decision.update({"decision": "rolled_back", "champion_after": target})
        if evaluation is not None:
            decision["verification"] = verify_champion(target, evaluation)
        return write_decision(decision)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("pipeline", help="evaluate, retrain and promote when eligible")
    p.add_argument("--dataset-version", choices=VERSIONS, required=True)
    r = sub.add_parser("rollback", help="move the champion alias back to an approved version")
    r.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        if args.operation == "pipeline":
            run_pipeline(args.dataset_version)
        else:
            run_rollback(args.version)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # show the real cause in the CI log
        print(f"CT error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
