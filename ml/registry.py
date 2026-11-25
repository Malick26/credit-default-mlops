"""MLflow Model Registry helpers: the champion alias, approval and rollback."""
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

MODEL_NAME = "credit-default"
CHAMPION = "champion"


def get_champion(client: MlflowClient) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except MlflowException:
        return None  # no registered model or no champion yet


def register_approved(client: MlflowClient, run_id: str, dataset_version: str, roc_auc: float) -> str:
    """Register the model of a run as an approved version and return its number."""
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(MODEL_NAME)
    version = client.create_model_version(MODEL_NAME, f"runs:/{run_id}/model", run_id=run_id)
    for key, value in {"approved": "true", "dataset_version": dataset_version,
                       "approved_roc_auc": f"{roc_auc:.6f}"}.items():
        client.set_model_version_tag(MODEL_NAME, version.version, key, value)
    return str(version.version)


def set_champion(client: MlflowClient, version: str) -> None:
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)


def check_rollback_target(client: MlflowClient, version: str) -> ModelVersion:
    """The target must exist and have been approved before; otherwise nothing moves."""
    try:
        target = client.get_model_version(MODEL_NAME, version)
    except MlflowException as exc:
        raise ValueError(f"model version {version} does not exist") from exc
    if target.tags.get("approved") != "true":
        raise ValueError(f"model version {version} was never approved")
    return target
