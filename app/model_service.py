"""Loads the approved model selected by the champion alias and serves predictions."""
import logging
import threading
from datetime import datetime, timezone

import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from app.schema import EXAMPLE_CLIENT, FEATURES

log = logging.getLogger("model")


class ModelService:
    def __init__(self, model_name: str, alias: str):
        self.model_name = model_name
        self.alias = alias
        self._lock = threading.Lock()
        self._model = None
        self.version: str | None = None
        self.run_id: str | None = None
        self.loaded_at: str | None = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> str:
        """Resolve the alias to a concrete version, load it and check it can predict.

        The new model replaces the old one only after the check passes, so a failed
        refresh keeps serving the previous version.
        """
        client = MlflowClient()
        version = client.get_model_version_by_alias(self.model_name, self.alias)
        model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version.version}")
        self._predict_proba(model, [EXAMPLE_CLIENT])  # readiness check
        with self._lock:
            self._model = model
            self.version = str(version.version)
            self.run_id = version.run_id
            self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.error = None
        log.info("Loaded %s version %s (alias %s)", self.model_name, self.version, self.alias)
        return self.version

    def try_load(self) -> None:
        try:
            self.load()
        except Exception as exc:  # the app stays up but reports not ready
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("Model loading failed: %s", self.error)

    @staticmethod
    def _predict_proba(model, rows: list[dict]) -> list[float]:
        frame = pd.DataFrame(rows, columns=FEATURES).astype("float64")
        return [float(p) for p in model.predict_proba(frame)[:, 1]]

    def predict(self, rows: list[dict]) -> tuple[list[float], str]:
        with self._lock:
            model, version = self._model, self.version
        if model is None:
            raise RuntimeError("model not loaded")
        return self._predict_proba(model, rows), version

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "alias": self.alias,
            "version": self.version,
            "run_id": self.run_id,
            "loaded_at": self.loaded_at,
            "ready": self.ready,
            "error": self.error,
        }
