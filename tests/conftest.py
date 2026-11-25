import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_registry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A fresh local registry (SQLite + local artifacts) for each test."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)  # local artifacts go to tmp_path/mlruns
    monkeypatch.setenv("RELOAD_TOKEN", "test-token")
    return tmp_path


@pytest.fixture
def app_client(registry):
    """The app started against a registry where v1 is champion and a newer v2 exists."""
    build_registry(champion=1, versions=2)
    from app.main import app, service
    service._model = None  # each test starts from a cold app
    with TestClient(app) as client:
        yield client
