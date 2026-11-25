"""Application tests: predictions, input validation, loading the approved model."""
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from app.schema import EXAMPLE_CLIENT, FEATURES
from ml.data import FEATURES as TRAINING_FEATURES
from ml.registry import MODEL_NAME, set_champion
from tests.helpers import build_registry


def test_app_and_training_use_the_same_features():
    assert FEATURES == TRAINING_FEATURES


# 1. Successful predictions and their output format
@pytest.mark.parametrize("changes", [
    {},                                                        # typical client
    {"limit_bal": 10_000, "age": 18, "pay_1": -2},             # lowest allowed values
    {"limit_bal": 2_000_000, "age": 100, "pay_1": 9},          # highest allowed values
    {"bill_amt1": -5_000.5},                                   # negative bill (overpayment)
    {"limit_bal": 50_000.0, "pay_amt1": 1234.56},              # floats are accepted
])
def test_predict_returns_a_probability_from_the_champion(app_client, changes):
    response = app_client.post("/predict", json={**EXAMPLE_CLIENT, **changes})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"probability_default", "prediction", "threshold", "model_name", "model_version"}
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["prediction"] == int(body["probability_default"] >= body["threshold"])
    assert body["model_name"] == MODEL_NAME
    assert body["model_version"] == "1"


# 2. Malformed or out-of-range inputs are rejected before reaching the model
@pytest.mark.parametrize("payload", [
    {k: v for k, v in EXAMPLE_CLIENT.items() if k != "age"},   # missing field
    {**EXAMPLE_CLIENT, "age": 17},                              # too young
    {**EXAMPLE_CLIENT, "age": 101},                             # too old
    {**EXAMPLE_CLIENT, "sex": 3},                               # unknown category
    {**EXAMPLE_CLIENT, "limit_bal": 0},                         # credit limit must be > 0
    {**EXAMPLE_CLIENT, "pay_1": 10},                            # repayment status out of range
    {**EXAMPLE_CLIENT, "pay_amt1": -1},                         # negative payment
    {**EXAMPLE_CLIENT, "age": "thirty"},                        # wrong type
    {**EXAMPLE_CLIENT, "income": 1000},                         # unknown field
    {},                                                         # empty body
])
def test_invalid_input_is_rejected(app_client, payload):
    response = app_client.post("/predict", json=payload)
    assert response.status_code == 422


# 3. The app serves the approved champion, not the newest registered version
def test_loads_champion_even_when_a_newer_version_exists(app_client):
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    assert max(int(v.version) for v in versions) == 2  # a newer version exists

    info = app_client.get("/model-info").json()
    assert info["version"] == "1"
    assert info["alias"] == "champion"
    assert app_client.get("/ready").json() == {"status": "ready", "model_version": "1"}


def test_reload_follows_the_alias_and_needs_the_token(app_client):
    set_champion(MlflowClient(), "2")  # a promotion happened
    assert app_client.post("/admin/reload").status_code == 401
    assert app_client.post("/admin/reload", headers={"X-Reload-Token": "wrong"}).status_code == 401

    response = app_client.post("/admin/reload", headers={"X-Reload-Token": "test-token"})
    assert response.json() == {"previous_version": "1", "version": "2"}
    assert app_client.post("/predict", json=EXAMPLE_CLIENT).json()["model_version"] == "2"


def test_not_ready_without_a_champion(registry):
    build_registry(champion=0, versions=1)  # a model exists but nothing is approved as champion
    from app.main import app, service
    service._model = None
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=EXAMPLE_CLIENT).status_code == 503
