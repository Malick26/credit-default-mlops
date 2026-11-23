"""Credit default web app: a form, a JSON API and the model's identity.

Settings (environment variables):
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
    MODEL_NAME (default credit-default), MODEL_ALIAS (default champion)
    DECISION_THRESHOLD (default 0.5), RELOAD_TOKEN (enables POST /admin/reload)
"""
import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.model_service import ModelService
from app.schema import Client, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_NAME = os.environ.get("MODEL_NAME", "credit-default")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))
STATIC = Path(__file__).parent / "static"

service = ModelService(MODEL_NAME, MODEL_ALIAS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.try_load()
    yield


app = FastAPI(title="Credit default prediction", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    """Liveness: the process answers."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: only OK once the approved model is loaded and has passed a test prediction."""
    if not service.ready:
        return JSONResponse(status_code=503, content={"status": "not ready", "error": service.error})
    return {"status": "ready", "model_version": service.version}


@app.get("/model-info")
def model_info():
    """The model version actually loaded, and the commit the image was built from."""
    return {**service.info(), "app_commit": os.environ.get("APP_GIT_SHA", "unknown")}


@app.post("/predict", response_model=Prediction)
def predict(client: Client):
    if not service.ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    [probability], version = service.predict([client.model_dump()])
    return Prediction(
        probability_default=round(probability, 4),
        prediction=int(probability >= THRESHOLD),
        threshold=THRESHOLD,
        model_name=MODEL_NAME,
        model_version=version,
    )


@app.post("/admin/reload")
def reload_model(x_reload_token: str = Header(default="")):
    """Re-resolve the alias after a promotion or a rollback (called by the CT workflow)."""
    expected = os.environ.get("RELOAD_TOKEN", "")
    if not expected or not hmac.compare_digest(x_reload_token, expected):
        raise HTTPException(status_code=401, detail="Invalid reload token")
    previous = service.version
    try:
        service.load()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Reload failed, still serving {previous}: {exc}") from exc
    return {"previous_version": previous, "version": service.version}
