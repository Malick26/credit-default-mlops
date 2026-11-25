"""Fill an MLflow server with a test model so a container can be smoke-tested in CI.

Registers two approved versions and puts the champion alias on version 1, so the smoke
test also proves the app loads the champion and not the newest version.

    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python -m tests.smoke.make_registry
"""
from tests.helpers import build_registry

if __name__ == "__main__":
    build_registry(champion=1, versions=2)
    print("Test registry ready: champion -> version 1")
