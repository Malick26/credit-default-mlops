"""HTTP smoke test against a running app (a CI container or the live production URL).

    python -m tests.smoke.smoke_test http://localhost:8000 --expect-version 1

Waits for /ready, then checks /model-info and a real prediction.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from app.schema import EXAMPLE_CLIENT


def call(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
        return 0, {"error": str(err)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--expect-version", help="model version that must be served")
    parser.add_argument("--timeout", type=int, default=300, help="seconds to wait for readiness")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    deadline = time.time() + args.timeout
    while True:
        status, body = call(f"{base}/ready")
        if status == 200:
            break
        if time.time() > deadline:
            print(f"Not ready after {args.timeout}s: {status} {body}")
            return 1
        time.sleep(5)
    print("ready:", body)

    _, info = call(f"{base}/model-info")
    print("model-info:", info)
    status, prediction = call(f"{base}/predict", EXAMPLE_CLIENT)
    print("predict:", status, prediction)

    errors = []
    if status != 200 or not 0 <= prediction.get("probability_default", -1) <= 1:
        errors.append("prediction failed")
    if args.expect_version and prediction.get("model_version") != args.expect_version:
        errors.append(f"served version {prediction.get('model_version')}, expected {args.expect_version}")
    if args.expect_version and info.get("version") != args.expect_version:
        errors.append(f"/model-info reports {info.get('version')}, expected {args.expect_version}")
    for error in errors:
        print("FAILED:", error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
