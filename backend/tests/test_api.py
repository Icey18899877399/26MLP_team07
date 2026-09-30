from fastapi.testclient import TestClient

from backend.contracts import ExperimentConfig
from backend.integration.ml_backend import (
    MLExecutionError,
    MLRequestError,
    PackageMLBackend,
)
from backend.main import create_app
from tests.fakes import FakeMLBackend


def _missing_importer(_name: str):
    raise ModuleNotFoundError(r"test package missing at C:\private\ml_core")


class RequestRejectingBackend(FakeMLBackend):
    def run_experiment(self, config: ExperimentConfig):
        raise MLRequestError(f"unsupported combination: {config.model}")


class ExecutionFailingBackend(FakeMLBackend):
    def run_experiment(self, config: ExperimentConfig):
        raise MLExecutionError(r"internal failure at C:\private\dataset.csv")


def test_health_reports_ml_unavailable_without_faking_success() -> None:
    backend = PackageMLBackend(importer=_missing_importer)
    client = TestClient(create_app(backend))

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ml_backend"]["available"] is False


def test_unavailable_ml_endpoint_returns_503() -> None:
    backend = PackageMLBackend(importer=_missing_importer)
    client = TestClient(create_app(backend))

    response = client.get("/api/models")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "ml_backend_unavailable"
    assert "private" not in response.text


def test_routes_delegate_to_fake_backend() -> None:
    client = TestClient(create_app(FakeMLBackend()))

    assert client.get("/api/models").json()[0]["id"] == "logistic_regression.optimized"
    assert client.get("/api/datasets").json()[0]["id"] == "wdbc"

    response = client.post(
        "/api/experiments",
        json={
            "model": "logistic_regression.optimized",
            "dataset": "wdbc",
            "params": {"max_iter": 500},
            "test_size": 0.2,
            "random_state": 42,
        },
    )

    assert response.status_code == 200
    assert response.json()["metrics"] == {"accuracy": 0.9}
    assert response.json()["metadata"]["source"] == "test-double"


def test_experiment_request_is_strictly_validated() -> None:
    client = TestClient(create_app(FakeMLBackend()))

    response = client.post(
        "/api/experiments",
        json={
            "model": "logistic_regression.optimized",
            "dataset": "wdbc",
            "test_size": 1.2,
            "unexpected": True,
        },
    )

    assert response.status_code == 422
    locations = {tuple(item["loc"]) for item in response.json()["detail"]}
    assert ("body", "test_size") in locations
    assert ("body", "unexpected") in locations


def test_domain_rejection_returns_documented_400() -> None:
    client = TestClient(create_app(RequestRejectingBackend()))

    response = client.post(
        "/api/experiments",
        json={"model": "unknown", "dataset": "wdbc"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "code": "ml_request_rejected",
            "message": "unsupported combination: unknown",
        }
    }


def test_execution_failure_returns_generic_502_without_internal_path() -> None:
    client = TestClient(create_app(ExecutionFailingBackend()))

    response = client.post(
        "/api/experiments",
        json={
            "model": "logistic_regression.optimized",
            "dataset": "wdbc",
        },
    )

    assert response.status_code == 502
    body = response.json()
    assert body["detail"]["code"] == "ml_execution_failed"
    assert body["detail"]["message"] == "ML 包执行失败"
    assert "private" not in response.text
