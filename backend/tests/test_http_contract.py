from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings
from tests.fakes import FakeMLBackend


def _client(origins: list[str] | None = None) -> TestClient:
    settings = Settings(
        _env_file=None,
        cors_origins=origins
        or ["http://localhost:5173", "http://127.0.0.1:5173"],
    )
    return TestClient(create_app(FakeMLBackend(), settings=settings))


def test_allowed_origin_receives_cors_header() -> None:
    response = _client().get(
        "/api/health", headers={"Origin": "http://localhost:5173"}
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in response.headers


def test_unlisted_origin_does_not_receive_cors_header() -> None:
    response = _client().get(
        "/api/health", headers={"Origin": "https://untrusted.example"}
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_preflight_allows_documented_method_and_header() -> None:
    response = _client().options(
        "/api/experiments",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_openapi_documents_public_endpoints_and_errors() -> None:
    document = _client().get("/openapi.json").json()
    paths = document["paths"]

    assert set(paths) >= {
        "/api/health",
        "/api/models",
        "/api/datasets",
        "/api/experiments",
    }
    assert set(paths["/api/experiments"]["post"]["responses"]) >= {
        "200",
        "400",
        "422",
        "502",
        "503",
    }
    assert "ErrorResponse" in document["components"]["schemas"]

    schemas = document["components"]["schemas"]
    experiment_fields = schemas["ExperimentConfig"]["properties"]
    result_fields = schemas["ExperimentResult"]["properties"]
    assert "variant" not in experiment_fields
    assert set(experiment_fields) == {
        "model",
        "dataset",
        "params",
        "test_size",
        "random_state",
    }
    assert set(result_fields) == {
        "run_id",
        "model",
        "dataset",
        "task",
        "effective_params",
        "metrics",
        "artifacts",
        "metadata",
    }
