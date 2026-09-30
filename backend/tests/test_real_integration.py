from importlib.util import find_spec

import pytest
from fastapi.testclient import TestClient

from backend.contracts import ExperimentConfig
from backend.integration.ml_backend import PackageMLBackend
from backend.main import create_app


@pytest.fixture(scope="module")
def real_backend() -> PackageMLBackend:
    if find_spec("ml_core") is None:
        pytest.skip("ml_core is not installed; install the monorepo root package first")
    backend = PackageMLBackend()
    assert backend.status().available, backend.status().detail
    return backend


@pytest.mark.integration
def test_real_ml_core_discovery(real_backend: PackageMLBackend) -> None:
    assert [item.id for item in real_backend.list_models()] == [
        "cart_decision_tree.optimized",
        "dbscan.optimized",
        "gaussian_naive_bayes.optimized",
        "gbdt_regression.optimized",
        "isolation_forest.optimized",
        "kmeans.optimized",
        "knn.optimized",
        "linear_regression.optimized",
        "logistic_regression.optimized",
        "mlp_regression.optimized",
        "one_class_svm.optimized",
        "random_forest.optimized",
    ]
    assert [item.id for item in real_backend.list_datasets()] == [
        "23_mammography",
        "6_cardio",
        "california_housing",
        "concrete",
        "seeds",
        "wdbc",
    ]


@pytest.mark.integration
def test_real_ml_core_experiment(real_backend: PackageMLBackend) -> None:
    result = real_backend.run_experiment(
        ExperimentConfig(
            model="kmeans.optimized",
            dataset="seeds",
            params={"n_init": 2, "max_iter": 50},
        )
    )

    assert result.model == "kmeans.optimized"
    assert result.dataset == "seeds"
    assert result.task == "clustering"
    assert "inertia" in result.metrics


@pytest.mark.integration
def test_real_ml_core_through_http(real_backend: PackageMLBackend) -> None:
    client = TestClient(create_app(real_backend))

    health = client.get("/api/health")
    response = client.post(
        "/api/experiments",
        json={
            "model": "logistic_regression.optimized",
            "dataset": "wdbc",
            "params": {"max_iter": 100},
            "test_size": 0.2,
            "random_state": 42,
        },
    )

    assert health.json()["ml_backend"]["available"] is True
    assert response.status_code == 200
    assert response.json()["task"] == "classification"
    assert "accuracy" in response.json()["metrics"]


@pytest.mark.integration
def test_real_ml_core_anomaly_through_http(real_backend: PackageMLBackend) -> None:
    client = TestClient(create_app(real_backend))

    response = client.post(
        "/api/experiments",
        json={
            "model": "isolation_forest.optimized",
            "dataset": "6_cardio",
            "params": {"n_estimators": 10},
        },
    )

    assert response.status_code == 200
    assert response.json()["task"] == "anomaly_detection"
    assert "anomaly_f1" in response.json()["metrics"]
