"""HTTP tests use a distinct module name from the core single-fit tests."""
from fastapi.testclient import TestClient

from backend.main import create_app


def test_interactive_http_catalog_training_and_validation(tmp_path):
    client = TestClient(create_app(original_jobs_dir=tmp_path))
    response = client.get("/api/interactive-experiments")
    assert response.status_code == 200
    assert len(response.json()) == 12
    response = client.post("/api/interactive-experiments", json={
        "model": "knn.optimized", "dataset": "wdbc", "params": {"n_neighbors": 3},
    })
    assert response.status_code == 200
    assert response.json()["metadata"]["visualizations"]
    assert response.json()["metadata"]["dataset_source"] == "data/classification/wdbc/wdbc.data"
    response = client.post("/api/interactive-experiments", json={
        "model": "knn.optimized", "dataset": "wdbc", "params": {"n_neighbors": 0},
    })
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "ml_request_rejected"
