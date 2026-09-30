"""Real HTTP coverage: persistence, untrusted tables, selection and actual fits."""
import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


def test_data_library_route_is_registered(tmp_path):
    client = TestClient(create_app(original_jobs_dir=tmp_path))
    assert client.get("/api/data-library/wdbc").status_code == 200


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(original_jobs_dir=tmp_path / "jobs", dataset_upload_dir=tmp_path / "uploads"))


def upload(client, content=None, filename="measurements.csv"):
    content = content or "x1,x2,label\n" + "\n".join(f"{i},{i % 5},{'left' if i < 20 else 'right'}" for i in range(40))
    response = client.post("/api/data-library/uploads", json={"filename": filename, "content": content})
    assert response.status_code == 200, response.text
    return response.json()


def selection(item, task="classification", target="label"):
    return {"dataset_id": item["id"], "feature_columns": ["x1", "x2"], "target_column": target, "task": task}


def test_original_schema_and_all_seven_registered_sources(client):
    response = client.get("/api/data-library")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 7
    for item in items:
        detail = client.get(f"/api/data-library/{item['id']}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["preview"]
        assert len(detail.json()["columns"]) == item["column_count"]
    detail = client.get("/api/data-library/wdbc").json()
    assert detail["row_count"] == 569
    assert detail["column_count"] == 32
    assert detail["default_target"] == "diagnosis"
    assert detail["source"] == "original"
    assert len(detail["preview"]) <= 20


def test_upload_persists_with_generated_id_after_app_recreation(client, tmp_path):
    item = upload(client)
    assert item["id"].startswith("upload_")
    assert item["name"] == "measurements.csv"
    assert item["row_count"] == 40
    assert item["columns"][0] == {"name": "x1", "dtype": "number", "missing_count": 0, "unique_count": 40}
    reopened = TestClient(create_app(dataset_upload_dir=tmp_path / "uploads", original_jobs_dir=tmp_path / "jobs2"))
    assert reopened.get(f"/api/data-library/{item['id']}").json() == item
    assert item["id"] in [row["id"] for row in reopened.get("/api/data-library").json()]


@pytest.mark.parametrize("filename,content", [
    ("../escape.csv", "a,b\n1,2"), ("..\\escape.csv", "a,b\n1,2"),
    ("bad.exe", "a,b\n1,2"), ("C:escape.csv", "a,b\n1,2"),
    ("bad.csv", "a,a\n1,2"), ("bad.csv", "a,b\n1,2,3"),
    ("bad.csv", "a,b\n1"), ("bad.csv", "a,b\n"),
    ("bad.csv", 'a,b\n"unterminated,2'), ("bad.csv", "a,,b\n1,2,3"),
    ("big.csv", "a\n" + "x" * (5 * 1024 * 1024)),
    ("wide.csv", ",".join(f"c{i}" for i in range(129)) + "\n" + ",".join("1" for _ in range(129))),
    ("long.csv", "a\n" + "1\n" * 20001),
], ids=["slash", "backslash", "extension", "drive", "duplicate", "extra-cell", "missing-cell", "empty", "quote", "empty-header", "bytes", "columns", "rows"])
def test_rejected_upload_creates_no_files(client, tmp_path, filename, content):
    response = client.post("/api/data-library/uploads", json={"filename": filename, "content": content})
    assert response.status_code == 400, response.text
    assert not list((tmp_path / "uploads").glob("*"))
    assert not (tmp_path / "escape.csv").exists()


def test_tsv_and_missing_schema_are_visible(client):
    item = upload(client, "x1\tx2\tlabel\n1\t\ta\n2\t4\tb", "sample.tsv")
    assert item["columns"][1]["missing_count"] == 1
    result = client.post("/api/data-library/compatibility", json=selection(item)).json()
    assert not result["models"]
    assert result["issues"]


@pytest.mark.parametrize("task,target", [("classification", None), ("classification", "x1"), ("regression", "label"), ("anomaly_detection", "label")])
def test_incompatible_target_is_explained(client, task, target):
    item = upload(client)
    response = client.post("/api/data-library/compatibility", json=selection(item, task, target))
    assert response.status_code == 200
    assert not response.json()["models"]
    assert response.json()["issues"]


def test_knn_uses_uploaded_rows_features_and_labels(client):
    item = upload(client)
    request = selection(item)
    compatible = client.post("/api/data-library/compatibility", json=request).json()
    assert len(compatible["models"]) == 5
    response = client.post("/api/data-library/experiments", json={**request, "model": "knn.optimized", "params": {"n_neighbors": 1}, "random_state": 7})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["dataset"] == item["id"]
    assert result["metadata"]["sample_count"] == 40
    assert result["metadata"]["train_sample_count"] + result["metadata"]["test_sample_count"] == 40
    assert result["metadata"]["feature_names"] == ["x1", "x2"]
    assert result["metrics"]["accuracy"] >= 0.75
    charts = json.dumps(result["metadata"]["visualizations"], ensure_ascii=False)
    assert "left" in charts and "right" in charts
    assert "良性" not in charts


@pytest.mark.parametrize("model,task,params", [
    ("linear_regression.optimized", "regression", {"max_iter": 20}),
    ("kmeans.optimized", "clustering", {"n_clusters": 2}),
    ("dbscan.optimized", "clustering", {}),
    ("isolation_forest.optimized", "anomaly_detection", {"n_estimators": 5}),
    ("one_class_svm.optimized", "anomaly_detection", {"max_iter": 5}),
])
def test_other_tasks_real_training_without_fake_labels(client, model, task, params):
    item = upload(client, "x1,x2,label\n" + "\n".join(f"{i},{i%3},{2*i+1}" for i in range(30)))
    target = "label" if task == "regression" else None
    response = client.post("/api/data-library/experiments", json={**selection(item, task, target), "model": model, "params": params})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metadata"]["visualizations"]
    if task == "clustering":
        assert "adjusted_rand_index" not in result["metrics"] and "ari" not in result["metrics"]
    if task == "anomaly_detection":
        assert "anomaly_f1" not in result["metrics"]
    if task == "regression":
        assert result["metadata"]["train_sample_cap"] is None


def test_labeled_anomaly_has_independent_holdout(client):
    item = upload(client, "x1,x2,label\n" + "\n".join(f"{i},{i%3},{int(i>=24)}" for i in range(30)))
    response = client.post("/api/data-library/experiments", json={**selection(item, "anomaly_detection"), "model": "one_class_svm.optimized", "params": {"max_iter": 5}})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metadata"]["train_test_overlap_count"] == 0
    assert result["metadata"]["train_anomaly_count"] == 0
    assert result["metadata"]["evaluation_sample_count"] < 30
    assert "anomaly_f1" in result["metrics"]


def test_nonfinite_categorical_leaking_features_and_unknown_ids_rejected(client):
    item = upload(client, "x1,x2,label\nNaN,no,0\n1,yes,1\n2,no,0\n3,yes,1")
    request = selection(item)
    assert not client.post("/api/data-library/compatibility", json=request).json()["models"]
    assert client.post("/api/data-library/experiments", json={**request, "model": "knn.optimized"}).status_code == 400
    item = upload(client)
    request = {**selection(item), "feature_columns": ["x1", "label"]}
    assert not client.post("/api/data-library/compatibility", json=request).json()["models"]
    assert client.get("/api/data-library/not_registered").status_code == 404


def test_large_table_keeps_isolation_forest_compatible_with_visible_svm_warning(client):
    item = upload(client, "x1,x2\n" + "\n".join(f"{i},{i%7}" for i in range(513)))
    request = selection(item, "anomaly_detection", None)
    result = client.post("/api/data-library/compatibility", json=request).json()
    assert result["models"] == ["isolation_forest.optimized"]
    assert result["issues"] == []
    assert "512" in result["warnings"][0]
    response = client.post("/api/data-library/experiments", json={**request, "model": "one_class_svm.optimized"})
    assert response.status_code == 400
    assert "512" in response.json()["detail"]["message"]


def test_identifier_is_not_suggested_as_numeric_feature(client):
    result = client.get("/api/data-library/wdbc").json()
    assert result["columns"][0]["suggested_role"] == "identifier"


@pytest.mark.parametrize("model,task,params", [
    ("logistic_regression.optimized", "classification", {"max_iter": 5}),
    ("gaussian_naive_bayes.optimized", "classification", {}),
    ("cart_decision_tree.optimized", "classification", {}),
    ("random_forest.optimized", "classification", {"n_estimators": 3}),
    ("gbdt_regression.optimized", "regression", {"n_estimators": 3}),
    ("mlp_regression.optimized", "regression", {"max_iter": 5, "hidden_layer_sizes": [4, 2]}),
])
def test_remaining_models_really_fit_custom_tables(client, model, task, params):
    item = upload(client, "x1,x2,label\n" + "\n".join(f"{i},{i%5},{int(i>=20) if task=='classification' else 2*i+1}" for i in range(40)))
    response = client.post("/api/data-library/experiments", json={**selection(item, task), "model": model, "params": params})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metadata"]["sample_count"] == 40
    assert result["metadata"]["visualizations"]
    assert result["metrics"]
    for name, value in params.items():
        assert result["effective_params"][name] == value


@pytest.mark.parametrize("filename,delimiter", [("labels.csv", ","), ("labels.tsv", "\t")])
def test_exact_numeric_label_tokens_are_not_collapsed(client, filename, delimiter):
    labels = ["9007199254740992", "9007199254740993", "9007199254740994"] * 3
    content = delimiter.join(["x1", "x2", "label"]) + "\n" + "\n".join(delimiter.join([str(i), str(i % 2), label]) for i, label in enumerate(labels))
    item = upload(client, content, filename)
    assert item["columns"][2]["unique_count"] == 3
    assert [row["label"] for row in item["preview"]] == labels
    reopened = client.get(f"/api/data-library/{item['id']}").json()
    assert reopened["preview"] == item["preview"]
    compatible = client.post("/api/data-library/compatibility", json=selection(item)).json()
    assert compatible["models"] == []
    assert compatible["issues"]
    result = client.post("/api/data-library/experiments", json={**selection(item, "clustering"), "model": "kmeans.optimized", "params": {"n_clusters": 3}})
    assert result.status_code == 200, result.text
    assert result.json()["metadata"]["label_names"] == labels[:3]


def test_zero_prefixed_binary_labels_keep_distinct_identity(client):
    labels = ["01"] * 10 + ["1"] * 10
    item = upload(client, "x1,x2,label\n" + "\n".join(f"{i},{i%3},{label}" for i, label in enumerate(labels)))
    assert item["columns"][2]["unique_count"] == 2
    assert [row["label"] for row in item["preview"]] == labels
    result = client.post("/api/data-library/experiments", json={**selection(item), "model": "knn.optimized", "params": {"n_neighbors": 1}})
    assert result.status_code == 200, result.text
    assert result.json()["metadata"]["label_names"] == ["01", "1"]


@pytest.mark.parametrize("model", ["isolation_forest.optimized", "one_class_svm.optimized"])
def test_unlabeled_anomaly_rejects_ignored_test_size(client, model):
    item = upload(client)
    response = client.post("/api/data-library/experiments", json={**selection(item, "anomaly_detection", None), "model": model, "test_size": 0.3})
    assert response.status_code == 400
    assert "test_size" in response.json()["detail"]["message"]
