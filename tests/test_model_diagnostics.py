"""Diagnostics must expose fitted quantities, including custom column names."""
import importlib
import json
from types import SimpleNamespace

import pytest

from ml_core.interactive import _CLASSES
from Models.kmeans_optimized import OptimizedKMeansScratch
from Models.dbscan_optimized import OptimizedDBSCANScratch

CASES = [
    ("logistic_regression", {"max_iter": 8}, "logistic_coefficients"),
    ("knn", {"n_neighbors": 3}, "knn_neighborhood"),
    ("gaussian_naive_bayes", {}, "gnb_gaussian_profiles"),
    ("cart_decision_tree", {"max_depth": 3}, "cart_tree"),
    ("random_forest", {"n_estimators": 3, "max_depth": 3, "n_jobs": 1}, "rf_tree_diversity"),
    ("linear_regression", {"max_iter": 8}, "linear_coefficients"),
    ("gbdt_regression", {"n_estimators": 3, "max_depth": 2}, "gbdt_stage_loss"),
    ("mlp_regression", {"max_iter": 8, "hidden_layer_sizes": (3,)}, "mlp_architecture"),
    ("kmeans", {"n_clusters": 2, "n_init": 1, "max_iter": 8}, "kmeans_center_distances"),
    ("dbscan", {"eps": 0.5, "min_samples": 3}, "dbscan_density_roles"),
    ("isolation_forest", {"n_estimators": 3, "max_samples": 12}, "if_score_rank"),
    ("one_class_svm", {"max_iter": 8}, "ocsvm_margins"),
]
X = [[float(i), float((i * 7) % 11)] for i in range(24)]
Y = [int(i >= 12) for i in range(24)]


def helper():
    assert importlib.util.find_spec("ml_core.adapters.model_charts") is not None, "fitted-model diagnostics are missing"
    return importlib.import_module("ml_core.adapters.model_charts").model_charts


@pytest.mark.parametrize("name,params,chart_id", CASES)
def test_all_models_emit_distinct_nonempty_fitted_diagnostics(name, params, chart_id):
    build = helper()
    model_class = {**_CLASSES, "kmeans": OptimizedKMeansScratch, "dbscan": OptimizedDBSCANScratch}[name]
    model = model_class(**params)
    if name in {"kmeans", "dbscan", "isolation_forest", "one_class_svm"}:
        model.fit(X)
    else:
        model.fit(X, Y)
    charts = build(name + ".optimized", model, train_features=X, evaluation_features=X,
                   train_targets=Y, evaluation_targets=Y, predictions=model.predict(X),
                   feature_names=("temperature", "pressure"), class_names=("cold", "hot"))
    json.dumps(charts, allow_nan=False)
    by_id = {c["id"]: c for c in charts}
    assert chart_id in by_id
    assert all(c.get("category") == "model_diagnostic" for c in charts)
    assert any(s["data"] for s in by_id[chart_id]["option"]["series"])
    if name in {"logistic_regression", "linear_regression"}:
        assert by_id[chart_id]["option"]["series"][0]["data"] == pytest.approx(model.weights)
        assert by_id[chart_id]["option"]["xAxis"]["data"] == ["temperature", "pressure"]
    elif name == "cart_decision_tree":
        root = by_id[chart_id]["option"]["series"][0]["data"][0]
        assert root["value"] == 24
        assert "temperature" in root["name"] or "pressure" in root["name"]
    elif name == "gbdt_regression":
        assert [v[1] for v in by_id[chart_id]["option"]["series"][0]["data"]] == model.train_loss_
    elif name == "dbscan":
        counts = by_id[chart_id]["option"]["series"][0]["data"]
        assert sum(counts) == 24
        assert counts[0] == len(model.core_sample_indices_)
    elif name == "one_class_svm":
        values = sorted(v[1] for v in by_id[chart_id]["option"]["series"][0]["data"])
        assert values == pytest.approx(sorted(model.decision_function(X)))


def test_knn_neighborhood_uses_model_metric_and_training_indices():
    model = _CLASSES["knn"](n_neighbors=2, standardize=False, p=1).fit([[0, 0], [2, 0], [9, 0]], [0, 1, 1])
    chart = helper()("knn", model, train_features=[[0, 0], [2, 0], [9, 0]],
                     evaluation_features=[[0.5, 0]])[0]
    assert chart["id"] == "knn_neighborhood"
    assert chart["option"]["series"][0]["data"] == [0.5, 1.5]


def test_custom_labels_and_one_dimensional_cluster_projection():
    from ml_core.adapters.charts import classification_charts, clustering_charts
    chart = classification_charts([0, 1], [0, 1], [0.1, 0.9], class_names=("cat", "dog"))[0]
    assert chart["option"]["xAxis"]["data"] == ["cat (0)", "dog (1)"]
    cluster = clustering_charts([[1], [2]], [0, 1], feature_names=("temperature",))[0]
    assert "temperature" in cluster["option"]["xAxis"]["name"]
    assert cluster["option"]["series"][0]["data"][0][1] == 0


def test_original_feature_labels_come_from_registered_dataset_schema():
    from ml_core import ExperimentConfig, run_experiment
    result = run_experiment(ExperimentConfig(model="linear_regression.optimized", dataset="concrete", params={"max_iter": 1}))
    chart = next(c for c in result.metadata["visualizations"] if c["id"] == "linear_coefficients")
    assert chart["option"]["xAxis"]["data"][0] == "Cement"


@pytest.mark.parametrize("name", ["kmeans", "dbscan"])
def test_unlabeled_cluster_runs_omit_external_evaluation(name):
    from ml_core.registry import get_model_spec
    from ml_core.types import DatasetInfo, ExperimentConfig
    spec = get_model_spec(name + ".optimized")
    dataset = SimpleNamespace(info=DatasetInfo("upload", "upload", "clustering", 24, 2, False),
                              features=X, targets=(), feature_names=("temperature", "pressure"))
    result = spec.runner(ExperimentConfig(model=name + ".optimized", dataset="upload"), dataset,
                         {**spec.info.default_params, **({"n_clusters": 2} if name == "kmeans" else {})})
    assert "adjusted_rand_index" not in result.metrics
    assert any(c["id"].startswith(name + "_") for c in result.metadata["visualizations"])
