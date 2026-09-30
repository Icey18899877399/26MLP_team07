"""Real single-fit contract: original sources, explicit protocols and live charts."""
import hashlib
import json

import pytest
import ml_core


def catalog():
    assert hasattr(ml_core, "list_interactive_experiments"), "interactive catalog is missing"
    return {item["id"]: item for item in ml_core.list_interactive_experiments()}


def test_interactive_catalog_uses_original_sources_and_training_defaults():
    items = catalog()
    assert len(items) == 12
    assert items["mlp_regression"]["default_params"]["max_iter"] == 300
    assert items["mlp_regression"]["default_params"]["batch_size"] == 64
    assert items["logistic_regression"]["default_params"]["max_iter"] == 600
    assert items["kmeans"]["default_params"]["max_iter"] == 200
    assert items["isolation_forest"]["random_state"] == 13
    assert items["one_class_svm"]["random_state"] == 17
    assert items["one_class_svm"]["default_params"]["max_iter"] == 120
    for item in items.values():
        assert item["dataset"] in [d["id"] for d in item["datasets"]]
        assert all((ml_core.original_root() / d["path"]).is_file() for d in item["datasets"])
        assert "单次" in item["protocol"]
        assert set(item["default_params"]) <= set(item["parameter_descriptions"])
    assert items["mlp_regression"]["datasets"][0]["id"] == "concrete"
    assert items["one_class_svm"]["datasets"][0]["path"].endswith("6_cardio.npz")


SMOKE = [
    ("cart_decision_tree", {}), ("gaussian_naive_bayes", {}), ("knn", {}),
    ("logistic_regression", {"max_iter": 2}), ("random_forest", {"n_estimators": 2}),
    ("gbdt_regression", {"n_estimators": 2}), ("linear_regression", {"max_iter": 2}),
    ("mlp_regression", {"max_iter": 2}), ("dbscan", {}),
    ("kmeans", {"n_init": 1, "max_iter": 2}),
    ("isolation_forest", {"n_estimators": 2}), ("one_class_svm", {"max_iter": 2}),
]


@pytest.mark.parametrize("algorithm,params", SMOKE)
def test_all_twelve_train_real_original_data_and_return_live_charts(algorithm, params):
    item = catalog()[algorithm]
    source = ml_core.original_root() / item["datasets"][0]["path"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = ml_core.run_interactive_experiment(ml_core.ExperimentConfig(
        model=item["model"], dataset=item["dataset"], params=params,
        test_size=item["test_size"], random_state=item["random_state"],
    ))
    assert result.model == item["model"]
    assert result.metadata["dataset_source"] == item["datasets"][0]["path"]
    assert result.metadata["dataset_sha256"] == digest
    assert result.metadata["visualizations"]
    assert result.metadata["execution_mode"] == "interactive_single_fit"
    assert result.metadata["sample_count"] in (569, 1030, 210, 1831)
    assert all(result.effective_params[key] == value for key, value in params.items())
    if algorithm == "one_class_svm":
        assert result.metadata["train_sample_count"] == 72
        assert result.metadata["test_sample_count"] == 367
        assert result.metadata["train_test_overlap_count"] == 0
    if algorithm == "isolation_forest":
        # 1655 normal + 176 anomaly: rounded per-class train sizes 993 + 106.
        assert result.metadata["train_sample_count"] == 1099
        assert result.metadata["test_sample_count"] == 366
        assert result.metadata["train_test_overlap_count"] == 0
    if result.task == "anomaly_detection":
        charts = {chart["id"]: chart for chart in result.metadata["visualizations"]}
        assert all("独立测试集" in chart["title"] and "样本内评估" not in chart["title"] for chart in charts.values())
        distribution = charts["anomaly_score_distribution"]["option"]["series"]
        cells = charts["anomaly_confusion_matrix"]["option"]["series"][0]["data"]
        assert sum(sum(series["data"]) for series in distribution) == result.metadata["test_sample_count"]
        assert sum(cell[2] for cell in cells) == result.metadata["test_sample_count"]
    if result.task == "regression":
        assert result.metadata["train_sample_count"] == 824
        assert result.metadata["train_sample_cap"] is None
    json.dumps(result.to_dict(), allow_nan=False)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_interactive_rejects_foreign_dataset_and_unknown_parameters():
    catalog()
    with pytest.raises(ml_core.IncompatibleDatasetError):
        ml_core.run_interactive_experiment(ml_core.ExperimentConfig(
            model="mlp_regression.optimized", dataset="california_housing"))
    with pytest.raises(ml_core.InvalidParameterError):
        ml_core.run_interactive_experiment(ml_core.ExperimentConfig(
            model="knn.optimized", dataset="wdbc", params={"teammate_preset": True}))
    with pytest.raises(ml_core.InvalidConfigError):
        ml_core.run_interactive_experiment(ml_core.ExperimentConfig(
            model="one_class_svm.optimized", dataset="6_cardio", test_size=0.3))


def test_gbdt_optional_patience_enables_real_internal_validation_and_early_stopping():
    item = catalog()["gbdt_regression"]
    assert item["default_params"].get("n_iter_no_change", "missing") is None
    config = dict(model=item["model"], dataset=item["dataset"])
    disabled = ml_core.run_interactive_experiment(ml_core.ExperimentConfig(**config,
        params={"n_estimators": 8, "learning_rate": 2.0}))
    enabled = ml_core.run_interactive_experiment(ml_core.ExperimentConfig(**config,
        params={"n_estimators": 8, "learning_rate": 2.0, "n_iter_no_change": 1, "validation_fraction": 0.2}))
    assert disabled.effective_params["n_iter_no_change"] is None
    assert enabled.effective_params["n_iter_no_change"] == 1
    disabled_loss = next(c for c in disabled.metadata["visualizations"] if c["id"] == "training_loss")["option"]["series"]
    enabled_loss = next(c for c in enabled.metadata["visualizations"] if c["id"] == "training_loss")["option"]["series"]
    assert len(disabled_loss) == 1
    assert len(enabled_loss) == 2
    assert len(enabled_loss[0]["data"]) < len(disabled_loss[0]["data"])
    assert enabled.metrics != disabled.metrics


@pytest.mark.parametrize("patience", [True, False, 0, -1, 1.5, "2"])
def test_gbdt_patience_rejects_nonpositive_or_noninteger_values(patience):
    with pytest.raises(ml_core.InvalidParameterError, match="n_iter_no_change"):
        ml_core.run_interactive_experiment(ml_core.ExperimentConfig(
            model="gbdt_regression.optimized", dataset="concrete",
            params={"n_iter_no_change": patience, "n_estimators": 2}))
