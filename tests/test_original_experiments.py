from pathlib import Path

import pytest

from ml_core import (
    build_original_command,
    list_original_datasets,
    list_original_experiments,
    resolve_original_asset,
)


def test_catalog_contains_original_gallery_and_sources():
    experiments = list_original_experiments()
    assert len(experiments) == 12
    assert {item["task"] for item in experiments} == {
        "classification", "regression", "clustering", "anomaly_detection"
    }
    assert sum(len(item["figures"]) for item in experiments) == 72
    assert sum(len(item["source_data"]) for item in experiments) == 18
    assert all(len(item["figures"]) == 6 for item in experiments)
    assert all(item["script"] == f'visualization.{item["id"]}_figures' for item in experiments)
    assert all(resolve_original_asset(item["id"], item["figures"][0]["name"]).read_bytes().startswith(b"\x89PNG") for item in experiments)


def test_dataset_catalog_distinguishes_archives_from_used_inputs():
    datasets = {item["id"]: item for item in list_original_datasets()}
    assert set(datasets) == {"wdbc", "concrete", "seeds", "6_cardio", "23_mammography", "adult", "california_housing"}
    assert datasets["adult"]["archived_only"] is True
    assert datasets["california_housing"]["archived_only"] is True
    assert datasets["wdbc"]["used_by"] == ["cart_decision_tree", "gaussian_naive_bayes", "knn", "logistic_regression", "random_forest"]


def test_original_asset_allowlist_rejects_unknown_and_traversal():
    for experiment_id, filename in [
        ("unknown", "01_class_distribution.png"),
        ("logistic_regression", "../../pyproject.toml"),
        ("logistic_regression", "source_data/../01_class_distribution.png"),
        ("logistic_regression", "not_an_original.png"),
    ]:
        with pytest.raises((KeyError, ValueError)):
            resolve_original_asset(experiment_id, filename)


def test_commands_keep_original_cli_defaults(tmp_path: Path):
    for item in list_original_experiments():
        command = build_original_command(item["id"], tmp_path / item["id"])
        assert command[1:3] == ["-m", item["script"]]
        assert "--quick" not in command
        assert not any(flag in command for flag in ["--seed", "--folds", "--max-iter", "--tuning-samples"])
        assert command[-1] == str(tmp_path / item["id"])
    oc = build_original_command("one_class_svm", tmp_path / "oc")
    assert "--data-dir" in oc and "--output-dir" in oc
    iso = build_original_command("isolation_forest", tmp_path / "iso")
    assert "--cardio" in iso and "--mammography" in iso


def test_command_rejects_missing_original_source_data(tmp_path: Path, monkeypatch):
    import ml_core.original_runner as original_runner

    monkeypatch.setattr(original_runner, "original_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        build_original_command("logistic_regression", tmp_path / "output")
