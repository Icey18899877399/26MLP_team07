from dataclasses import dataclass
from types import ModuleType

import pytest

from backend.integration.ml_backend import (
    MLBackendUnavailable,
    MLExecutionError,
    MLRequestError,
    PackageMLBackend,
)
from backend.contracts import ExperimentConfig


@dataclass
class PackageExperimentConfig:
    model: str
    variant: str
    dataset: str
    params: dict
    test_size: float
    random_state: int


class PackageMLCoreError(Exception):
    pass


class PackageInvalidExperimentError(PackageMLCoreError):
    pass


class PackageExperimentExecutionError(PackageMLCoreError):
    pass


def _contract_module() -> ModuleType:
    module = ModuleType("ml_core")
    module.ModelSpec = dict
    module.DatasetSpec = dict
    module.ExperimentConfig = PackageExperimentConfig
    module.ExperimentResult = dict
    module.MLCoreError = PackageMLCoreError
    module.InvalidExperimentError = PackageInvalidExperimentError
    module.ExperimentExecutionError = PackageExperimentExecutionError
    module.list_models = lambda: [
        {
            "id": "kmeans",
            "name": "K-Means",
            "task_type": "clustering",
            "variants": ["base", "optimized"],
            "parameters": {},
        }
    ]
    module.list_datasets = lambda: [
        {"id": "seeds", "name": "Seeds", "task_type": "clustering"}
    ]
    module.run_experiment = lambda config: {
        "model": config.model,
        "variant": config.variant,
        "dataset": config.dataset,
        "metrics": {"silhouette": 0.61},
        "diagnostics": {"labels": [0, 1, 1]},
    }
    return module


def test_package_boundary_calls_only_public_contract() -> None:
    backend = PackageMLBackend(importer=lambda _name: _contract_module())

    assert backend.status().available is True
    assert backend.list_models()[0].id == "kmeans"
    assert backend.list_datasets()[0].id == "seeds"

    result = backend.run_experiment(
        ExperimentConfig(model="kmeans", dataset="seeds", variant="optimized")
    )
    assert result.metrics["silhouette"] == 0.61


def test_missing_public_export_marks_package_unavailable() -> None:
    incomplete_module = ModuleType("ml_core")
    backend = PackageMLBackend(importer=lambda _name: incomplete_module)

    assert backend.status().available is False
    with pytest.raises(MLBackendUnavailable):
        backend.list_models()


def test_invalid_package_result_is_rejected_at_boundary() -> None:
    module = _contract_module()
    module.list_models = lambda: [{"id": "missing-required-fields"}]
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLExecutionError):
        backend.list_models()


def test_invalid_experiment_error_is_a_safe_request_error() -> None:
    module = _contract_module()

    def reject(_config):
        raise PackageInvalidExperimentError("model and dataset are incompatible")

    module.run_experiment = reject
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLRequestError, match="model and dataset are incompatible"):
        backend.run_experiment(ExperimentConfig(model="kmeans", dataset="seeds"))


def test_config_constructor_domain_error_is_a_safe_request_error() -> None:
    module = _contract_module()

    def reject_config(**_values):
        raise PackageInvalidExperimentError("invalid model parameter")

    module.ExperimentConfig = reject_config
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLRequestError, match="invalid model parameter"):
        backend.run_experiment(ExperimentConfig(model="kmeans", dataset="seeds"))


def test_execution_error_does_not_expose_internal_package_message() -> None:
    module = _contract_module()

    def fail(_config):
        raise PackageExperimentExecutionError(r"failed at C:\private\data.csv")

    module.run_experiment = fail
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLExecutionError, match="ML experiment execution failed") as error:
        backend.run_experiment(ExperimentConfig(model="kmeans", dataset="seeds"))
    assert "private" not in str(error.value)


def test_invalid_experiment_result_is_rejected_as_execution_error() -> None:
    module = _contract_module()
    module.run_experiment = lambda _config: {
        "model": "kmeans",
        "variant": "base",
        "dataset": "seeds",
        "metrics": {"silhouette": float("nan")},
    }
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLExecutionError, match="invalid experiment result"):
        backend.run_experiment(ExperimentConfig(model="kmeans", dataset="seeds"))


def test_invalid_exception_hierarchy_marks_contract_unavailable() -> None:
    module = _contract_module()
    module.InvalidExperimentError = ValueError
    backend = PackageMLBackend(importer=lambda _name: module)

    assert backend.status().available is False
    assert "must inherit MLCoreError" in backend.status().detail
