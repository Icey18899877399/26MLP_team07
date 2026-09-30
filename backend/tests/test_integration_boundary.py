from dataclasses import dataclass
from types import ModuleType

import pytest

from backend.contracts import ExperimentConfig
from backend.integration.ml_backend import (
    MLBackendUnavailable,
    MLExecutionError,
    MLRequestError,
    PackageMLBackend,
)


@dataclass
class PackageExperimentConfig:
    model: str
    dataset: str
    params: dict
    test_size: float | None
    random_state: int


class PackageMLCoreError(Exception):
    pass


class PackageUnknownModelError(PackageMLCoreError):
    pass


class PackageUnknownDatasetError(PackageMLCoreError):
    pass


class PackageIncompatibleDatasetError(PackageMLCoreError):
    pass


class PackageInvalidConfigError(PackageMLCoreError):
    pass


class PackageInvalidParameterError(PackageMLCoreError):
    pass


class PackageExperimentExecutionError(PackageMLCoreError):
    pass


def _contract_module() -> ModuleType:
    module = ModuleType("ml_core")
    module.ArtifactInfo = dict
    module.ModelInfo = dict
    module.DatasetInfo = dict
    module.ExperimentConfig = PackageExperimentConfig
    module.ExperimentResult = dict
    module.MLCoreError = PackageMLCoreError
    module.UnknownModelError = PackageUnknownModelError
    module.UnknownDatasetError = PackageUnknownDatasetError
    module.IncompatibleDatasetError = PackageIncompatibleDatasetError
    module.InvalidConfigError = PackageInvalidConfigError
    module.InvalidParameterError = PackageInvalidParameterError
    module.ExperimentExecutionError = PackageExperimentExecutionError
    module.list_models = lambda: [
        {
            "id": "kmeans.optimized",
            "display_name": "K-Means (Optimized)",
            "task": "clustering",
            "compatible_datasets": ["seeds"],
            "default_params": {"n_clusters": 3},
            "parameter_descriptions": {"n_clusters": "Cluster count"},
        }
    ]
    module.list_datasets = lambda: [
        {
            "id": "seeds",
            "display_name": "Seeds",
            "task": "clustering",
            "sample_count": 210,
            "feature_count": 7,
            "has_target": True,
        }
    ]
    module.run_experiment = lambda config: {
        "run_id": "run-1",
        "model": config.model,
        "dataset": config.dataset,
        "task": "clustering",
        "effective_params": {"n_clusters": 3, **config.params},
        "metrics": {"silhouette": 0.61},
        "artifacts": [],
        "metadata": {"samples": 210},
    }
    return module


def test_package_boundary_calls_only_public_contract() -> None:
    backend = PackageMLBackend(importer=lambda _name: _contract_module())

    assert backend.status().available is True
    assert backend.list_models()[0].id == "kmeans.optimized"
    assert backend.list_datasets()[0].id == "seeds"

    result = backend.run_experiment(
        ExperimentConfig(model="kmeans.optimized", dataset="seeds")
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


@pytest.mark.parametrize(
    "error_type",
    [
        PackageUnknownModelError,
        PackageUnknownDatasetError,
        PackageIncompatibleDatasetError,
        PackageInvalidConfigError,
        PackageInvalidParameterError,
    ],
)
def test_domain_errors_are_safe_request_errors(error_type) -> None:
    module = _contract_module()

    def reject(_config):
        raise error_type("request can be corrected")

    module.run_experiment = reject
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLRequestError, match="request can be corrected"):
        backend.run_experiment(
            ExperimentConfig(model="kmeans.optimized", dataset="seeds")
        )


def test_config_constructor_domain_error_is_a_safe_request_error() -> None:
    module = _contract_module()

    def reject_config(**_values):
        raise PackageInvalidConfigError("invalid experiment configuration")

    module.ExperimentConfig = reject_config
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLRequestError, match="invalid experiment configuration"):
        backend.run_experiment(
            ExperimentConfig(model="kmeans.optimized", dataset="seeds")
        )


def test_execution_error_does_not_expose_internal_package_message() -> None:
    module = _contract_module()

    def fail(_config):
        raise PackageExperimentExecutionError(r"failed at C:\private\data.csv")

    module.run_experiment = fail
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLExecutionError, match="ML experiment execution failed") as error:
        backend.run_experiment(
            ExperimentConfig(model="kmeans.optimized", dataset="seeds")
        )
    assert "private" not in str(error.value)


def test_invalid_experiment_result_is_rejected_as_execution_error() -> None:
    module = _contract_module()
    module.run_experiment = lambda _config: {
        "run_id": "run-1",
        "model": "kmeans.optimized",
        "dataset": "seeds",
        "task": "clustering",
        "effective_params": {},
        "metrics": {"silhouette": float("nan")},
        "artifacts": [],
        "metadata": {},
    }
    backend = PackageMLBackend(importer=lambda _name: module)

    with pytest.raises(MLExecutionError, match="invalid experiment result"):
        backend.run_experiment(
            ExperimentConfig(model="kmeans.optimized", dataset="seeds")
        )


def test_invalid_exception_hierarchy_marks_contract_unavailable() -> None:
    module = _contract_module()
    module.InvalidConfigError = ValueError
    backend = PackageMLBackend(importer=lambda _name: module)

    assert backend.status().available is False
    assert "must inherit MLCoreError" in backend.status().detail
