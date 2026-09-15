"""Private registry connecting stable IDs to experiment runners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from .datasets import LoadedDataset
from .types import ExperimentConfig, ExperimentResult, JSONValue, ModelInfo


Runner = Callable[
    [ExperimentConfig, LoadedDataset, dict[str, JSONValue]],
    ExperimentResult,
]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    info: ModelInfo
    runner: Runner | None = None


_MODEL_SPECS = {
    "kmeans.optimized": ModelSpec(
        info=ModelInfo(
            id="kmeans.optimized",
            display_name="Optimized K-Means",
            task="clustering",
            compatible_datasets=("seeds",),
            default_params={
                "n_clusters": 3,
                "init": "k-means++",
                "n_init": 10,
                "max_iter": 300,
                "tol": 1e-4,
                "standardize": True,
            },
            parameter_descriptions={
                "n_clusters": "Number of clusters.",
                "init": "Centroid initialization: 'random' or 'k-means++'.",
                "n_init": "Independent initializations; the lowest-inertia run wins.",
                "max_iter": "Maximum iterations per initialization.",
                "tol": "Non-negative centroid-shift stopping tolerance.",
                "standardize": "Whether to standardize features before clustering.",
            },
        )
    ),
    "logistic_regression.optimized": ModelSpec(
        info=ModelInfo(
            id="logistic_regression.optimized",
            display_name="Optimized Logistic Regression",
            task="classification",
            compatible_datasets=("wdbc",),
            default_params={
                "learning_rate": 0.1,
                "max_iter": 1000,
                "threshold": 0.5,
                "l2": 0.0,
                "tol": 1e-8,
                "standardize": True,
                "class_weight": None,
            },
            parameter_descriptions={
                "learning_rate": "Positive gradient-descent learning rate.",
                "max_iter": "Positive maximum training iterations.",
                "threshold": "Classification threshold between zero and one.",
                "l2": "Non-negative L2 regularization strength.",
                "tol": "Non-negative early-stopping tolerance.",
                "standardize": "Whether to standardize training features.",
                "class_weight": "Either null or 'balanced'.",
            },
        )
    ),
}


def model_catalog() -> tuple[ModelInfo, ...]:
    """Return detached metadata copies in stable identifier order."""

    return tuple(
        replace(
            _MODEL_SPECS[model_id].info,
            default_params=dict(_MODEL_SPECS[model_id].info.default_params),
            parameter_descriptions=dict(
                _MODEL_SPECS[model_id].info.parameter_descriptions
            ),
        )
        for model_id in sorted(_MODEL_SPECS)
    )


def get_model_spec(model_id: str) -> ModelSpec | None:
    return _MODEL_SPECS.get(model_id)


def register_runner(model_id: str, runner: Runner) -> None:
    """Attach an internal runner while preserving the public specification."""

    spec = _MODEL_SPECS[model_id]
    _MODEL_SPECS[model_id] = replace(spec, runner=runner)
