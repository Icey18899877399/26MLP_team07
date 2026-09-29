"""Regression experiment adapters."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from Models.gbdt_regression_optimized import OptimizedGBDTRegressorScratch
from Models.linear_regression_optimized import OptimizedLinearRegressionScratch
from Models.mlp_regression_optimized import OptimizedMLPRegressorScratch

from ..datasets import LoadedDataset
from ..errors import InvalidParameterError
from ..types import ExperimentConfig, ExperimentResult, JSONValue
from ._common import (
    merge_model_kwargs,
    plain_split,
    regression_metrics,
    require_boolean,
    require_in_range,
    require_non_negative_number,
    require_positive_integer,
    require_positive_number,
)


def _fit_evaluate_regressor(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
    model_class: type,
    *,
    extra_kwargs: dict[str, JSONValue] | None = None,
    metadata_extra: Callable[[object], dict[str, JSONValue]] | None = None,
) -> ExperimentResult:
    """回归共享骨架：随机切分 → fit → predict → 回归指标 → 结果组装。"""

    test_size = 0.2 if config.test_size is None else float(config.test_size)
    train_indices, test_indices = plain_split(
        len(dataset.features),
        test_size=test_size,
        random_state=config.random_state,
    )
    train_features = [dataset.features[index] for index in train_indices]
    train_targets = [float(dataset.targets[index]) for index in train_indices]
    test_features = [dataset.features[index] for index in test_indices]
    test_targets = [float(dataset.targets[index]) for index in test_indices]

    model = model_class(**merge_model_kwargs(effective_params, extra_kwargs or {}))
    model.fit(train_features, train_targets)
    predictions = [float(value) for value in model.predict(test_features)]
    metrics = regression_metrics(test_targets, predictions)

    metadata: dict[str, JSONValue] = {
        "train_sample_count": len(train_indices),
        "test_sample_count": len(test_indices),
        "feature_count": dataset.info.feature_count,
        "random_state": config.random_state,
        "test_size": test_size,
    }
    if metadata_extra is not None:
        metadata.update(metadata_extra(model))

    return ExperimentResult(
        run_id=str(uuid4()),
        model=config.model,
        dataset=config.dataset,
        task="regression",
        effective_params=dict(effective_params),
        metrics=metrics,
        metadata=metadata,
    )


def run_linear_regression(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate the optimized linear regression (Adam minibatch)."""

    _validate_linear_params(effective_params)
    return _fit_evaluate_regressor(
        config,
        dataset,
        effective_params,
        OptimizedLinearRegressionScratch,
        extra_kwargs={"random_state": config.random_state},
        metadata_extra=lambda model: {"iteration_count": int(model.n_iter)},
    )


def _validate_linear_params(params: dict[str, JSONValue]) -> None:
    require_positive_number(params, "learning_rate")
    require_positive_integer(params, "max_iter")
    require_positive_integer(params, "batch_size")
    require_non_negative_number(params, "l2")
    require_boolean(params, "standardize")
    require_boolean(params, "standardize_target")


def run_gbdt_regression(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate the optimized GBDT regressor."""

    _validate_gbdt_params(effective_params)
    return _fit_evaluate_regressor(
        config,
        dataset,
        effective_params,
        OptimizedGBDTRegressorScratch,
        extra_kwargs={"random_state": config.random_state},
        metadata_extra=lambda model: {
            "trees_fitted": int(model.n_estimators_),
            **(
                {"final_train_loss": float(model.train_loss_[-1])}
                if model.train_loss_
                else {}
            ),
        },
    )


def _validate_gbdt_params(params: dict[str, JSONValue]) -> None:
    require_positive_integer(params, "n_estimators")
    require_positive_number(params, "learning_rate")
    require_positive_integer(params, "max_depth")
    require_positive_integer(params, "min_samples_split")
    require_positive_integer(params, "min_samples_leaf")
    require_in_range(params, "subsample", 0.0, 1.0, left_open=True, right_open=False)
    require_in_range(params, "validation_fraction", 0.0, 1.0)
    require_non_negative_number(params, "l2_regularization")


def run_mlp_regression(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate the optimized MLP regressor.

    hidden_layer_sizes 为元组（非 JSON 安全），由适配器固定为 (16, 8)。
    """

    _validate_mlp_params(effective_params)
    return _fit_evaluate_regressor(
        config,
        dataset,
        effective_params,
        OptimizedMLPRegressorScratch,
        extra_kwargs={
            "hidden_layer_sizes": (16, 8),
            "random_state": config.random_state,
        },
        metadata_extra=lambda model: {"iteration_count": int(model.n_iter_)},
    )


def _validate_mlp_params(params: dict[str, JSONValue]) -> None:
    if params["activation"] not in ("relu", "tanh"):
        raise InvalidParameterError("activation 必须是 'relu' 或 'tanh'")
    require_positive_number(params, "learning_rate")
    require_positive_integer(params, "max_iter")
    require_positive_integer(params, "batch_size")
    require_non_negative_number(params, "l2")
    require_in_range(params, "validation_fraction", 0.0, 1.0)
    require_positive_integer(params, "n_iter_no_change")
    require_boolean(params, "standardize")
    require_boolean(params, "standardize_target")
    require_positive_number(params, "gradient_clip")
