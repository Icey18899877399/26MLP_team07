"""Classification experiment adapters."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from Models.cart_decision_tree_optimized import OptimizedCARTClassifierScratch
from Models.gaussian_naive_bayes_optimized import OptimizedGaussianNaiveBayesScratch
from Models.knn_optimized import OptimizedKNNScratch
from Models.logistic_regression_optimized import OptimizedLogisticRegressionScratch
from Models.random_forest_optimized import OptimizedRandomForestClassifierScratch

from ..datasets import LoadedDataset
from ..errors import InvalidParameterError
from ..types import ExperimentConfig, ExperimentResult, JSONValue
from ._common import (
    merge_model_kwargs,
    require_boolean,
    require_bounded_number,
    require_non_negative_number,
    require_optional_positive_integer_or_none,
    require_positive_integer,
    require_positive_number,
    stratified_split,
)


def _classification_metrics(
    targets: list[int],
    predictions: list[int],
) -> dict[str, int | float]:
    tn = sum(target == 0 and prediction == 0 for target, prediction in zip(targets, predictions))
    fp = sum(target == 0 and prediction == 1 for target, prediction in zip(targets, predictions))
    fn = sum(target == 1 and prediction == 0 for target, prediction in zip(targets, predictions))
    tp = sum(target == 1 and prediction == 1 for target, prediction in zip(targets, predictions))
    total = len(targets)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _fit_evaluate_classifier(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
    model_class: type,
    *,
    extra_kwargs: dict[str, JSONValue] | None = None,
    metadata_extra: Callable[[object], dict[str, JSONValue]] | None = None,
) -> ExperimentResult:
    """分类共享骨架：分层切分 → fit → predict → 分类指标 → 结果组装。"""

    test_size = 0.2 if config.test_size is None else float(config.test_size)
    train_indices, test_indices = stratified_split(
        dataset.targets,
        test_size=test_size,
        random_state=config.random_state,
    )
    train_features = [dataset.features[index] for index in train_indices]
    train_targets = [dataset.targets[index] for index in train_indices]
    test_features = [dataset.features[index] for index in test_indices]
    test_targets = [dataset.targets[index] for index in test_indices]

    model = model_class(**merge_model_kwargs(effective_params, extra_kwargs or {}))
    model.fit(train_features, train_targets)
    predictions = model.predict(test_features)
    metrics = _classification_metrics(test_targets, predictions)

    metadata: dict[str, JSONValue] = {
        "train_sample_count": len(train_indices),
        "test_sample_count": len(test_indices),
        "feature_count": dataset.info.feature_count,
        "positive_label": 1,
        "random_state": config.random_state,
        "test_size": test_size,
    }
    if metadata_extra is not None:
        metadata.update(metadata_extra(model))

    return ExperimentResult(
        run_id=str(uuid4()),
        model=config.model,
        dataset=config.dataset,
        task="classification",
        effective_params=dict(effective_params),
        metrics=metrics,
        metadata=metadata,
    )


def run_logistic_regression(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate optimized logistic regression on a stratified split."""

    _validate_logistic_params(effective_params)
    return _fit_evaluate_classifier(
        config,
        dataset,
        effective_params,
        OptimizedLogisticRegressionScratch,
        metadata_extra=lambda model: {"iteration_count": model.n_iter},
    )


def _validate_logistic_params(params: dict[str, JSONValue]) -> None:
    require_positive_number(params, "learning_rate")
    require_positive_integer(params, "max_iter")
    require_bounded_number(params, "threshold", minimum=0.0, maximum=1.0)
    require_non_negative_number(params, "l2")
    require_non_negative_number(params, "tol")
    require_boolean(params, "standardize")
    if params["class_weight"] not in (None, "balanced"):
        raise InvalidParameterError("class_weight 必须是 null 或 'balanced'")


def run_knn(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate optimized KNN on a stratified split."""

    _validate_knn_params(effective_params)
    return _fit_evaluate_classifier(
        config,
        dataset,
        effective_params,
        OptimizedKNNScratch,
    )


def _validate_knn_params(params: dict[str, JSONValue]) -> None:
    require_positive_integer(params, "n_neighbors")
    require_positive_number(params, "p")
    if float(params["p"]) < 1.0:
        raise InvalidParameterError("p 必须是不小于 1 的有限数值")
    if params["weights"] not in ("uniform", "distance"):
        raise InvalidParameterError("weights 必须是 'uniform' 或 'distance'")
    require_boolean(params, "standardize")


def run_gaussian_naive_bayes(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate optimized Gaussian Naive Bayes on a stratified split."""

    _validate_gaussian_nb_params(effective_params)
    return _fit_evaluate_classifier(
        config,
        dataset,
        effective_params,
        OptimizedGaussianNaiveBayesScratch,
    )


def _validate_gaussian_nb_params(params: dict[str, JSONValue]) -> None:
    require_positive_number(params, "var_smoothing")


def run_cart_decision_tree(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate the optimized CART classifier on a stratified split."""

    _validate_cart_params(effective_params)
    return _fit_evaluate_classifier(
        config,
        dataset,
        effective_params,
        OptimizedCARTClassifierScratch,
        extra_kwargs={"random_state": config.random_state},
        metadata_extra=lambda model: {
            "tree_depth": int(model.tree_depth_),
            "leaf_count": int(model.n_leaves_),
        },
    )


def _validate_cart_params(params: dict[str, JSONValue]) -> None:
    require_optional_positive_integer_or_none(params, "max_depth")
    require_positive_integer(params, "min_samples_split")
    require_positive_integer(params, "min_samples_leaf")
    require_non_negative_number(params, "min_impurity_decrease")
    require_non_negative_number(params, "ccp_alpha")


def run_random_forest(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate the optimized random forest on a stratified split.

    n_jobs 固定为 1（Windows 下进程池 spawn 不安全），其余危险参数由适配器注入。
    """

    _validate_random_forest_params(effective_params)
    return _fit_evaluate_classifier(
        config,
        dataset,
        effective_params,
        OptimizedRandomForestClassifierScratch,
        extra_kwargs={
            "n_jobs": 1,
            "max_features": "sqrt",
            "class_weight": None,
            "ccp_alpha": 0.0,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "max_samples": None,
            "oob_score": True,
            "random_state": config.random_state,
        },
    )


def _validate_random_forest_params(params: dict[str, JSONValue]) -> None:
    require_positive_integer(params, "n_estimators")
    require_optional_positive_integer_or_none(params, "max_depth")
    require_positive_integer(params, "min_samples_split")
    require_positive_integer(params, "min_samples_leaf")
    if params["voting"] not in ("soft", "hard"):
        raise InvalidParameterError("voting 必须是 'soft' 或 'hard'")
