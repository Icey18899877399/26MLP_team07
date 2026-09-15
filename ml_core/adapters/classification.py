"""Classification experiment adapters."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from uuid import uuid4

from Models.logistic_regression_optimized import OptimizedLogisticRegressionScratch

from ..datasets import LoadedDataset
from ..errors import InvalidParameterError
from ..types import ExperimentConfig, ExperimentResult, JSONValue


def run_logistic_regression(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Train and evaluate optimized logistic regression on a stratified split."""

    _validate_logistic_params(effective_params)
    test_size = 0.2 if config.test_size is None else float(config.test_size)
    train_indices, test_indices = _stratified_split(
        dataset.targets,
        test_size=test_size,
        random_state=config.random_state,
    )
    train_features = [dataset.features[index] for index in train_indices]
    train_targets = [dataset.targets[index] for index in train_indices]
    test_features = [dataset.features[index] for index in test_indices]
    test_targets = [dataset.targets[index] for index in test_indices]

    model = OptimizedLogisticRegressionScratch(**effective_params)
    model.fit(train_features, train_targets)
    predictions = model.predict(test_features)
    metrics = _classification_metrics(test_targets, predictions)

    return ExperimentResult(
        run_id=str(uuid4()),
        model=config.model,
        dataset=config.dataset,
        task="classification",
        effective_params=dict(effective_params),
        metrics=metrics,
        metadata={
            "train_sample_count": len(train_indices),
            "test_sample_count": len(test_indices),
            "feature_count": dataset.info.feature_count,
            "iteration_count": model.n_iter,
            "positive_label": 1,
            "random_state": config.random_state,
            "test_size": test_size,
        },
    )


def _validate_logistic_params(params: dict[str, JSONValue]) -> None:
    _require_positive_number(params, "learning_rate")
    _require_positive_integer(params, "max_iter")
    _require_bounded_number(params, "threshold", minimum=0.0, maximum=1.0)
    _require_non_negative_number(params, "l2")
    _require_non_negative_number(params, "tol")
    if not isinstance(params["standardize"], bool):
        raise InvalidParameterError("standardize must be a boolean")
    if params["class_weight"] not in (None, "balanced"):
        raise InvalidParameterError("class_weight must be null or 'balanced'")


def _require_positive_integer(params: dict[str, JSONValue], name: str) -> None:
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidParameterError(f"{name} must be a positive integer")


def _require_positive_number(params: dict[str, JSONValue], name: str) -> None:
    value = params[name]
    if not _is_finite_number(value) or float(value) <= 0.0:
        raise InvalidParameterError(f"{name} must be a positive finite number")


def _require_non_negative_number(params: dict[str, JSONValue], name: str) -> None:
    value = params[name]
    if not _is_finite_number(value) or float(value) < 0.0:
        raise InvalidParameterError(f"{name} must be a non-negative finite number")


def _require_bounded_number(
    params: dict[str, JSONValue],
    name: str,
    minimum: float,
    maximum: float,
) -> None:
    value = params[name]
    if (
        not _is_finite_number(value)
        or float(value) < minimum
        or float(value) > maximum
    ):
        raise InvalidParameterError(
            f"{name} must be a finite number between {minimum} and {maximum}"
        )


def _is_finite_number(value: JSONValue) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _stratified_split(
    targets: tuple[int, ...],
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[target].append(index)

    generator = random.Random(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for target in sorted(grouped):
        indices = list(grouped[target])
        generator.shuffle(indices)
        test_count = max(1, min(len(indices) - 1, round(len(indices) * test_size)))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    generator.shuffle(train_indices)
    generator.shuffle(test_indices)
    return train_indices, test_indices


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
