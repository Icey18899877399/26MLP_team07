"""Anomaly detection experiment adapters.

数据集标签约定：0 = 正常，1 = 异常。
模型输出约定：1 = 正常，-1 = 异常（IsolationForest / OneClassSVM）。
task = "anomaly_detection"；test_size 对该任务无意义，适配器忽略。
"""

from __future__ import annotations

import random
from uuid import uuid4

from Models.isolation_forest_optimized import OptimizedIsolationForestScratch
from Models.one_class_svm_optimized import OptimizedOneClassSVMScratch

from ..datasets import LoadedDataset
from ..errors import InvalidParameterError
from ..types import ExperimentConfig, ExperimentResult, JSONValue
from ._common import (
    is_finite_number,
    merge_model_kwargs,
    require_boolean,
    require_in_range,
    require_positive_integer,
)

# 单类 SVM 的 O(n^2) 核矩阵：训练集（仅正常样本）子采样上限
_OCSVM_TRAIN_CAP = 300


def _anomaly_metrics(
    targets: tuple[int | float, ...] | list[int | float],
    predictions: list[int],
) -> dict[str, int | float]:
    """按异常类（正类）计算检出指标。"""

    true_anomaly = [int(t) == 1 for t in targets]
    pred_anomaly = [p == -1 for p in predictions]
    tp = sum(a and b for a, b in zip(true_anomaly, pred_anomaly))
    fp = sum((not a) and b for a, b in zip(true_anomaly, pred_anomaly))
    fn = sum(a and (not b) for a, b in zip(true_anomaly, pred_anomaly))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_anomalies": tp + fn,
        "detected_anomalies": tp + fp,
        "anomaly_recall": round(recall, 8),
        "anomaly_precision": round(precision, 8),
        "anomaly_f1": round(f1, 8),
    }


def run_isolation_forest(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """全量数据上训练孤立森林并全量评估。"""

    _validate_isolation_forest_params(effective_params)
    model = OptimizedIsolationForestScratch(
        **merge_model_kwargs(
            effective_params,
            {
                "max_depth": None,
                "n_split_candidates": 3,
                "random_state": config.random_state,
            },
        )
    )
    predictions = model.fit_predict(dataset.features)
    metrics = _anomaly_metrics(dataset.targets, predictions)

    return ExperimentResult(
        run_id=str(uuid4()),
        model=config.model,
        dataset=config.dataset,
        task="anomaly_detection",
        effective_params=dict(effective_params),
        metrics=metrics,
        metadata={
            "sample_count": len(dataset.features),
            "feature_count": dataset.info.feature_count,
            "threshold": float(model.threshold_),
            "anomaly_rate": round(float(sum(1 for p in predictions if p == -1)) / len(predictions), 8),
            "random_state": config.random_state,
        },
    )


def _validate_isolation_forest_params(params: dict[str, JSONValue]) -> None:
    require_positive_integer(params, "n_estimators")
    max_samples = params["max_samples"]
    if isinstance(max_samples, bool) or not is_finite_number(max_samples):
        raise InvalidParameterError("max_samples 必须是正整数或 (0,1] 的比例")
    if isinstance(max_samples, int):
        if max_samples <= 0:
            raise InvalidParameterError("max_samples 必须是正整数或 (0,1] 的比例")
    elif not (0.0 < float(max_samples) <= 1.0):
        raise InvalidParameterError("max_samples 必须是正整数或 (0,1] 的比例")
    require_in_range(params, "contamination", 0.0, 0.5, left_open=True, right_open=False)
    max_features = params["max_features"]
    if isinstance(max_features, bool) or not is_finite_number(max_features):
        raise InvalidParameterError("max_features 必须是正整数或 (0,1] 的比例")
    if isinstance(max_features, int):
        if max_features <= 0:
            raise InvalidParameterError("max_features 必须是正整数或 (0,1] 的比例")
    elif not (0.0 < float(max_features) <= 1.0):
        raise InvalidParameterError("max_features 必须是正整数或 (0,1] 的比例")


def run_one_class_svm(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """仅用正常样本训练单类 SVM（子采样上限 300），全量评估。"""

    _validate_one_class_svm_params(effective_params)
    normal_indices = [index for index, target in enumerate(dataset.targets) if int(target) == 0]
    generator = random.Random(config.random_state)
    generator.shuffle(normal_indices)
    train_indices = normal_indices[:_OCSVM_TRAIN_CAP]
    train_features = [dataset.features[index] for index in train_indices]

    model = OptimizedOneClassSVMScratch(
        **merge_model_kwargs(effective_params, {"kernel": "rbf"})
    )
    model.fit(train_features)
    predictions = model.predict(dataset.features)
    metrics = _anomaly_metrics(dataset.targets, predictions)

    return ExperimentResult(
        run_id=str(uuid4()),
        model=config.model,
        dataset=config.dataset,
        task="anomaly_detection",
        effective_params=dict(effective_params),
        metrics=metrics,
        metadata={
            "sample_count": len(dataset.features),
            "feature_count": dataset.info.feature_count,
            "train_sample_count": len(train_indices),
            "train_anomaly_count": 0,
            "random_state": config.random_state,
        },
    )


def _validate_one_class_svm_params(params: dict[str, JSONValue]) -> None:
    require_in_range(params, "nu", 0.0, 1.0, left_open=True, right_open=False)
    gamma = params["gamma"]
    if gamma != "scale" and (
        isinstance(gamma, bool) or not is_finite_number(gamma) or float(gamma) <= 0.0
    ):
        raise InvalidParameterError("gamma 必须是 'scale' 或正的有限数值")
    require_positive_integer(params, "max_iter")
    require_boolean(params, "standardize")
