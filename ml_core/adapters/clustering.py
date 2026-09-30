"""Clustering experiment adapters."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from uuid import uuid4

from Models.dbscan_optimized import OptimizedDBSCANScratch
from Models.kmeans_optimized import OptimizedKMeansScratch

from ..datasets import LoadedDataset
from ..errors import InvalidParameterError
from ..types import ExperimentConfig, ExperimentResult, JSONValue
from ._common import (
    is_finite_number,
    merge_model_kwargs,
    require_boolean,
    require_positive_integer,
)
from .charts import clustering_charts
from .model_charts import dataset_feature_names, model_charts


def adjusted_rand_index(reference: list[int] | tuple[int, ...], predicted: list[int]) -> float:
    """Compute the adjusted Rand index without a third-party dependency."""

    if len(reference) != len(predicted):
        raise ValueError("reference and predicted labels must have the same length")
    if len(reference) < 2:
        return 1.0

    contingency: dict[tuple[int, int], int] = defaultdict(int)
    reference_counts: Counter[int] = Counter()
    predicted_counts: Counter[int] = Counter()
    for reference_label, predicted_label in zip(reference, predicted):
        contingency[(reference_label, predicted_label)] += 1
        reference_counts[reference_label] += 1
        predicted_counts[predicted_label] += 1

    joint_pairs = sum(_choose_two(count) for count in contingency.values())
    reference_pairs = sum(_choose_two(count) for count in reference_counts.values())
    predicted_pairs = sum(_choose_two(count) for count in predicted_counts.values())
    total_pairs = _choose_two(len(reference))
    expected = reference_pairs * predicted_pairs / total_pairs
    maximum = 0.5 * (reference_pairs + predicted_pairs)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0
    return (joint_pairs - expected) / denominator


def _choose_two(count: int) -> int:
    return count * (count - 1) // 2


def run_kmeans(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Fit optimized K-Means and normalize its clustering report."""

    _validate_kmeans_params(effective_params, sample_count=len(dataset.features))
    model = OptimizedKMeansScratch(
        **effective_params,
        random_state=config.random_state,
    )
    labels = model.fit_predict(dataset.features)
    cluster_sizes = Counter(labels)

    return ExperimentResult(
        run_id=str(uuid4()),
        model=config.model,
        dataset=config.dataset,
        task="clustering",
        effective_params=dict(effective_params),
        metrics={
            "inertia": float(model.inertia_),
            **({"adjusted_rand_index": adjusted_rand_index(dataset.targets, labels)} if dataset.targets else {}),
            "n_clusters": int(model.n_clusters),
        },
        metadata={
            "sample_count": len(dataset.features),
            "feature_count": dataset.info.feature_count,
            "iteration_count": model.n_iter_,
            "evaluation_protocol": "in_sample：全体样本聚类；有参考标签时计算外部 ARI，无参考标签时不计算",
            "visualizations": clustering_charts(dataset.features, labels, feature_names=dataset_feature_names(dataset))
                + model_charts(config.model, model, train_features=dataset.features, evaluation_features=dataset.features,
                    predictions=labels, feature_names=dataset_feature_names(dataset)),
            "random_state": config.random_state,
            "cluster_sizes": {
                str(cluster): cluster_sizes.get(cluster, 0)
                for cluster in range(model.n_clusters)
            },
        },
    )


def run_dbscan(
    config: ExperimentConfig,
    dataset: LoadedDataset,
    effective_params: dict[str, JSONValue],
) -> ExperimentResult:
    """Fit optimized DBSCAN (eps 作用于标准化空间) and normalize its report."""

    _validate_dbscan_params(effective_params)
    model = OptimizedDBSCANScratch(
        **merge_model_kwargs(effective_params, {"algorithm": "kd_tree", "leaf_size": 20})
    )
    labels = model.fit_predict(dataset.features)
    cluster_sizes = Counter(label for label in labels if label >= 0)

    return ExperimentResult(
        run_id=str(uuid4()),
        model=config.model,
        dataset=config.dataset,
        task="clustering",
        effective_params=dict(effective_params),
        metrics={
            "n_clusters": int(model.n_clusters_),
            "noise_points": int(sum(1 for label in labels if label == -1)),
            **({"adjusted_rand_index": adjusted_rand_index(dataset.targets, labels)} if dataset.targets else {}),
        },
        metadata={
            "sample_count": len(dataset.features),
            "feature_count": dataset.info.feature_count,
            "evaluation_protocol": "in_sample：全体样本聚类；有参考标签时计算外部 ARI，无参考标签时不计算",
            "visualizations": clustering_charts(dataset.features, labels, feature_names=dataset_feature_names(dataset))
                + model_charts(config.model, model, train_features=dataset.features, evaluation_features=dataset.features,
                    predictions=labels, feature_names=dataset_feature_names(dataset)),
            "cluster_sizes": {
                str(cluster): cluster_sizes.get(cluster, 0)
                for cluster in range(int(model.n_clusters_))
            },
        },
    )


def _validate_kmeans_params(
    params: dict[str, JSONValue],
    sample_count: int,
) -> None:
    require_positive_integer(params, "n_clusters")
    if int(params["n_clusters"]) > sample_count:
        raise InvalidParameterError("n_clusters 不能超过数据集样本数")
    if params["init"] not in ("random", "k-means++"):
        raise InvalidParameterError("init 必须是 'random' 或 'k-means++'")
    require_positive_integer(params, "n_init")
    require_positive_integer(params, "max_iter")
    tol = params["tol"]
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not math.isfinite(tol)
        or tol < 0.0
    ):
        raise InvalidParameterError("tol 必须是非负有限数值")
    require_boolean(params, "standardize")


def _validate_dbscan_params(params: dict[str, JSONValue]) -> None:
    eps = params["eps"]
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(eps)
        or float(eps) <= 0.0
    ):
        raise InvalidParameterError("eps 必须是正的有限数值")
    require_positive_integer(params, "min_samples")
    require_boolean(params, "standardize")
