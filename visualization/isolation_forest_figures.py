"""Generate six publication-style PNG figures for the scratch Isolation Forest models."""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 300,
}
matplotlib.rcParams.update(PLOT_STYLE)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.isolation_forest import IsolationForestScratch
from Models.isolation_forest_optimized import OptimizedIsolationForestScratch


FIGURE_FILENAMES = (
    "01_isolation_mechanism.png",
    "02_score_landscape.png",
    "03_score_distribution.png",
    "04_roc_pr_curves.png",
    "05_parameter_sensitivity.png",
    "06_benchmark_comparison.png",
)

PALETTES = {
    "mechanism": ("#315A7D", "#E49B43", "#AAB7C4", "#B94055"),
    "score_landscape": ("#173F5F", "#3CAEA3", "#F6D55C", "#ED553B"),
    "score_distribution": ("#236B8E", "#D87959", "#9FB7C8", "#6F334D"),
    "ranking_curves": ("#514A9D", "#D59B2D", "#B7B7C8", "#2E294E"),
    "parameter_sensitivity": ("#1B7837", "#A6D96A", "#762A83", "#E08214"),
    "benchmark": ("#31572C", "#BC4749", "#8FA6B8", "#6B5B73"),
}


def load_npz_dataset(path):
    """Load an NPZ anomaly dataset; NumPy is used only for file I/O."""
    with np.load(Path(path)) as archive:
        X = archive["X"].astype(float).tolist()
        y = archive["y"].astype(int).tolist()
    return X, y


def stratified_split(X, y, seed=0, train_fraction=0.6, validation_fraction=0.2):
    """Create deterministic train/validation/test partitions for each class."""
    generator = random.Random(seed)
    partitions = {"train": [], "validation": [], "test": []}
    for label in sorted(set(y)):
        indices = [index for index, value in enumerate(y) if value == label]
        generator.shuffle(indices)
        train_count = max(1, round(len(indices) * train_fraction))
        validation_count = max(1, round(len(indices) * validation_fraction))
        if train_count + validation_count >= len(indices):
            validation_count = 1
            train_count = max(1, len(indices) - 2)
        partitions["train"].extend(indices[:train_count])
        partitions["validation"].extend(
            indices[train_count : train_count + validation_count]
        )
        partitions["test"].extend(indices[train_count + validation_count :])

    report = {}
    for name, indices in partitions.items():
        indices.sort()
        report[f"{name}_indices"] = indices
        report[f"{name}_X"] = [list(X[index]) for index in indices]
        report[f"{name}_y"] = [int(y[index]) for index in indices]
    return report


def fit_standardizer(X):
    sample_count = len(X)
    feature_count = len(X[0])
    means = [
        sum(row[feature] for row in X) / sample_count
        for feature in range(feature_count)
    ]
    scales = []
    for feature, mean in enumerate(means):
        variance = sum((row[feature] - mean) ** 2 for row in X) / sample_count
        scale = math.sqrt(variance)
        scales.append(scale if scale > 0.0 else 1.0)
    return means, scales


def transform_with_standardizer(X, means, scales):
    return [
        [
            (value - means[feature]) / scales[feature]
            for feature, value in enumerate(row)
        ]
        for row in X
    ]


def binary_ranking_curves(y_true, scores):
    """Compute tie-aware ROC and precision-recall curves without ML packages."""
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("both classes are required")

    ordered = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    groups = []
    for score, label in ordered:
        if not groups or score != groups[-1][0]:
            groups.append([score, 0, 0])
        groups[-1][1 if label == 1 else 2] += 1

    true_positive = 0
    false_positive = 0
    roc_fpr = [0.0]
    roc_tpr = [0.0]
    pr_recall = [0.0]
    pr_precision = [1.0]
    average_precision = 0.0
    previous_recall = 0.0

    for _, positive_count, negative_count in groups:
        true_positive += positive_count
        false_positive += negative_count
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        roc_fpr.append(false_positive / negatives)
        roc_tpr.append(recall)
        pr_recall.append(recall)
        pr_precision.append(precision)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall

    roc_auc = sum(
        (roc_fpr[index] - roc_fpr[index - 1])
        * (roc_tpr[index] + roc_tpr[index - 1])
        / 2.0
        for index in range(1, len(roc_fpr))
    )
    return {
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "pr_recall": pr_recall,
        "pr_precision": pr_precision,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }


def binary_classification_metrics(y_true, predictions):
    true_positive = sum(
        label == 1 and prediction == -1
        for label, prediction in zip(y_true, predictions)
    )
    false_positive = sum(
        label == 0 and prediction == -1
        for label, prediction in zip(y_true, predictions)
    )
    false_negative = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(y_true, predictions)
    )
    true_negative = len(y_true) - true_positive - false_positive - false_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (true_positive + true_negative) / len(y_true),
    }


def generate_oblique_data(n_inliers=180, n_outliers=20, seed=0):
    """Generate a diagonal normal manifold with off-manifold anomalies."""
    generator = random.Random(seed)
    X = []
    y = []
    for _ in range(n_inliers):
        coordinate = generator.uniform(-2.6, 2.6)
        perpendicular_noise = generator.gauss(0.0, 0.11)
        X.append(
            [coordinate + perpendicular_noise, coordinate - perpendicular_noise]
        )
        y.append(0)
    for index in range(n_outliers):
        side = -1.0 if index % 2 == 0 else 1.0
        coordinate = side * generator.uniform(1.0, 2.45)
        X.append(
            [coordinate, -coordinate + generator.gauss(0.0, 0.12)]
        )
        y.append(1)
    return X, y


def _stratified_subset(X, y, limit, seed):
    if limit is None or limit >= len(X):
        return X, y, list(range(len(X)))
    generator = random.Random(seed)
    anomaly_count = max(3, round(limit * sum(y) / len(y)))
    normal_count = limit - anomaly_count
    normal_indices = [index for index, label in enumerate(y) if label == 0]
    anomaly_indices = [index for index, label in enumerate(y) if label == 1]
    selected = generator.sample(normal_indices, normal_count)
    selected += generator.sample(anomaly_indices, min(anomaly_count, len(anomaly_indices)))
    selected.sort()
    return (
        [X[index] for index in selected],
        [y[index] for index in selected],
        selected,
    )


def _prepare_dataset(path, seed, limit=None):
    X, y = load_npz_dataset(path)
    X, y, record_ids = _stratified_subset(X, y, limit, seed)
    split = stratified_split(X, y, seed=seed)
    for name in ("train", "validation", "test"):
        split[f"{name}_record_ids"] = [
            record_ids[index] for index in split[f"{name}_indices"]
        ]
    means, scales = fit_standardizer(split["train_X"])
    for name in ("train", "validation", "test"):
        split[f"{name}_X"] = transform_with_standardizer(
            split[f"{name}_X"], means, scales
        )
    split["means"] = means
    split["scales"] = scales
    split["sample_count"] = len(X)
    split["feature_count"] = len(X[0])
    split["anomaly_count"] = sum(y)
    return split


def _prevalence(y):
    return sum(y) / len(y)


def _fit_timed(model, X):
    start = time.perf_counter()
    model.fit(X)
    return time.perf_counter() - start


def _timed_candidate_sweep(
    candidates,
    build_model,
    train_X,
    validation_X,
    validation_y,
    seed,
    repeats,
):
    """Measure candidates repeatedly while rotating their execution order."""
    warmup = build_model(candidates[0], seed + 10_000)
    warmup.fit(train_X)
    measurements = [
        {"fit_times": [], "average_precisions": [], "roc_aucs": []}
        for _ in candidates
    ]
    indices = list(range(len(candidates)))
    for repeat in range(repeats):
        shift = repeat % len(indices)
        for index in indices[shift:] + indices[:shift]:
            model = build_model(candidates[index], seed + repeat)
            fit_time = _fit_timed(model, train_X)
            ranking = binary_ranking_curves(
                validation_y,
                model.score_samples(validation_X),
            )
            measurements[index]["fit_times"].append(fit_time)
            measurements[index]["average_precisions"].append(
                ranking["average_precision"]
            )
            measurements[index]["roc_aucs"].append(ranking["roc_auc"])

    rows = []
    for candidate, values in zip(candidates, measurements):
        median, q1, q3 = _median_iqr(values["fit_times"])
        rows.append(
            {
                **candidate,
                "average_precision": statistics.mean(
                    values["average_precisions"]
                ),
                "average_precision_std": statistics.stdev(
                    values["average_precisions"]
                )
                if repeats > 1
                else 0.0,
                "roc_auc": statistics.mean(values["roc_aucs"]),
                "roc_auc_std": statistics.stdev(values["roc_aucs"])
                if repeats > 1
                else 0.0,
                "fit_time_median": median,
                "fit_time_q1": q1,
                "fit_time_q3": q3,
            }
        )
    return rows


def _evaluate(model, X, y):
    start = time.perf_counter()
    scores = model.score_samples(X)
    predictions = [-1 if score > model.threshold_ else 1 for score in scores]
    predict_time = time.perf_counter() - start
    ranking = binary_ranking_curves(y, scores)
    thresholded = binary_classification_metrics(y, predictions)
    return {
        "scores": scores,
        "predictions": predictions,
        "predict_time": predict_time,
        **ranking,
        **thresholded,
    }


def _parameter_search(split, seed, quick):
    max_samples_values = [32, 64] if quick else [64, 128, 256]
    max_features_values = [0.5, 1.0] if quick else [0.35, 0.65, 1.0]
    attempt_values = [1, 3] if quick else [1, 3, 5]
    estimator_count = 12 if quick else 60
    timing_repeats = 2 if quick else 3
    contamination = _prevalence(split["train_y"])
    baseline_candidates = [
        {"max_samples": max_samples} for max_samples in max_samples_values
    ]
    baseline_grid = _timed_candidate_sweep(
        baseline_candidates,
        lambda row, model_seed: IsolationForestScratch(
            n_estimators=estimator_count,
            max_samples=row["max_samples"],
            contamination=contamination,
            random_state=model_seed,
        ),
        split["train_X"],
        split["validation_X"],
        split["validation_y"],
        seed,
        timing_repeats,
    )
    baseline_selected = max(
        baseline_grid,
        key=lambda row: (row["average_precision"], -row["max_samples"]),
    )
    grid_candidates = [
        {"max_samples": max_samples, "max_features": max_features}
        for max_samples in max_samples_values
        for max_features in max_features_values
    ]
    grid = _timed_candidate_sweep(
        grid_candidates,
        lambda row, model_seed: OptimizedIsolationForestScratch(
            n_estimators=estimator_count,
            max_samples=row["max_samples"],
            contamination=contamination,
            max_features=row["max_features"],
            n_split_candidates=3,
            random_state=model_seed,
        ),
        split["train_X"],
        split["validation_X"],
        split["validation_y"],
        seed,
        timing_repeats,
    )
    selected_grid = max(
        grid,
        key=lambda row: (
            row["average_precision"],
            -row["max_samples"],
            -row["max_features"],
        ),
    )
    attempt_candidates = [
        {"n_split_candidates": attempt_count}
        for attempt_count in attempt_values
    ]
    attempts = _timed_candidate_sweep(
        attempt_candidates,
        lambda row, model_seed: OptimizedIsolationForestScratch(
            n_estimators=estimator_count,
            max_samples=selected_grid["max_samples"],
            contamination=contamination,
            max_features=selected_grid["max_features"],
            n_split_candidates=row["n_split_candidates"],
            random_state=model_seed,
        ),
        split["train_X"],
        split["validation_X"],
        split["validation_y"],
        seed,
        timing_repeats,
    )
    selected_attempt = max(
        attempts,
        key=lambda row: (
            row["average_precision"],
            -row["n_split_candidates"],
        ),
    )
    return {
        "selection_split": "validation",
        "test_used_for_tuning": False,
        "selection_rule": (
            "maximize validation AUPRC; deterministic complexity tie-break"
        ),
        "timing_protocol": (
            "warm-up; rotated configuration order; median and IQR"
        ),
        "selection_record_indices": list(split["validation_record_ids"]),
        "n_estimators": estimator_count,
        "baseline_grid": baseline_grid,
        "baseline_selected": {
            "max_samples": baseline_selected["max_samples"],
            "n_estimators": estimator_count,
        },
        "grid": grid,
        "attempts": attempts,
        "max_samples_values": max_samples_values,
        "max_features_values": max_features_values,
        "selected": {
            "max_samples": selected_grid["max_samples"],
            "max_features": selected_grid["max_features"],
            "n_split_candidates": selected_attempt["n_split_candidates"],
            "n_estimators": estimator_count,
        },
    }


def _mechanism_report(seed, quick):
    X, y = generate_oblique_data(
        n_inliers=100 if quick else 180,
        n_outliers=12 if quick else 20,
        seed=seed,
    )
    contamination = _prevalence(y)
    tree_count = 30 if quick else 100
    sample_count = min(len(X), 64 if quick else 128)
    basic = IsolationForestScratch(
        n_estimators=tree_count,
        max_samples=sample_count,
        contamination=contamination,
        random_state=seed,
    ).fit(X)
    optimized = OptimizedIsolationForestScratch(
        n_estimators=tree_count,
        max_samples=sample_count,
        contamination=contamination,
        max_features=1.0,
        n_split_candidates=3,
        random_state=seed,
    ).fit(X)

    resolution = 35 if quick else 65
    x_min = min(row[0] for row in X) - 0.45
    x_max = max(row[0] for row in X) + 0.45
    y_min = min(row[1] for row in X) - 0.45
    y_max = max(row[1] for row in X) + 0.45
    x_values = np.linspace(x_min, x_max, resolution).tolist()
    y_values = np.linspace(y_min, y_max, resolution).tolist()
    grid = [[x_value, y_value] for y_value in y_values for x_value in x_values]
    basic_grid = basic.score_samples(grid)
    optimized_grid = optimized.score_samples(grid)
    return {
        "X": X,
        "y": y,
        "basic_model": basic,
        "optimized_model": optimized,
        "basic_predictions": basic.predict(X),
        "optimized_predictions": optimized.predict(X),
        "x_values": x_values,
        "y_values": y_values,
        "basic_grid": basic_grid,
        "optimized_grid": optimized_grid,
        "resolution": resolution,
    }


def _cardio_reports(split, tuning, seed, quick):
    settings = tuning["selected"]
    estimator_count = tuning["n_estimators"]
    contamination = _prevalence(split["train_y"])
    basic = IsolationForestScratch(
        n_estimators=estimator_count,
        max_samples=tuning["baseline_selected"]["max_samples"],
        contamination=contamination,
        random_state=seed,
    )
    optimized = OptimizedIsolationForestScratch(
        n_estimators=estimator_count,
        max_samples=settings["max_samples"],
        contamination=contamination,
        max_features=settings["max_features"],
        n_split_candidates=settings["n_split_candidates"],
        random_state=seed,
    )
    basic_fit_time = _fit_timed(basic, split["train_X"])
    optimized_fit_time = _fit_timed(optimized, split["train_X"])
    basic_result = _evaluate(basic, split["test_X"], split["test_y"])
    optimized_result = _evaluate(optimized, split["test_X"], split["test_y"])
    basic_result["fit_time"] = basic_fit_time
    optimized_result["fit_time"] = optimized_fit_time
    basic_result["model_threshold"] = float(basic.threshold_)
    optimized_result["model_threshold"] = float(optimized.threshold_)
    shared = {
        "dataset": "Cardio",
        "sample_count": split["sample_count"],
        "feature_count": split["feature_count"],
        "n_estimators": estimator_count,
        "test_y": split["test_y"],
        "train_prevalence": contamination,
        "threshold_calibration": "oracle training prevalence",
        "selection_split": "validation",
        "test_used_for_tuning": False,
        "methods": {
            "Isolation Forest": basic_result,
            "Extended IF": optimized_result,
        },
    }
    return shared


def _mean_std(values):
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _median_iqr(values):
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), quartiles[0], quartiles[2]


def _benchmark_dataset(path, name, tuning, seed, quick, limit):
    seeds = [seed, seed + 17] if quick else [seed, seed + 17, seed + 31]
    metrics = {
        "Isolation Forest": [],
        "Extended IF": [],
    }
    sample_count = feature_count = anomaly_count = None
    estimator_count = tuning["n_estimators"]
    split = _prepare_dataset(path, seed, limit=limit)
    sample_count = split["sample_count"]
    feature_count = split["feature_count"]
    anomaly_count = split["anomaly_count"]
    contamination = _prevalence(split["train_y"])
    warm_count = min(32, len(split["train_X"]))
    IsolationForestScratch(
        n_estimators=2,
        max_samples=warm_count,
        contamination=contamination,
        random_state=seed,
    ).fit(split["train_X"][:warm_count])
    OptimizedIsolationForestScratch(
        n_estimators=2,
        max_samples=warm_count,
        contamination=contamination,
        max_features=tuning["selected"]["max_features"],
        n_split_candidates=tuning["selected"]["n_split_candidates"],
        random_state=seed,
    ).fit(split["train_X"][:warm_count])

    for run_index, run_seed in enumerate(seeds):
        models = {
            "Isolation Forest": IsolationForestScratch(
                n_estimators=estimator_count,
                max_samples=tuning["baseline_selected"]["max_samples"],
                contamination=contamination,
                random_state=run_seed,
            ),
            "Extended IF": OptimizedIsolationForestScratch(
                n_estimators=estimator_count,
                max_samples=tuning["selected"]["max_samples"],
                contamination=contamination,
                max_features=tuning["selected"]["max_features"],
                n_split_candidates=tuning["selected"]["n_split_candidates"],
                random_state=run_seed,
            ),
        }
        method_order = list(models)
        if run_index % 2 == 1:
            method_order.reverse()
        for method in method_order:
            model = models[method]
            fit_time = _fit_timed(model, split["train_X"])
            result = _evaluate(model, split["test_X"], split["test_y"])
            result["fit_time"] = fit_time
            metrics[method].append(result)

    summary = {}
    fields = (
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "fit_time",
        "predict_time",
    )
    for method, runs in metrics.items():
        summary[method] = {"runs": runs}
        for field in fields:
            mean, standard_deviation = _mean_std([run[field] for run in runs])
            summary[method][f"{field}_mean"] = mean
            summary[method][f"{field}_std"] = standard_deviation
            if field in ("fit_time", "predict_time"):
                median, q1, q3 = _median_iqr([run[field] for run in runs])
                summary[method][f"{field}_median"] = median
                summary[method][f"{field}_q1"] = q1
                summary[method][f"{field}_q3"] = q3
    return {
        "name": name,
        "sample_count": sample_count,
        "feature_count": feature_count,
        "anomaly_count": anomaly_count,
        "n_estimators": estimator_count,
        "split_seed": seed,
        "test_indices": list(split["test_record_ids"]),
        "seeds": seeds,
        "methods": summary,
        "threshold_calibration": "oracle training prevalence",
    }


def run_experiment(cardio_path, mammography_path, seed=13, quick=False):
    """Run the full evidence pipeline and return data used by all six figures."""
    cardio_limit = 220 if quick else None
    mammography_limit = 320 if quick else None
    cardio_split = _prepare_dataset(cardio_path, seed, limit=cardio_limit)
    tuning = _parameter_search(cardio_split, seed, quick)
    mechanism = _mechanism_report(seed, quick)
    cardio = _cardio_reports(cardio_split, tuning, seed, quick)
    cardio_benchmark = _benchmark_dataset(
        cardio_path,
        "Cardio",
        tuning,
        seed,
        quick,
        cardio_limit,
    )
    mammography_benchmark = _benchmark_dataset(
        mammography_path,
        "Mammography",
        tuning,
        seed,
        quick,
        mammography_limit,
    )
    tuning_test_overlap = set(tuning["selection_record_indices"]) & set(
        cardio_benchmark["test_indices"]
    )
    benchmark = {
        "datasets": {
            "Cardio": cardio_benchmark,
            "Mammography": mammography_benchmark,
        },
        "parameters_transferred_from": "Cardio validation",
        "test_used_for_tuning": False,
        "tuning_test_overlap_count": len(tuning_test_overlap),
        "timing_protocol": (
            "warm-up; alternating fit order; one scoring pass; "
            "median and IQR across model seeds"
        ),
    }
    return {
        "figure_contract": {
            "core_conclusion": (
                "Test whether sparse EIF hyperplanes reduce axis bias while "
                "preserving stable anomaly ranking across imbalanced datasets."
            ),
            "archetype": "quantitative grid with a mechanism-led opening",
            "backend": "Python",
            "output": "six independent 300 dpi PNG files",
        },
        "figures": {
            "mechanism": mechanism,
            "score_landscape": mechanism,
            "score_distribution": cardio,
            "ranking_curves": cardio,
            "parameter_sensitivity": tuning,
            "benchmark": benchmark,
        },
    }


def _scatter_oblique(ax, data, palette):
    X, y = data["X"], data["y"]
    normal = [row for row, label in zip(X, y) if label == 0]
    anomaly = [row for row, label in zip(X, y) if label == 1]
    ax.scatter(
        [row[0] for row in normal],
        [row[1] for row in normal],
        s=15,
        color=palette[2],
        alpha=0.78,
        label="Normal",
        edgecolors="none",
    )
    ax.scatter(
        [row[0] for row in anomaly],
        [row[1] for row in anomaly],
        s=31,
        facecolors="none",
        edgecolors=palette[3],
        linewidths=1.2,
        label="Anomaly",
    )


def _draw_axis_tree(ax, node, bounds, color, depth=0, max_depth=1):
    if node.is_leaf or depth >= max_depth:
        return
    x_min, x_max, y_min, y_max = bounds
    alpha = 0.9 - 0.2 * depth
    if node.feature == 0:
        ax.plot([node.split, node.split], [y_min, y_max], color=color, lw=1.2, alpha=alpha)
        _draw_axis_tree(ax, node.left, (x_min, node.split, y_min, y_max), color, depth + 1, max_depth)
        _draw_axis_tree(ax, node.right, (node.split, x_max, y_min, y_max), color, depth + 1, max_depth)
    elif node.feature == 1:
        ax.plot([x_min, x_max], [node.split, node.split], color=color, lw=1.2, alpha=alpha)
        _draw_axis_tree(ax, node.left, (x_min, x_max, y_min, node.split), color, depth + 1, max_depth)
        _draw_axis_tree(ax, node.right, (x_min, x_max, node.split, y_max), color, depth + 1, max_depth)


def _draw_hyperplanes(ax, node, bounds, color, depth=0, max_depth=1):
    if node.is_leaf or depth >= max_depth:
        return
    weights = dict(node.normal)
    weight_x = weights.get(0, 0.0)
    weight_y = weights.get(1, 0.0)
    x_min, x_max, y_min, y_max = bounds
    alpha = 0.9 - 0.2 * depth
    if abs(weight_y) > 1e-12:
        xs = [x_min, x_max]
        ys = [
            (node.offset - weight_x * x_value) / weight_y
            for x_value in xs
        ]
        ax.plot(xs, ys, color=color, lw=1.2, alpha=alpha)
    elif abs(weight_x) > 1e-12:
        x_value = node.offset / weight_x
        ax.plot([x_value, x_value], [y_min, y_max], color=color, lw=1.2, alpha=alpha)
    _draw_hyperplanes(ax, node.left, bounds, color, depth + 1, max_depth)
    _draw_hyperplanes(ax, node.right, bounds, color, depth + 1, max_depth)


def _save(fig, output, filename):
    path = output / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_mechanism(report, output):
    palette = PALETTES["mechanism"]
    X = report["X"]
    bounds = (
        min(row[0] for row in X) - 0.3,
        max(row[0] for row in X) + 0.3,
        min(row[1] for row in X) - 0.3,
        max(row[1] for row in X) + 0.3,
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.0), constrained_layout=True)
    _scatter_oblique(axes[0], report, palette)
    _draw_axis_tree(
        axes[0], report["basic_model"].estimators_[0], bounds, palette[0]
    )
    axes[0].set_title("Root split: axis-aligned tree")
    _scatter_oblique(axes[1], report, palette)
    _draw_hyperplanes(
        axes[1], report["optimized_model"].estimators_[0], bounds, palette[1]
    )
    axes[1].set_title("Root split: EIF hyperplane tree")
    for index, ax in enumerate(axes):
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="Feature 1", ylabel="Feature 2")
        ax.text(-0.04, 1.02, chr(97 + index), transform=ax.transAxes, va="bottom", fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2)
    fig.suptitle("A fixed first-tree root split illustrates axis and oblique isolation geometry")
    return _save(fig, output, FIGURE_FILENAMES[0])


def _plot_score_landscape(report, output):
    palette = PALETTES["score_landscape"]
    cmap = LinearSegmentedColormap.from_list(
        "isolation_score",
        ["#F5F7F4", palette[1], palette[0]],
    )
    resolution = report["resolution"]
    X_grid, Y_grid = np.meshgrid(report["x_values"], report["y_values"])
    score_arrays = [
        np.asarray(report["basic_grid"]).reshape(resolution, resolution),
        np.asarray(report["optimized_grid"]).reshape(resolution, resolution),
    ]
    low = min(float(array.min()) for array in score_arrays)
    high = max(float(array.max()) for array in score_arrays)
    levels = np.linspace(low, high, 18)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.05), constrained_layout=True)
    models = [report["basic_model"], report["optimized_model"]]
    titles = ["Isolation Forest", "Extended Isolation Forest"]
    contour = None
    for index, (ax, scores, model, title) in enumerate(zip(axes, score_arrays, models, titles)):
        contour = ax.contourf(X_grid, Y_grid, scores, levels=levels, cmap=cmap, extend="both")
        ax.contour(
            X_grid,
            Y_grid,
            scores,
            levels=[model.threshold_],
            colors=[palette[3]],
            linewidths=1.4,
        )
        _scatter_oblique(ax, report, palette)
        ax.set(title=title, xlabel="Feature 1", ylabel="Feature 2")
        ax.text(-0.04, 1.02, chr(97 + index), transform=ax.transAxes, va="bottom", fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(contour, ax=axes, shrink=0.84, pad=0.02)
    colorbar.set_label("Anomaly score (higher = more anomalous)")
    fig.suptitle("Hyperplane isolation reshapes anomaly scores around an oblique normal manifold")
    return _save(fig, output, FIGURE_FILENAMES[1])


def _plot_score_distribution(report, output):
    palette = PALETTES["score_distribution"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True, sharey=True)
    y = report["test_y"]
    for index, (method, result) in enumerate(report["methods"].items()):
        normal_scores = [score for score, label in zip(result["scores"], y) if label == 0]
        anomaly_scores = [score for score, label in zip(result["scores"], y) if label == 1]
        all_scores = normal_scores + anomaly_scores
        bins = np.linspace(min(all_scores), max(all_scores), 18)
        axes[index].hist(normal_scores, bins=bins, density=True, color=palette[2], alpha=0.72, label=f"Normal (n={len(normal_scores)})")
        axes[index].hist(anomaly_scores, bins=bins, density=True, histtype="step", linewidth=1.8, color=palette[1], label=f"Anomaly (n={len(anomaly_scores)})")
        axes[index].axvline(
            result["model_threshold"],
            color=palette[3],
            ls="--",
            lw=1.2,
            label="Training-calibrated threshold",
        )
        axes[index].set(title=method, xlabel="Anomaly score")
        axes[index].text(
            0.97,
            0.96,
            f"AP = {result['average_precision']:.3f}\nAUROC = {result['roc_auc']:.3f}",
            transform=axes[index].transAxes,
            ha="right",
            va="top",
        )
        axes[index].legend(loc="upper left")
        axes[index].text(-0.04, 1.02, chr(97 + index), transform=axes[index].transAxes, va="bottom", fontweight="bold")
    axes[0].set_ylabel("Density")
    fig.suptitle("Held-out Cardio scores; dashed thresholds use oracle training prevalence")
    return _save(fig, output, FIGURE_FILENAMES[2])


def _plot_ranking_curves(report, output):
    palette = PALETTES["ranking_curves"]
    method_colors = {"Isolation Forest": palette[0], "Extended IF": palette[1]}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    for method, result in report["methods"].items():
        axes[0].plot(result["roc_fpr"], result["roc_tpr"], lw=1.8, color=method_colors[method], label=f"{method} (AUC={result['roc_auc']:.3f})")
        axes[1].step(result["pr_recall"], result["pr_precision"], where="post", lw=1.8, color=method_colors[method], label=f"{method} (AP={result['average_precision']:.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1.0, color=palette[2], label="Chance")
    prevalence = sum(report["test_y"]) / len(report["test_y"])
    axes[1].axhline(prevalence, ls="--", lw=1.0, color=palette[2], label=f"Prevalence={prevalence:.3f}")
    axes[0].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    for index, ax in enumerate(axes):
        ax.legend(loc="lower right" if index == 0 else "upper right")
        ax.text(-0.04, 1.02, chr(97 + index), transform=ax.transAxes, va="bottom", fontweight="bold")
    fig.suptitle("Threshold-free ranking performance on the untouched Cardio test set")
    return _save(fig, output, FIGURE_FILENAMES[3])


def _heatmap(ax, matrix, x_labels, y_labels, cmap, title, formatter):
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(x_labels)), [str(value) for value in x_labels])
    ax.set_yticks(range(len(y_labels)), [str(value) for value in y_labels])
    ax.set(xlabel="max_features", ylabel="max_samples", title=title)
    midpoint = (float(np.nanmin(matrix)) + float(np.nanmax(matrix))) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, formatter(value), ha="center", va="center", color="white" if value > midpoint else "black", fontsize=8)
    return image


def _plot_parameter_sensitivity(report, output):
    palette = PALETTES["parameter_sensitivity"]
    sample_values = report["max_samples_values"]
    feature_values = report["max_features_values"]
    ap_matrix = np.zeros((len(sample_values), len(feature_values)))
    time_matrix = np.zeros_like(ap_matrix)
    for row in report["grid"]:
        y_index = sample_values.index(row["max_samples"])
        x_index = feature_values.index(row["max_features"])
        ap_matrix[y_index, x_index] = row["average_precision"]
        time_matrix[y_index, x_index] = row["fit_time_median"]
    ap_cmap = LinearSegmentedColormap.from_list("ap_map", ["#F4F8EC", palette[1], palette[0]])
    time_cmap = LinearSegmentedColormap.from_list("time_map", ["#FAF3E8", "#D9A066", palette[2]])
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.65), constrained_layout=True)
    _heatmap(axes[0], ap_matrix, feature_values, sample_values, ap_cmap, "Validation AUPRC", lambda value: f"{value:.3f}")
    _heatmap(axes[1], time_matrix, feature_values, sample_values, time_cmap, "Median fit time (s)", lambda value: f"{value:.3f}")
    selected = report["selected"]
    selected_xy = (
        feature_values.index(selected["max_features"]),
        sample_values.index(selected["max_samples"]),
    )
    for ax in axes[:2]:
        ax.add_patch(Rectangle((selected_xy[0] - 0.5, selected_xy[1] - 0.5), 1, 1, fill=False, edgecolor=palette[3], linewidth=2.0))
    attempts = report["attempts"]
    x = [row["n_split_candidates"] for row in attempts]
    ap = [row["average_precision"] for row in attempts]
    timing = [row["fit_time_median"] for row in attempts]
    timing_error = [
        [row["fit_time_median"] - row["fit_time_q1"] for row in attempts],
        [row["fit_time_q3"] - row["fit_time_median"] for row in attempts],
    ]
    ap_line, = axes[2].plot(
        x, ap, marker="o", color=palette[0], lw=1.7, label="AUPRC"
    )
    axes[2].set(xlabel="Maximum split attempts", ylabel="Validation AUPRC", title="Invalid-split retry trade-off")
    twin = axes[2].twinx()
    timing_handle = twin.errorbar(
        x,
        timing,
        yerr=timing_error,
        marker="s",
        capsize=2.5,
        color=palette[3],
        lw=1.5,
        label="Fit time, median (IQR)",
    )
    twin.set_ylabel("Median fit time (s)")
    axes[2].legend(
        [ap_line, timing_handle],
        ["AUPRC", "Fit time, median (IQR)"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
    )
    for index, ax in enumerate(axes):
        ax.text(-0.04, 1.02, chr(97 + index), transform=ax.transAxes, va="bottom", fontweight="bold")
    fig.suptitle(
        f"Validation sweeps use {report['n_estimators']} trees and repeated model seeds; time shows median (IQR where visible)"
    )
    return _save(fig, output, FIGURE_FILENAMES[4])


def _plot_benchmark(report, output):
    palette = PALETTES["benchmark"]
    methods = ["Isolation Forest", "Extended IF"]
    colors = [palette[0], palette[1]]
    datasets = list(report["datasets"])
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), constrained_layout=True)
    panels = [
        ("average_precision", "AUPRC"),
        ("f1", "F1 at oracle training-prevalence threshold"),
    ]
    positions = np.arange(len(datasets))
    width = 0.34
    for panel_index, (field, title) in enumerate(panels):
        ax = axes[panel_index]
        for method_index, method in enumerate(methods):
            means = [report["datasets"][dataset]["methods"][method][f"{field}_mean"] for dataset in datasets]
            errors = [report["datasets"][dataset]["methods"][method][f"{field}_std"] for dataset in datasets]
            ax.bar(positions + (method_index - 0.5) * width, means, width, yerr=errors, capsize=3, color=colors[method_index], alpha=0.88, label=method)
        ax.set_xticks(positions, datasets)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.text(-0.04, 1.02, chr(97 + panel_index), transform=ax.transAxes, va="bottom", fontweight="bold")

    timing_groups = [
        (dataset, field)
        for dataset in datasets
        for field in ("fit_time", "predict_time")
    ]
    timing_positions = np.arange(len(timing_groups))
    for method_index, method in enumerate(methods):
        medians = []
        lower_errors = []
        upper_errors = []
        for dataset, field in timing_groups:
            result = report["datasets"][dataset]["methods"][method]
            median = result[f"{field}_median"]
            medians.append(median)
            lower_errors.append(median - result[f"{field}_q1"])
            upper_errors.append(result[f"{field}_q3"] - median)
        axes[2].bar(
            timing_positions + (method_index - 0.5) * width,
            medians,
            width,
            yerr=np.asarray([lower_errors, upper_errors]),
            capsize=3,
            color=colors[method_index],
            alpha=0.88,
            label=method,
        )
    axes[2].set_xticks(
        timing_positions,
        [f"{dataset}\n{'fit' if field == 'fit_time' else 'score'}" for dataset, field in timing_groups],
    )
    axes[2].set_title("Fit and single-pass score time (s; median, IQR)")
    axes[2].set_yscale("log")
    axes[2].text(-0.04, 1.02, "c", transform=axes[2].transAxes, va="bottom", fontweight="bold")
    axes[0].set_ylabel("Mean ± SD")
    axes[0].legend(loc="upper left")
    run_count = len(next(iter(report["datasets"].values()))["seeds"])
    fig.suptitle(
        f"Cross-dataset benchmark: quality mean ± SD; time median (IQR) across {run_count} model seeds"
    )
    return _save(fig, output, FIGURE_FILENAMES[5])


def render_all(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    figures = report["figures"]
    return [
        _plot_mechanism(figures["mechanism"], output),
        _plot_score_landscape(figures["score_landscape"], output),
        _plot_score_distribution(figures["score_distribution"], output),
        _plot_ranking_curves(figures["ranking_curves"], output),
        _plot_parameter_sensitivity(figures["parameter_sensitivity"], output),
        _plot_benchmark(figures["benchmark"], output),
    ]


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cardio",
        type=Path,
        default=PROJECT_ROOT / "data" / "anomaly" / "6_cardio.npz",
    )
    parser.add_argument(
        "--mammography",
        type=Path,
        default=PROJECT_ROOT / "data" / "anomaly" / "23_mammography.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "figures" / "isolation_forest",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--quick", action="store_true")
    return parser


def main():
    arguments = build_argument_parser().parse_args()
    report = run_experiment(
        cardio_path=arguments.cardio,
        mammography_path=arguments.mammography,
        seed=arguments.seed,
        quick=arguments.quick,
    )
    paths = render_all(report, arguments.output)
    print(f"Generated {len(paths)} PNG figures in {arguments.output}")


if __name__ == "__main__":
    main()
