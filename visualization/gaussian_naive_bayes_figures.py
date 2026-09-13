"""为手写高斯朴素贝叶斯生成六张可复现实验图。"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.gaussian_naive_bayes import GaussianNaiveBayesScratch
from Models.gaussian_naive_bayes_optimized import OptimizedGaussianNaiveBayesScratch


FEATURE_NAMES = [
    "Mean radius", "Mean texture", "Mean perimeter", "Mean area",
    "Mean smoothness", "Mean compactness", "Mean concavity",
    "Mean concave points", "Mean symmetry", "Mean fractal dimension",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE",
    "Smoothness SE", "Compactness SE", "Concavity SE",
    "Concave points SE", "Symmetry SE", "Fractal dimension SE",
    "Worst radius", "Worst texture", "Worst perimeter", "Worst area",
    "Worst smoothness", "Worst compactness", "Worst concavity",
    "Worst concave points", "Worst symmetry", "Worst fractal dimension",
]

FIGURE_PALETTES = {
    "distribution": ["#2A9D8F", "#E76F51", "#264653", "#E9C46A"],
    "surface": ["#5E60CE", "#FFB703", "#E8E6F3", "#FFF1C1"],
    "assumptions": ["#3D5A80", "#EE6C4D", "#DCEAF4", "#F4D6CC"],
    "benchmark": ["#A7A9AC", "#7B2CBF", "#2A9D8F", "#252525"],
    "diagnostics": ["#003049", "#D62828", "#F77F00", "#669BBC"],
    "optimization": ["#F6BD60", "#84A59D", "#9C89B8", "#F28482"],
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_wdbc(path):
    features = []
    targets = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if row:
                targets.append(1 if row[1] == "M" else 0)
                features.append([float(value) for value in row[2:]])
    return features, targets


def stratified_split(features, targets, test_size=0.2, seed=42):
    generator = random.Random(seed)
    train_indices = []
    test_indices = []
    for label in sorted(set(targets)):
        indices = [index for index, target in enumerate(targets) if target == label]
        generator.shuffle(indices)
        test_count = max(1, round(len(indices) * test_size))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    generator.shuffle(train_indices)
    generator.shuffle(test_indices)
    return (
        [features[index] for index in train_indices],
        [features[index] for index in test_indices],
        [targets[index] for index in train_indices],
        [targets[index] for index in test_indices],
    )


def stratified_k_folds(targets, n_splits=5, seed=42):
    generator = random.Random(seed)
    folds = [[] for _ in range(n_splits)]
    for label in sorted(set(targets)):
        indices = [index for index, target in enumerate(targets) if target == label]
        generator.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % n_splits].append(index)
    return folds


def classification_metrics(y_true, y_pred):
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(y_true, y_pred))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(y_true, y_pred))
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
    accuracy = (tn + tp) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": [[tn, fp], [fn, tp]],
    }


def _score_groups(y_true, scores):
    ordered = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    groups = []
    for score, target in ordered:
        if not groups or score != groups[-1][0]:
            groups.append([score, 0, 0])
        groups[-1][1 if target == 1 else 2] += 1
    return groups


def roc_curve(y_true, scores):
    positives = sum(y_true)
    negatives = len(y_true) - positives
    true_positive = 0
    false_positive = 0
    fpr = [0.0]
    tpr = [0.0]
    for _, group_positive, group_negative in _score_groups(y_true, scores):
        true_positive += group_positive
        false_positive += group_negative
        tpr.append(true_positive / positives)
        fpr.append(false_positive / negatives)
    auc = sum(
        (fpr[index] - fpr[index - 1]) * (tpr[index] + tpr[index - 1]) / 2.0
        for index in range(1, len(fpr))
    )
    return {"fpr": fpr, "tpr": tpr, "auc": auc}


def precision_recall_curve(y_true, scores):
    positives = sum(y_true)
    true_positive = 0
    false_positive = 0
    recall = [0.0]
    precision = [1.0]
    average_precision = 0.0
    for _, group_positive, group_negative in _score_groups(y_true, scores):
        previous_recall = true_positive / positives
        true_positive += group_positive
        false_positive += group_negative
        current_recall = true_positive / positives
        current_precision = true_positive / (true_positive + false_positive)
        recall.append(current_recall)
        precision.append(current_precision)
        average_precision += (current_recall - previous_recall) * current_precision
    return {"recall": recall, "precision": precision, "ap": average_precision}


def calibration_curve(y_true, scores, n_bins=8):
    bins = [[] for _ in range(n_bins)]
    for target, score in zip(y_true, scores):
        index = min(n_bins - 1, int(score * n_bins))
        bins[index].append((target, score))
    mean_probability = []
    positive_rate = []
    for values in bins:
        if values:
            mean_probability.append(sum(score for _, score in values) / len(values))
            positive_rate.append(sum(target for target, _ in values) / len(values))
    return {"mean_probability": mean_probability, "positive_rate": positive_rate}


def probability_curves(y_true, scores, n_bins=8):
    return {
        "roc": roc_curve(y_true, scores),
        "pr": precision_recall_curve(y_true, scores),
        "calibration": calibration_curve(y_true, scores, n_bins=n_bins),
        "brier": sum((score - target) ** 2 for target, score in zip(y_true, scores))
        / len(y_true),
    }


def _population_mean(values):
    return sum(values) / len(values)


def _population_variance(values):
    mean = _population_mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def fisher_scores(features, targets):
    scores = []
    for feature_index in range(len(features[0])):
        negative = [row[feature_index] for row, target in zip(features, targets) if target == 0]
        positive = [row[feature_index] for row, target in zip(features, targets) if target == 1]
        mean_gap = _population_mean(negative) - _population_mean(positive)
        denominator = _population_variance(negative) + _population_variance(positive)
        scores.append(mean_gap**2 / denominator if denominator else 0.0)
    return scores


def _correlation_matrix(features, indices):
    columns = [[row[index] for row in features] for index in indices]
    means = [_population_mean(column) for column in columns]
    scales = [math.sqrt(_population_variance(column)) for column in columns]
    matrix = []
    for left, left_mean, left_scale in zip(columns, means, scales):
        row = []
        for right, right_mean, right_scale in zip(columns, means, scales):
            denominator = len(left) * left_scale * right_scale
            covariance_sum = sum(
                (left_value - left_mean) * (right_value - right_mean)
                for left_value, right_value in zip(left, right)
            )
            row.append(covariance_sum / denominator if denominator else 0.0)
        matrix.append(row)
    return matrix


def _normal_qq(values):
    mean = _population_mean(values)
    scale = math.sqrt(_population_variance(values))
    observed = sorted((value - mean) / scale for value in values)
    normal = NormalDist()
    theoretical = [
        normal.inv_cdf((position + 0.5) / len(values))
        for position in range(len(values))
    ]
    return {"theoretical": theoretical, "observed": observed}


def _positive_probabilities(model, features):
    positive_index = model.classes.index(1)
    return [row[positive_index] for row in model.predict_proba(features)]


def _cross_validated_metrics(features, targets, smoothing, positive_prior, folds):
    metrics = []
    for validation_indices in folds:
        validation_set = set(validation_indices)
        training_indices = [index for index in range(len(targets)) if index not in validation_set]
        class_prior = None
        if positive_prior is not None:
            class_prior = {0: 1.0 - positive_prior, 1: positive_prior}
        model = OptimizedGaussianNaiveBayesScratch(
            var_smoothing=smoothing,
            class_prior=class_prior,
        ).fit(
            [features[index] for index in training_indices],
            [targets[index] for index in training_indices],
        )
        predictions = model.predict([features[index] for index in validation_indices])
        metrics.append(
            classification_metrics(
                [targets[index] for index in validation_indices],
                predictions,
            )
        )
    return {
        name: sum(record[name] for record in metrics) / len(metrics)
        for name in ("balanced_accuracy", "precision", "recall", "f1")
    }


def tune_parameters(features, targets, n_splits=5, seed=42):
    folds = stratified_k_folds(targets, n_splits=n_splits, seed=seed)
    smoothing_values = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    smoothing_records = [
        _cross_validated_metrics(features, targets, value, None, folds)
        for value in smoothing_values
    ]
    best_smoothing_index = max(
        range(len(smoothing_values)),
        key=lambda index: smoothing_records[index]["balanced_accuracy"],
    )
    best_smoothing = smoothing_values[best_smoothing_index]

    prior_values = [0.30, 0.40, 0.50, 0.60, 0.70]
    prior_records = [
        _cross_validated_metrics(features, targets, best_smoothing, value, folds)
        for value in prior_values
    ]
    best_prior_index = max(
        range(len(prior_values)),
        key=lambda index: prior_records[index]["balanced_accuracy"],
    )
    return {
        "smoothing": {
            "values": smoothing_values,
            "scores": [record["balanced_accuracy"] for record in smoothing_records],
            "best": best_smoothing,
        },
        "priors": {
            "values": prior_values,
            "precision": [record["precision"] for record in prior_records],
            "recall": [record["recall"] for record in prior_records],
            "best": prior_values[best_prior_index],
        },
    }


def _feature_fit_report(features, targets, feature_index):
    model = GaussianNaiveBayesScratch().fit(features, targets)
    values = {
        "Benign": [row[feature_index] for row, target in zip(features, targets) if target == 0],
        "Malignant": [row[feature_index] for row, target in zip(features, targets) if target == 1],
    }
    all_values = values["Benign"] + values["Malignant"]
    span = max(all_values) - min(all_values)
    grid = [min(all_values) - 0.05 * span + 1.1 * span * index / 199 for index in range(200)]
    densities = {}
    means = {}
    for label, name in ((0, "Benign"), (1, "Malignant")):
        class_index = model.classes.index(label)
        mean = model.means[class_index][feature_index]
        variance = model.variances[class_index][feature_index]
        means[name] = mean
        densities[name] = [model._gaussian_density(value, mean, variance) for value in grid]
    return {
        "feature_name": FEATURE_NAMES[feature_index],
        "values": values,
        "grid": grid,
        "densities": densities,
        "means": means,
    }


def _surface_report(features, targets, feature_indices, smoothing, positive_prior):
    reduced = [[row[index] for index in feature_indices] for row in features]
    model = OptimizedGaussianNaiveBayesScratch(
        var_smoothing=smoothing,
        class_prior={0: 1.0 - positive_prior, 1: positive_prior},
    ).fit(reduced, targets)
    x_data = [row[0] for row in reduced]
    y_data = [row[1] for row in reduced]
    x_span = max(x_data) - min(x_data)
    y_span = max(y_data) - min(y_data)
    x_values = [min(x_data) - 0.05 * x_span + 1.1 * x_span * index / 89 for index in range(90)]
    y_values = [min(y_data) - 0.05 * y_span + 1.1 * y_span * index / 89 for index in range(90)]
    grid = [[x_value, y_value] for y_value in y_values for x_value in x_values]
    grid_probabilities = _positive_probabilities(model, grid)
    probabilities = [
        grid_probabilities[index * len(x_values):(index + 1) * len(x_values)]
        for index in range(len(y_values))
    ]
    return {
        "x_values": x_values,
        "y_values": y_values,
        "probabilities": probabilities,
        "train_x": reduced,
        "train_y": targets,
        "feature_names": [FEATURE_NAMES[index] for index in feature_indices],
    }


def _assumption_report(features, targets, top_indices):
    top_feature = top_indices[0]
    qq = {}
    correlations = {}
    for label, name in ((0, "Benign"), (1, "Malignant")):
        class_features = [row for row, target in zip(features, targets) if target == label]
        qq[name] = _normal_qq([row[top_feature] for row in class_features])
        correlations[name] = _correlation_matrix(class_features, top_indices)
    return {
        "qq": qq,
        "correlations": correlations,
        "feature_names": [FEATURE_NAMES[index] for index in top_indices],
    }


def _stability_report():
    dimensions = [2, 5, 10, 20, 50, 100, 200, 500]
    base_finite = []
    optimized_finite = []
    underflow_dimension = None
    for dimension in dimensions:
        features = [
            [0.0] * dimension,
            [0.1] * dimension,
            [1.0] * dimension,
            [1.1] * dimension,
        ]
        targets = [0, 0, 1, 1]
        sample = [0.55] * dimension
        base = GaussianNaiveBayesScratch().fit(features, targets)
        likelihood = max(base._class_likelihoods(sample))
        if likelihood == 0.0:
            if underflow_dimension is None:
                underflow_dimension = dimension
            base_finite.append(0.0)
        else:
            base_finite.append(1.0)

        optimized = OptimizedGaussianNaiveBayesScratch().fit(features, targets)
        probabilities = optimized.predict_proba([sample])[0]
        optimized_finite.append(
            1.0 if all(math.isfinite(value) for value in probabilities) else 0.0
        )
    return {
        "dimensions": dimensions,
        "base_finite": base_finite,
        "optimized_finite": optimized_finite,
        "underflow_dimension": underflow_dimension,
    }


def _incremental_report(X_train, y_train, X_test, y_test, smoothing, positive_prior, seed):
    indices = list(range(len(y_train)))
    random.Random(seed).shuffle(indices)
    model = OptimizedGaussianNaiveBayesScratch(
        var_smoothing=smoothing,
        class_prior={0: 1.0 - positive_prior, 1: positive_prior},
    )
    samples_seen = []
    scores = []
    checkpoints = [10, 20, 40, 80, 160, 320, len(indices)]
    previous = 0
    for checkpoint in checkpoints:
        current = min(checkpoint, len(indices))
        if current <= previous:
            continue
        batch = indices[previous:current]
        model.partial_fit(
            [X_train[index] for index in batch],
            [y_train[index] for index in batch],
            classes=[0, 1] if previous == 0 else None,
        )
        samples_seen.append(current)
        scores.append(
            classification_metrics(y_test, model.predict(X_test))["balanced_accuracy"]
        )
        previous = current
    batch_model = OptimizedGaussianNaiveBayesScratch(
        var_smoothing=smoothing,
        class_prior={0: 1.0 - positive_prior, 1: positive_prior},
    ).fit(X_train, y_train)
    return {
        "samples_seen": samples_seen,
        "balanced_accuracy": scores,
        "batch_balanced_accuracy": classification_metrics(
            y_test,
            batch_model.predict(X_test),
        )["balanced_accuracy"],
    }


def run_experiment(data_path, seed=42, n_splits=5):
    features, targets = load_wdbc(data_path)
    X_train, X_test, y_train, y_test = stratified_split(
        features,
        targets,
        test_size=0.2,
        seed=seed,
    )
    feature_scores = fisher_scores(X_train, y_train)
    ranked_features = sorted(
        range(len(FEATURE_NAMES)),
        key=lambda index: feature_scores[index],
        reverse=True,
    )
    tuning = tune_parameters(X_train, y_train, n_splits=n_splits, seed=seed)
    best_smoothing = tuning["smoothing"]["best"]
    best_prior = tuning["priors"]["best"]

    base_model = GaussianNaiveBayesScratch().fit(X_train, y_train)
    optimized_model = OptimizedGaussianNaiveBayesScratch(
        var_smoothing=best_smoothing,
        class_prior={0: 1.0 - best_prior, 1: best_prior},
    ).fit(X_train, y_train)
    majority_label = max(set(y_train), key=y_train.count)
    predictions = {
        "Majority": [majority_label] * len(y_test),
        "Base": base_model.predict(X_test),
        "Optimized": optimized_model.predict(X_test),
    }
    metrics = {
        name: classification_metrics(y_test, values)
        for name, values in predictions.items()
    }
    diagnostics = {
        "Base": probability_curves(y_test, _positive_probabilities(base_model, X_test)),
        "Optimized": probability_curves(
            y_test,
            _positive_probabilities(optimized_model, X_test),
        ),
    }
    tuning["stability"] = _stability_report()
    tuning["incremental"] = _incremental_report(
        X_train,
        y_train,
        X_test,
        y_test,
        best_smoothing,
        best_prior,
        seed,
    )
    return {
        "metadata": {
            "seed": seed,
            "folds": n_splits,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "best_smoothing": best_smoothing,
            "best_positive_prior": best_prior,
        },
        "feature_fit": _feature_fit_report(X_train, y_train, ranked_features[0]),
        "surface": _surface_report(
            X_train,
            y_train,
            ranked_features[:2],
            best_smoothing,
            best_prior,
        ),
        "assumptions": _assumption_report(X_train, y_train, ranked_features[:8]),
        "benchmark": {
            "confusion": {
                "Base": metrics["Base"]["confusion"],
                "Optimized": metrics["Optimized"]["confusion"],
            },
            "metrics": metrics,
        },
        "diagnostics": diagnostics,
        "optimization": tuning,
    }


def _style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.55)
    axis.set_axisbelow(True)


def _save_png(figure, output_dir, stem):
    path = Path(output_dir) / f"{stem}.png"
    if not figure.get_constrained_layout():
        figure.tight_layout(pad=1.2)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _plot_feature_fit(report):
    data = report["feature_fit"]
    palette = FIGURE_PALETTES["distribution"]
    figure, axis = plt.subplots(figsize=(6.2, 3.6))
    _style_axis(axis)
    for name, color, hatch in (
        ("Benign", palette[0], "//"),
        ("Malignant", palette[1], "\\\\"),
    ):
        axis.hist(
            data["values"][name],
            bins=18,
            density=True,
            color=color,
            alpha=0.24,
            edgecolor=color,
            linewidth=0.6,
            hatch=hatch,
            label=f"{name} observations",
        )
        axis.plot(
            data["grid"],
            data["densities"][name],
            color=color,
            linewidth=2.0,
            label=f"{name} Gaussian fit",
        )
        axis.axvline(data["means"][name], color=color, linestyle=":", linewidth=1.1)
    axis.set_xlabel(data["feature_name"])
    axis.set_ylabel("Probability density")
    axis.set_title("Class-conditional Gaussian likelihood")
    axis.legend(ncols=2)
    return figure


def _plot_surface(report):
    data = report["surface"]
    palette = FIGURE_PALETTES["surface"]
    figure, axis = plt.subplots(figsize=(6.0, 4.6))
    color_map = LinearSegmentedColormap.from_list(
        "posterior",
        [palette[2], palette[0], palette[1], palette[3]],
    )
    contour = axis.contourf(
        data["x_values"],
        data["y_values"],
        data["probabilities"],
        levels=[0.0, 0.25, 0.5, 0.75, 1.0],
        cmap=color_map,
        alpha=0.72,
    )
    axis.contour(
        data["x_values"],
        data["y_values"],
        data["probabilities"],
        levels=[0.5],
        colors="#202020",
        linewidths=1.5,
    )
    for label, name, color, marker in (
        (0, "Benign", palette[0], "o"),
        (1, "Malignant", palette[1], "^"),
    ):
        points = [point for point, target in zip(data["train_x"], data["train_y"]) if target == label]
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            s=16,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.80,
            label=name,
        )
    colorbar = figure.colorbar(contour, ax=axis, pad=0.02)
    colorbar.set_label("Posterior P(malignant | x)")
    axis.set_xlabel(data["feature_names"][0])
    axis.set_ylabel(data["feature_names"][1])
    axis.set_title("Two-feature explanatory posterior surface")
    axis.legend(loc="best")
    return figure


def _plot_assumptions(report):
    data = report["assumptions"]
    palette = FIGURE_PALETTES["assumptions"]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(9.0, 3.1),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]},
    )
    _style_axis(axes[0])
    for name, color, marker in (
        ("Benign", palette[0], "o"),
        ("Malignant", palette[1], "s"),
    ):
        qq = data["qq"][name]
        axes[0].scatter(qq["theoretical"], qq["observed"], s=10, color=color, alpha=0.65, marker=marker, label=name)
    axes[0].plot([-3, 3], [-3, 3], color="#555555", linestyle="--", linewidth=0.9)
    axes[0].set(xlabel="Theoretical quantile", ylabel="Observed standardized quantile", xlim=(-3, 3), ylim=(-3, 3))
    axes[0].set_title("Gaussian assumption")
    axes[0].legend(loc="upper left")

    correlation_map = LinearSegmentedColormap.from_list(
        "correlation",
        [palette[0], "#FFFFFF", palette[1]],
    )
    images = []
    short_names = [name.replace("Mean ", "").replace("Worst ", "W. ") for name in data["feature_names"]]
    for axis, name in zip(axes[1:], ("Benign", "Malignant")):
        image = axis.imshow(data["correlations"][name], cmap=correlation_map, vmin=-1.0, vmax=1.0)
        images.append(image)
        axis.set_xticks(range(len(short_names)), short_names, rotation=55, ha="right")
        axis.set_yticks(range(len(short_names)), short_names)
        axis.set_title(f"{name}: conditional correlation")
        for spine in axis.spines.values():
            spine.set_visible(False)
    colorbar = figure.colorbar(images[-1], ax=axes[1:], fraction=0.025, pad=0.03)
    colorbar.set_label("Pearson r")
    figure.suptitle("Audit of Gaussian and conditional-independence assumptions", y=1.02, fontsize=9)
    return figure


def _draw_confusion(axis, matrix, title, color_map, text_color):
    maximum = max(max(row) for row in matrix)
    axis.imshow(matrix, cmap=color_map, vmin=0, vmax=maximum)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > maximum / 2 else text_color,
            )
    axis.set_xticks([0, 1], ["Benign", "Malignant"])
    axis.set_yticks([0, 1], ["Benign", "Malignant"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)


def _plot_benchmark(report):
    data = report["benchmark"]
    palette = FIGURE_PALETTES["benchmark"]
    figure = plt.figure(figsize=(9.0, 3.2))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.8])
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    base_map = LinearSegmentedColormap.from_list("base_confusion", ["#FFFFFF", palette[1]])
    optimized_map = LinearSegmentedColormap.from_list("optimized_confusion", ["#FFFFFF", palette[2]])
    _draw_confusion(axes[0], data["confusion"]["Base"], "Base GNB", base_map, palette[3])
    _draw_confusion(axes[1], data["confusion"]["Optimized"], "Optimized GNB", optimized_map, palette[3])

    _style_axis(axes[2])
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Balanced\naccuracy", "Precision", "Recall", "F1"]
    positions = list(range(len(metric_names)))
    width = 0.24
    for offset, name, color, hatch in (
        (-width, "Majority", palette[0], ".."),
        (0.0, "Base", palette[1], "//"),
        (width, "Optimized", palette[2], ""),
    ):
        axes[2].bar(
            [position + offset for position in positions],
            [data["metrics"][name][metric] for metric in metric_names],
            width=width,
            color=color,
            edgecolor=palette[3],
            linewidth=0.45,
            hatch=hatch,
            label=name,
        )
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("Test-set score")
    axes[2].set_title("Benchmark metrics")
    axes[2].legend(ncols=3, loc="upper center")
    figure.suptitle("Test-set classification benchmark", y=1.02, fontsize=9)
    return figure


def _plot_diagnostics(report):
    data = report["diagnostics"]
    palette = FIGURE_PALETTES["diagnostics"]
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for axis in axes:
        _style_axis(axis)
    for name, color, style in (
        ("Base", palette[0], "--"),
        ("Optimized", palette[1], "-"),
    ):
        values = data[name]
        axes[0].plot(values["roc"]["fpr"], values["roc"]["tpr"], color=color, linestyle=style, linewidth=1.7, label=f"{name} (AUC={values['roc']['auc']:.3f})")
        axes[1].step(values["pr"]["recall"], values["pr"]["precision"], where="post", color=color, linestyle=style, linewidth=1.7, label=f"{name} (AP={values['pr']['ap']:.3f})")
        calibration = values["calibration"]
        axes[2].plot(calibration["mean_probability"], calibration["positive_rate"], color=color, linestyle=style, marker="o", markersize=3.5, linewidth=1.5, label=f"{name} (Brier={values['brier']:.3f})")
    axes[0].plot([0, 1], [0, 1], color=palette[3], linestyle=":", linewidth=0.9)
    axes[2].plot([0, 1], [0, 1], color=palette[2], linestyle=":", linewidth=0.9, label="Perfect calibration")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02), title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision–recall curve")
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed positive rate", xlim=(0, 1), ylim=(0, 1.02), title="Calibration")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")
    axes[2].legend(loc="upper left")
    figure.suptitle("Probability-quality diagnostics", y=1.02, fontsize=9)
    return figure


def _plot_optimization(report):
    data = report["optimization"]
    palette = FIGURE_PALETTES["optimization"]
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
    axes = [axis for row in axes for axis in row]
    for axis in axes:
        _style_axis(axis)

    smoothing = data["smoothing"]
    axes[0].plot(smoothing["values"], smoothing["scores"], color=palette[0], marker="o", markersize=4, linewidth=1.6)
    axes[0].axvline(smoothing["best"], color=palette[3], linestyle="--", linewidth=1.0, label=f"Selected: {smoothing['best']:.0e}")
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Variance smoothing", ylabel="CV balanced accuracy", title="Variance-smoothing selection")
    axes[0].legend(loc="best")

    priors = data["priors"]
    axes[1].plot(priors["values"], priors["precision"], color=palette[1], marker="s", linewidth=1.6, label="Precision")
    axes[1].plot(priors["values"], priors["recall"], color=palette[3], marker="o", linewidth=1.6, label="Recall")
    axes[1].axvline(priors["best"], color="#555555", linestyle="--", linewidth=1.0, label=f"Selected: {priors['best']:.2f}")
    axes[1].set(xlabel="Malignant-class prior", ylabel="Cross-validation score", ylim=(0, 1.02), title="Prior-sensitive trade-off")
    axes[1].legend(loc="best")

    stability = data["stability"]
    axes[2].step(stability["dimensions"], stability["base_finite"], where="post", color=palette[2], marker="s", linewidth=1.5, label="Base direct product")
    axes[2].step(stability["dimensions"], stability["optimized_finite"], where="post", color=palette[1], marker="o", linewidth=1.5, label="Optimized log domain")
    if stability["underflow_dimension"] is not None:
        axes[2].axvline(stability["underflow_dimension"], color=palette[3], linestyle=":", linewidth=1.0, label=f"Underflow at {stability['underflow_dimension']} features")
    axes[2].set_yticks([0.0, 1.0], ["Failed", "Finite"])
    axes[2].set(xlabel="Number of features", ylabel="Posterior computation", ylim=(-0.08, 1.08), title="High-dimensional numerical stability")
    axes[2].legend(loc="best")

    incremental = data["incremental"]
    axes[3].plot(incremental["samples_seen"], incremental["balanced_accuracy"], color=palette[1], marker="o", linewidth=1.7, label="Incremental partial_fit")
    axes[3].axhline(incremental["batch_balanced_accuracy"], color=palette[3], linestyle="--", linewidth=1.2, label="One-shot fit")
    axes[3].set(xlabel="Training samples seen", ylabel="Test balanced accuracy", ylim=(0, 1.02), title="Incremental-learning convergence")
    axes[3].legend(loc="best")
    figure.suptitle("Optimization evidence", y=1.01, fontsize=9)
    return figure


def render_six_figures(report, output_dir):
    """分别导出六张300 DPI PNG，不生成其他图形格式。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("01_gaussian_feature_fit", _plot_feature_fit),
        ("02_posterior_decision_surface", _plot_surface),
        ("03_assumption_diagnostics", _plot_assumptions),
        ("04_benchmark_performance", _plot_benchmark),
        ("05_probability_diagnostics", _plot_diagnostics),
        ("06_optimization_analysis", _plot_optimization),
    ]
    return [_save_png(plotter(report), output_dir, stem) for stem, plotter in plots]


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate six Gaussian Naive Bayes figures.")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "classification" / "wdbc" / "wdbc.data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "figures" / "gaussian_naive_bayes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    report = run_experiment(args.data, seed=args.seed, n_splits=args.folds)
    paths = render_six_figures(report, args.output)
    metadata = report["metadata"]
    print(f"Generated {len(paths)} PNG files in {args.output}")
    print(
        f"Selected var_smoothing={metadata['best_smoothing']:.0e}, "
        f"malignant prior={metadata['best_positive_prior']:.2f}"
    )


if __name__ == "__main__":
    main()
