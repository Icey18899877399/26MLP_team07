"""Generate six publication-style figures for the hand-written random forests."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
}
matplotlib.rcParams.update(PLOT_STYLE)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


FIGURE_FILENAMES = (
    "01_bootstrap_oob_mechanism.png",
    "02_tree_count_convergence.png",
    "03_hyperparameter_response.png",
    "04_tree_diversity_and_ensemble_gain.png",
    "05_feature_importance_stability.png",
    "06_final_benchmark_comparison.png",
)

# Independent, color-blind-conscious palettes tailored to each figure's role.
FIGURE_PALETTES = {
    FIGURE_FILENAMES[0]: ("#006D77", "#E29578", "#83C5BE", "#FFDDD2", "#243B4A"),
    FIGURE_FILENAMES[1]: ("#355070", "#E56B6F", "#6D597A", "#EAAC8B", "#B8C0D9"),
    FIGURE_FILENAMES[2]: ("#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"),
    FIGURE_FILENAMES[3]: ("#5E548E", "#F4A261", "#9F86C0", "#E9D8A6", "#231942"),
    FIGURE_FILENAMES[4]: ("#1B998B", "#D1495B", "#8AC926", "#FFCA3A", "#2D3047"),
    FIGURE_FILENAMES[5]: ("#003049", "#00B4D8", "#F77F00", "#D62828", "#669BBC"),
}

# Sequential blue scale: rank magnitude is ordered, not diverging.
RANK_CMAP_COLORS = ("#F7FBFF", "#6BAED6", "#08306B")

WDBC_FEATURE_NAMES = [
    f"{stat} {measure}"
    for stat in ("Mean", "SE", "Worst")
    for measure in (
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave points", "symmetry",
        "fractal dimension",
    )
]


def classification_metrics(y_true, y_pred):
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "confusion": [[tn, fp], [fn, tp]],
        "accuracy": (tn + tp) / len(y_true),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "precision": precision,
        "recall": sensitivity,
        "f1": f1,
    }


def probability_curves(y_true, scores):
    ordered = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(label == 1 for label in y_true)
    negatives = len(y_true) - positives
    tp = fp = 0
    roc_x, roc_y = [0.0], [0.0]
    pr_x, pr_y = [0.0], [1.0]
    average_precision = 0.0
    previous_recall = 0.0
    position = 0
    while position < len(ordered):
        score = ordered[position][0]
        tied_labels = []
        while position < len(ordered) and ordered[position][0] == score:
            tied_labels.append(ordered[position][1])
            position += 1
        tp += sum(label == 1 for label in tied_labels)
        fp += sum(label != 1 for label in tied_labels)
        recall = tp / positives if positives else 0.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        roc_x.append(fp / negatives if negatives else 0.0)
        roc_y.append(recall)
        pr_x.append(recall)
        pr_y.append(precision)
    auc = sum(
        (roc_x[i] - roc_x[i - 1]) * (roc_y[i] + roc_y[i - 1]) / 2.0
        for i in range(1, len(roc_x))
    )
    return {
        "roc": {"x": roc_x, "y": roc_y, "auc": auc},
        "pr": {"x": pr_x, "y": pr_y, "ap": average_precision},
    }


def pairwise_disagreement(first, second):
    return sum(a != b for a, b in zip(first, second)) / len(first)


def error_correlation(y_true, first, second):
    first_error = [a != b for a, b in zip(y_true, first)]
    second_error = [a != b for a, b in zip(y_true, second)]
    n11 = sum(a and b for a, b in zip(first_error, second_error))
    n10 = sum(a and not b for a, b in zip(first_error, second_error))
    n01 = sum(not a and b for a, b in zip(first_error, second_error))
    n00 = len(y_true) - n11 - n10 - n01
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / denominator if denominator else 0.0


def _panel(axis, label):
    axis.text(-0.09, 1.04, label, transform=axis.transAxes, fontsize=12, fontweight="bold")


def _save(fig, path):
    fig.savefig(path, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _luminance(color):
    red, green, blue, _ = matplotlib.colors.to_rgba(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _annotate_heatmap(axis, matrix, image):
    norm, cmap = image.norm, image.cmap
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            color = "white" if _luminance(cmap(norm(value))) < 0.54 else "#222222"
            axis.text(column, row, f"{value:.3f}" if isinstance(value, float) else str(value), ha="center", va="center", color=color, fontsize=8)


def _sample_tick_positions(sample_count, maximum_ticks=12):
    step = max(1, math.ceil(sample_count / maximum_ticks))
    return list(range(0, sample_count, step))


def _configure_runtime_axis(axis):
    axis.set_yscale("log")
    axis.set_ylabel("Fit time (ms; log scale, startup included)")
    axis.set_xlabel("Synthetic training samples (12 features)")
    axis.set_title("Synthetic runtime scaling (5 trees; one run)")


def _plot_bootstrap(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["bootstrap"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35), layout="constrained", gridspec_kw={"width_ratios": [1.55, 1, 1]})
    cmap = ListedColormap([palette[3], palette[2], palette[0]])
    axes[0].imshow(item["membership"], aspect="auto", cmap=cmap, vmin=0, vmax=2)
    axes[0].set(xlabel="Training sample", ylabel="Tree", title="Bootstrap membership")
    axes[0].set_yticks(range(len(item["membership"])), [f"T{i + 1}" for i in range(len(item["membership"]))])
    tick_positions = _sample_tick_positions(len(item["sample_labels"]))
    axes[0].set_xticks(tick_positions, [f"{i + 1}\n{item['sample_labels'][i]}" for i in tick_positions])
    axes[0].tick_params(length=0)
    axes[0].text(0.02, -0.31, "0 = OOB   1 = sampled   2 = repeated", transform=axes[0].transAxes, color=palette[4], fontsize=7)
    bins = list(range(min(item["oob_counts"]), max(item["oob_counts"]) + 2))
    axes[1].hist(item["oob_counts"], bins=bins, color=palette[1], edgecolor="white", align="left")
    axes[1].set(xlabel="OOB trees per sample", ylabel="Samples", title="OOB coverage")
    tree_probabilities = item["example_tree_probabilities"]
    axes[2].scatter(
        range(1, len(tree_probabilities) + 1),
        tree_probabilities,
        c=[palette[0] if probability >= 0.5 else palette[1] for probability in tree_probabilities],
        s=30,
        zorder=3,
    )
    axes[2].axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    axes[2].axhline(item["example_probability"], color=palette[4], linewidth=1.2)
    axes[2].text(
        0.04,
        0.96,
        f"All-tree mean P(M) = {item['example_probability']:.2f}\ntrue = {'M' if item['example_true'] else 'B'} ({item['example_scope']})",
        transform=axes[2].transAxes,
        va="top",
        color=palette[4],
    )
    axes[2].set(xlabel="Tree", ylabel="Leaf probability P(M)", title="Soft-vote aggregation", ylim=(-0.05, 1.05))
    for label, axis in zip("abc", axes):
        _panel(axis, label)
    _save(fig, path)


def _plot_convergence(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["convergence"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45), layout="constrained")
    for mean_key, std_key, label, color, marker in (
        ("cv_mean", "cv_std", "Five-fold CV", palette[0], "o"),
        ("oob_mean", "oob_std", "OOB", palette[1], "s"),
    ):
        means, stds = item[mean_key], item[std_key]
        axes[0].plot(item["tree_counts"], means, marker=marker, color=color, linewidth=1.7, label=label)
        axes[0].fill_between(item["tree_counts"], [a - b for a, b in zip(means, stds)], [a + b for a, b in zip(means, stds)], color=color, alpha=0.14)
    axes[0].axvline(item["selected_count"], color=palette[2], linestyle="--", linewidth=1.1, label=f"Selected = {item['selected_count']}")
    axes[0].set(xlabel="Number of trees", ylabel="Balanced accuracy", title="Predictive convergence")
    axes[0].set_xlim(0, max(item["tree_counts"]) * 1.08)
    axes[0].legend()
    axes[1].plot(item["tree_counts"], item["coverage_mean"], marker="D", color=palette[2], linewidth=1.8)
    axes[1].axhline(1.0, color="#888888", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="Number of trees", ylabel="OOB coverage", title="Reliability of OOB estimate", ylim=(0, 1.04))
    for label, axis in zip("ab", axes):
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        _panel(axis, label)
    _save(fig, path)


def _plot_hyperparameters(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["hyperparameters"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.65), layout="constrained")
    score_cmap = LinearSegmentedColormap.from_list("rf_score", ["#F6F1E9", palette[2], palette[1], palette[0]])
    leaf_cmap = LinearSegmentedColormap.from_list("rf_leaf", ["#F8F3EA", palette[2], palette[3], palette[4]])
    score_image = axes[0].imshow(item["score_matrix"], cmap=score_cmap, aspect="auto")
    leaf_image = axes[1].imshow(item["leaf_matrix"], cmap=leaf_cmap, aspect="auto")
    for axis in axes:
        axis.set_xticks(range(len(item["feature_labels"])), item["feature_labels"])
        axis.set_yticks(range(len(item["depth_labels"])), item["depth_labels"])
        axis.set_xlabel("max_features")
        axis.set_ylabel("max_depth")
        axis.tick_params(length=0)
    axes[0].set_title("Five-fold balanced accuracy")
    axes[1].set_title("Mean leaves per tree")
    _annotate_heatmap(axes[0], item["score_matrix"], score_image)
    _annotate_heatmap(axes[1], item["leaf_matrix"], leaf_image)
    selected_row, selected_column = item["selected"]
    axes[0].scatter([selected_column], [selected_row], marker="s", s=400, facecolors="none", edgecolors="white", linewidths=1.8)
    fig.colorbar(score_image, ax=axes[0], shrink=0.78, label="Balanced accuracy")
    fig.colorbar(leaf_image, ax=axes[1], shrink=0.78, label="Leaves")
    for label, axis in zip("ab", axes):
        _panel(axis, label)
    _save(fig, path)


def _plot_diversity(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["diversity"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.35), layout="constrained")
    axes[0].scatter(item["disagreement"], item["error_correlation"], color=palette[0], alpha=0.72, edgecolor="white", linewidth=0.4)
    axes[0].axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    axes[0].set(xlabel="Pairwise disagreement", ylabel="Error correlation", title="Tree diversity")
    axes[1].hist(item["individual_scores"], bins=min(8, len(item["individual_scores"])), color=palette[2], edgecolor="white")
    axes[1].axvline(item["ensemble_score"], color=palette[1], linewidth=2, label=f"Forest = {item['ensemble_score']:.3f}")
    axes[1].set(xlabel="Balanced accuracy", ylabel="Trees", title="Individual vs ensemble")
    axes[1].legend()
    axes[2].plot(item["tree_counts"], item["cumulative_scores"], color=palette[0], marker="o", markersize=3.5, linewidth=1.7)
    axes[2].axhline(item["ensemble_score"], color=palette[1], linestyle="--", linewidth=1)
    axes[2].set(xlabel="Trees aggregated", ylabel="Balanced accuracy", title="Gain from aggregation")
    for label, axis in zip("abc", axes):
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.45)
        _panel(axis, label)
    _save(fig, path)


def _plot_importance(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["importance"]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.2), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
    positions = list(range(len(item["feature_names"])))
    for name, offset, color, marker in (("Base", -0.13, palette[1], "o"), ("Optimized", 0.13, palette[0], "s")):
        axes[0].errorbar(item[name]["mean"], [value + offset for value in positions], xerr=item[name]["std"], fmt=marker, color=color, capsize=2.5, linewidth=1, label=name)
    axes[0].set_yticks(positions, item["feature_names"])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Mean importance ± SD across trees", title="Feature importance stability")
    axes[0].legend()
    rank_cmap = LinearSegmentedColormap.from_list("rf_rank", RANK_CMAP_COLORS)
    rank_image = axes[1].imshow(item["rank_matrix"], cmap=rank_cmap, aspect="auto")
    midpoint = (min(map(min, item["rank_matrix"])) + max(map(max, item["rank_matrix"]))) / 2
    for row_index, row in enumerate(item["rank_matrix"]):
        for column_index, rank in enumerate(row):
            axes[1].text(column_index, row_index, str(rank), ha="center", va="center", fontsize=6.5, color="white" if rank > midpoint else "#1F2933")
    axes[1].set_xticks(positions, [name.replace("Worst ", "W. ").replace("Mean ", "M. ") for name in item["feature_names"]], rotation=38, ha="right")
    axes[1].set_yticks(range(len(item["rank_matrix"])), [f"Fold {i + 1}" for i in range(len(item["rank_matrix"]))])
    axes[1].set(xlabel="Feature", ylabel="Cross-validation split", title="Rank stability across folds")
    axes[1].tick_params(length=0)
    fig.colorbar(rank_image, ax=axes[1], shrink=0.76, label="Rank (lower is better)")
    for label, axis in zip("ab", axes):
        _panel(axis, label)
    _save(fig, path)


def _draw_confusion(axis, matrix, colors, title, score):
    maximum = max(max(row) for row in matrix)
    cmap = LinearSegmentedColormap.from_list("confusion", ["#FFFFFF", colors[0]])
    axis.imshow(matrix, cmap=cmap, vmin=0, vmax=maximum)
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row][column] > 0.55 * maximum else "#222222"
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center", fontsize=11, color=color)
    axis.set_xticks([0, 1], ["B", "M"])
    axis.set_yticks([0, 1], ["B", "M"])
    axis.set(xlabel="Predicted", ylabel="Actual", title=f"{title} | BA={score:.3f}")


def _plot_benchmark(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["benchmark"]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.4), layout="constrained")
    _draw_confusion(axes[0, 0], item["Base"]["confusion"], (palette[4],), "Base", item["Base"]["metrics"]["balanced_accuracy"])
    _draw_confusion(axes[0, 1], item["Optimized"]["confusion"], (palette[1],), "Optimized", item["Optimized"]["metrics"]["balanced_accuracy"])
    metric_names = ["balanced_accuracy", "f1", "auc", "ap"]
    x_positions = list(range(len(metric_names)))
    width = 0.36
    axes[0, 2].bar([x - width / 2 for x in x_positions], [item["Base"]["metrics"][key] for key in metric_names], width, color=palette[4], label="Base")
    axes[0, 2].bar([x + width / 2 for x in x_positions], [item["Optimized"]["metrics"][key] for key in metric_names], width, color=palette[1], label="Optimized")
    axes[0, 2].set_xticks(x_positions, ["BA", "F1", "AUC", "AP"])
    axes[0, 2].set_ylim(0.75, 1.0)
    axes[0, 2].set(ylabel="Score", title="Final test metrics")
    axes[0, 2].legend()
    for name, color in (("Base", palette[4]), ("Optimized", palette[1])):
        axes[1, 0].plot(item[name]["roc"]["x"], item[name]["roc"]["y"], color=color, linewidth=1.7, label=f"{name} ({item[name]['metrics']['auc']:.3f})")
        axes[1, 1].plot(item[name]["pr"]["x"], item[name]["pr"]["y"], color=color, linewidth=1.7, label=f"{name} ({item[name]['metrics']['ap']:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.8)
    axes[1, 0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1.02))
    axes[1, 1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve", xlim=(0, 1), ylim=(0, 1.02))
    runtime = item["runtime"]
    for key, label, color, marker in (
        ("base_ms", "Base", palette[4], "o"),
        ("optimized_serial_ms", "Optimized serial", palette[2], "s"),
        ("optimized_parallel_ms", "Optimized parallel", palette[1], "D"),
    ):
        axes[1, 2].plot(runtime["sizes"], runtime[key], color=color, marker=marker, linewidth=1.6, label=label)
    _configure_runtime_axis(axes[1, 2])
    for axis in axes.flat:
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.45)
    axes[1, 0].legend(loc="lower right")
    axes[1, 1].legend(loc="lower left")
    axes[1, 2].legend()
    for label, axis in zip("abcdef", axes.flat):
        _panel(axis, label)
    _save(fig, path)


def render_six_figures(report, output_dir):
    matplotlib.rcParams.update(PLOT_STYLE)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in FIGURE_FILENAMES]
    for path in paths:
        if path.is_file():
            path.unlink()
    plotters = (_plot_bootstrap, _plot_convergence, _plot_hyperparameters, _plot_diversity, _plot_importance, _plot_benchmark)
    for plotter, path in zip(plotters, paths):
        plotter(report, path)
    return paths


def load_wdbc(path):
    features, targets = [], []
    with Path(path).open("r", encoding="utf-8") as source:
        for row in csv.reader(source):
            if row:
                targets.append(1 if row[1] == "M" else 0)
                features.append([float(value) for value in row[2:]])
    return features, targets


def stratified_split(features, targets, test_fraction=0.2, seed=42):
    generator = random.Random(seed)
    train_indices, test_indices = [], []
    for label in sorted(set(targets)):
        indices = [index for index, target in enumerate(targets) if target == label]
        generator.shuffle(indices)
        test_count = max(1, round(len(indices) * test_fraction))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    generator.shuffle(train_indices)
    generator.shuffle(test_indices)
    return (
        [features[i] for i in train_indices], [features[i] for i in test_indices],
        [targets[i] for i in train_indices], [targets[i] for i in test_indices],
    )


def _fold_indices(targets, folds, seed):
    generator = random.Random(seed)
    buckets = [[] for _ in range(folds)]
    for label in sorted(set(targets)):
        indices = [index for index, target in enumerate(targets) if target == label]
        generator.shuffle(indices)
        for position, index in enumerate(indices):
            buckets[position % folds].append(index)
    return buckets


def _mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _aligned_tree_probabilities(tree, samples, classes):
    rows = tree.predict_proba(samples)
    positions = {label: index for index, label in enumerate(tree.classes)}
    return [[row[positions[label]] if label in positions else 0.0 for label in classes] for row in rows]


def _prefix_probabilities(model, samples, tree_count, soft=True):
    totals = [[0.0] * len(model.classes) for _ in samples]
    for tree in model.estimators_[:tree_count]:
        if soft:
            votes = _aligned_tree_probabilities(tree, samples, model.classes)
        else:
            votes = [[1.0 if prediction == label else 0.0 for label in model.classes] for prediction in tree.predict(samples)]
        for total, vote in zip(totals, votes):
            for index, value in enumerate(vote):
                total[index] += value
    return [[value / tree_count for value in row] for row in totals]


def _predictions_from_probabilities(probabilities, classes):
    return [classes[max(range(len(classes)), key=row.__getitem__)] for row in probabilities]


def _prefix_oob(model, features, targets, tree_count):
    totals = [[0.0] * len(model.classes) for _ in targets]
    counts = [0] * len(targets)
    all_indices = set(range(len(targets)))
    for tree, sampled in zip(model.estimators_[:tree_count], model.bootstrap_indices_[:tree_count]):
        indices = sorted(all_indices - set(sampled))
        votes = _aligned_tree_probabilities(tree, [features[i] for i in indices], model.classes)
        for sample_index, vote in zip(indices, votes):
            counts[sample_index] += 1
            for class_index, value in enumerate(vote):
                totals[sample_index][class_index] += value
    covered = [i for i, count in enumerate(counts) if count]
    probabilities = [[value / counts[i] for value in totals[i]] for i in covered]
    predictions = _predictions_from_probabilities(probabilities, model.classes)
    score = classification_metrics([targets[i] for i in covered], predictions)["balanced_accuracy"] if covered else 0.0
    return score, len(covered) / len(targets)


def _cv_forest(features, targets, folds, seed, parameters, keep_models=False):
    from Models.random_forest_optimized import OptimizedRandomForestClassifierScratch

    scores, leaves, models = [], [], []
    all_indices = set(range(len(targets)))
    for validation in _fold_indices(targets, folds, seed):
        validation_set = set(validation)
        training = sorted(all_indices - validation_set)
        model = OptimizedRandomForestClassifierScratch(n_jobs=1, oob_score=True, random_state=seed, **parameters)
        model.fit([features[i] for i in training], [targets[i] for i in training])
        predictions = model.predict([features[i] for i in validation])
        scores.append(classification_metrics([targets[i] for i in validation], predictions)["balanced_accuracy"])
        leaves.append(statistics.mean(tree.n_leaves_ for tree in model.estimators_))
        if keep_models:
            models.append((model, training, validation))
    mean, std = _mean_sd(scores)
    return {"mean": mean, "std": std, "leaves": statistics.mean(leaves), "models": models}


def _convergence_report(features, targets, folds, seed):
    from Models.random_forest_optimized import OptimizedRandomForestClassifierScratch

    counts = [1, 5, 10, 20, 40]
    cv_values = [[] for _ in counts]
    oob_values = [[] for _ in counts]
    coverage_values = [[] for _ in counts]
    fold_models = []
    all_indices = set(range(len(targets)))
    for fold_number, validation in enumerate(_fold_indices(targets, folds, seed)):
        validation_set = set(validation)
        training = sorted(all_indices - validation_set)
        train_x, train_y = [features[i] for i in training], [targets[i] for i in training]
        validation_x, validation_y = [features[i] for i in validation], [targets[i] for i in validation]
        model = OptimizedRandomForestClassifierScratch(
            n_estimators=max(counts), max_depth=4, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", ccp_alpha=0.003,
            n_jobs=1, random_state=seed + fold_number,
        ).fit(train_x, train_y)
        fold_models.append(model)
        for position, count in enumerate(counts):
            probabilities = _prefix_probabilities(model, validation_x, count, soft=True)
            predictions = _predictions_from_probabilities(probabilities, model.classes)
            cv_values[position].append(classification_metrics(validation_y, predictions)["balanced_accuracy"])
            oob_score, coverage = _prefix_oob(model, train_x, train_y, count)
            oob_values[position].append(oob_score)
            coverage_values[position].append(coverage)
    cv_summary = [_mean_sd(values) for values in cv_values]
    oob_summary = [_mean_sd(values) for values in oob_values]
    coverage_mean = [statistics.mean(values) for values in coverage_values]
    maximum = max(value[0] for value in cv_summary)
    eligible = [count for count, (mean, _), coverage in zip(counts, cv_summary, coverage_mean) if mean >= maximum - 0.002 and coverage >= 0.99]
    selected = min(eligible) if eligible else counts[max(range(len(counts)), key=lambda i: cv_summary[i][0])]
    return {
        "tree_counts": counts,
        "cv_mean": [value[0] for value in cv_summary],
        "cv_std": [value[1] for value in cv_summary],
        "oob_mean": [value[0] for value in oob_summary],
        "oob_std": [value[1] for value in oob_summary],
        "coverage_mean": coverage_mean,
        "selected_count": selected,
    }, fold_models


def _hyperparameter_report(features, targets, folds, seed):
    depths = [2, 4, 6, None]
    feature_options = [0.25, "sqrt", None]
    scores, leaves = [], []
    for depth in depths:
        score_row, leaf_row = [], []
        for option in feature_options:
            result = _cv_forest(
                features, targets, folds, seed,
                {"n_estimators": 12, "max_depth": depth, "min_samples_leaf": 2, "max_features": option, "class_weight": "balanced", "ccp_alpha": 0.003},
            )
            score_row.append(result["mean"])
            leaf_row.append(result["leaves"])
        scores.append(score_row)
        leaves.append(leaf_row)
    candidates = [(scores[row][column], -leaves[row][column], -row, -column, row, column) for row in range(len(depths)) for column in range(len(feature_options))]
    _, _, _, _, selected_row, selected_column = max(candidates)
    return {
        "depth_labels": ["None" if value is None else str(value) for value in depths],
        "feature_labels": ["25%", "sqrt", "all"],
        "score_matrix": scores,
        "leaf_matrix": leaves,
        "selected": [selected_row, selected_column],
        "depth_value": depths[selected_row],
        "feature_value": feature_options[selected_column],
    }


def _tree_importance_summary(model):
    columns = list(zip(*(tree.feature_importances_ for tree in model.estimators_)))
    return {
        "mean": [statistics.mean(column) for column in columns],
        "std": [statistics.stdev(column) if len(column) > 1 else 0.0 for column in columns],
    }


def _rank_vector(values):
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _diagnostics(model, features, targets):
    probabilities = model.predict_proba(features)
    positive_index = model.classes.index(1)
    predictions = _predictions_from_probabilities(probabilities, model.classes)
    metrics = classification_metrics(targets, predictions)
    curves = probability_curves(targets, [row[positive_index] for row in probabilities])
    metrics.update({"auc": curves["roc"]["auc"], "ap": curves["pr"]["ap"]})
    return {"confusion": metrics["confusion"], "metrics": metrics, "roc": curves["roc"], "pr": curves["pr"]}


def _synthetic_runtime(seed, n_estimators=5):
    from Models.random_forest import RandomForestClassifierScratch
    from Models.random_forest_optimized import OptimizedRandomForestClassifierScratch

    generator = random.Random(seed)
    sizes = [80, 160, 320]
    base_ms, serial_ms, parallel_ms = [], [], []
    for size in sizes:
        features = [[generator.uniform(-1, 1) for _ in range(12)] for _ in range(size)]
        targets = [int(row[0] + 0.7 * row[1] - 0.4 * row[2] + generator.gauss(0, 0.2) > 0) for row in features]
        started = time.perf_counter()
        RandomForestClassifierScratch(n_estimators=n_estimators, max_depth=4, random_state=seed).fit(features, targets)
        base_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        OptimizedRandomForestClassifierScratch(n_estimators=n_estimators, max_depth=4, n_jobs=1, random_state=seed).fit(features, targets)
        serial_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        OptimizedRandomForestClassifierScratch(n_estimators=n_estimators, max_depth=4, n_jobs=2, random_state=seed).fit(features, targets)
        parallel_ms.append((time.perf_counter() - started) * 1000)
    return {"sizes": sizes, "base_ms": base_ms, "optimized_serial_ms": serial_ms, "optimized_parallel_ms": parallel_ms}


def run_experiment(data_path, seed=42, folds=5):
    from Models.random_forest import RandomForestClassifierScratch
    from Models.random_forest_optimized import OptimizedRandomForestClassifierScratch

    features, targets = load_wdbc(data_path)
    train_x, test_x, train_y, test_y = stratified_split(features, targets, seed=seed)
    convergence, _ = _convergence_report(train_x, train_y, folds, seed)
    hyper = _hyperparameter_report(train_x, train_y, folds, seed)
    tree_count = convergence["selected_count"]
    common = {"n_estimators": tree_count, "max_depth": hyper["depth_value"], "max_features": hyper["feature_value"], "random_state": seed}
    base = RandomForestClassifierScratch(**common).fit(train_x, train_y)
    optimized = OptimizedRandomForestClassifierScratch(
        **common, min_samples_leaf=2, class_weight="balanced", ccp_alpha=0.003,
        voting="soft", oob_score=True, n_jobs=2,
    ).fit(train_x, train_y)

    selected_cv = _cv_forest(
        train_x,
        train_y,
        folds,
        seed,
        {
            "n_estimators": tree_count,
            "max_depth": hyper["depth_value"],
            "min_samples_leaf": 2,
            "max_features": hyper["feature_value"],
            "class_weight": "balanced",
            "ccp_alpha": 0.003,
        },
        keep_models=True,
    )
    selected_fold_models = selected_cv["models"]
    diagnostic_model, diagnostic_training, diagnostic_validation = selected_fold_models[0]
    diagnostic_train_x = [train_x[i] for i in diagnostic_training]
    diagnostic_train_y = [train_y[i] for i in diagnostic_training]
    diagnostic_x = [train_x[i] for i in diagnostic_validation]
    diagnostic_y = [train_y[i] for i in diagnostic_validation]

    sample_indices = list(range(min(36, len(diagnostic_train_x))))
    membership = []
    for sampled in diagnostic_model.bootstrap_indices_[:10]:
        membership.append([min(2, sampled.count(index)) for index in sample_indices])
    vote_sample = diagnostic_x[0]
    positive_index = diagnostic_model.classes.index(1)
    example_tree_probabilities = [
        _aligned_tree_probabilities(tree, [vote_sample], diagnostic_model.classes)[0][positive_index]
        for tree in diagnostic_model.estimators_
    ]
    example_probability = diagnostic_model.predict_proba([vote_sample])[0][positive_index]
    bootstrap_report = {
        "membership": membership,
        "sample_labels": ["M" if diagnostic_train_y[i] else "B" for i in sample_indices],
        "oob_counts": [sum(index not in sampled for sampled in diagnostic_model.bootstrap_indices_) for index in range(len(diagnostic_train_x))],
        "example_tree_probabilities": example_tree_probabilities,
        "example_probability": example_probability,
        "example_true": diagnostic_y[0],
        "example_scope": "CV validation",
    }

    tree_predictions = [tree.predict(diagnostic_x) for tree in diagnostic_model.estimators_]
    disagreements, correlations = [], []
    for first in range(len(tree_predictions)):
        for second in range(first + 1, len(tree_predictions)):
            disagreements.append(pairwise_disagreement(tree_predictions[first], tree_predictions[second]))
            correlations.append(error_correlation(diagnostic_y, tree_predictions[first], tree_predictions[second]))
    individual_scores = [classification_metrics(diagnostic_y, prediction)["balanced_accuracy"] for prediction in tree_predictions]
    cumulative_scores = []
    for count in range(1, tree_count + 1):
        probabilities = _prefix_probabilities(diagnostic_model, diagnostic_x, count, soft=True)
        cumulative_scores.append(classification_metrics(diagnostic_y, _predictions_from_probabilities(probabilities, diagnostic_model.classes))["balanced_accuracy"])
    ensemble_score = classification_metrics(diagnostic_y, diagnostic_model.predict(diagnostic_x))["balanced_accuracy"]

    base_importance = _tree_importance_summary(base)
    optimized_importance = _tree_importance_summary(optimized)
    top_indices = sorted(range(len(WDBC_FEATURE_NAMES)), key=lambda index: optimized_importance["mean"][index], reverse=True)[:10]
    rank_matrix = []
    for model, _, _ in selected_fold_models:
        ranks = _rank_vector(model.feature_importances_)
        rank_matrix.append([ranks[index] for index in top_indices])

    base_diagnostics = _diagnostics(base, test_x, test_y)
    optimized_diagnostics = _diagnostics(optimized, test_x, test_y)
    return {
        "bootstrap": bootstrap_report,
        "convergence": convergence,
        "hyperparameters": hyper,
        "diversity": {
            "disagreement": disagreements,
            "error_correlation": correlations,
            "individual_scores": individual_scores,
            "ensemble_score": ensemble_score,
            "tree_counts": list(range(1, tree_count + 1)),
            "cumulative_scores": cumulative_scores,
        },
        "importance": {
            "feature_names": [WDBC_FEATURE_NAMES[index] for index in top_indices],
            "Base": {"mean": [base_importance["mean"][index] for index in top_indices], "std": [base_importance["std"][index] for index in top_indices]},
            "Optimized": {"mean": [optimized_importance["mean"][index] for index in top_indices], "std": [optimized_importance["std"][index] for index in top_indices]},
            "rank_matrix": rank_matrix,
        },
        "benchmark": {
            "Base": base_diagnostics,
            "Optimized": optimized_diagnostics,
            "runtime": _synthetic_runtime(seed),
        },
    }


def build_parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate six random-forest PNG figures.")
    parser.add_argument("--data", type=Path, default=project_root / "data" / "classification" / "wdbc" / "wdbc.data")
    parser.add_argument("--output", type=Path, default=project_root / "figures" / "random_forest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    report = run_experiment(arguments.data, arguments.seed, arguments.folds)
    paths = render_six_figures(report, arguments.output)
    print(f"Generated {len(paths)} PNG figures in {arguments.output}")


if __name__ == "__main__":
    main()
