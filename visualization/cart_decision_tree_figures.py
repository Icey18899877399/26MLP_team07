"""Generate six publication-style figures for the hand-written CART models.

Only Matplotlib is used for plotting.  Model fitting and every reported metric are
computed by the project's own CART implementations and Python's standard library.
"""

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
    "savefig.dpi": 300,
}
matplotlib.rcParams.update(PLOT_STYLE)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


FIGURE_FILENAMES = (
    "01_split_gain_mechanism.png",
    "02_tree_structure_comparison.png",
    "03_decision_boundary_comparison.png",
    "04_pruning_validation_curve.png",
    "05_test_set_diagnostics.png",
    "06_optimization_evidence.png",
)

# Each figure deliberately has its own visual identity while remaining color-blind safe.
FIGURE_PALETTES = {
    FIGURE_FILENAMES[0]: ("#2A9D8F", "#E76F51", "#264653", "#E9C46A"),
    FIGURE_FILENAMES[1]: ("#457B9D", "#E63946", "#A8DADC", "#F1FAEE"),
    FIGURE_FILENAMES[2]: ("#6A4C93", "#F4A261", "#CDB4DB", "#FFD6A5"),
    FIGURE_FILENAMES[3]: ("#0077B6", "#D62828", "#90E0EF", "#F77F00"),
    FIGURE_FILENAMES[4]: ("#3A86FF", "#FF006E", "#8338EC", "#FFBE0B"),
    FIGURE_FILENAMES[5]: ("#386641", "#BC4749", "#6A994E", "#F2E8CF"),
}

WDBC_FEATURE_NAMES = [
    f"{stat} {measure}"
    for stat in ("Mean", "SE", "Worst")
    for measure in (
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave points",
        "symmetry",
        "fractal dimension",
    )
]


def _gini(targets):
    if not targets:
        return 0.0
    return 1.0 - sum((targets.count(label) / len(targets)) ** 2 for label in set(targets))


def split_gain_curve(values, targets):
    """Return every midpoint threshold and its exact Gini information gain."""
    unique_values = sorted(set(values))
    thresholds = [(a + b) / 2.0 for a, b in zip(unique_values, unique_values[1:])]
    parent = _gini(list(targets))
    gains = []
    for threshold in thresholds:
        left = [target for value, target in zip(values, targets) if value <= threshold]
        right = [target for value, target in zip(values, targets) if value > threshold]
        weighted = (len(left) * _gini(left) + len(right) * _gini(right)) / len(targets)
        gains.append(parent - weighted)
    selected = thresholds[max(range(len(gains)), key=gains.__getitem__)] if gains else None
    return {"thresholds": thresholds, "gains": gains, "selected_threshold": selected}


def classification_metrics(y_true, y_pred):
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "confusion": [[tn, fp], [fn, tp]],
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def probability_curves(y_true, scores):
    """Compute empirical ROC/PR points, trapezoidal AUC, and average precision."""
    ordered = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(label == 1 for label in y_true)
    negatives = len(y_true) - positives
    tp = fp = 0
    roc_x = [0.0]
    roc_y = [0.0]
    pr_x = [0.0]
    pr_y = [1.0]
    ap = 0.0
    for _, label in ordered:
        if label == 1:
            tp += 1
            ap += (tp / (tp + fp)) / positives if positives else 0.0
        else:
            fp += 1
        recall = tp / positives if positives else 0.0
        precision = tp / (tp + fp) if tp + fp else 1.0
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
        "pr": {"x": pr_x, "y": pr_y, "ap": ap},
    }


def _save(fig, path):
    fig.savefig(path, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _panel_label(axis, label):
    axis.text(-0.08, 1.04, label, transform=axis.transAxes, fontsize=12, fontweight="bold")


def _annotate_selected_threshold(axis, threshold, color):
    return axis.text(
        0.31,
        0.94,
        f"selected = {threshold:.3g}",
        transform=axis.transAxes,
        color=color,
        va="top",
    )


def _plot_split_gain(report, path):
    palette = FIGURE_PALETTES[path.name]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), layout="constrained")
    item = report["split_gain"]
    for row, (class_name, values) in enumerate(item["values"].items()):
        axes[0].scatter(values, [row] * len(values), s=22, alpha=0.72, color=palette[row])
    axes[0].axvline(item["selected_threshold"], color=palette[2], linestyle="--", linewidth=1.4)
    axes[0].set_yticks(range(len(item["values"])), list(item["values"].keys()))
    axes[0].set_xlabel(item["feature_name"])
    axes[0].set_title("Samples at the root node")
    axes[0].grid(axis="x", color="#DDDDDD", linewidth=0.5)
    _annotate_selected_threshold(axes[0], item["selected_threshold"], palette[2])
    axes[1].plot(item["thresholds"], item["gains"], color=palette[0], linewidth=1.8)
    best_index = max(range(len(item["gains"])), key=item["gains"].__getitem__)
    axes[1].scatter(
        [item["thresholds"][best_index]], [item["gains"][best_index]],
        s=46, color=palette[1], zorder=3, label="Maximum gain",
    )
    axes[1].set(xlabel="Candidate threshold", ylabel="Gini information gain", title="Exhaustive split search")
    axes[1].grid(color="#DDDDDD", linewidth=0.5)
    axes[1].legend(frameon=False)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    _save(fig, path)


def _tree_layout(root, maximum_depth=3):
    nodes, edges = [], []
    next_leaf = [0]

    def visit(node, depth):
        collapsed = depth >= maximum_depth and node.get("left") is not None
        if node.get("left") is None or collapsed:
            x = next_leaf[0]
            next_leaf[0] += 1
        else:
            left_x = visit(node["left"], depth + 1)
            right_x = visit(node["right"], depth + 1)
            x = (left_x + right_x) / 2.0
            edges.extend([(x, -depth, left_x, -(depth + 1)), (x, -depth, right_x, -(depth + 1))])
        nodes.append((node, x, -depth, collapsed))
        return x

    visit(root, 0)
    return nodes, edges


def _draw_tree(axis, tree, feature_names, palette, title, metadata):
    nodes, edges = _tree_layout(tree)
    for x1, y1, x2, y2 in edges:
        axis.plot([x1, x2], [y1, y2], color="#888888", linewidth=0.8, zorder=0)
    for node, x, y, collapsed in nodes:
        leaf = node.get("left") is None
        if leaf:
            class_name = "Malignant" if node["prediction"] == 1 else "Benign"
            text = f"{class_name}\nn = {node['sample_count']}\nP(M) = {node['probabilities'][1]:.2f}"
            face = palette[3]
        else:
            feature = feature_names[node["feature_index"]]
            text = (
                f"{feature} ≤ {node['threshold']:.3g}\n"
                f"Gini = {node['impurity']:.2f}; n = {node['sample_count']}\n"
                f"P(M) = {node['probabilities'][1]:.2f}"
            )
            if collapsed:
                text += "\n…"
            face = palette[2]
        axis.text(
            x, y, text, ha="center", va="center", fontsize=6.8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": face, "edgecolor": palette[0], "linewidth": 0.8},
            zorder=2,
        )
    axis.set_title(f"{title}\ndepth = {metadata['depth']}; leaves = {metadata['leaves']}")
    axis.set_xlim(-0.8, max(item[1] for item in nodes) + 0.8)
    axis.set_ylim(-3.55, 0.45)
    axis.axis("off")


def _plot_trees(report, path):
    palette = FIGURE_PALETTES[path.name]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), layout="constrained")
    trees = report["trees"]
    for axis, name, label in zip(axes, ("Base", "Optimized"), ("a", "b")):
        _draw_tree(axis, trees[name], trees["feature_names"], palette, name, trees["metadata"][name])
        _panel_label(axis, label)
    _save(fig, path)


def _plot_boundaries(report, path):
    palette = FIGURE_PALETTES[path.name]
    cmap = ListedColormap([palette[2], palette[3]])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.55), sharex=True, sharey=True, layout="constrained")
    item = report["boundaries"]
    for axis, name, label in zip(axes, ("Base", "Optimized"), ("a", "b")):
        axis.contourf(item["x_values"], item["y_values"], item[name], levels=[-0.5, 0.5, 1.5], cmap=cmap, alpha=0.55)
        for class_value, marker, color in ((0, "o", palette[0]), (1, "^", palette[1])):
            points = [row for row, target in zip(item["train_x"], item["train_y"]) if target == class_value]
            axis.scatter(
                [row[0] for row in points], [row[1] for row in points], marker=marker,
                s=20, color=color, edgecolor="white", linewidth=0.35,
                label="Benign" if class_value == 0 else "Malignant",
            )
        axis.set_title(name)
        axis.set_xlabel(item["feature_names"][0])
        axis.grid(color="white", linewidth=0.45, alpha=0.8)
        _panel_label(axis, label)
    axes[0].set_ylabel(item["feature_names"][1])
    axes[1].legend(frameon=False, loc="best")
    _save(fig, path)


def _plot_pruning(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["pruning"]
    positions = list(range(len(item["alphas"])))
    labels = [f"{value:g}" for value in item["alphas"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), layout="constrained")
    axes[0].errorbar(
        positions, item["score_mean"], yerr=item["score_std"], color=palette[0],
        marker="o", markersize=4, capsize=3, linewidth=1.5,
    )
    best_index = item["alphas"].index(item["best_alpha"])
    axes[0].scatter([best_index], [item["score_mean"][best_index]], color=palette[1], s=50, zorder=3, label="Selected")
    axes[0].set(ylabel="CV balanced accuracy", xlabel="Cost-complexity alpha", title="Five-fold validation")
    axes[0].set_xticks(positions, labels, rotation=30)
    axes[0].legend(frameon=False)
    axes[1].plot(positions, item["leaves_mean"], marker="o", color=palette[1], label="Leaves")
    axes[1].plot(positions, item["depth_mean"], marker="s", color=palette[3], label="Depth")
    axes[1].axvline(best_index, color=palette[0], linestyle="--", linewidth=1)
    axes[1].set(ylabel="Mean structural complexity", xlabel="Cost-complexity alpha", title="Tree simplification")
    axes[1].set_xticks(positions, labels, rotation=30)
    axes[1].legend(frameon=False)
    for label, axis in zip(("a", "b"), axes):
        axis.grid(color="#DDDDDD", linewidth=0.5)
        _panel_label(axis, label)
    _save(fig, path)


def _draw_confusion(axis, matrix, palette, title, score):
    maximum = max(max(row) for row in matrix)
    axis.imshow(matrix, cmap=ListedColormap(["#FFFFFF", palette[0]]), vmin=0, vmax=maximum)
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row][column] > maximum * 0.55 else "#222222"
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center", fontsize=12, color=color)
    axis.set_xticks([0, 1], ["Benign", "Malignant"])
    axis.set_yticks([0, 1], ["Benign", "Malignant"])
    axis.set(xlabel="Predicted", ylabel="Actual", title=f"{title} | balanced accuracy = {score:.3f}")


def _plot_diagnostics(report, path):
    palette = FIGURE_PALETTES[path.name]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5), layout="constrained")
    for column, name in enumerate(("Base", "Optimized")):
        item = report["diagnostics"][name]
        _draw_confusion(axes[0, column], item["confusion"], palette, name, item["balanced_accuracy"])
        _panel_label(axes[0, column], "ab"[column])
    for name, color in (("Base", palette[0]), ("Optimized", palette[1])):
        roc = report["diagnostics"][name]["roc"]
        pr = report["diagnostics"][name]["pr"]
        axes[1, 0].plot(roc["x"], roc["y"], color=color, linewidth=1.7, label=f"{name} (AUC={roc['auc']:.3f})")
        axes[1, 1].plot(pr["x"], pr["y"], color=color, linewidth=1.7, label=f"{name} (AP={pr['ap']:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.8)
    axes[1, 0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[1, 1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
    for label, axis in zip(("c", "d"), axes[1]):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(color="#DDDDDD", linewidth=0.5)
        axis.legend(frameon=False, loc="lower left")
        _panel_label(axis, label)
    _save(fig, path)


def _plot_optimization(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["optimization"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.65), layout="constrained")
    runtime = item["runtime"]
    axes[0].plot(runtime["sizes"], runtime["base_ms"], marker="o", color=palette[1], label="Brute-force scan")
    axes[0].plot(runtime["sizes"], runtime["optimized_ms"], marker="s", color=palette[0], label="Sorted scan")
    axes[0].set(xlabel="Training samples", ylabel="Fit time (ms)", title="Runtime scaling")
    axes[0].legend(frameon=False)
    ablation = item["ablation"]
    names = [row["name"] for row in ablation]
    scores = [row["balanced_accuracy"] for row in ablation]
    axes[1].bar(range(len(names)), scores, color=[palette[0], palette[2], palette[3], palette[1], palette[0]][: len(names)], width=0.72)
    axes[1].set_xticks(range(len(names)), names, rotation=35, ha="right")
    axes[1].set_ylim(max(0.0, min(scores) - 0.08), min(1.0, max(scores) + 0.04))
    axes[1].set(ylabel="CV balanced accuracy", title="Optimization ablation")
    importance = item["importance"]
    y_positions = list(range(len(importance["feature_names"])))
    for name, offset, color, marker in (("Base", -0.12, palette[1], "o"), ("Optimized", 0.12, palette[0], "s")):
        axes[2].errorbar(
            importance[name]["mean"], [y + offset for y in y_positions], xerr=importance[name]["std"],
            fmt=marker, color=color, capsize=2.5, linewidth=1, label=name,
        )
    axes[2].set_yticks(y_positions, importance["feature_names"])
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Mean importance ± SD", title="Five-fold stability")
    axes[2].legend(frameon=False)
    for label, axis in zip(("a", "b", "c"), axes):
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        _panel_label(axis, label)
    _save(fig, path)


def render_six_figures(report, output_dir):
    """Render the fixed six-file PNG contract and return output paths."""
    # Other visualization modules may mutate Matplotlib's process-global style.
    matplotlib.rcParams.update(PLOT_STYLE)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.iterdir():
        if existing.is_file():
            existing.unlink()
    paths = [output_dir / name for name in FIGURE_FILENAMES]
    plotters = (
        _plot_split_gain,
        _plot_trees,
        _plot_boundaries,
        _plot_pruning,
        _plot_diagnostics,
        _plot_optimization,
    )
    for plotter, path in zip(plotters, paths):
        plotter(report, path)
    return paths


def load_wdbc(path):
    features, targets = [], []
    with Path(path).open("r", encoding="utf-8") as source:
        for row in csv.reader(source):
            if not row:
                continue
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


def _positive_scores(model, features):
    positive_index = model.classes.index(1)
    return [row[positive_index] for row in model.predict_proba(features)]


def _cv_evaluate(features, targets, folds, seed, parameters, keep_importance=False):
    from Models.cart_decision_tree_optimized import OptimizedCARTClassifierScratch

    scores, leaves, depths, importances = [], [], [], []
    all_indices = set(range(len(targets)))
    for validation in _fold_indices(targets, folds, seed):
        validation_set = set(validation)
        training = sorted(all_indices - validation_set)
        model = OptimizedCARTClassifierScratch(**parameters)
        model.fit([features[i] for i in training], [targets[i] for i in training])
        predictions = model.predict([features[i] for i in validation])
        scores.append(classification_metrics([targets[i] for i in validation], predictions)["balanced_accuracy"])
        leaves.append(model.n_leaves_)
        depths.append(model.tree_depth_)
        if keep_importance:
            importances.append(model.feature_importances_)
    score_mean, score_sd = _mean_sd(scores)
    result = {
        "score_mean": score_mean,
        "score_std": score_sd,
        "leaves_mean": statistics.mean(leaves),
        "depth_mean": statistics.mean(depths),
    }
    if keep_importance:
        result["importances"] = importances
    return result


def _select_best(candidates, evaluations, complexity_key=None):
    indices = list(range(len(candidates)))
    if complexity_key is None:
        return max(indices, key=lambda i: evaluations[i]["score_mean"])
    return max(indices, key=lambda i: (evaluations[i]["score_mean"], -complexity_key(candidates[i])))


def _tree_dict(node, classes):
    probabilities_by_label = {
        label: probability for label, probability in zip(classes, node.probabilities)
    }
    return {
        "prediction": node.prediction,
        "probabilities": [probabilities_by_label.get(0, 0.0), probabilities_by_label.get(1, 0.0)],
        "impurity": node.impurity,
        "sample_count": node.sample_count,
        "feature_index": node.feature_index,
        "threshold": node.threshold,
        "left": _tree_dict(node.left, classes) if node.left is not None else None,
        "right": _tree_dict(node.right, classes) if node.right is not None else None,
    }


def _diagnostic_record(model, features, targets):
    metrics = classification_metrics(targets, model.predict(features))
    metrics.update(probability_curves(targets, _positive_scores(model, features)))
    return metrics


def _importance_summary(rows):
    columns = list(zip(*rows))
    return {
        "mean": [statistics.mean(column) for column in columns],
        "std": [statistics.stdev(column) if len(column) > 1 else 0.0 for column in columns],
    }


def _synthetic_runtime(seed):
    from Models.cart_decision_tree import CARTClassifierScratch
    from Models.cart_decision_tree_optimized import OptimizedCARTClassifierScratch

    generator = random.Random(seed)
    sizes = [60, 120, 240, 480]
    base_ms, optimized_ms = [], []
    for size in sizes:
        x = [[generator.uniform(-1, 1) for _ in range(10)] for _ in range(size)]
        y = [int(row[0] + 0.7 * row[1] - 0.35 * row[2] + generator.gauss(0, 0.18) > 0) for row in x]
        started = time.perf_counter()
        CARTClassifierScratch(max_depth=4, min_samples_split=5).fit(x, y)
        base_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        OptimizedCARTClassifierScratch(max_depth=4, min_samples_split=5, random_state=seed).fit(x, y)
        optimized_ms.append((time.perf_counter() - started) * 1000)
    return {"sizes": sizes, "base_ms": base_ms, "optimized_ms": optimized_ms}


def _boundary_report(train_x, train_y, base_parameters, optimized_parameters, feature_indices, seed):
    from Models.cart_decision_tree import CARTClassifierScratch
    from Models.cart_decision_tree_optimized import OptimizedCARTClassifierScratch

    two_features = [[row[index] for index in feature_indices] for row in train_x]
    base = CARTClassifierScratch(**base_parameters).fit(two_features, train_y)
    optimized = OptimizedCARTClassifierScratch(**optimized_parameters).fit(two_features, train_y)
    x_column = [row[0] for row in two_features]
    y_column = [row[1] for row in two_features]
    x_pad = (max(x_column) - min(x_column)) * 0.05
    y_pad = (max(y_column) - min(y_column)) * 0.05
    grid_size = 90
    x_values = [min(x_column) - x_pad + i * (max(x_column) - min(x_column) + 2 * x_pad) / (grid_size - 1) for i in range(grid_size)]
    y_values = [min(y_column) - y_pad + i * (max(y_column) - min(y_column) + 2 * y_pad) / (grid_size - 1) for i in range(grid_size)]
    grid = [[x_value, y_value] for y_value in y_values for x_value in x_values]

    def matrix(model):
        flat = model.predict(grid)
        return [flat[i * grid_size : (i + 1) * grid_size] for i in range(grid_size)]

    return {
        "x_values": x_values,
        "y_values": y_values,
        "Base": matrix(base),
        "Optimized": matrix(optimized),
        "train_x": two_features,
        "train_y": train_y,
        "feature_names": [WDBC_FEATURE_NAMES[index] for index in feature_indices],
    }


def run_experiment(data_path, seed=42, folds=5):
    """Run leakage-free CART selection and assemble all six figure data blocks."""
    from Models.cart_decision_tree import CARTClassifierScratch
    from Models.cart_decision_tree_optimized import OptimizedCARTClassifierScratch

    features, targets = load_wdbc(data_path)
    train_x, test_x, train_y, test_y = stratified_split(features, targets, seed=seed)

    depth_candidates = [2, 3, 4, 5, 6]
    depth_evaluations = [
        _cv_evaluate(train_x, train_y, folds, seed, {"max_depth": depth, "random_state": seed})
        for depth in depth_candidates
    ]
    depth = depth_candidates[_select_best(depth_candidates, depth_evaluations, lambda value: value)]
    leaf_candidates = [1, 3, 5, 8]
    leaf_evaluations = [
        _cv_evaluate(train_x, train_y, folds, seed, {"max_depth": depth, "min_samples_leaf": leaf, "random_state": seed})
        for leaf in leaf_candidates
    ]
    leaf = leaf_candidates[_select_best(leaf_candidates, leaf_evaluations, lambda value: -value)]
    weight_candidates = [None, "balanced"]
    weight_evaluations = [
        _cv_evaluate(train_x, train_y, folds, seed, {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight, "random_state": seed})
        for weight in weight_candidates
    ]
    weight = weight_candidates[_select_best(weight_candidates, weight_evaluations)]
    alpha_candidates = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
    alpha_evaluations = [
        _cv_evaluate(
            train_x, train_y, folds, seed,
            {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight, "ccp_alpha": alpha, "random_state": seed},
        )
        for alpha in alpha_candidates
    ]
    alpha = alpha_candidates[_select_best(alpha_candidates, alpha_evaluations, lambda value: -value)]

    base_parameters = {"max_depth": depth, "min_samples_split": 2}
    optimized_parameters = {
        "max_depth": depth,
        "min_samples_split": 2,
        "min_samples_leaf": leaf,
        "class_weight": weight,
        "ccp_alpha": alpha,
        "random_state": seed,
    }
    base = CARTClassifierScratch(**base_parameters).fit(train_x, train_y)
    optimized = OptimizedCARTClassifierScratch(**optimized_parameters).fit(train_x, train_y)

    root_index = base.root.feature_index
    curve = split_gain_curve([row[root_index] for row in train_x], train_y)
    curve.update(
        {
            "feature_name": WDBC_FEATURE_NAMES[root_index],
            "values": {
                "Benign": [row[root_index] for row, target in zip(train_x, train_y) if target == 0],
                "Malignant": [row[root_index] for row, target in zip(train_x, train_y) if target == 1],
            },
        }
    )

    base_cv_parameters = {"max_depth": depth, "random_state": seed}
    base_cv = _cv_evaluate(train_x, train_y, folds, seed, base_cv_parameters, keep_importance=True)
    optimized_cv = _cv_evaluate(train_x, train_y, folds, seed, optimized_parameters, keep_importance=True)
    base_importance = _importance_summary(base_cv["importances"])
    optimized_importance = _importance_summary(optimized_cv["importances"])
    top_indices = sorted(range(len(WDBC_FEATURE_NAMES)), key=lambda i: optimized_importance["mean"][i], reverse=True)[:8]
    boundary_indices = top_indices[:2]

    pre_parameters = {"max_depth": depth, "min_samples_leaf": leaf, "random_state": seed}
    weighted_parameters = {**pre_parameters, "class_weight": weight}
    pre_eval = _cv_evaluate(train_x, train_y, folds, seed, pre_parameters)
    weighted_eval = _cv_evaluate(train_x, train_y, folds, seed, weighted_parameters)
    ablation = [
        {"name": "Base config", "balanced_accuracy": base_cv["score_mean"], "leaves": base_cv["leaves_mean"]},
        {"name": "+ Fast scan", "balanced_accuracy": base_cv["score_mean"], "leaves": base_cv["leaves_mean"]},
        {"name": "+ Pre-pruning", "balanced_accuracy": pre_eval["score_mean"], "leaves": pre_eval["leaves_mean"]},
        {"name": "+ Class weights", "balanced_accuracy": weighted_eval["score_mean"], "leaves": weighted_eval["leaves_mean"]},
        {"name": "+ CCP pruning", "balanced_accuracy": optimized_cv["score_mean"], "leaves": optimized_cv["leaves_mean"]},
    ]
    return {
        "split_gain": curve,
        "trees": {
            "Base": _tree_dict(base.root, base.classes),
            "Optimized": _tree_dict(optimized.root, optimized.classes),
            "metadata": {
                "Base": {"depth": base.tree_depth_, "leaves": base.n_leaves_},
                "Optimized": {"depth": optimized.tree_depth_, "leaves": optimized.n_leaves_},
            },
            "feature_names": WDBC_FEATURE_NAMES,
        },
        "boundaries": _boundary_report(train_x, train_y, base_parameters, optimized_parameters, boundary_indices, seed),
        "pruning": {
            "alphas": alpha_candidates,
            "score_mean": [item["score_mean"] for item in alpha_evaluations],
            "score_std": [item["score_std"] for item in alpha_evaluations],
            "leaves_mean": [item["leaves_mean"] for item in alpha_evaluations],
            "depth_mean": [item["depth_mean"] for item in alpha_evaluations],
            "best_alpha": alpha,
        },
        "diagnostics": {
            "Base": _diagnostic_record(base, test_x, test_y),
            "Optimized": _diagnostic_record(optimized, test_x, test_y),
        },
        "optimization": {
            "runtime": _synthetic_runtime(seed),
            "ablation": ablation,
            "importance": {
                "feature_names": [WDBC_FEATURE_NAMES[i] for i in top_indices],
                "Base": {"mean": [base_importance["mean"][i] for i in top_indices], "std": [base_importance["std"][i] for i in top_indices]},
                "Optimized": {"mean": [optimized_importance["mean"][i] for i in top_indices], "std": [optimized_importance["std"][i] for i in top_indices]},
            },
        },
    }


def build_parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate six CART decision-tree PNG figures.")
    parser.add_argument("--data", type=Path, default=project_root / "data" / "classification" / "wdbc" / "wdbc.data")
    parser.add_argument("--output", type=Path, default=project_root / "figures" / "cart_decision_tree")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    report = run_experiment(arguments.data, seed=arguments.seed, folds=arguments.folds)
    paths = render_six_figures(report, arguments.output)
    print(f"Generated {len(paths)} PNG figures in {arguments.output}")


if __name__ == "__main__":
    main()
