"""Generate six PNG figures for the hand-written K-Means models."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

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
    "legend.frameon": False,
    "savefig.dpi": 300,
}
matplotlib.rcParams.update(PLOT_STYLE)
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.kmeans import KMeansScratch
from Models.kmeans_optimized import OptimizedKMeansScratch


FIGURE_FILENAMES = (
    "01_cluster_distribution.png",
    "02_centroid_trajectory.png",
    "03_convergence_comparison.png",
    "04_cluster_number_selection.png",
    "05_initialization_stability.png",
    "06_benchmark_comparison.png",
)

FIGURE_PALETTES = {
    FIGURE_FILENAMES[0]: ("#264653", "#2A9D8F", "#E9C46A", "#E76F51", "#F4F1DE"),
    FIGURE_FILENAMES[1]: ("#5E548E", "#9F86C0", "#BE95C4", "#E0B1CB", "#231942"),
    FIGURE_FILENAMES[2]: ("#003049", "#D62828", "#F77F00", "#FCBF49", "#EAE2B7"),
    FIGURE_FILENAMES[3]: ("#3D405B", "#81B29A", "#E07A5F", "#F2CC8F", "#F4F1DE"),
    FIGURE_FILENAMES[4]: ("#006D77", "#83C5BE", "#E29578", "#FFDDD2", "#283D3B"),
    FIGURE_FILENAMES[5]: ("#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"),
}

FEATURE_NAMES = (
    "Area",
    "Perimeter",
    "Compactness",
    "Kernel length",
    "Kernel width",
    "Asymmetry coefficient",
    "Kernel groove length",
)


def load_seeds(path):
    """Load the seven measurements and hold out the last column as reference labels."""
    features, labels = [], []
    with Path(path).open("r", encoding="utf-8-sig") as source:
        for line in source:
            fields = line.split()
            if not fields:
                continue
            features.append([float(value) for value in fields[:7]])
            labels.append(int(fields[7]))
    return features, labels


def standardize_features(X):
    sample_count = len(X)
    feature_count = len(X[0])
    means = [sum(row[index] for row in X) / sample_count for index in range(feature_count)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in X) / sample_count
        scale = math.sqrt(variance)
        scales.append(scale if scale > 0.0 else 1.0)
    transformed = [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in X
    ]
    return transformed, means, scales


def _euclidean(first, second):
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _pairwise_distances(X):
    matrix = [[0.0] * len(X) for _ in X]
    for first in range(len(X)):
        for second in range(first + 1, len(X)):
            distance = _euclidean(X[first], X[second])
            matrix[first][second] = distance
            matrix[second][first] = distance
    return matrix


def silhouette_score(X, labels, distance_matrix=None):
    """Mean silhouette coefficient using Euclidean distance."""
    distances = distance_matrix if distance_matrix is not None else _pairwise_distances(X)
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    if len(groups) < 2:
        return 0.0

    scores = []
    for index, label in enumerate(labels):
        own = groups[label]
        if len(own) == 1:
            scores.append(0.0)
            continue
        a = sum(distances[index][other] for other in own if other != index) / (len(own) - 1)
        b = min(
            sum(distances[index][other] for other in members) / len(members)
            for other_label, members in groups.items()
            if other_label != label
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0.0 else 0.0)
    return sum(scores) / len(scores)


def purity_score(true_labels, cluster_labels):
    groups = {}
    for truth, cluster in zip(true_labels, cluster_labels):
        groups.setdefault(cluster, []).append(truth)
    correct = sum(max(Counter(group).values()) for group in groups.values())
    return correct / len(true_labels)


def _combination_two(value):
    return value * (value - 1) / 2.0


def _contingency(true_labels, cluster_labels):
    truth_values = sorted(set(true_labels))
    cluster_values = sorted(set(cluster_labels))
    truth_index = {value: index for index, value in enumerate(truth_values)}
    cluster_index = {value: index for index, value in enumerate(cluster_values)}
    table = [[0] * len(cluster_values) for _ in truth_values]
    for truth, cluster in zip(true_labels, cluster_labels):
        table[truth_index[truth]][cluster_index[cluster]] += 1
    return table


def adjusted_rand_index(true_labels, cluster_labels):
    table = _contingency(true_labels, cluster_labels)
    row_totals = [sum(row) for row in table]
    column_totals = [sum(table[row][column] for row in range(len(table))) for column in range(len(table[0]))]
    pairs = _combination_two(len(true_labels))
    index = sum(_combination_two(value) for row in table for value in row)
    row_pairs = sum(_combination_two(value) for value in row_totals)
    column_pairs = sum(_combination_two(value) for value in column_totals)
    expected = row_pairs * column_pairs / pairs if pairs else 0.0
    maximum = 0.5 * (row_pairs + column_pairs)
    denominator = maximum - expected
    if denominator == 0.0:
        equivalent = all(
            (true_labels[first] == true_labels[second])
            == (cluster_labels[first] == cluster_labels[second])
            for first in range(len(true_labels))
            for second in range(first + 1, len(true_labels))
        )
        return 1.0 if equivalent else 0.0
    return (index - expected) / denominator


def normalized_mutual_information(true_labels, cluster_labels):
    table = _contingency(true_labels, cluster_labels)
    total = len(true_labels)
    row_totals = [sum(row) for row in table]
    column_totals = [sum(table[row][column] for row in range(len(table))) for column in range(len(table[0]))]
    mutual_information = 0.0
    for row, row_total in zip(table, row_totals):
        for value, column_total in zip(row, column_totals):
            if value:
                mutual_information += value / total * math.log(value * total / (row_total * column_total))
    truth_entropy = -sum(value / total * math.log(value / total) for value in row_totals if value)
    cluster_entropy = -sum(value / total * math.log(value / total) for value in column_totals if value)
    if truth_entropy == 0.0 or cluster_entropy == 0.0:
        return 1.0 if truth_entropy == 0.0 and cluster_entropy == 0.0 else 0.0
    denominator = math.sqrt(truth_entropy * cluster_entropy)
    return mutual_information / denominator


def mean_and_std(values):
    return statistics.mean(values), statistics.pstdev(values)


def trace_lloyd_centers(X, n_clusters=3, max_iter=100, seed=42):
    """Replay the basic Lloyd loop and retain the center positions for teaching."""
    model = KMeansScratch(n_clusters=n_clusters, max_iter=max_iter, random_state=seed)
    generator = random.Random(seed)
    centers = [list(X[index]) for index in generator.sample(range(len(X)), n_clusters)]
    center_history = [[list(center) for center in centers]]
    inertia_history = []
    previous_labels = None
    for _ in range(max_iter):
        labels = model._assign(X, centers)
        centers = model._recompute_centers(X, labels, centers)
        labels = model._assign(X, centers)
        inertia_history.append(model._inertia(X, labels, centers))
        center_history.append([list(center) for center in centers])
        if labels == previous_labels:
            break
        previous_labels = labels
    return {
        "centers": center_history,
        "labels": labels,
        "inertia_history": inertia_history,
    }


def _original_value(value, feature, means, scales):
    return value * scales[feature] + means[feature]


def _model_metrics(X, truth, predicted, distance_matrix=None):
    return {
        "silhouette": silhouette_score(X, predicted, distance_matrix),
        "purity": purity_score(truth, predicted),
        "ari": adjusted_rand_index(truth, predicted),
        "nmi": normalized_mutual_information(truth, predicted),
    }


def _fit_timed(model, X):
    start = time.perf_counter()
    model.fit(X)
    return (time.perf_counter() - start) * 1000.0


def run_experiment(
    data_path,
    seed=42,
    n_clusters=3,
    k_values=range(2, 9),
    stability_runs=30,
    benchmark_runs=10,
):
    features, truth = load_seeds(data_path)
    scaled, means, scales = standardize_features(features)
    distance_matrix = _pairwise_distances(scaled)
    x_feature, y_feature = 0, 6

    base = KMeansScratch(n_clusters=n_clusters, max_iter=200, random_state=seed).fit(scaled)
    optimized = OptimizedKMeansScratch(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=200,
        tol=1e-5,
        standardize=False,
        random_state=seed,
    ).fit(scaled)
    optimized_single = OptimizedKMeansScratch(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=1,
        max_iter=200,
        tol=1e-5,
        standardize=False,
        random_state=seed,
    ).fit(scaled)

    cluster_centers = [
        [
            _original_value(center[x_feature], x_feature, means, scales),
            _original_value(center[y_feature], y_feature, means, scales),
        ]
        for center in optimized.cluster_centers_
    ]
    cluster_map = {
        "x": [row[x_feature] for row in features],
        "y": [row[y_feature] for row in features],
        "labels": list(optimized.labels_),
        "centers": cluster_centers,
        "feature_names": [FEATURE_NAMES[x_feature], FEATURE_NAMES[y_feature]],
        "inertia": optimized.inertia_,
        "silhouette": silhouette_score(scaled, optimized.labels_, distance_matrix),
    }

    trace = trace_lloyd_centers(scaled, n_clusters=n_clusters, max_iter=200, seed=seed)
    paths = []
    for cluster in range(n_clusters):
        paths.append(
            [
                [
                    _original_value(centers[cluster][x_feature], x_feature, means, scales),
                    _original_value(centers[cluster][y_feature], y_feature, means, scales),
                ]
                for centers in trace["centers"]
            ]
        )
    trajectory = {
        "x": [row[x_feature] for row in features],
        "y": [row[y_feature] for row in features],
        "labels": trace["labels"],
        "feature_names": [FEATURE_NAMES[x_feature], FEATURE_NAMES[y_feature]],
        "paths": paths,
    }

    selection = {"k_values": [], "inertia": [], "silhouette": []}
    for k in k_values:
        model = OptimizedKMeansScratch(
            n_clusters=k,
            init="k-means++",
            n_init=8,
            max_iter=200,
            tol=1e-5,
            standardize=False,
            random_state=seed,
        ).fit(scaled)
        selection["k_values"].append(k)
        selection["inertia"].append(model.inertia_)
        selection["silhouette"].append(silhouette_score(scaled, model.labels_, distance_matrix))
    best_index = max(range(len(selection["k_values"])), key=lambda index: selection["silhouette"][index])
    selection["selected_k"] = selection["k_values"][best_index]
    selection["reference_k"] = len(set(truth))

    stability_methods = ("Random", "K-Means++", "K-Means++ (n_init=10)")
    stability = {"methods": list(stability_methods), "values": {name: [] for name in stability_methods}}
    for run_seed in range(stability_runs):
        settings = (
            ("Random", "random", 1),
            ("K-Means++", "k-means++", 1),
            ("K-Means++ (n_init=10)", "k-means++", 10),
        )
        for name, init, n_init in settings:
            model = OptimizedKMeansScratch(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=200,
                tol=1e-5,
                standardize=False,
                random_state=run_seed,
            ).fit(scaled)
            stability["values"][name].append(model.inertia_)

    benchmark_methods = ("Base K-Means", "Optimized K-Means")
    benchmark = {"methods": list(benchmark_methods), "runs": {name: [] for name in benchmark_methods}}
    for run_seed in range(benchmark_runs):
        models = (
            ("Base K-Means", KMeansScratch(n_clusters=n_clusters, max_iter=200, random_state=run_seed)),
            (
                "Optimized K-Means",
                OptimizedKMeansScratch(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=10,
                    max_iter=200,
                    tol=1e-5,
                    standardize=False,
                    random_state=run_seed,
                ),
            ),
        )
        for name, model in models:
            fit_ms = _fit_timed(model, scaled)
            record = {
                "seed": run_seed,
                **_model_metrics(scaled, truth, model.labels_, distance_matrix),
                "fit_ms": fit_ms,
            }
            benchmark["runs"][name].append(record)

    return {
        "metadata": {
            "sample_count": len(features),
            "feature_count": len(features[0]),
            "n_clusters": n_clusters,
            "reference_classes": len(set(truth)),
            "seed": seed,
            "stability_runs": stability_runs,
            "benchmark_runs": benchmark_runs,
        },
        "cluster_map": cluster_map,
        "trajectory": trajectory,
        "convergence": {
            "base_history": list(base.inertia_history_),
            "optimized_history": list(optimized_single.inertia_history_),
            "base_label": "Random initialization (1 start)",
            "optimized_label": "K-Means++ (1 start)",
        },
        "selection": selection,
        "stability": stability,
        "benchmark": benchmark,
    }


def _style_axis(axis, grid_axis="y"):
    axis.grid(axis=grid_axis, color="#E6E6E6", linewidth=0.55, zorder=0)
    axis.tick_params(length=3, width=0.7)
    axis.set_axisbelow(True)


def _save_png(figure, output_dir, filename):
    path = Path(output_dir) / filename
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _write_csv(path, header, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(header)
        writer.writerows(rows)


def _plot_cluster_distribution(report):
    item = report["cluster_map"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[0]]
    figure, axis = plt.subplots(figsize=(6.3, 4.4), layout="constrained")
    for cluster in sorted(set(item["labels"])):
        indices = [index for index, label in enumerate(item["labels"]) if label == cluster]
        axis.scatter(
            [item["x"][index] for index in indices],
            [item["y"][index] for index in indices],
            s=30,
            color=palette[cluster % 3],
            alpha=0.72,
            edgecolors="white",
            linewidth=0.35,
            label=f"Cluster {cluster + 1}",
        )
    axis.scatter(
        [center[0] for center in item["centers"]],
        [center[1] for center in item["centers"]],
        marker="*",
        s=190,
        color=palette[3],
        edgecolors="#2B2B2B",
        linewidth=0.8,
        label="Centroid",
        zorder=5,
    )
    axis.text(
        0.02,
        0.98,
        (
            f"SSE = {item['inertia']:.2f}\n"
            f"Silhouette = {item['silhouette']:.3f}\n"
            "Fit: all 7 standardized features\n"
            "View: 2 original-scale features"
        ),
        transform=axis.transAxes,
        va="top",
        color=palette[0],
    )
    axis.set(
        xlabel=item["feature_names"][0],
        ylabel=item["feature_names"][1],
        title="K-Means clustering of the Seeds data",
    )
    _style_axis(axis)
    axis.legend(ncols=2, loc="lower right")
    return figure


def _plot_centroid_trajectory(report):
    item = report["trajectory"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[1]]
    figure, axis = plt.subplots(figsize=(6.3, 4.4), layout="constrained")
    axis.scatter(item["x"], item["y"], s=18, color=palette[3], alpha=0.38, edgecolors="none")
    for cluster, path in enumerate(item["paths"]):
        color = palette[cluster]
        xs = [point[0] for point in path]
        ys = [point[1] for point in path]
        axis.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=4, alpha=0.92)
        axis.scatter(xs[0], ys[0], marker="X", s=78, color=color, edgecolors="white", linewidth=0.6, zorder=4)
        axis.scatter(xs[-1], ys[-1], marker="*", s=155, color=color, edgecolors=palette[4], linewidth=0.7, zorder=5)
        axis.text(xs[-1], ys[-1], f"  C{cluster + 1}", color=color, va="center", fontweight="bold")
    axis.scatter([], [], marker="X", s=60, color=palette[4], label="Initial centroid")
    axis.scatter([], [], marker="*", s=100, color=palette[4], label="Final centroid")
    axis.set(
        xlabel=item["feature_names"][0],
        ylabel=item["feature_names"][1],
        title="Centroid movement during Lloyd iteration",
    )
    axis.text(
        0.98,
        0.03,
        "Fit: 7 standardized features; view: 2 original-scale features",
        transform=axis.transAxes,
        ha="right",
        color="#666666",
        fontsize=8,
    )
    _style_axis(axis)
    axis.legend(loc="best")
    return figure


def _plot_convergence(report):
    item = report["convergence"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[2]]
    figure, axis = plt.subplots(figsize=(6.3, 4.1), layout="constrained")
    for history, label, color, marker in (
        (item["base_history"], item["base_label"], palette[0], "o"),
        (item["optimized_history"], item["optimized_label"], palette[1], "s"),
    ):
        iterations = list(range(1, len(history) + 1))
        axis.plot(iterations, history, color=color, marker=marker, markersize=4, linewidth=1.9, label=label)
        axis.scatter(iterations[-1], history[-1], color=color, s=38, zorder=4)
        axis.text(iterations[-1], history[-1], f"  {history[-1]:.2f}", color=color, va="center")
    axis.set(xlabel="Iteration", ylabel="Within-cluster SSE (standardized space)", title="Single-start objective paths on the same data")
    axis.xaxis.get_major_locator().set_params(integer=True)
    _style_axis(axis)
    axis.legend()
    return figure


def _plot_cluster_number_selection(report):
    item = report["selection"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[3]]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), layout="constrained")
    axes[0].plot(item["k_values"], item["inertia"], color=palette[0], marker="o", linewidth=1.8)
    reference_k = item["reference_k"]
    axes[0].axvline(reference_k, color=palette[2], linestyle="--", linewidth=1.2, label=f"Reference K = {reference_k}")
    axes[0].set(xlabel="Number of clusters, K", ylabel="Within-cluster SSE", title="Elbow criterion")
    axes[0].legend()
    axes[1].plot(item["k_values"], item["silhouette"], color=palette[1], marker="s", linewidth=1.8)
    chosen = item["selected_k"]
    chosen_index = item["k_values"].index(chosen)
    axes[1].scatter([chosen], [item["silhouette"][chosen_index]], s=62, color=palette[2], zorder=4, label=f"Maximum at K = {chosen}")
    axes[1].axvline(reference_k, color=palette[3], linestyle="--", linewidth=1.0, label=f"Reference K = {reference_k}")
    axes[1].set(xlabel="Number of clusters, K", ylabel="Mean silhouette coefficient", title="Separation criterion")
    axes[1].legend()
    for axis in axes:
        axis.xaxis.get_major_locator().set_params(integer=True)
        _style_axis(axis)
    return figure


def _jitter_offsets(count, width=0.15):
    if count <= 1:
        return [0.0]
    return [width * (2.0 * index / (count - 1) - 1.0) for index in range(count)]


def _plot_initialization_stability(report):
    item = report["stability"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[4]]
    figure, axis = plt.subplots(figsize=(7.1, 4.2), layout="constrained")
    raw_values = [item["values"][method] for method in item["methods"]]
    best_sse = min(value for samples in raw_values for value in samples)
    values = [[value - best_sse for value in samples] for samples in raw_values]
    positions = list(range(1, len(values) + 1))
    boxes = axis.boxplot(values, positions=positions, widths=0.48, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.2})
    for box, color in zip(boxes["boxes"], palette[:3]):
        box.set_facecolor(color)
        box.set_alpha(0.55)
        box.set_edgecolor(color)
    for position, samples, color in zip(positions, values, palette[:3]):
        offsets = _jitter_offsets(len(samples))
        axis.scatter([position + offset for offset in offsets], samples, s=18, color=color, alpha=0.70, edgecolors="white", linewidth=0.25)
        mean, _ = mean_and_std(samples)
        axis.scatter([position], [mean], marker="D", s=42, color=palette[4], zorder=5)
    axis.scatter([], [], marker="D", s=36, color=palette[4], label="Mean SSE")
    display_names = ["Random", "K-Means++", "K-Means++\n(10 starts)"]
    axis.set_xticks(positions, display_names)
    axis.set(ylabel="Excess SSE above the best observed result", title="Initialization robustness across random seeds")
    _style_axis(axis)
    axis.legend(loc="upper right")
    return figure


def _plot_benchmark(report):
    item = report["benchmark"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[5]]
    methods = item["methods"]
    colors = (palette[0], palette[3])
    metrics = ("silhouette", "purity", "ari", "nmi")
    labels = ("Silhouette", "Purity", "ARI", "NMI")
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), layout="constrained", gridspec_kw={"width_ratios": [1.7, 0.8]})
    positions = list(range(len(metrics)))
    width = 0.34
    for model_index, (method, color) in enumerate(zip(methods, colors)):
        means, errors = [], []
        for metric in metrics:
            mean, std = mean_and_std([run[metric] for run in item["runs"][method]])
            means.append(mean)
            errors.append(std)
        offsets = [position + (model_index - 0.5) * width for position in positions]
        axes[0].bar(offsets, means, width=width, yerr=errors, capsize=3, color=color, alpha=0.86, label=method, edgecolor="white", linewidth=0.6)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set(ylabel="Score (mean ± SD)", title="Clustering quality across seeds")
    axes[0].legend(loc="lower right")

    runtime_means, runtime_errors = [], []
    for method in methods:
        mean, std = mean_and_std([run["fit_ms"] for run in item["runs"][method]])
        runtime_means.append(mean)
        runtime_errors.append(std)
    bars = axes[1].bar(range(len(methods)), runtime_means, yerr=runtime_errors, capsize=3, color=colors, alpha=0.86)
    axes[1].set_xticks(range(len(methods)), ["Base", "Optimized"])
    axes[1].set(ylabel="Fit time (ms; mean ± SD)", title="Computational cost")
    top = max(value + error for value, error in zip(runtime_means, runtime_errors))
    margin = 0.035 * top if top else 0.1
    axes[1].set_ylim(0.0, top + 3.0 * margin)
    for bar, value, error in zip(bars, runtime_means, runtime_errors):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + error + margin,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for axis in axes:
        _style_axis(axis)
    return figure


def render_six_figures(report, output_dir):
    """Export six independent 300-DPI PNG figures and their source tables."""
    output_dir = Path(output_dir)
    source_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    plotters = (
        _plot_cluster_distribution,
        _plot_centroid_trajectory,
        _plot_convergence,
        _plot_cluster_number_selection,
        _plot_initialization_stability,
        _plot_benchmark,
    )
    paths = [
        _save_png(plotter(report), output_dir, filename)
        for filename, plotter in zip(FIGURE_FILENAMES, plotters)
    ]

    cluster_map = report["cluster_map"]
    metadata = report["metadata"]
    cluster_rows = [
        (
            "sample",
            index,
            x,
            y,
            label,
            cluster_map["feature_names"][0],
            cluster_map["feature_names"][1],
            "7-feature standardized",
            "2-feature original scale",
            cluster_map["inertia"],
            cluster_map["silhouette"],
            metadata["n_clusters"],
            metadata["seed"],
        )
        for index, (x, y, label) in enumerate(zip(cluster_map["x"], cluster_map["y"], cluster_map["labels"]))
    ]
    cluster_rows.extend(
        (
            "centroid",
            index,
            center[0],
            center[1],
            index,
            cluster_map["feature_names"][0],
            cluster_map["feature_names"][1],
            "7-feature standardized",
            "2-feature original scale",
            cluster_map["inertia"],
            cluster_map["silhouette"],
            metadata["n_clusters"],
            metadata["seed"],
        )
        for index, center in enumerate(cluster_map["centers"])
    )
    _write_csv(
        source_dir / "01_cluster_distribution.csv",
        [
            "type",
            "index",
            "x",
            "y",
            "cluster",
            "feature_x",
            "feature_y",
            "clustering_space",
            "display_space",
            "sse",
            "silhouette",
            "n_clusters",
            "seed",
        ],
        cluster_rows,
    )

    trajectory = report["trajectory"]
    trajectory_rows = [
        (
            "sample",
            "",
            label,
            x,
            y,
            trajectory["feature_names"][0],
            trajectory["feature_names"][1],
            "7-feature standardized",
            "2-feature original scale",
            metadata["n_clusters"],
            metadata["seed"],
        )
        for x, y, label in zip(trajectory["x"], trajectory["y"], trajectory["labels"])
    ]
    for cluster, path in enumerate(trajectory["paths"]):
        trajectory_rows.extend(
            (
                "centroid",
                iteration,
                cluster,
                point[0],
                point[1],
                trajectory["feature_names"][0],
                trajectory["feature_names"][1],
                "7-feature standardized",
                "2-feature original scale",
                metadata["n_clusters"],
                metadata["seed"],
            )
            for iteration, point in enumerate(path)
        )
    _write_csv(
        source_dir / "02_centroid_trajectory.csv",
        [
            "type",
            "iteration",
            "cluster",
            "x",
            "y",
            "feature_x",
            "feature_y",
            "clustering_space",
            "display_space",
            "n_clusters",
            "seed",
        ],
        trajectory_rows,
    )

    convergence_rows = []
    for model_name, history in (
        (report["convergence"]["base_label"], report["convergence"]["base_history"]),
        (report["convergence"]["optimized_label"], report["convergence"]["optimized_history"]),
    ):
        convergence_rows.extend((model_name, iteration, value) for iteration, value in enumerate(history, start=1))
    _write_csv(source_dir / "03_convergence_comparison.csv", ["model", "iteration", "sse"], convergence_rows)

    selection = report["selection"]
    _write_csv(
        source_dir / "04_cluster_number_selection.csv",
        ["k", "sse", "silhouette", "selected_by_silhouette", "reference_k"],
        [
            (k, inertia, silhouette, k == selection["selected_k"], k == selection["reference_k"])
            for k, inertia, silhouette in zip(selection["k_values"], selection["inertia"], selection["silhouette"])
        ],
    )

    stability = report["stability"]
    stability_rows = []
    stability_best = min(
        value
        for method in stability["methods"]
        for value in stability["values"][method]
    )
    for method in stability["methods"]:
        stability_rows.extend(
            (method, seed, value, stability_best, value - stability_best)
            for seed, value in enumerate(stability["values"][method])
        )
    _write_csv(
        source_dir / "05_initialization_stability.csv",
        ["method", "seed", "sse", "best_sse", "excess_sse"],
        stability_rows,
    )

    benchmark = report["benchmark"]
    benchmark_rows = []
    for method in benchmark["methods"]:
        benchmark_rows.extend(
            (method, run["seed"], run["silhouette"], run["purity"], run["ari"], run["nmi"], run["fit_ms"])
            for run in benchmark["runs"][method]
        )
    _write_csv(
        source_dir / "06_benchmark_comparison.csv",
        ["model", "seed", "silhouette", "purity", "ari", "nmi", "fit_ms"],
        benchmark_rows,
    )
    return paths


def _print_summary(report, output_dir):
    selection = report["selection"]
    print(f"Output: {output_dir}")
    print(f"Figures: {len(FIGURE_FILENAMES)} PNG")
    print(f"Silhouette-selected K: {selection['selected_k']}")


def main():
    parser = argparse.ArgumentParser(description="Generate six K-Means experiment figures.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "clustering" / "seeds" / "seeds_dataset.txt",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figures" / "kmeans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--stability-runs", type=int, default=30)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    args = parser.parse_args()

    report = run_experiment(
        args.data,
        seed=args.seed,
        n_clusters=args.clusters,
        stability_runs=args.stability_runs,
        benchmark_runs=args.benchmark_runs,
    )
    render_six_figures(report, args.output)
    _print_summary(report, args.output)


if __name__ == "__main__":
    main()
