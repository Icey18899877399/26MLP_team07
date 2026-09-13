"""Generate six PNG figures for the hand-written DBSCAN models."""

from __future__ import annotations

import argparse
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
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 300,
}
matplotlib.rcParams.update(PLOT_STYLE)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.dbscan import DBSCANScratch
from Models.dbscan_optimized import OptimizedDBSCANScratch


FIGURE_FILENAMES = (
    "01_cluster_result.png",
    "02_density_mechanism.png",
    "03_k_distance_curve.png",
    "04_parameter_sensitivity.png",
    "05_standardization_comparison.png",
    "06_benchmark_comparison.png",
)

FIGURE_PALETTES = {
    FIGURE_FILENAMES[0]: ("#264653", "#2A9D8F", "#E9C46A", "#E76F51", "#4D4D4D"),
    FIGURE_FILENAMES[1]: ("#3C1642", "#7B2CBF", "#C77DFF", "#F1E3F3", "#4D4D4D"),
    FIGURE_FILENAMES[2]: ("#003049", "#669BBC", "#F77F00", "#FCBF49", "#EAE2B7"),
    FIGURE_FILENAMES[3]: ("#006D77", "#83C5BE", "#E29578", "#FFDDD2", "#283D3B"),
    FIGURE_FILENAMES[4]: ("#355070", "#6D597A", "#B56576", "#EAAC8B", "#4D4D4D"),
    FIGURE_FILENAMES[5]: ("#1D3557", "#457B9D", "#E63946", "#A8DADC", "#6C757D"),
}

SEED_FEATURE_NAMES = (
    "Area",
    "Perimeter",
    "Compactness",
    "Kernel length",
    "Kernel width",
    "Asymmetry coefficient",
    "Kernel groove length",
)


def generate_two_moons(
    samples_per_moon=120,
    noise_samples=30,
    jitter=0.045,
    random_state=42,
):
    """Create two curved clusters plus uniformly distributed noise."""
    if samples_per_moon < 2 or noise_samples < 0 or jitter < 0.0:
        raise ValueError("invalid synthetic-data parameters")
    generator = random.Random(random_state)
    X, labels = [], []
    for index in range(samples_per_moon):
        angle = math.pi * index / (samples_per_moon - 1)
        X.append(
            [
                math.cos(angle) + generator.gauss(0.0, jitter),
                math.sin(angle) + generator.gauss(0.0, jitter),
            ]
        )
        labels.append(0)
    for index in range(samples_per_moon):
        angle = math.pi * index / (samples_per_moon - 1)
        X.append(
            [
                1.0 - math.cos(angle) + generator.gauss(0.0, jitter),
                0.5 - math.sin(angle) + generator.gauss(0.0, jitter),
            ]
        )
        labels.append(1)
    for _ in range(noise_samples):
        X.append([generator.uniform(-1.4, 2.4), generator.uniform(-0.8, 1.3)])
        labels.append(-1)
    return X, labels


def load_seeds(path):
    """Load seven seed measurements and their reference varieties."""
    X, labels = [], []
    with Path(path).open("r", encoding="utf-8-sig") as source:
        for line in source:
            fields = line.split()
            if not fields:
                continue
            X.append([float(value) for value in fields[:7]])
            labels.append(int(fields[7]))
    if not X:
        raise ValueError("the Seeds dataset is empty")
    return X, labels


def standardize_features(X):
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
    transformed = [
        [
            (value - means[feature]) / scales[feature]
            for feature, value in enumerate(row)
        ]
        for row in X
    ]
    return transformed, means, scales


def _squared_distance(first, second):
    return sum((left - right) ** 2 for left, right in zip(first, second))


def _pairwise_distances(X):
    matrix = [[0.0] * len(X) for _ in X]
    for left in range(len(X)):
        for right in range(left + 1, len(X)):
            distance = math.sqrt(_squared_distance(X[left], X[right]))
            matrix[left][right] = distance
            matrix[right][left] = distance
    return matrix


def k_distances(X, min_samples=5):
    """Return each sample's DBSCAN k-distance, counting itself as neighbor one."""
    if min_samples < 1 or min_samples > len(X):
        raise ValueError("min_samples must be between 1 and the sample count")
    matrix = _pairwise_distances(X)
    position = min_samples - 1
    return [sorted(row)[position] for row in matrix]


def find_knee(sorted_distances):
    """Find the point furthest below the line joining curve endpoints."""
    values = list(sorted_distances)
    if not values:
        raise ValueError("sorted_distances must not be empty")
    if len(values) < 3 or values[-1] == values[0]:
        index = len(values) - 1
        return index, values[index]

    span = values[-1] - values[0]
    last = len(values) - 1
    departures = []
    for index, value in enumerate(values):
        x = index / last
        y = (value - values[0]) / span
        departures.append(x - y)
    knee_index = max(range(len(values)), key=lambda index: departures[index])
    return knee_index, values[knee_index]


def _combination_two(value):
    return value * (value - 1) / 2.0


def _contingency(truth, predicted):
    truth_counts = Counter(truth)
    predicted_counts = Counter(predicted)
    joint = Counter(zip(truth, predicted))
    return truth_counts, predicted_counts, joint


def adjusted_rand_index(truth, predicted):
    truth_counts, predicted_counts, joint = _contingency(truth, predicted)
    sample_count = len(truth)
    if sample_count < 2:
        return 1.0
    total_pairs = _combination_two(sample_count)
    joint_pairs = sum(_combination_two(value) for value in joint.values())
    truth_pairs = sum(_combination_two(value) for value in truth_counts.values())
    predicted_pairs = sum(
        _combination_two(value) for value in predicted_counts.values()
    )
    expected = truth_pairs * predicted_pairs / total_pairs
    denominator = 0.5 * (truth_pairs + predicted_pairs) - expected
    if denominator == 0.0:
        return 1.0
    return max(-1.0, min(1.0, (joint_pairs - expected) / denominator))


def normalized_mutual_information(truth, predicted):
    truth_counts, predicted_counts, joint = _contingency(truth, predicted)
    sample_count = len(truth)
    if sample_count == 0:
        return 0.0
    mutual_information = 0.0
    for (truth_label, predicted_label), count in joint.items():
        mutual_information += count / sample_count * math.log(
            count * sample_count
            / (truth_counts[truth_label] * predicted_counts[predicted_label])
        )
    truth_entropy = -sum(
        count / sample_count * math.log(count / sample_count)
        for count in truth_counts.values()
    )
    predicted_entropy = -sum(
        count / sample_count * math.log(count / sample_count)
        for count in predicted_counts.values()
    )
    denominator = math.sqrt(truth_entropy * predicted_entropy)
    if denominator == 0.0:
        return 1.0 if truth_entropy == predicted_entropy else 0.0
    return max(0.0, min(1.0, mutual_information / denominator))


def silhouette_score(X, labels):
    """Compute silhouette on clustered samples; DBSCAN noise is excluded."""
    kept = [index for index, label in enumerate(labels) if label != -1]
    clusters = sorted({labels[index] for index in kept})
    if len(kept) < 2 or len(clusters) < 2:
        return 0.0
    compact_X = [X[index] for index in kept]
    compact_labels = [labels[index] for index in kept]
    distances = _pairwise_distances(compact_X)
    values = []
    for index, label in enumerate(compact_labels):
        same = [
            other
            for other, other_label in enumerate(compact_labels)
            if other_label == label and other != index
        ]
        if not same:
            values.append(0.0)
            continue
        within = sum(distances[index][other] for other in same) / len(same)
        between = min(
            sum(
                distances[index][other]
                for other, other_label in enumerate(compact_labels)
                if other_label == cluster
            )
            / compact_labels.count(cluster)
            for cluster in clusters
            if cluster != label
        )
        scale = max(within, between)
        values.append((between - within) / scale if scale > 0.0 else 0.0)
    return sum(values) / len(values)


def coverage_score(labels):
    return sum(label != -1 for label in labels) / len(labels) if labels else 0.0


def select_parameters(
    eps_values,
    min_samples_values,
    cluster_counts,
    silhouette_scores,
    noise_rates,
):
    """Select a non-degenerate grid cell using Silhouette × Coverage only."""
    best = None
    for row, min_samples in enumerate(min_samples_values):
        for column, eps in enumerate(eps_values):
            if cluster_counts[row][column] < 2:
                continue
            coverage = 1.0 - noise_rates[row][column]
            score = silhouette_scores[row][column] * coverage
            rank = (score, -eps, -min_samples)
            if best is None or rank > best[0]:
                best = (rank, eps, min_samples)
    return None if best is None else (best[1], best[2])


def _cluster_metrics(X, truth, labels):
    return {
        "ARI": adjusted_rand_index(truth, labels),
        "NMI": normalized_mutual_information(truth, labels),
        "Silhouette": silhouette_score(X, labels),
        "Coverage": coverage_score(labels),
    }


def _neighbor_counts(X, eps):
    radius_squared = eps ** 2
    return [
        sum(_squared_distance(sample, candidate) <= radius_squared for candidate in X)
        for sample in X
    ]


def _mean_runtime(factory, X, runs):
    timings, labels = [], None
    for _ in range(runs):
        model = factory()
        started = time.perf_counter()
        model.fit(X)
        timings.append(time.perf_counter() - started)
        labels = model.labels_
    deviation = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return statistics.mean(timings), deviation, labels


def run_experiment(
    data_path,
    samples_per_moon=120,
    noise_samples=30,
    min_samples=5,
    eps_values=None,
    min_samples_values=(3, 5, 7, 9, 11),
    benchmark_sizes=(200, 400, 800, 1200),
    benchmark_runs=3,
    random_state=42,
):
    """Run the calculations used by all six figures."""
    synthetic_X, synthetic_truth = generate_two_moons(
        samples_per_moon=samples_per_moon,
        noise_samples=noise_samples,
        random_state=random_state,
    )
    arc_spacing = math.pi / (samples_per_moon - 1)
    synthetic_eps = max(0.16, 2.25 * arc_spacing)
    synthetic_model = OptimizedDBSCANScratch(
        eps=synthetic_eps,
        min_samples=min_samples,
        algorithm="kd_tree",
        standardize=False,
    ).fit(synthetic_X)
    core_set = set(synthetic_model.core_sample_indices_)
    border_indices = [
        index
        for index, label in enumerate(synthetic_model.labels_)
        if label != -1 and index not in core_set
    ]
    noise_indices = [
        index for index, label in enumerate(synthetic_model.labels_) if label == -1
    ]

    seeds_X, seeds_truth = load_seeds(data_path)
    standardized_X, _, _ = standardize_features(seeds_X)
    sorted_k = sorted(k_distances(standardized_X, min_samples))
    knee_index, standardized_eps = find_knee(sorted_k)
    standardized_eps = max(standardized_eps, 1e-12)

    raw_k = sorted(k_distances(seeds_X, min_samples))
    _, raw_eps = find_knee(raw_k)
    raw_eps = max(raw_eps, 1e-12)
    basic_model = DBSCANScratch(eps=raw_eps, min_samples=min_samples).fit(seeds_X)

    if eps_values is None:
        eps_values = tuple(
            standardized_eps * factor for factor in (0.65, 0.82, 1.0, 1.18, 1.35)
        )
    eps_values = list(eps_values)
    min_samples_values = list(min_samples_values)
    cluster_counts, silhouette_values, noise_rates = [], [], []
    for current_min_samples in min_samples_values:
        count_row, silhouette_row, noise_row = [], [], []
        for eps in eps_values:
            model = OptimizedDBSCANScratch(
                eps=eps,
                min_samples=current_min_samples,
                algorithm="kd_tree",
                standardize=False,
            ).fit(standardized_X)
            count_row.append(model.n_clusters_)
            silhouette = silhouette_score(standardized_X, model.labels_)
            coverage = coverage_score(model.labels_)
            silhouette_row.append(silhouette)
            noise_row.append(1.0 - coverage)
        cluster_counts.append(count_row)
        silhouette_values.append(silhouette_row)
        noise_rates.append(noise_row)

    selected = select_parameters(
        eps_values,
        min_samples_values,
        cluster_counts,
        silhouette_values,
        noise_rates,
    )
    if selected is None:
        selected_eps = standardized_eps
        selected_min_samples = min_samples
    else:
        selected_eps, selected_min_samples = selected
    optimized_model = OptimizedDBSCANScratch(
        eps=selected_eps,
        min_samples=selected_min_samples,
        algorithm="kd_tree",
        standardize=True,
    ).fit(seeds_X)

    basic_metrics = _cluster_metrics(seeds_X, seeds_truth, basic_model.labels_)
    optimized_metrics = _cluster_metrics(
        standardized_X,
        seeds_truth,
        optimized_model.labels_,
    )

    benchmark_sizes = list(benchmark_sizes)
    largest_size = max(benchmark_sizes)
    largest_noise = max(2, largest_size // 10)
    largest_moon = max(2, (largest_size - largest_noise) // 2)
    largest_noise = largest_size - 2 * largest_moon
    benchmark_pool, _ = generate_two_moons(
        samples_per_moon=largest_moon,
        noise_samples=largest_noise,
        random_state=random_state + 9973,
    )
    random.Random(random_state + 6151).shuffle(benchmark_pool)
    benchmark_eps = max(0.16, 2.25 * math.pi / (largest_moon - 1))

    sizes, basic_seconds, kd_tree_seconds, agreements = [], [], [], []
    basic_std_seconds, kd_tree_std_seconds = [], []
    for size in benchmark_sizes:
        benchmark_X = benchmark_pool[:size]
        basic_time, basic_std, basic_labels = _mean_runtime(
            lambda: DBSCANScratch(benchmark_eps, min_samples),
            benchmark_X,
            benchmark_runs,
        )
        kd_time, kd_std, kd_labels = _mean_runtime(
            lambda: OptimizedDBSCANScratch(
                benchmark_eps,
                min_samples,
                algorithm="kd_tree",
                standardize=False,
            ),
            benchmark_X,
            benchmark_runs,
        )
        sizes.append(len(benchmark_X))
        basic_seconds.append(basic_time)
        kd_tree_seconds.append(kd_time)
        basic_std_seconds.append(basic_std)
        kd_tree_std_seconds.append(kd_std)
        agreements.append(basic_labels == kd_labels)

    quality_metrics = {
        name: [basic_metrics[name], optimized_metrics[name]]
        for name in ("ARI", "NMI", "Silhouette", "Coverage")
    }
    return {
        "metadata": {
            "seed": random_state,
            "seed_samples": len(seeds_X),
            "seed_features": len(seeds_X[0]),
            "synthetic_samples": len(synthetic_X),
            "min_samples": min_samples,
            "benchmark_runs": benchmark_runs,
        },
        "synthetic": {
            "X": synthetic_X,
            "truth": synthetic_truth,
            "labels": synthetic_model.labels_,
            "core_indices": synthetic_model.core_sample_indices_,
            "eps": synthetic_eps,
            "min_samples": min_samples,
        },
        "mechanism": {
            "X": synthetic_X,
            "labels": synthetic_model.labels_,
            "neighbor_counts": _neighbor_counts(synthetic_X, synthetic_eps),
            "core_indices": synthetic_model.core_sample_indices_,
            "border_indices": border_indices,
            "noise_indices": noise_indices,
            "eps": synthetic_eps,
            "min_samples": min_samples,
        },
        "k_distance": {
            "distances": sorted_k,
            "knee_index": knee_index,
            "recommended_eps": standardized_eps,
            "min_samples": min_samples,
        },
        "parameter_grid": {
            "eps_values": eps_values,
            "min_samples_values": min_samples_values,
            "cluster_counts": cluster_counts,
            "silhouette_scores": silhouette_values,
            "noise_rates": noise_rates,
            "selected_eps": selected_eps,
            "selected_min_samples": selected_min_samples,
        },
        "scaling": {
            "comparison": "raw_knee_vs_scaled_tuned",
            "display_coordinates": "original",
            "x": [row[0] for row in seeds_X],
            "y": [row[6] for row in seeds_X],
            "feature_names": (SEED_FEATURE_NAMES[0], SEED_FEATURE_NAMES[6]),
            "truth": seeds_truth,
            "raw_labels": basic_model.labels_,
            "standardized_labels": optimized_model.labels_,
            "raw_eps": raw_eps,
            "raw_min_samples": min_samples,
            "standardized_eps": selected_eps,
            "standardized_min_samples": selected_min_samples,
            "raw_cluster_count": basic_model.n_clusters_,
            "standardized_cluster_count": optimized_model.n_clusters_,
            "raw_metrics": basic_metrics,
            "standardized_metrics": optimized_metrics,
        },
        "benchmark": {
            "sizes": sizes,
            "basic_seconds": basic_seconds,
            "kd_tree_seconds": kd_tree_seconds,
            "basic_std_seconds": basic_std_seconds,
            "kd_tree_std_seconds": kd_tree_std_seconds,
            "label_agreement": agreements,
            "runs": benchmark_runs,
            "quality": {
                "methods": [
                    "Basic DBSCAN (raw + knee)",
                    "Optimized DBSCAN (scaled + tuned)",
                ],
                "metrics": quality_metrics,
            },
        },
    }


def _style_axis(axis, grid_axis=None):
    axis.set_facecolor("white")
    axis.tick_params(length=3, width=0.8, color="#555555")
    if grid_axis:
        axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.55)
        axis.set_axisbelow(True)


def _cluster_colors(labels, palette):
    clusters = sorted(label for label in set(labels) if label != -1)
    return {label: palette[index % (len(palette) - 1)] for index, label in enumerate(clusters)}


def _scatter_partition(axis, X, labels, palette, core_indices=None):
    colors = _cluster_colors(labels, palette)
    core_set = set(core_indices or [])
    border_indices = [
        index
        for index, label in enumerate(labels)
        if label != -1 and index not in core_set
    ]
    for label, color in colors.items():
        members = [index for index, value in enumerate(labels) if value == label]
        core_members = [index for index in members if index in core_set]
        other_members = [index for index in members if index not in core_set]
        if core_members:
            axis.scatter(
                [X[index][0] for index in core_members],
                [X[index][1] for index in core_members],
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.92,
            )
        if other_members:
            axis.scatter(
                [X[index][0] for index in other_members],
                [X[index][1] for index in other_members],
                s=34,
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
            )
    noise = [index for index, label in enumerate(labels) if label == -1]
    if noise:
        axis.scatter(
            [X[index][0] for index in noise],
            [X[index][1] for index in noise],
            s=28,
            marker="x",
            color=palette[-1],
            linewidth=1.1,
        )
    return border_indices


def _save_png(figure, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _plot_cluster_result(report):
    data = report["synthetic"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[0]]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)

    _scatter_partition(axes[0], data["X"], data["truth"], palette)
    axes[0].set_title("Reference structure")
    axes[0].set(xlabel="Feature 1", ylabel="Feature 2")
    _style_axis(axes[0])

    _scatter_partition(
        axes[1],
        data["X"],
        data["labels"],
        palette,
        data["core_indices"],
    )
    axes[1].set_title(
        f"DBSCAN recovery  ($eps$={data['eps']:.3f}, min_samples={data['min_samples']})"
    )
    axes[1].set(xlabel="Feature 1", ylabel="Feature 2")
    _style_axis(axes[1])
    handles = [
        Line2D([], [], marker="o", linestyle="", color=palette[0], label="Core sample"),
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="white",
            markeredgecolor=palette[1], label="Border sample",
        ),
        Line2D([], [], marker="x", linestyle="", color=palette[-1], label="Noise"),
    ]
    axes[1].legend(handles=handles, loc="lower right")
    figure.suptitle("Density clustering recovers curved structure and isolates noise", fontsize=12)
    return figure


def _plot_density_mechanism(report):
    data = report["mechanism"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[1]]
    figure, axis = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
    X = data["X"]
    axis.scatter(
        [row[0] for row in X],
        [row[1] for row in X],
        s=18,
        color=palette[3],
        edgecolor="none",
        alpha=0.72,
    )
    groups = (
        ("Core", data["core_indices"], palette[0], "o"),
        ("Border", data["border_indices"], palette[1], "s"),
        ("Noise", data["noise_indices"], palette[-1], "X"),
    )
    for label, indices, color, marker in groups:
        if not indices:
            continue
        if label == "Core":
            index = max(indices, key=lambda item: data["neighbor_counts"][item])
        else:
            index = indices[0]
        point = X[index]
        axis.add_patch(
            Circle(
                point,
                data["eps"],
                facecolor=color,
                edgecolor=color,
                linewidth=1.4,
                alpha=0.12,
            )
        )
        axis.scatter(
            [point[0]], [point[1]], s=85, marker=marker, color=color,
            edgecolor="white", linewidth=0.7, zorder=5,
        )
        axis.annotate(
            f"{label}: {data['neighbor_counts'][index]} neighbors",
            xy=point,
            xytext=(10, 13),
            textcoords="offset points",
            color=color,
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        )
    axis.set(
        xlabel="Feature 1",
        ylabel="Feature 2",
        title=(
            "Local density rule: neighborhood count determines sample type\n"
            f"$eps$={data['eps']:.3f}; min_samples={data['min_samples']} (self included)"
        ),
    )
    axis.set_aspect("equal", adjustable="datalim")
    _style_axis(axis)
    return figure


def _plot_k_distance(report):
    data = report["k_distance"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[2]]
    ranks = list(range(1, len(data["distances"]) + 1))
    knee_rank = data["knee_index"] + 1
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    axis.plot(ranks, data["distances"], color=palette[0], linewidth=2.2)
    axis.fill_between(ranks, data["distances"], color=palette[1], alpha=0.16)
    axis.axhline(
        data["recommended_eps"], color=palette[2], linestyle="--", linewidth=1.4
    )
    axis.scatter(
        [knee_rank], [data["recommended_eps"]], s=55,
        color=palette[3], edgecolor=palette[0], linewidth=0.8, zorder=4,
    )
    axis.annotate(
        f"Initial candidate: $eps$={data['recommended_eps']:.3f}",
        xy=(knee_rank, data["recommended_eps"]),
        xytext=(-95, 32),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": palette[2], "lw": 1.0},
        color=palette[2],
    )
    axis.set(
        xlabel="Samples sorted by k-distance",
        ylabel=f"Distance to neighbor {data['min_samples']}",
        title="The k-distance knee provides a data-driven radius candidate",
    )
    _style_axis(axis, "y")
    return figure


def _plot_parameter_sensitivity(report):
    data = report["parameter_grid"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[3]]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    panels = (
        ("Cluster count", data["cluster_counts"], "{:.0f}", palette[:3]),
        ("Silhouette", data["silhouette_scores"], "{:.2f}", palette[1:4]),
        ("Noise fraction", data["noise_rates"], "{:.2f}", (palette[3], palette[2], palette[0])),
    )
    selected_column = data["eps_values"].index(data["selected_eps"])
    selected_row = data["min_samples_values"].index(data["selected_min_samples"])
    for axis, (title, values, formatter, colors) in zip(axes, panels):
        color_map = LinearSegmentedColormap.from_list(title, colors)
        image = axis.imshow(values, aspect="auto", cmap=color_map)
        for row, row_values in enumerate(values):
            for column, value in enumerate(row_values):
                normalized = image.norm(value)
                axis.text(
                    column,
                    row,
                    formatter.format(value),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if normalized > 0.62 else "#1F1F1F",
                )
        axis.add_patch(
            Rectangle(
                (selected_column - 0.48, selected_row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor=palette[-1],
                linewidth=2.0,
            )
        )
        axis.set_xticks(range(len(data["eps_values"])))
        axis.set_xticklabels([f"{value:.2f}" for value in data["eps_values"]], rotation=35)
        axis.set_yticks(range(len(data["min_samples_values"])))
        axis.set_yticklabels(data["min_samples_values"])
        axis.set(xlabel="$eps$", ylabel="min_samples", title=title)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.colorbar(image, ax=axis, shrink=0.72, pad=0.025)
    figure.suptitle(
        "Parameter sensitivity: outlined cell maximizes Silhouette × Coverage",
        fontsize=12,
    )
    return figure


def _metric_summary(metrics):
    return (
        f"ARI={metrics['ARI']:.2f}   NMI={metrics['NMI']:.2f}\n"
        f"Silhouette={metrics['Silhouette']:.2f}   Coverage={metrics['Coverage']:.2f}"
    )


def _plot_standardization(report):
    data = report["scaling"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[4]]
    X = list(zip(data["x"], data["y"]))
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    panels = (
        (
            "Raw + knee",
            data["raw_labels"],
            data["raw_metrics"],
            data["raw_eps"],
            data["raw_min_samples"],
            data["raw_cluster_count"],
        ),
        (
            "Scaled + tuned",
            data["standardized_labels"],
            data["standardized_metrics"],
            data["standardized_eps"],
            data["standardized_min_samples"],
            data["standardized_cluster_count"],
        ),
    )
    for axis, (title, labels, metrics, eps, min_samples, cluster_count) in zip(axes, panels):
        _scatter_partition(axis, X, labels, palette)
        axis.set(
            xlabel=data["feature_names"][0],
            ylabel=data["feature_names"][1],
            title=(
                f"{title}\n$eps$={eps:.3f}, min_samples={min_samples}, "
                f"clusters={cluster_count}\n{_metric_summary(metrics)}"
            ),
        )
        _style_axis(axis)
    figure.suptitle(
        "DBSCAN workflows on Seeds (displayed in original Area–groove coordinates)",
        fontsize=12,
    )
    return figure


def _plot_benchmark(report):
    data = report["benchmark"]
    quality = data["quality"]
    palette = FIGURE_PALETTES[FIGURE_FILENAMES[5]]
    figure, axes = plt.subplots(
        1, 2, figsize=(10.3, 4.2), gridspec_kw={"width_ratios": (1.0, 1.18)},
        constrained_layout=True,
    )

    metric_names = list(quality["metrics"])
    positions = list(range(len(metric_names)))
    width = 0.34
    for method_index, (method, color) in enumerate(
        zip(quality["methods"], palette[:2])
    ):
        values = [quality["metrics"][name][method_index] for name in metric_names]
        offsets = [position + (method_index - 0.5) * width for position in positions]
        bars = axes[0].bar(
            offsets,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=method,
        )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.025 if value >= 0.0 else -0.055),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=palette[-1],
            )
    axes[0].axhline(0.0, color=palette[-1], linewidth=0.7)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(metric_names, rotation=22, ha="right")
    axes[0].set_ylim(-0.15, 1.10)
    axes[0].set(ylabel="Score", title="Clustering quality on Seeds")
    axes[0].legend(loc="lower right")
    _style_axis(axes[0], "y")

    axes[1].errorbar(
        data["sizes"], data["basic_seconds"], yerr=data["basic_std_seconds"],
        marker="o", markersize=5, linewidth=2.0, capsize=3,
        color=palette[0], label="Brute-force baseline",
    )
    axes[1].errorbar(
        data["sizes"], data["kd_tree_seconds"], yerr=data["kd_tree_std_seconds"],
        marker="s", markersize=5, linewidth=2.0, capsize=3,
        color=palette[2], label="Exact KD-tree",
    )
    axes[1].set_yscale("log")
    agreement = sum(data["label_agreement"])
    axes[1].text(
        0.04,
        0.94,
        f"Exact label agreement: {agreement}/{len(data['label_agreement'])}\n"
        f"Mean ± SD of {data['runs']} run(s)",
        transform=axes[1].transAxes,
        va="top",
        color=palette[-1],
        fontsize=8,
    )
    axes[1].set(
        xlabel="Number of samples",
        ylabel="Fit time (s, log scale)",
        title="Search efficiency at fixed clustering semantics",
    )
    axes[1].legend(loc="lower right")
    _style_axis(axes[1], "both")
    figure.suptitle(
        "Scaling/tuning and exact KD-tree separate quality and efficiency gains",
        fontsize=12,
    )
    return figure


def render_six_figures(report, output_dir):
    plotters = (
        _plot_cluster_result,
        _plot_density_mechanism,
        _plot_k_distance,
        _plot_parameter_sensitivity,
        _plot_standardization,
        _plot_benchmark,
    )
    return [
        _save_png(plotter(report), output_dir, filename)
        for filename, plotter in zip(FIGURE_FILENAMES, plotters)
    ]


def _print_summary(report, output_dir):
    benchmark = report["benchmark"]
    print(f"Generated {len(FIGURE_FILENAMES)} PNG figures in {Path(output_dir)}")
    print(
        "KD-tree label agreement: "
        f"{sum(benchmark['label_agreement'])}/{len(benchmark['label_agreement'])}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate six publication-style DBSCAN PNG figures."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "clustering" / "seeds" / "seeds_dataset.txt",
        help="Path to the Seeds dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "figures" / "dbscan",
        help="Directory for the six PNG files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples value.",
    )
    args = parser.parse_args()
    report = run_experiment(
        args.data,
        min_samples=args.min_samples,
        random_state=args.seed,
    )
    render_six_figures(report, args.output)
    _print_summary(report, args.output)


if __name__ == "__main__":
    main()
