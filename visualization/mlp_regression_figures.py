"""Generate six PNG figures for the hand-written MLP regressors."""

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle


FIGURE_FILENAMES = (
    "01_network_and_nonlinear_fitting.png",
    "02_training_dynamics_and_early_stopping.png",
    "03_hyperparameter_performance.png",
    "04_feature_interpretation.png",
    "05_prediction_and_residual_diagnostics.png",
    "06_final_benchmark_comparison.png",
)

FIGURE_PALETTES = {
    FIGURE_FILENAMES[0]: ("#23395B", "#406E8E", "#8EA8C3", "#F2A65A", "#F7F3E8"),
    FIGURE_FILENAMES[1]: ("#5B2A86", "#A06CD5", "#D9B8FF", "#F18F01", "#F4EDF7"),
    FIGURE_FILENAMES[2]: ("#1B4965", "#5FA8D3", "#CAE9FF", "#FF6B6B", "#102A43"),
    FIGURE_FILENAMES[3]: ("#006D77", "#83C5BE", "#FFDDD2", "#E29578", "#243B4A"),
    FIGURE_FILENAMES[4]: ("#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"),
    FIGURE_FILENAMES[5]: ("#0B3954", "#087E8B", "#BFD7EA", "#FF5A5F", "#F4D35E"),
}

FEATURE_NAMES = [
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "Superplasticizer",
    "Coarse aggregate",
    "Fine aggregate",
    "Age",
]
DISPLAY_BEST = "Best-grid MLP"
BASE_CANDIDATES = (
    {"hidden_size": 8, "learning_rate": 0.0005, "max_iter": 500},
    {"hidden_size": 16, "learning_rate": 0.001, "max_iter": 500},
    {"hidden_size": 24, "learning_rate": 0.002, "max_iter": 500},
)
OPTIMIZED_ARCHITECTURES = ((8,), (16,), (16, 8))
OPTIMIZED_LEARNING_RATES = (0.001, 0.005, 0.02)
OPTIMIZED_ACTIVATIONS = ("tanh", "relu")


class _MeanRegressor:
    def fit(self, X, y):
        self.mean_ = sum(y) / len(y)
        return self

    def predict(self, X):
        return [self.mean_] * len(X)


def regression_metrics(y_true, y_pred):
    errors = [prediction - target for prediction, target in zip(y_pred, y_true)]
    mae = sum(abs(error) for error in errors) / len(errors)
    mse = sum(error * error for error in errors) / len(errors)
    mean_target = sum(y_true) / len(y_true)
    total = sum((target - mean_target) ** 2 for target in y_true)
    residual = sum(error * error for error in errors)
    return {
        "mae": mae,
        "rmse": math.sqrt(mse),
        "r2": 1.0 - residual / total if total else 0.0,
    }


def split_indices(sample_count, test_fraction=0.2, seed=42):
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    test_count = max(1, round(sample_count * test_fraction))
    return indices[test_count:], indices[:test_count]


def kfold_indices(sample_count, folds=5, seed=42):
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    buckets = [[] for _ in range(folds)]
    for position, index in enumerate(indices):
        buckets[position % folds].append(index)
    return buckets


def load_concrete_csv(path):
    features, targets = [], []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        next(reader)
        for row in reader:
            if row:
                features.append([float(value) for value in row[:8]])
                targets.append(float(row[8]))
    return features, targets


def _subset(values, indices):
    return [values[index] for index in indices]


def _mean(values):
    return sum(values) / len(values)


def _rmse(targets, predictions):
    return regression_metrics(targets, predictions)["rmse"]


def _quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) <= 2:
        return ordered[0] if fraction <= 0.5 else ordered[-1]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def partial_dependence(model, X, feature_index, grid):
    means, lows, highs = [], [], []
    for value in grid:
        replaced = [list(row) for row in X]
        for row in replaced:
            row[feature_index] = value
        predictions = model.predict(replaced)
        means.append(_mean(predictions))
        lows.append(_quantile(predictions, 0.1))
        highs.append(_quantile(predictions, 0.9))
    return {"grid": list(grid), "mean": means, "low": lows, "high": highs}


def permutation_importance(model, X, y, repeats=5, seed=42):
    """Return RMSE increases after shuffling each validation feature."""
    baseline = _rmse(y, model.predict(X))
    generator = random.Random(seed)
    feature_count = len(X[0])
    all_increases = [[] for _ in range(feature_count)]
    for feature_index in range(feature_count):
        original = [row[feature_index] for row in X]
        for _ in range(repeats):
            shuffled = list(original)
            generator.shuffle(shuffled)
            permuted = [list(row) for row in X]
            for row, value in zip(permuted, shuffled):
                row[feature_index] = value
            all_increases[feature_index].append(
                _rmse(y, model.predict(permuted)) - baseline
            )
    return {
        "mean": [_mean(values) for values in all_increases],
        "std": [statistics.pstdev(values) for values in all_increases],
    }


def _linear_baseline(seed=42):
    from Models.linear_regression_optimized import OptimizedLinearRegressionScratch

    return OptimizedLinearRegressionScratch(
        learning_rate=0.03,
        max_iter=5_000,
        tol=1e-10,
        batch_size=None,
        standardize=True,
        standardize_target=True,
        random_state=seed,
    )


def _base_model(parameters, seed):
    from Models.mlp_regression import MLPRegressorScratch

    return MLPRegressorScratch(random_state=seed, **parameters)


def _optimized_model(parameters, seed, max_iter):
    from Models.mlp_regression_optimized import OptimizedMLPRegressorScratch

    return OptimizedMLPRegressorScratch(
        hidden_layer_sizes=parameters["hidden_layer_sizes"],
        activation=parameters["activation"],
        learning_rate=parameters["learning_rate"],
        max_iter=max_iter,
        batch_size=64,
        l2=0.0005,
        tol=1e-5,
        validation_fraction=0.2,
        n_iter_no_change=25,
        standardize=True,
        standardize_target=True,
        gradient_clip=5.0,
        random_state=seed,
    )


def _panel(axis, label):
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_network(axis, layer_sizes, palette):
    visible_counts = [min(size, 6) for size in layer_sizes]
    x_positions = [index / (len(layer_sizes) - 1) for index in range(len(layer_sizes))]
    nodes = []
    for x_position, count in zip(x_positions, visible_counts):
        if count == 1:
            y_positions = [0.5]
        else:
            y_positions = [0.15 + 0.7 * index / (count - 1) for index in range(count)]
        nodes.append([(x_position, y_position) for y_position in y_positions])
    for left, right in zip(nodes, nodes[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                axis.plot([x1, x2], [y1, y2], color=palette[2], linewidth=0.55, alpha=0.45, zorder=1)
    for layer_index, layer in enumerate(nodes):
        color = palette[0] if layer_index in (0, len(nodes) - 1) else palette[1]
        for x_position, y_position in layer:
            axis.add_patch(Circle((x_position, y_position), 0.035, facecolor=color, edgecolor="white", linewidth=0.8, zorder=2))
        if layer_sizes[layer_index] > visible_counts[layer_index]:
            axis.text(x_positions[layer_index], 0.07, "...", ha="center", rotation=90, color=palette[0], fontsize=10)
        axis.text(x_positions[layer_index], 0.96, str(layer_sizes[layer_index]), ha="center", va="top", color=palette[0], fontweight="bold")
    axis.text(0.0, 0.02, "Input", ha="center")
    axis.text(0.5, 0.02, "Hidden layer(s)", ha="center")
    axis.text(1.0, 0.02, "Output", ha="center")
    axis.set(xlim=(-0.10, 1.10), ylim=(0, 1), title="Hand-written feed-forward network")
    axis.set_axis_off()


def _plot_mechanism(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["mechanism"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), layout="constrained", gridspec_kw={"width_ratios": [0.9, 1.35]})
    _draw_network(axes[0], item["layer_sizes"], palette)
    axes[0].text(0.5, -0.08, f"Activation: {item['activation']}  |  backpropagation coded from scratch", transform=axes[0].transAxes, ha="center", fontsize=8, color=palette[0])
    stage_colors = (palette[2], palette[3], palette[0])
    axes[1].scatter(item["sample_x"], item["sample_y"], s=28, color="#8A8A8A", alpha=0.60, edgecolors="white", linewidth=0.35, label="Teaching samples")
    for stage, color in zip(item["stages"], stage_colors):
        axes[1].plot(stage["grid"], stage["predicted"], color=color, linewidth=1.8, label=f"Epoch {stage['epoch']} (RMSE {stage['rmse']:.2f})")
    axes[1].set(xlabel="Standardized input", ylabel="Response", title="Nonlinear function learning (teaching view, not benchmark)")
    axes[1].grid(color="#E4E4E4", linewidth=0.45)
    axes[1].legend()
    for label, axis in zip("ab", axes):
        _panel(axis, label)
    _save(fig, path)


def _plot_convergence(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["convergence"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65), layout="constrained")
    axes[0].plot(item["base_epochs"], item["base_loss"], color=palette[0], linewidth=1.8)
    axes[0].set(xlabel="Epoch", ylabel="Training MSE (native objective)", title="Base MLP: full-batch gradient descent", yscale="log")
    axes[1].plot(item["optimized_epochs"], item["optimized_train_loss"], color=palette[1], linewidth=1.7, label="Training")
    axes[1].plot(item["optimized_epochs"], item["optimized_validation_loss"], color=palette[3], linewidth=1.7, label="Internal validation")
    best_index = item["optimized_epochs"].index(item["best_iteration"])
    best_loss = item["optimized_validation_loss"][best_index]
    axes[1].axvline(item["best_iteration"], color=palette[0], linestyle="--", linewidth=1.0)
    axes[1].scatter([item["best_iteration"]], [best_loss], color=palette[0], s=32, zorder=5, label=f"Best epoch = {item['best_iteration']}")
    axes[1].axvline(item["stop_iteration"], color="#888888", linestyle=":", linewidth=1.0, label=f"Stopped = {item['stop_iteration']}")
    axes[1].set(xlabel="Epoch", ylabel="Standardized objective loss", title="Optimized MLP: Adam and early stopping", yscale="log")
    axes[1].legend()
    for label, axis in zip("ab", axes):
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.45)
        _panel(axis, label)
    _save(fig, path)


def _heatmap_text_color(value, minimum, maximum):
    return "white" if value <= (minimum + maximum) / 2.0 else "#102A43"


def tuning_experiment_label(item):
    return (
        f"Bounded {item['folds']}-fold CV on representative "
        f"outer-training subset (n={item['sample_count']})"
    )


def tuning_colormap(palette):
    return LinearSegmentedColormap.from_list(
        "mlp_tuning_palette",
        [palette[0], palette[1], palette[2]],
    )


def _plot_tuning(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["tuning"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), layout="constrained")
    images = []
    all_values = [value for matrix in item["cv_mean"].values() for row in matrix for value in row]
    minimum, maximum = min(all_values), max(all_values)
    for axis, activation, label in zip(axes, ("tanh", "relu"), "ab"):
        matrix = item["cv_mean"][activation]
        image = axis.imshow(
            matrix,
            cmap=tuning_colormap(palette),
            aspect="auto",
            vmin=minimum,
            vmax=maximum,
        )
        images.append(image)
        axis.set_xticks(range(len(item["learning_rates"])), [f"{value:g}" for value in item["learning_rates"]])
        axis.set_yticks(range(len(item["architectures"])), item["architectures"])
        axis.set(xlabel="Learning rate", ylabel="Hidden-layer sizes", title=f"{activation} activation")
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=_heatmap_text_color(value, minimum, maximum))
        if activation == item["selected_activation"]:
            selected_column = item["learning_rates"].index(item["selected_learning_rate"])
            selected_row = item["architectures"].index(item["selected_architecture"])
            axis.add_patch(Rectangle((selected_column - 0.48, selected_row - 0.48), 0.96, 0.96, fill=False, edgecolor=palette[3], linewidth=2.2))
        _panel(axis, label)
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Mean CV RMSE (MPa)")
    fig.suptitle(tuning_experiment_label(item), fontsize=10)
    _save(fig, path)


def _plot_interpretation(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["interpretation"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1, 1]})
    order = sorted(range(len(item["importances"])), key=lambda index: item["importances"][index])
    names = [item["feature_names"][index] for index in order]
    values = [item["importances"][index] for index in order]
    errors = [item["importance_std"][index] for index in order]
    axes[0].barh(range(len(values)), values, xerr=errors, color=palette[0], alpha=0.88, error_kw={"elinewidth": 0.8, "capsize": 2})
    axes[0].axvline(0, color="#777777", linewidth=0.8)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].set(xlabel="Validation RMSE increase (MPa)", title="Permutation importance")
    for axis, dependence, color, label in zip(axes[1:], item["partial_dependence"], (palette[1], palette[3]), "bc"):
        axis.fill_between(dependence["grid"], dependence["low"], dependence["high"], color=color, alpha=0.18, label="10–90% sample spread")
        axis.plot(dependence["grid"], dependence["mean"], color=color, linewidth=2.0, label="Partial dependence")
        axis.set(xlabel=dependence["feature_name"], ylabel="Predicted strength (MPa)", title=f"Dependence on {dependence['feature_name']}")
        axis.legend(fontsize=7)
        _panel(axis, label)
    for axis in axes:
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.45)
    _panel(axes[0], "a")
    _save(fig, path)


def _plot_diagnostics(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["diagnostics"]
    methods = ("Base MLP", DISPLAY_BEST)
    colors = (palette[0], palette[3])
    all_values = list(item["true"])
    for method in methods:
        all_values.extend(item[method]["predicted"])
    low, high = min(all_values), max(all_values)
    margin = 0.04 * (high - low)
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 7.0), layout="constrained")
    for axis, method, color, label in zip(axes[0], methods, colors, "ab"):
        values = item[method]
        axis.scatter(item["true"], values["predicted"], s=22, color=color, alpha=0.66, edgecolors="white", linewidth=0.3)
        axis.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
        metrics = values["metrics"]
        axis.text(0.04, 0.95, f"RMSE = {metrics['rmse']:.2f} MPa\nMAE = {metrics['mae']:.2f} MPa\nR² = {metrics['r2']:.3f}", transform=axis.transAxes, va="top", color=palette[2])
        axis.set(xlabel="Measured strength (MPa)", ylabel="Predicted strength (MPa)", title=method, xlim=(low - margin, high + margin), ylim=(low - margin, high + margin))
        axis.set_aspect("equal", adjustable="box")
        _panel(axis, label)
    for method, color, marker in zip(methods, colors, ("o", "s")):
        values = item[method]
        axes[1, 0].scatter(values["predicted"], values["residuals"], s=19, color=color, marker=marker, alpha=0.55, label=method)
        axes[1, 1].hist(values["residuals"], bins=18, density=True, color=color, alpha=0.42, label=method)
    axes[1, 0].axhline(0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 0].set(xlabel="Fitted strength (MPa)", ylabel="Residual (MPa)", title="Held-out residual pattern")
    axes[1, 1].axvline(0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 1].set(xlabel="Residual (MPa)", ylabel="Density", title="Held-out residual distribution")
    for axis, label in zip(axes[1], "cd"):
        axis.legend()
        _panel(axis, label)
    _save(fig, path)


def _plot_benchmark(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["benchmark"]
    methods = item["methods"]
    colors = ["#A8A8A8", palette[2], palette[0], palette[3]]
    positions = benchmark_box_positions(methods)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), layout="constrained")
    width = 0.36
    axes[0, 0].bar([position - width / 2 for position in positions], item["rmse"], width, color=colors, label="RMSE")
    axes[0, 0].bar([position + width / 2 for position in positions], item["mae"], width, color=colors, alpha=0.55, hatch="//", label="MAE")
    axes[0, 0].set(ylabel="Error (MPa)", title="Held-out prediction error")
    axes[0, 0].legend()
    axes[0, 1].bar(positions, item["r2"], color=colors)
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 1].set(ylabel="R²", title="Held-out explained variance")
    cv_values = [item["cv_rmse"][method] for method in methods]
    boxes = axes[0, 2].boxplot(
        cv_values,
        positions=positions,
        tick_labels=methods,
        patch_artist=True,
        showmeans=True,
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axes[0, 2].set(
        ylabel="RMSE (MPa)",
        title=benchmark_cv_title(item["folds"]),
    )
    mlp_methods = ["Base MLP", DISPLAY_BEST]
    for method, color, marker, x_position in zip(mlp_methods, (palette[0], palette[3]), ("o", "s"), (0, 1)):
        values = item["seed_rmse"][method]
        jitter = jitter_offsets(len(values))
        axes[1, 0].scatter([x_position + offset for offset in jitter], values, color=color, marker=marker, s=32, alpha=0.82)
        axes[1, 0].plot([x_position - 0.12, x_position + 0.12], [_mean(values)] * 2, color=color, linewidth=2.0)
    axes[1, 0].set_xticks([0, 1], mlp_methods)
    axes[1, 0].set(
        ylabel="Held-out RMSE (MPa)",
        title=robustness_title(item["seed_runs"]),
    )
    axes[1, 1].bar(positions, item["fit_seconds"], color=colors)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        ylabel="Fit time (s; log scale)",
        title=timing_title(item["timing_repeats"]),
    )
    base_rmse = item["rmse"][methods.index("Base MLP")]
    optimized_rmse = item["rmse"][methods.index(DISPLAY_BEST)]
    gain = 100.0 * (base_rmse - optimized_rmse) / base_rmse
    axes[1, 2].barh([0], [gain], color=palette[3], height=0.45)
    axes[1, 2].axvline(0, color="#777777", linewidth=0.8)
    axes[1, 2].text(gain, 0, f"  {gain:.1f}%", va="center", color=palette[0], fontweight="bold")
    axes[1, 2].set(
        xlabel="RMSE reduction vs Base MLP (%)",
        title=pipeline_gain_title(),
        yticks=[],
    )
    axes[1, 2].text(
        0.02,
        0.05,
        "Standardization + Adam + mini-batches\n+ L2 + clipping + early stopping + grid search",
        transform=axes[1, 2].transAxes,
        fontsize=7,
        color=palette[0],
        va="bottom",
    )
    for axis in axes.flat:
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.45)
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1]):
        axis.set_xticks(positions, methods, rotation=15)
    for label, axis in zip("abcdef", axes.flat):
        _panel(axis, label)
    _save(fig, path)


def benchmark_box_positions(methods):
    return list(range(len(methods)))


def jitter_offsets(count, span=0.16):
    if count == 1:
        return [0.0]
    return [
        -span / 2.0 + span * index / (count - 1)
        for index in range(count)
    ]


def benchmark_cv_title(folds):
    return (
        f"Descriptive post-selection {folds}-fold RMSE\n"
        "(settings selected on overlapping training data)"
    )


def robustness_title(seed_runs):
    return f"Initialization robustness ({seed_runs} seeds)"


def timing_title(timing_repeats):
    return f"Median training cost ({timing_repeats} runs)"


def pipeline_gain_title():
    return "Full-pipeline gain"


def render_six_figures(report, output_dir):
    matplotlib.rcParams.update(PLOT_STYLE)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / filename for filename in FIGURE_FILENAMES]
    for path in paths:
        if path.is_file():
            path.unlink()
    plotters = (
        _plot_mechanism,
        _plot_convergence,
        _plot_tuning,
        _plot_interpretation,
        _plot_diagnostics,
        _plot_benchmark,
    )
    for plotter, path in zip(plotters, paths):
        plotter(report, path)
    return paths


def _cross_validation_scores(features, targets, folds, model_factory, seed):
    validation_folds = kfold_indices(len(targets), folds=folds, seed=seed)
    all_indices = set(range(len(targets)))
    scores = []
    for fold_index, validation in enumerate(validation_folds):
        training = sorted(all_indices - set(validation))
        model = model_factory(fold_index)
        model.fit(_subset(features, training), _subset(targets, training))
        predictions = model.predict(_subset(features, validation))
        scores.append(_rmse(_subset(targets, validation), predictions))
    return scores


def _representative_training_subset(features, targets, limit):
    if limit is None or limit >= len(targets):
        return features, targets
    ordered = sorted(range(len(targets)), key=targets.__getitem__)
    selected = [ordered[round(index * (len(ordered) - 1) / (limit - 1))] for index in range(limit)]
    return _subset(features, selected), _subset(targets, selected)


def _tune_base(features, targets, folds, seed):
    candidates = []
    for parameters in BASE_CANDIDATES:
        scores = _cross_validation_scores(
            features,
            targets,
            folds,
            lambda fold_index, p=parameters: _base_model(p, seed + fold_index),
            seed,
        )
        candidates.append({"parameters": parameters, "scores": scores, "mean": _mean(scores)})
    return min(candidates, key=lambda item: item["mean"])


def _tune_optimized(features, targets, folds, seed, tuning_max_iter):
    matrices = {activation: [] for activation in OPTIMIZED_ACTIVATIONS}
    candidates = []
    for activation in OPTIMIZED_ACTIVATIONS:
        for architecture in OPTIMIZED_ARCHITECTURES:
            row = []
            for learning_rate in OPTIMIZED_LEARNING_RATES:
                parameters = {
                    "hidden_layer_sizes": architecture,
                    "activation": activation,
                    "learning_rate": learning_rate,
                }
                scores = _cross_validation_scores(
                    features,
                    targets,
                    folds,
                    lambda fold_index, p=parameters: _optimized_model(p, seed + fold_index, tuning_max_iter),
                    seed,
                )
                mean_score = _mean(scores)
                row.append(mean_score)
                candidates.append({"parameters": parameters, "scores": scores, "mean": mean_score})
            matrices[activation].append(row)
    selected = min(candidates, key=lambda item: item["mean"])
    return {
        "architectures": [str(value) for value in OPTIMIZED_ARCHITECTURES],
        "learning_rates": list(OPTIMIZED_LEARNING_RATES),
        "cv_mean": matrices,
        "selected_activation": selected["parameters"]["activation"],
        "selected_architecture": str(selected["parameters"]["hidden_layer_sizes"]),
        "selected_learning_rate": selected["parameters"]["learning_rate"],
        "selected_parameters": selected["parameters"],
        "selected_scores": selected["scores"],
        "folds": folds,
    }


def _teaching_report():
    from Models.mlp_regression import MLPRegressorScratch

    sample_x = [-2.0 + 4.0 * index / 40 for index in range(41)]
    sample_y = [
        1.5 + 1.2 * math.tanh(1.3 * value) + 0.15 * math.sin(3.0 * value)
        for value in sample_x
    ]
    target_mean = _mean(sample_y)
    target_scale = math.sqrt(_mean([(value - target_mean) ** 2 for value in sample_y]))
    standardized_y = [(value - target_mean) / target_scale for value in sample_y]
    grid = [-2.0 + 4.0 * index / 120 for index in range(121)]
    stages = []
    for epoch in (1, 40, 400):
        model = MLPRegressorScratch(hidden_size=8, learning_rate=0.03, max_iter=epoch, random_state=13).fit([[value] for value in sample_x], standardized_y)
        predicted = [value * target_scale + target_mean for value in model.predict([[value] for value in grid])]
        fitted = [value * target_scale + target_mean for value in model.predict([[value] for value in sample_x])]
        stages.append({"epoch": epoch, "grid": grid, "predicted": predicted, "rmse": _rmse(sample_y, fitted)})
    return {"layer_sizes": [1, 8, 1], "activation": "tanh", "sample_x": sample_x, "sample_y": sample_y, "stages": stages}


def _interpretation_report(model, features, validation_X, validation_y, seed):
    importance = permutation_importance(model, validation_X, validation_y, repeats=5, seed=seed)
    top_indices = sorted(range(len(importance["mean"])), key=importance["mean"].__getitem__, reverse=True)[:2]
    dependence_reports = []
    for feature_index in top_indices:
        values = [row[feature_index] for row in features]
        low, high = _quantile(values, 0.05), _quantile(values, 0.95)
        grid = [low + (high - low) * position / 24 for position in range(25)]
        dependence = partial_dependence(model, features, feature_index, grid)
        dependence["feature_name"] = FEATURE_NAMES[feature_index]
        dependence_reports.append(dependence)
    return {
        "feature_names": FEATURE_NAMES,
        "importances": importance["mean"],
        "importance_std": importance["std"],
        "partial_dependence": dependence_reports,
    }


def _benchmark_cv(features, targets, folds, base_parameters, optimized_parameters, seed, final_max_iter):
    validation_folds = kfold_indices(len(targets), folds=folds, seed=seed)
    all_indices = set(range(len(targets)))
    scores = {"Mean": [], "Linear": [], "Base MLP": [], DISPLAY_BEST: []}
    for fold_index, validation in enumerate(validation_folds):
        training = sorted(all_indices - set(validation))
        train_X, train_y = _subset(features, training), _subset(targets, training)
        validation_X, validation_y = _subset(features, validation), _subset(targets, validation)
        mean_value = _mean(train_y)
        scores["Mean"].append(_rmse(validation_y, [mean_value] * len(validation_y)))
        linear = _linear_baseline(seed + fold_index).fit(train_X, train_y)
        scores["Linear"].append(_rmse(validation_y, linear.predict(validation_X)))
        base = _base_model(base_parameters, seed + fold_index).fit(train_X, train_y)
        scores["Base MLP"].append(_rmse(validation_y, base.predict(validation_X)))
        optimized = _optimized_model(optimized_parameters, seed + fold_index, final_max_iter).fit(train_X, train_y)
        scores[DISPLAY_BEST].append(_rmse(validation_y, optimized.predict(validation_X)))
    return scores


def _median_fit_seconds(factory, features, targets, repeats=3):
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        factory().fit(features, targets)
        durations.append(max(time.perf_counter() - started, 1e-6))
    return statistics.median(durations)


def run_experiment(
    data_path,
    seed=42,
    folds=5,
    tuning_samples=240,
    tuning_max_iter=80,
    final_max_iter=300,
    seed_runs=5,
    timing_repeats=3,
):
    features, targets = load_concrete_csv(data_path)
    train_indices, test_indices = split_indices(len(targets), 0.2, seed)
    train_X, train_y = _subset(features, train_indices), _subset(targets, train_indices)
    test_X, test_y = _subset(features, test_indices), _subset(targets, test_indices)
    tuning_X, tuning_y = _representative_training_subset(train_X, train_y, tuning_samples)

    base_selection = _tune_base(tuning_X, tuning_y, folds, seed)
    tuning = _tune_optimized(tuning_X, tuning_y, folds, seed, tuning_max_iter)
    tuning["sample_count"] = len(tuning_y)
    optimized_parameters = tuning.pop("selected_parameters")
    tuning.pop("selected_scores")

    base = _base_model(base_selection["parameters"], seed).fit(train_X, train_y)
    optimized = _optimized_model(optimized_parameters, seed, final_max_iter).fit(train_X, train_y)
    linear = _linear_baseline(seed).fit(train_X, train_y)
    mean_value = _mean(train_y)

    predictions_by_method = {
        "Mean": [mean_value] * len(test_y),
        "Linear": linear.predict(test_X),
        "Base MLP": base.predict(test_X),
        DISPLAY_BEST: optimized.predict(test_X),
    }
    methods = ["Mean", "Linear", "Base MLP", DISPLAY_BEST]
    metrics = {method: regression_metrics(test_y, predictions_by_method[method]) for method in methods}

    inner_train_indices, inner_validation_indices = split_indices(len(train_y), 0.2, seed + 101)
    interpretation_model = _optimized_model(optimized_parameters, seed + 101, final_max_iter).fit(
        _subset(train_X, inner_train_indices),
        _subset(train_y, inner_train_indices),
    )
    interpretation = _interpretation_report(
        interpretation_model,
        _subset(train_X, inner_train_indices),
        _subset(train_X, inner_validation_indices),
        _subset(train_y, inner_validation_indices),
        seed + 101,
    )

    cv_rmse = _benchmark_cv(train_X, train_y, folds, base_selection["parameters"], optimized_parameters, seed, final_max_iter)
    seed_rmse = {"Base MLP": [], DISPLAY_BEST: []}
    for seed_offset in range(seed_runs):
        run_seed = seed + seed_offset
        base_seed_model = _base_model(base_selection["parameters"], run_seed).fit(train_X, train_y)
        optimized_seed_model = _optimized_model(optimized_parameters, run_seed, final_max_iter).fit(train_X, train_y)
        seed_rmse["Base MLP"].append(_rmse(test_y, base_seed_model.predict(test_X)))
        seed_rmse[DISPLAY_BEST].append(_rmse(test_y, optimized_seed_model.predict(test_X)))

    fit_seconds = [
        _median_fit_seconds(_MeanRegressor, train_X, train_y, timing_repeats),
        _median_fit_seconds(lambda: _linear_baseline(seed), train_X, train_y, timing_repeats),
        _median_fit_seconds(lambda: _base_model(base_selection["parameters"], seed), train_X, train_y, timing_repeats),
        _median_fit_seconds(lambda: _optimized_model(optimized_parameters, seed, final_max_iter), train_X, train_y, timing_repeats),
    ]

    diagnostics = {"true": test_y}
    for method in ("Base MLP", DISPLAY_BEST):
        predictions = predictions_by_method[method]
        diagnostics[method] = {
            "predicted": predictions,
            "residuals": [target - prediction for target, prediction in zip(test_y, predictions)],
            "metrics": metrics[method],
        }

    return {
        "mechanism": _teaching_report(),
        "convergence": {
            "base_epochs": list(range(1, len(base.loss_history_) + 1)),
            "base_loss": base.loss_history_,
            "optimized_epochs": list(range(1, len(optimized.train_loss_) + 1)),
            "optimized_train_loss": optimized.train_loss_,
            "optimized_validation_loss": optimized.validation_loss_,
            "best_iteration": optimized.best_iteration_,
            "stop_iteration": optimized.n_iter_,
        },
        "tuning": tuning,
        "interpretation": interpretation,
        "diagnostics": diagnostics,
        "benchmark": {
            "methods": methods,
            "rmse": [metrics[method]["rmse"] for method in methods],
            "mae": [metrics[method]["mae"] for method in methods],
            "r2": [metrics[method]["r2"] for method in methods],
            "cv_rmse": cv_rmse,
            "seed_rmse": seed_rmse,
            "fit_seconds": fit_seconds,
            "folds": folds,
            "seed_runs": seed_runs,
            "timing_repeats": timing_repeats,
        },
    }


def build_parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate six MLP-regression PNG figures.")
    parser.add_argument("--data", type=Path, default=project_root / "data" / "regression" / "concrete" / "concrete_data.csv")
    parser.add_argument("--output", type=Path, default=project_root / "figures" / "mlp_regression")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tuning-samples", type=int, default=240)
    parser.add_argument("--tuning-max-iter", type=int, default=80)
    parser.add_argument("--final-max-iter", type=int, default=300)
    parser.add_argument("--seed-runs", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=3)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    report = run_experiment(
        arguments.data,
        arguments.seed,
        arguments.folds,
        arguments.tuning_samples,
        arguments.tuning_max_iter,
        arguments.final_max_iter,
        arguments.seed_runs,
        arguments.timing_repeats,
    )
    paths = render_six_figures(report, arguments.output)
    print(f"Generated {len(paths)} PNG figures in {arguments.output}")


if __name__ == "__main__":
    main()
