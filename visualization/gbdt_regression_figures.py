"""Generate six PNG figures for the hand-written GBDT regressors."""

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
from matplotlib.patches import Rectangle


FIGURE_FILENAMES = (
    "01_stagewise_residual_fitting.png",
    "02_convergence_and_early_stopping.png",
    "03_hyperparameter_landscape.png",
    "04_feature_importance_and_dependence.png",
    "05_prediction_and_residual_diagnostics.png",
    "06_final_benchmark_comparison.png",
)

FIGURE_PALETTES = {
    FIGURE_FILENAMES[0]: ("#264653", "#E76F51", "#2A9D8F", "#E9C46A", "#EEF4F1"),
    FIGURE_FILENAMES[1]: ("#5E548E", "#F4A261", "#9F86C0", "#231942", "#F3E9DC"),
    FIGURE_FILENAMES[2]: ("#003049", "#669BBC", "#F77F00", "#D9E2EC", "#780000"),
    FIGURE_FILENAMES[3]: ("#006D77", "#E29578", "#83C5BE", "#243B4A", "#FFDDD2"),
    FIGURE_FILENAMES[4]: ("#355070", "#E56B6F", "#6D597A", "#EAAC8B", "#F1E9DA"),
    FIGURE_FILENAMES[5]: ("#0B3954", "#087E8B", "#FF5A5F", "#BFD7EA", "#F4D35E"),
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
DISPLAY_BEST = "Best-grid GBDT"

OPTIMIZED_LEARNING_RATES = (0.03, 0.07, 0.12)
OPTIMIZED_DEPTHS = (1, 2, 3)
BASE_CANDIDATES = (
    {"n_estimators": 8, "learning_rate": 0.10, "max_depth": 1},
    {"n_estimators": 8, "learning_rate": 0.10, "max_depth": 2},
    {"n_estimators": 12, "learning_rate": 0.15, "max_depth": 2},
)


class _MeanRegressor:
    """Small baseline used to time the actual mean-fitting operation."""

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


def staged_predictions(model, X):
    predictions = [model.init_prediction_] * len(X)
    stages = [list(predictions)]
    for tree in model.estimators_:
        updates = tree.predict(X)
        predictions = [
            prediction + model.learning_rate * update
            for prediction, update in zip(predictions, updates)
        ]
        stages.append(list(predictions))
    return stages


def stage_staircase(model, round_number, low, high):
    """Return exact tree split edges and interval predictions for one GBDT stage."""
    thresholds = sorted(
        {
            tree.root.threshold
            for tree in model.estimators_[:round_number]
            if not tree.root.is_leaf and low < tree.root.threshold < high
        }
    )
    edges = [low, *thresholds, high]
    midpoints = [
        (left + right) / 2.0
        for left, right in zip(edges, edges[1:])
    ]
    predictions = [model.init_prediction_] * len(midpoints)
    for tree in model.estimators_[:round_number]:
        updates = tree.predict([[value] for value in midpoints])
        predictions = [
            prediction + model.learning_rate * update
            for prediction, update in zip(predictions, updates)
        ]
    return {"edges": edges, "predicted": predictions}


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
        means.append(sum(predictions) / len(predictions))
        lows.append(_quantile(predictions, 0.1))
        highs.append(_quantile(predictions, 0.9))
    return {"grid": list(grid), "mean": means, "low": lows, "high": highs}


def stopping_annotation(selected_round, round_limit):
    if selected_round == round_limit:
        return f"Run limit = {round_limit}"
    return f"Selected round = {selected_round}"


def heatmap_text_color(value, minimum, maximum):
    midpoint = (minimum + maximum) / 2.0
    return "white" if value <= midpoint else "#003049"


def _subset(values, indices):
    return [values[index] for index in indices]


def _mean(values):
    return sum(values) / len(values)


def _rmse(targets, predictions):
    return math.sqrt(
        sum(
            (target - prediction) ** 2
            for target, prediction in zip(targets, predictions)
        )
        / len(targets)
    )


def _linear_baseline(seed=42):
    """Create a standardized full-batch linear baseline with convergence stopping."""
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


def _plot_mechanism(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["mechanism"]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), layout="constrained")
    fig.suptitle(
        "One-feature teaching view of stagewise residual fitting (not a benchmark)",
        fontsize=10,
    )
    stage_colors = (palette[0], palette[1], palette[2], palette[3])
    for axis, stage, color, label in zip(axes.flat, item["stages"], stage_colors, "abcd"):
        axis.scatter(item["sample_x"], item["sample_y"], s=16, color="#A7A7A7", alpha=0.45, edgecolors="none", label="Training samples")
        axis.stairs(
            stage["predicted"],
            stage["edges"],
            baseline=None,
            color=color,
            linewidth=1.8,
            label="Stage prediction",
        )
        axis.set(
            xlabel=item["feature_name"],
            ylabel="Compressive strength (MPa)",
            title=f"Round {stage['round']}  |  RMSE = {stage['rmse']:.2f} MPa",
        )
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.45)
        axis.legend()
        _panel(axis, label)
    _save(fig, path)


def _plot_convergence(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["convergence"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65), layout="constrained")
    for axis, suffix, title in (
        (axes[0], "train", "Training convergence"),
        (axes[1], "validation", "Internal validation convergence"),
    ):
        axis.plot(item["base_rounds"], item[f"base_{suffix}"], color=palette[0], marker="o", markersize=3, linewidth=1.5, label="Base GBDT")
        axis.plot(item["optimized_rounds"], item[f"optimized_{suffix}"], color=palette[1], marker="s", markersize=3, linewidth=1.7, label=DISPLAY_BEST)
        axis.set(xlabel="Boosting round", ylabel="RMSE (MPa)", title=title)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.45)
        axis.legend()
    best_index = item["optimized_rounds"].index(item["best_round"])
    best_rmse = item["optimized_validation"][best_index]
    axes[1].axvline(item["best_round"], color=palette[3], linestyle="--", linewidth=1.1)
    axes[1].scatter(
        [item["best_round"]],
        [best_rmse],
        color=palette[3],
        s=32,
        zorder=5,
        label=stopping_annotation(item["best_round"], max(item["optimized_rounds"])),
    )
    axes[1].legend()
    for label, axis in zip("ab", axes):
        _panel(axis, label)
    _save(fig, path)


def _plot_tuning(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["tuning"]
    fig, axis = plt.subplots(figsize=(6.2, 4.6), layout="constrained")
    image = axis.imshow(item["cv_mean"], cmap="Blues_r", aspect="auto")
    axis.set_xticks(range(len(item["learning_rates"])), [f"{value:g}" for value in item["learning_rates"]])
    axis.set_yticks(range(len(item["depths"])), [str(value) for value in item["depths"]])
    axis.set(
        xlabel="Learning rate",
        ylabel="Maximum tree depth",
        title="Bounded five-fold CV grid (best tested cell outlined)",
    )
    heatmap_minimum = min(min(row) for row in item["cv_mean"])
    heatmap_maximum = max(max(row) for row in item["cv_mean"])
    for row_index, row in enumerate(item["cv_mean"]):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=heatmap_text_color(value, heatmap_minimum, heatmap_maximum),
                fontsize=9,
            )
    selected_column = item["learning_rates"].index(item["selected_learning_rate"])
    selected_row = item["depths"].index(item["selected_depth"])
    axis.add_patch(Rectangle((selected_column - 0.48, selected_row - 0.48), 0.96, 0.96, fill=False, edgecolor=palette[2], linewidth=2.2))
    colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Mean CV RMSE (MPa)")
    _panel(axis, "a")
    _save(fig, path)


def _plot_interpretation(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["interpretation"]
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.8), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1, 1]})
    order = sorted(range(len(item["importances"])), key=lambda index: item["importances"][index])
    names = [item["feature_names"][index] for index in order]
    values = [item["importances"][index] for index in order]
    axes[0].barh(range(len(values)), values, color=palette[0], alpha=0.9)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].set(xlabel="Normalized gain importance", title="Feature importance")
    for axis, dependence, color, label in zip(axes[1:], item["partial_dependence"], (palette[1], palette[2]), "bc"):
        axis.fill_between(dependence["grid"], dependence["low"], dependence["high"], color=color, alpha=0.18, label="10–90% sample heterogeneity")
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
    methods = ("Base GBDT", DISPLAY_BEST)
    colors = (palette[0], palette[1])
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
        axis.text(0.04, 0.95, f"RMSE = {metrics['rmse']:.2f} MPa\nMAE = {metrics['mae']:.2f} MPa\n$R^2$ = {metrics['r2']:.3f}", transform=axis.transAxes, va="top", color=palette[2])
        axis.set(xlabel="Measured strength (MPa)", ylabel="Predicted strength (MPa)", title=method, xlim=(low - margin, high + margin), ylim=(low - margin, high + margin))
        axis.set_aspect("equal", adjustable="box")
        _panel(axis, label)
    for method, color, marker in zip(methods, colors, ("o", "s")):
        values = item[method]
        axes[1, 0].scatter(values["predicted"], values["residuals"], s=19, color=color, marker=marker, alpha=0.55, label=method)
        axes[1, 1].hist(values["residuals"], bins=18, density=True, color=color, alpha=0.45, label=method)
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
    colors = ["#A8A8A8", palette[3], palette[0], palette[1]]
    positions = list(range(len(methods)))
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), layout="constrained")
    width = 0.36
    axes[0, 0].bar([position - width / 2 for position in positions], item["rmse"], width, color=colors, alpha=0.95, label="RMSE")
    axes[0, 0].bar([position + width / 2 for position in positions], item["mae"], width, color=colors, alpha=0.55, hatch="//", label="MAE")
    axes[0, 0].set_xticks(positions, methods, rotation=14)
    axes[0, 0].set(ylabel="Error (MPa)", title="Held-out prediction error")
    axes[0, 0].legend()
    axes[0, 1].bar(positions, item["r2"], color=colors)
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 1].set_xticks(positions, methods, rotation=14)
    axes[0, 1].set(ylabel="$R^2$", title="Held-out coefficient of determination")
    cv_values = [item["cv_rmse"][method] for method in methods]
    boxes = axes[1, 0].boxplot(cv_values, tick_labels=methods, patch_artist=True, showmeans=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axes[1, 0].tick_params(axis="x", rotation=14)
    axes[1, 0].set(ylabel="RMSE (MPa)", title="Post-selection five-fold variability\n(descriptive; settings chosen on training data)")
    axes[1, 1].bar(positions, item["fit_seconds"], color=colors)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(positions, methods, rotation=14)
    axes[1, 1].set(ylabel="Fit time (s; log scale)", title="Median training cost (3 runs)")
    for axis in axes.flat:
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.45)
    for label, axis in zip("abcd", axes.flat):
        _panel(axis, label)
    _save(fig, path)


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


def _tune_base(features, targets, folds, seed):
    from Models.gbdt_regression import GBDTRegressorScratch

    candidates = []
    for parameters in BASE_CANDIDATES:
        scores = _cross_validation_scores(
            features,
            targets,
            folds,
            lambda _fold, p=parameters: GBDTRegressorScratch(**p),
            seed,
        )
        candidates.append({"parameters": parameters, "scores": scores, "mean": _mean(scores)})
    return min(candidates, key=lambda item: item["mean"])


def _optimized_model(learning_rate, depth, n_estimators, seed):
    from Models.gbdt_regression_optimized import OptimizedGBDTRegressorScratch

    return OptimizedGBDTRegressorScratch(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=depth,
        min_samples_leaf=3,
        l2_regularization=1.0,
        subsample=0.8,
        max_features=None,
        n_iter_no_change=None,
        random_state=seed,
    )


def _tune_optimized(features, targets, folds, max_estimators, seed):
    matrix = []
    fold_scores = {}
    for depth in OPTIMIZED_DEPTHS:
        row = []
        for learning_rate in OPTIMIZED_LEARNING_RATES:
            scores = _cross_validation_scores(
                features,
                targets,
                folds,
                lambda fold, lr=learning_rate, d=depth: _optimized_model(lr, d, max_estimators, seed + fold),
                seed,
            )
            fold_scores[(learning_rate, depth)] = scores
            row.append(_mean(scores))
        matrix.append(row)
    best_row, best_column = min(
        (
            (row_index, column_index)
            for row_index in range(len(OPTIMIZED_DEPTHS))
            for column_index in range(len(OPTIMIZED_LEARNING_RATES))
        ),
        key=lambda position: matrix[position[0]][position[1]],
    )
    return {
        "learning_rates": list(OPTIMIZED_LEARNING_RATES),
        "depths": list(OPTIMIZED_DEPTHS),
        "cv_mean": matrix,
        "selected_learning_rate": OPTIMIZED_LEARNING_RATES[best_column],
        "selected_depth": OPTIMIZED_DEPTHS[best_row],
        "fold_scores": fold_scores,
    }


def _curve_report(features, targets, base_parameters, optimized_parameters, max_estimators, seed):
    from Models.gbdt_regression import GBDTRegressorScratch

    training_indices, validation_indices = split_indices(len(targets), 0.2, seed + 1)
    training_X, training_y = _subset(features, training_indices), _subset(targets, training_indices)
    validation_X, validation_y = _subset(features, validation_indices), _subset(targets, validation_indices)
    base = GBDTRegressorScratch(**base_parameters).fit(training_X, training_y)
    optimized = _optimized_model(
        optimized_parameters["learning_rate"],
        optimized_parameters["max_depth"],
        max_estimators,
        seed,
    ).fit(training_X, training_y)
    base_train_stages = staged_predictions(base, training_X)[1:]
    base_validation_stages = staged_predictions(base, validation_X)[1:]
    optimized_train_stages = staged_predictions(optimized, training_X)[1:]
    optimized_validation_stages = staged_predictions(optimized, validation_X)[1:]
    optimized_validation = [_rmse(validation_y, values) for values in optimized_validation_stages]
    best_round = min(range(1, len(optimized_validation) + 1), key=lambda round_number: optimized_validation[round_number - 1])
    return {
        "base_rounds": list(range(1, len(base.estimators_) + 1)),
        "base_train": [_rmse(training_y, values) for values in base_train_stages],
        "base_validation": [_rmse(validation_y, values) for values in base_validation_stages],
        "optimized_rounds": list(range(1, len(optimized.estimators_) + 1)),
        "optimized_train": [_rmse(training_y, values) for values in optimized_train_stages],
        "optimized_validation": optimized_validation,
        "best_round": best_round,
    }


def _mechanism_report(features, targets, feature_index):
    from Models.gbdt_regression import GBDTRegressorScratch

    ordered = sorted(zip(features, targets), key=lambda item: item[0][feature_index])
    one_feature_X = [[row[feature_index]] for row, _ in ordered]
    ordered_targets = [target for _, target in ordered]
    model = GBDTRegressorScratch(n_estimators=20, learning_rate=0.15, max_depth=1).fit(one_feature_X, ordered_targets)
    sample_stages = staged_predictions(model, one_feature_X)
    selected_rounds = (0, 1, 5, 20)
    low, high = one_feature_X[0][0], one_feature_X[-1][0]
    return {
        "feature_name": FEATURE_NAMES[feature_index],
        "sample_x": [row[0] for row in one_feature_X],
        "sample_y": ordered_targets,
        "stages": [
            {
                "round": round_number,
                **stage_staircase(model, round_number, low, high),
                "rmse": _rmse(ordered_targets, sample_stages[round_number]),
            }
            for round_number in selected_rounds
        ],
    }


def _interpretation_report(model, features):
    top_indices = sorted(
        range(len(model.feature_importances_)),
        key=model.feature_importances_.__getitem__,
        reverse=True,
    )[:2]
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
        "importances": model.feature_importances_,
        "partial_dependence": dependence_reports,
    }


def _benchmark_cv(features, targets, folds, base_parameters, optimized_parameters, seed):
    from Models.gbdt_regression import GBDTRegressorScratch

    validation_folds = kfold_indices(len(targets), folds=folds, seed=seed)
    all_indices = set(range(len(targets)))
    scores = {"Mean": [], "Linear": [], "Base GBDT": [], DISPLAY_BEST: []}
    for fold_index, validation in enumerate(validation_folds):
        training = sorted(all_indices - set(validation))
        train_X, train_y = _subset(features, training), _subset(targets, training)
        validation_X, validation_y = _subset(features, validation), _subset(targets, validation)
        mean_value = _mean(train_y)
        scores["Mean"].append(_rmse(validation_y, [mean_value] * len(validation_y)))
        linear = _linear_baseline(seed + fold_index).fit(train_X, train_y)
        scores["Linear"].append(_rmse(validation_y, linear.predict(validation_X)))
        base = GBDTRegressorScratch(**base_parameters).fit(train_X, train_y)
        scores["Base GBDT"].append(_rmse(validation_y, base.predict(validation_X)))
        optimized = _optimized_model(
            optimized_parameters["learning_rate"],
            optimized_parameters["max_depth"],
            optimized_parameters["n_estimators"],
            seed + fold_index,
        ).fit(train_X, train_y)
        scores[DISPLAY_BEST].append(_rmse(validation_y, optimized.predict(validation_X)))
    return scores


def _median_fit_seconds(factory, features, targets, repeats=3):
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        factory().fit(features, targets)
        durations.append(max(time.perf_counter() - started, 1e-6))
    return statistics.median(durations)


def run_experiment(data_path, seed=42, folds=5, max_estimators=60):
    from Models.gbdt_regression import GBDTRegressorScratch

    features, targets = load_concrete_csv(data_path)
    train_indices, test_indices = split_indices(len(targets), 0.2, seed)
    train_X, train_y = _subset(features, train_indices), _subset(targets, train_indices)
    test_X, test_y = _subset(features, test_indices), _subset(targets, test_indices)

    base_selection = _tune_base(train_X, train_y, folds, seed)
    tuning = _tune_optimized(train_X, train_y, folds, max_estimators, seed)
    optimized_parameters = {
        "learning_rate": tuning["selected_learning_rate"],
        "max_depth": tuning["selected_depth"],
    }
    convergence = _curve_report(train_X, train_y, base_selection["parameters"], optimized_parameters, max_estimators, seed)
    optimized_parameters["n_estimators"] = convergence["best_round"]

    base = GBDTRegressorScratch(**base_selection["parameters"]).fit(train_X, train_y)
    optimized = _optimized_model(
        optimized_parameters["learning_rate"],
        optimized_parameters["max_depth"],
        optimized_parameters["n_estimators"],
        seed,
    ).fit(train_X, train_y)
    linear = _linear_baseline(seed).fit(train_X, train_y)
    mean_value = _mean(train_y)

    mean_predictions = [mean_value] * len(test_y)
    linear_predictions = linear.predict(test_X)
    base_predictions = base.predict(test_X)
    optimized_predictions = optimized.predict(test_X)
    methods = ["Mean", "Linear", "Base GBDT", DISPLAY_BEST]
    predictions_by_method = {
        "Mean": mean_predictions,
        "Linear": linear_predictions,
        "Base GBDT": base_predictions,
        DISPLAY_BEST: optimized_predictions,
    }
    metrics = {
        method: regression_metrics(test_y, predictions_by_method[method])
        for method in methods
    }

    cv_rmse = _benchmark_cv(
        train_X,
        train_y,
        folds,
        base_selection["parameters"],
        optimized_parameters,
        seed,
    )
    fit_seconds = [
        _median_fit_seconds(_MeanRegressor, train_X, train_y),
        _median_fit_seconds(lambda: _linear_baseline(seed), train_X, train_y),
        _median_fit_seconds(lambda: GBDTRegressorScratch(**base_selection["parameters"]), train_X, train_y),
        _median_fit_seconds(lambda: _optimized_model(optimized_parameters["learning_rate"], optimized_parameters["max_depth"], optimized_parameters["n_estimators"], seed), train_X, train_y),
    ]

    top_feature = max(range(len(optimized.feature_importances_)), key=optimized.feature_importances_.__getitem__)
    diagnostics = {"true": test_y}
    for method, predictions in (("Base GBDT", base_predictions), (DISPLAY_BEST, optimized_predictions)):
        diagnostics[method] = {
            "predicted": predictions,
            "residuals": [target - prediction for target, prediction in zip(test_y, predictions)],
            "metrics": metrics[method],
        }

    tuning.pop("fold_scores")
    return {
        "mechanism": _mechanism_report(train_X, train_y, top_feature),
        "convergence": convergence,
        "tuning": tuning,
        "interpretation": _interpretation_report(optimized, train_X),
        "diagnostics": diagnostics,
        "benchmark": {
            "methods": methods,
            "rmse": [metrics[method]["rmse"] for method in methods],
            "mae": [metrics[method]["mae"] for method in methods],
            "r2": [metrics[method]["r2"] for method in methods],
            "cv_rmse": cv_rmse,
            "fit_seconds": fit_seconds,
        },
    }


def build_parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate six GBDT-regression PNG figures.")
    parser.add_argument("--data", type=Path, default=project_root / "data" / "regression" / "concrete" / "concrete_data.csv")
    parser.add_argument("--output", type=Path, default=project_root / "figures" / "gbdt_regression")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-estimators", type=int, default=60)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    report = run_experiment(arguments.data, arguments.seed, arguments.folds, arguments.max_estimators)
    paths = render_six_figures(report, arguments.output)
    print(f"Generated {len(paths)} PNG figures in {arguments.output}")


if __name__ == "__main__":
    main()
