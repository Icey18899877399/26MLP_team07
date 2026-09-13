"""Generate six PNG figures for the hand-written linear regression models."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import time
from pathlib import Path
from statistics import NormalDist

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


FIGURE_FILENAMES = (
    "01_data_distribution_and_correlation.png",
    "02_optimization_convergence.png",
    "03_regularization_path.png",
    "04_actual_vs_predicted.png",
    "05_residual_diagnostics.png",
    "06_final_benchmark_comparison.png",
)

# Each figure has a restrained, color-blind-conscious palette of its own.
FIGURE_PALETTES = {
    FIGURE_FILENAMES[0]: ("#264653", "#E9C46A", "#2A9D8F", "#F4A261", "#E8F1F2"),
    FIGURE_FILENAMES[1]: ("#5E548E", "#F4A261", "#9F86C0", "#231942", "#E9D8A6"),
    FIGURE_FILENAMES[2]: ("#006D77", "#E29578", "#83C5BE", "#243B4A", "#FFDDD2"),
    FIGURE_FILENAMES[3]: ("#355070", "#E56B6F", "#6D597A", "#EAAC8B", "#B8C0D9"),
    FIGURE_FILENAMES[4]: ("#1B998B", "#D1495B", "#2D3047", "#FFCA3A", "#C5D86D"),
    FIGURE_FILENAMES[5]: ("#003049", "#00B4D8", "#F77F00", "#669BBC", "#D9E2EC"),
}

FEATURE_NAMES = [
    "Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
    "Coarse aggregate", "Fine aggregate", "Age",
]

OPTIMIZED_BATCH_SIZE = 32
OPTIMIZED_MAX_ITER = 1_000
OPTIMIZED_TOL = 1e-10


def regression_metrics(y_true, y_pred):
    errors = [prediction - target for prediction, target in zip(y_pred, y_true)]
    mae = sum(abs(error) for error in errors) / len(errors)
    mse = sum(error * error for error in errors) / len(errors)
    mean_target = sum(y_true) / len(y_true)
    total = sum((target - mean_target) ** 2 for target in y_true)
    residual = sum(error * error for error in errors)
    return {"mae": mae, "rmse": math.sqrt(mse), "r2": 1.0 - residual / total if total else 0.0}


def pearson_correlation(first, second):
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum((a - first_mean) * (b - second_mean) for a, b in zip(first, second))
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    return numerator / (first_scale * second_scale) if first_scale and second_scale else 0.0


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


def build_optimized_model(l2, seed, max_iter=OPTIMIZED_MAX_ITER):
    """Create the optimized estimator used by tuning, validation, and final fitting."""
    from Models.linear_regression_optimized import OptimizedLinearRegressionScratch

    return OptimizedLinearRegressionScratch(
        learning_rate=0.01,
        max_iter=max_iter,
        l2=l2,
        batch_size=OPTIMIZED_BATCH_SIZE,
        tol=OPTIMIZED_TOL,
        random_state=seed,
    )


def _subset(values, indices):
    return [values[index] for index in indices]


def _mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _safe_base_learning_rate(features):
    sample_count = len(features)
    summed_second_moments = sum(
        sum(row[column] * row[column] for row in features) / sample_count
        for column in range(len(features[0]))
    )
    return 0.1 / summed_second_moments


def _qq_points(residuals):
    ordered = sorted(residuals)
    count = len(ordered)
    expected = [NormalDist().inv_cdf((index + 0.625) / (count + 0.25)) for index in range(count)]
    return expected, ordered


def stop_annotation(observed_iteration, iteration_limit):
    if observed_iteration < iteration_limit:
        return f"Early stop = {observed_iteration}"
    return f"Run limit = {iteration_limit}"


def qq_reference(residuals, expected_low, expected_high):
    mean = statistics.mean(residuals)
    scale = statistics.stdev(residuals) if len(residuals) > 1 else 0.0
    return (
        (expected_low, mean + scale * expected_low),
        (expected_high, mean + scale * expected_high),
    )


def _panel(axis, label):
    axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="bottom")


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_data(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["data"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].hist(item["target_values"], bins=20, color=palette[2], edgecolor="white", alpha=0.9)
    mean_value = statistics.mean(item["target_values"])
    median_value = statistics.median(item["target_values"])
    axes[0].axvline(mean_value, color=palette[0], linewidth=1.5, label=f"Mean = {mean_value:.1f}")
    axes[0].axvline(median_value, color=palette[1], linewidth=1.5, linestyle="--", label=f"Median = {median_value:.1f}")
    axes[0].set(xlabel="Compressive strength (MPa)", ylabel="Samples", title="Target distribution")
    axes[0].legend()

    order = sorted(range(len(item["correlations"])), key=lambda index: item["correlations"][index])
    values = [item["correlations"][index] for index in order]
    names = [item["feature_names"][index] for index in order]
    colors = [palette[0] if value >= 0 else palette[3] for value in values]
    axes[1].barh(range(len(values)), values, color=colors)
    axes[1].axvline(0, color="#777777", linewidth=0.8)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].set(xlabel="Pearson correlation with strength", title="Training-set associations", xlim=(-1, 1))
    for row, value in enumerate(values):
        axes[1].text(value + (0.02 if value >= 0 else -0.02), row, f"{value:.2f}", ha="left" if value >= 0 else "right", va="center", fontsize=7.5)
    for label, axis in zip("ab", axes):
        _panel(axis, label)
    _save(fig, path)


def _plot_convergence(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["convergence"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), layout="constrained")
    for axis, suffix, title in ((axes[0], "train", "Training convergence"), (axes[1], "validation", "Internal validation convergence")):
        axis.plot(item["iterations"], item[f"base_{suffix}"], color=palette[0], marker="o", linewidth=1.6, label="Base GD")
        axis.plot(item["iterations"], item[f"optimized_{suffix}"], color=palette[1], marker="s", linewidth=1.8, label="Standardized Adam")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(xlabel="Training epochs (data passes)", ylabel="RMSE (MPa; log scale)", title=title)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.5)
        axis.legend()
    iteration_limit = max(item["iterations"])
    axes[1].axvline(item["early_stop"], color=palette[3], linestyle="--", linewidth=1.1, label=stop_annotation(item["early_stop"], iteration_limit))
    axes[1].legend()
    for label, axis in zip("ab", axes):
        _panel(axis, label)
    _save(fig, path)


def _plot_regularization(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["regularization"]
    positions = list(range(len(item["l2_values"])))
    labels = ["0" if value == 0 else f"{value:g}" for value in item["l2_values"]]
    selected_position = item["l2_values"].index(item["selected_l2"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.75), layout="constrained")
    axes[0].errorbar(positions, item["cv_mean"], yerr=item["cv_std"], color=palette[0], marker="o", capsize=3, linewidth=1.7)
    axes[0].axvline(selected_position, color=palette[1], linestyle="--", linewidth=1.1, label=f"Selected L2 = {item['selected_l2']:g}")
    axes[0].set_xticks(positions, labels)
    axes[0].set(xlabel="L2 strength", ylabel="Five-fold CV RMSE (MPa; mean ± fold SD)", title="Regularization selection")
    axes[0].legend()
    line_colors = [palette[0], palette[1], palette[2], palette[3], "#4C78A8", "#A05195", "#7A8B5A", "#C17C38"]
    for name, path_values, color in zip(item["coefficient_names"], item["coefficient_paths"], line_colors):
        axes[1].plot(positions, path_values, marker="o", markersize=3, linewidth=1.2, color=color, label=name)
    axes[1].axvline(selected_position, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].axhline(0, color="#AAAAAA", linewidth=0.7)
    axes[1].set_xticks(positions, labels)
    axes[1].set(xlabel="L2 strength", ylabel="Coefficient in standardized coordinates", title="Coefficient shrinkage")
    axes[1].legend(ncol=2, fontsize=6.8)
    for label, axis in zip("ab", axes):
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.45)
        _panel(axis, label)
    _save(fig, path)


def _plot_predictions(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["prediction"]
    all_values = item["true"] + item["Base"]["predicted"] + item["Optimized"]["predicted"]
    low, high = min(all_values), max(all_values)
    margin = 0.04 * (high - low)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.75), layout="constrained")
    for axis, name, color in zip(axes, ("Base", "Optimized"), (palette[0], palette[1])):
        values = item[name]
        axis.scatter(item["true"], values["predicted"], s=24, color=color, alpha=0.7, edgecolors="white", linewidth=0.35)
        axis.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
        metrics = values["metrics"]
        axis.text(0.04, 0.95, f"RMSE = {metrics['rmse']:.2f} MPa\nMAE = {metrics['mae']:.2f} MPa\n$R^2$ = {metrics['r2']:.3f}", transform=axis.transAxes, va="top", color=palette[3])
        axis.set(xlabel="Measured strength (MPa)", ylabel="Predicted strength (MPa)", title=name, xlim=(low - margin, high + margin), ylim=(low - margin, high + margin))
        axis.set_aspect("equal", adjustable="box")
        _panel(axis, "a" if name == "Base" else "b")
    _save(fig, path)


def _plot_residuals(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["residuals"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45), layout="constrained")
    for name, color, marker in (("Base", palette[0], "o"), ("Optimized", palette[1], "s")):
        values = item[name]
        axes[0].scatter(values["fitted"], values["residuals"], s=21, alpha=0.62, color=color, marker=marker, label=name)
        axes[1].hist(values["residuals"], bins=16, density=True, alpha=0.48, color=color, label=name)
        axes[2].scatter(values["qq_expected"], values["qq_observed"], s=17, alpha=0.68, color=color, marker=marker, label=name)
        expected_low, expected_high = min(values["qq_expected"]), max(values["qq_expected"])
        low_point, high_point = qq_reference(values["residuals"], expected_low, expected_high)
        axes[2].plot(
            [low_point[0], high_point[0]],
            [low_point[1], high_point[1]],
            color=color,
            linestyle="--",
            linewidth=0.9,
        )
    axes[0].axhline(0, color="#777777", linestyle="--", linewidth=0.9)
    axes[0].set(xlabel="Fitted strength (MPa)", ylabel="Residual (MPa)", title="Residual pattern")
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set(xlabel="Residual (MPa)", ylabel="Density", title="Residual distribution")
    axes[2].set(xlabel="Theoretical normal quantile", ylabel="Ordered residual (MPa)", title="Q–Q diagnostic")
    for axis in axes:
        axis.legend()
    for label, axis in zip("abc", axes):
        _panel(axis, label)
    _save(fig, path)


def _plot_benchmark(report, path):
    palette = FIGURE_PALETTES[path.name]
    item = report["benchmark"]
    methods = item["methods"]
    colors = ["#A8A8A8", palette[0], palette[1]]
    positions = list(range(len(methods)))
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.4), layout="constrained")
    width = 0.35
    axes[0, 0].bar([position - width / 2 for position in positions], item["rmse"], width, color=colors, alpha=0.95, label="RMSE")
    axes[0, 0].bar([position + width / 2 for position in positions], item["mae"], width, color=colors, alpha=0.55, hatch="//", label="MAE")
    axes[0, 0].set_xticks(positions, methods, rotation=12)
    axes[0, 0].set(ylabel="Error (MPa)", title="Held-out prediction error")
    axes[0, 0].legend()
    axes[0, 1].bar(positions, item["r2"], color=colors)
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 1].set_xticks(positions, methods, rotation=12)
    axes[0, 1].set(ylabel="$R^2$", title="Held-out coefficient of determination")
    cv_values = [item["cv_rmse"][method] for method in methods]
    boxes = axes[1, 0].boxplot(cv_values, tick_labels=methods, patch_artist=True, showmeans=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axes[1, 0].tick_params(axis="x", rotation=12)
    axes[1, 0].set(
        ylabel="RMSE (MPa)",
        title="Post-selection five-fold variability\n(descriptive; L2 chosen on full training split)",
    )
    axes[1, 1].bar(positions, item["fit_ms"], color=colors)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(positions, methods, rotation=12)
    axes[1, 1].set(ylabel="Fit time (ms; log scale)", title="Single-run training cost")
    for axis in axes.flat:
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.45)
    for label, axis in zip("abcd", axes.flat):
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
    plotters = (_plot_data, _plot_convergence, _plot_regularization, _plot_predictions, _plot_residuals, _plot_benchmark)
    for plotter, path in zip(plotters, paths):
        plotter(report, path)
    return paths


def _regularization_report(features, targets, folds, seed):
    l2_values = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    validation_folds = kfold_indices(len(targets), folds, seed)
    all_indices = set(range(len(targets)))
    scores_by_l2 = []
    for l2 in l2_values:
        fold_scores = []
        for validation in validation_folds:
            training = sorted(all_indices - set(validation))
            model = build_optimized_model(l2=l2, seed=seed)
            model.fit(_subset(features, training), _subset(targets, training))
            predictions = model.predict(_subset(features, validation))
            fold_scores.append(regression_metrics(_subset(targets, validation), predictions)["rmse"])
        scores_by_l2.append(fold_scores)
    summaries = [_mean_sd(values) for values in scores_by_l2]
    selected_position = min(range(len(l2_values)), key=lambda index: summaries[index][0])
    coefficient_paths = [[] for _ in FEATURE_NAMES]
    for l2 in l2_values:
        model = build_optimized_model(l2=l2, seed=seed).fit(features, targets)
        for feature_index, weight in enumerate(model.weights):
            coefficient_paths[feature_index].append(weight)
    return {
        "l2_values": l2_values,
        "cv_mean": [summary[0] for summary in summaries],
        "cv_std": [summary[1] for summary in summaries],
        "selected_l2": l2_values[selected_position],
        "coefficient_names": FEATURE_NAMES,
        "coefficient_paths": coefficient_paths,
        "cv_values": scores_by_l2,
        "selected_position": selected_position,
    }


def _convergence_report(features, targets, selected_l2, seed):
    from Models.linear_regression import LinearRegressionScratch

    training_indices, validation_indices = split_indices(len(targets), 0.2, seed + 1)
    train_x, train_y = _subset(features, training_indices), _subset(targets, training_indices)
    validation_x, validation_y = _subset(features, validation_indices), _subset(targets, validation_indices)
    iterations = [1, 5, 10, 25, 50, 100, 250, 500, 1_000]
    base_train, base_validation, optimized_train, optimized_validation = [], [], [], []
    base_learning_rate = _safe_base_learning_rate(train_x)
    last_optimized = None
    for count in iterations:
        base = LinearRegressionScratch(learning_rate=base_learning_rate, max_iter=count).fit(train_x, train_y)
        optimized = build_optimized_model(l2=selected_l2, seed=seed, max_iter=count).fit(train_x, train_y)
        last_optimized = optimized
        base_train.append(regression_metrics(train_y, base.predict(train_x))["rmse"])
        base_validation.append(regression_metrics(validation_y, base.predict(validation_x))["rmse"])
        optimized_train.append(regression_metrics(train_y, optimized.predict(train_x))["rmse"])
        optimized_validation.append(regression_metrics(validation_y, optimized.predict(validation_x))["rmse"])
    return {
        "iterations": iterations,
        "base_train": base_train,
        "base_validation": base_validation,
        "optimized_train": optimized_train,
        "optimized_validation": optimized_validation,
        "early_stop": last_optimized.n_iter,
    }


def _benchmark_cv(features, targets, selected_l2, folds, seed):
    from Models.linear_regression import LinearRegressionScratch

    validation_folds = kfold_indices(len(targets), folds, seed)
    all_indices = set(range(len(targets)))
    results = {"Mean predictor": [], "Base": [], "Optimized": []}
    for validation in validation_folds:
        training = sorted(all_indices - set(validation))
        train_x, train_y = _subset(features, training), _subset(targets, training)
        validation_x, validation_y = _subset(features, validation), _subset(targets, validation)
        mean_value = statistics.mean(train_y)
        results["Mean predictor"].append(regression_metrics(validation_y, [mean_value] * len(validation_y))["rmse"])
        base = LinearRegressionScratch(learning_rate=_safe_base_learning_rate(train_x), max_iter=1_000).fit(train_x, train_y)
        results["Base"].append(regression_metrics(validation_y, base.predict(validation_x))["rmse"])
        optimized = build_optimized_model(l2=selected_l2, seed=seed).fit(train_x, train_y)
        results["Optimized"].append(regression_metrics(validation_y, optimized.predict(validation_x))["rmse"])
    return results


def run_experiment(data_path, seed=42, folds=5):
    from Models.linear_regression import LinearRegressionScratch

    features, targets = load_concrete_csv(data_path)
    train_indices, test_indices = split_indices(len(targets), 0.2, seed)
    train_x, train_y = _subset(features, train_indices), _subset(targets, train_indices)
    test_x, test_y = _subset(features, test_indices), _subset(targets, test_indices)
    regularization = _regularization_report(train_x, train_y, folds, seed)
    selected_l2 = regularization["selected_l2"]
    convergence = _convergence_report(train_x, train_y, selected_l2, seed)

    started = time.perf_counter()
    mean_value = statistics.mean(train_y)
    mean_ms = max((time.perf_counter() - started) * 1_000, 0.001)
    mean_predictions = [mean_value] * len(test_y)
    started = time.perf_counter()
    base = LinearRegressionScratch(learning_rate=_safe_base_learning_rate(train_x), max_iter=2_000).fit(train_x, train_y)
    base_ms = (time.perf_counter() - started) * 1_000
    started = time.perf_counter()
    optimized = build_optimized_model(l2=selected_l2, seed=seed).fit(train_x, train_y)
    optimized_ms = (time.perf_counter() - started) * 1_000
    base_predictions = base.predict(test_x)
    optimized_predictions = optimized.predict(test_x)
    base_metrics = regression_metrics(test_y, base_predictions)
    optimized_metrics = regression_metrics(test_y, optimized_predictions)
    mean_metrics = regression_metrics(test_y, mean_predictions)

    residual_report = {}
    for name, predictions in (("Base", base_predictions), ("Optimized", optimized_predictions)):
        residuals = [target - prediction for target, prediction in zip(test_y, predictions)]
        qq_expected, qq_observed = _qq_points(residuals)
        residual_report[name] = {"fitted": predictions, "residuals": residuals, "qq_expected": qq_expected, "qq_observed": qq_observed}

    correlations = [pearson_correlation([row[index] for row in train_x], train_y) for index in range(len(FEATURE_NAMES))]
    cv_rmse = _benchmark_cv(train_x, train_y, selected_l2, folds, seed)
    regularization.pop("cv_values")
    regularization.pop("selected_position")
    return {
        "data": {"target_values": train_y, "feature_names": FEATURE_NAMES, "correlations": correlations},
        "convergence": convergence,
        "regularization": regularization,
        "prediction": {
            "true": test_y,
            "Base": {"predicted": base_predictions, "metrics": base_metrics},
            "Optimized": {"predicted": optimized_predictions, "metrics": optimized_metrics},
        },
        "residuals": residual_report,
        "benchmark": {
            "methods": ["Mean predictor", "Base", "Optimized"],
            "rmse": [mean_metrics["rmse"], base_metrics["rmse"], optimized_metrics["rmse"]],
            "mae": [mean_metrics["mae"], base_metrics["mae"], optimized_metrics["mae"]],
            "r2": [mean_metrics["r2"], base_metrics["r2"], optimized_metrics["r2"]],
            "cv_rmse": cv_rmse,
            "fit_ms": [mean_ms, base_ms, optimized_ms],
        },
    }


def build_parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate six linear-regression PNG figures.")
    parser.add_argument("--data", type=Path, default=project_root / "data" / "regression" / "concrete" / "concrete_data.csv")
    parser.add_argument("--output", type=Path, default=project_root / "figures" / "linear_regression")
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
