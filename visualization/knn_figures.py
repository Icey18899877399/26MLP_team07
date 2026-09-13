"""为手写 KNN 生成六张可复现实验图，Matplotlib 仅负责绘图。"""

import argparse
import csv
import heapq
import math
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Models.knn import KNNScratch
from Models.knn_optimized import OptimizedKNNScratch


FIGURE_PALETTES = {
    "neighborhood": ["#2A9D8F", "#E76F51", "#F4A261", "#264653"],
    "boundaries": ["#5E60CE", "#FFB703", "#E8E6F3", "#FFF1C1"],
    "validation": ["#3D5A80", "#EE6C4D", "#98C1D9", "#C44536"],
    "diagnostics": ["#7B2CBF", "#2A9D8F", "#A7A9AC"],
    "confusion": ["Purples", "BuGn", "#252525"],
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
    """读取 WDBC 数据，恶性 M 为正类 1，良性 B 为负类 0。"""
    features = []
    targets = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if row:
                targets.append(1 if row[1] == "M" else 0)
                features.append([float(value) for value in row[2:]])
    return features, targets


def stratified_split(features, targets, test_size=0.2, seed=42):
    """按类别分层切分训练集和测试集。"""
    generator = random.Random(seed)
    train_indices = []
    test_indices = []
    for label in (0, 1):
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
    """返回分层 K 折中每一折的验证样本下标。"""
    generator = random.Random(seed)
    folds = [[] for _ in range(n_splits)]
    for label in sorted(set(targets)):
        indices = [index for index, target in enumerate(targets) if target == label]
        generator.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % n_splits].append(index)
    for fold in folds:
        generator.shuffle(fold)
    return folds


def mean_and_std(values):
    """计算总体均值与总体标准差。"""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def classification_metrics(y_true, y_pred):
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(y_true, y_pred))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(y_true, y_pred))
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
    accuracy = (tn + tp) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
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
    return fpr, tpr, auc


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
    return recall, precision, average_precision


def _distance(first, second, p=2):
    if p == 1:
        return sum(abs(left - right) for left, right in zip(first, second))
    if p == 2:
        return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))
    return sum(abs(left - right) ** p for left, right in zip(first, second)) ** (1.0 / p)


def _rank_neighbors(sample, X_train, y_train, p=2):
    neighbors = [
        (_distance(sample, train_sample, p), index, label)
        for index, (train_sample, label) in enumerate(zip(X_train, y_train))
    ]
    neighbors.sort(key=lambda item: (item[0], item[1]))
    return neighbors


def _binary_vote(neighbors, k, weighted):
    selected = neighbors[:k]
    if weighted:
        exact_matches = [neighbor for neighbor in selected if neighbor[0] == 0.0]
        if exact_matches:
            selected = exact_matches
            weights = [1.0] * len(selected)
        else:
            weights = [1.0 / neighbor[0] for neighbor in selected]
    else:
        weights = [1.0] * len(selected)
    positive_score = sum(weight for weight, neighbor in zip(weights, selected) if neighbor[2] == 1)
    total_score = sum(weights)
    probability = positive_score / total_score
    return int(probability >= 0.5), probability


def _standardize_fold(X_train, X_valid):
    scaler = OptimizedKNNScratch(n_neighbors=1, standardize=True).fit(
        X_train, [0] * len(X_train)
    )
    return scaler.X_train, scaler._transform(X_valid)


def validation_curves(features, targets, k_values, n_splits=5, seed=42):
    """一次计算邻居排序，再得到多个 K 值的分层交叉验证结果。"""
    score_store = {
        "Base accuracy": [[] for _ in k_values],
        "Base F1": [[] for _ in k_values],
        "Optimized accuracy": [[] for _ in k_values],
        "Optimized F1": [[] for _ in k_values],
    }
    folds = stratified_k_folds(targets, n_splits=n_splits, seed=seed)

    for validation_indices in folds:
        validation_set = set(validation_indices)
        training_indices = [index for index in range(len(targets)) if index not in validation_set]
        X_train = [features[index] for index in training_indices]
        y_train = [targets[index] for index in training_indices]
        X_valid = [features[index] for index in validation_indices]
        y_valid = [targets[index] for index in validation_indices]
        scaled_train, scaled_valid = _standardize_fold(X_train, X_valid)

        base_rankings = [_rank_neighbors(sample, X_train, y_train) for sample in X_valid]
        optimized_rankings = [
            _rank_neighbors(sample, scaled_train, y_train) for sample in scaled_valid
        ]

        for position, k in enumerate(k_values):
            base_predictions = [_binary_vote(ranking, k, False)[0] for ranking in base_rankings]
            optimized_predictions = [
                _binary_vote(ranking, k, True)[0] for ranking in optimized_rankings
            ]
            base_metrics = classification_metrics(y_valid, base_predictions)
            optimized_metrics = classification_metrics(y_valid, optimized_predictions)
            score_store["Base accuracy"][position].append(base_metrics["accuracy"])
            score_store["Base F1"][position].append(base_metrics["f1"])
            score_store["Optimized accuracy"][position].append(optimized_metrics["accuracy"])
            score_store["Optimized F1"][position].append(optimized_metrics["f1"])

    series = {}
    for name, scores_by_k in score_store.items():
        summaries = [mean_and_std(scores) for scores in scores_by_k]
        series[name] = {
            "mean": [summary[0] for summary in summaries],
            "std": [summary[1] for summary in summaries],
        }
    optimized_f1 = series["Optimized F1"]["mean"]
    best_position = max(range(len(k_values)), key=lambda index: optimized_f1[index])
    return {"k_values": list(k_values), "series": series, "best_k": k_values[best_position]}


def _positive_scores(model, X):
    positive_index = model.classes.index(1)
    return [probabilities[positive_index] for probabilities in model.predict_proba(X)]


def _curve_report(y_true, scores):
    fpr, tpr, auc = roc_curve(y_true, scores)
    recall, precision, average_precision = precision_recall_curve(y_true, scores)
    return {
        "roc": {"fpr": fpr, "tpr": tpr, "auc": auc},
        "pr": {"recall": recall, "precision": precision, "ap": average_precision},
    }


def _two_feature_report(X_train, X_test, y_train, y_test, k):
    feature_indices = [20, 27]
    feature_names = ["Worst radius (standardized)", "Worst concave points (standardized)"]
    training_2d = [[row[index] for index in feature_indices] for row in X_train]
    test_2d = [[row[index] for index in feature_indices] for row in X_test]
    model = OptimizedKNNScratch(
        n_neighbors=k, p=2, weights="distance", standardize=True
    ).fit(training_2d, y_train)
    scaled_test = model._transform(test_2d)
    probabilities = _positive_scores(model, test_2d)
    query_index = min(range(len(test_2d)), key=lambda index: abs(probabilities[index] - 0.5))
    query = scaled_test[query_index]
    neighbors = model._nearest_neighbors(query)
    neighbor_records = []
    for distance, index, label in neighbors:
        neighbor_records.append(
            {
                "index": index,
                "distance": distance,
                "weight": 1.0 / distance if distance > 0.0 else 1.0,
                "label": label,
            }
        )

    neighborhood = {
        "train_x": model.X_train,
        "train_y": list(y_train),
        "query": query,
        "query_label": y_test[query_index],
        "predicted_label": model.predict([test_2d[query_index]])[0],
        "neighbors": neighbor_records,
        "feature_names": feature_names,
    }

    x_values = _linear_space(
        min(row[0] for row in model.X_train) - 0.5,
        max(row[0] for row in model.X_train) + 0.5,
        70,
    )
    y_values = _linear_space(
        min(row[1] for row in model.X_train) - 0.5,
        max(row[1] for row in model.X_train) + 0.5,
        70,
    )
    grid = [[x_value, y_value] for y_value in y_values for x_value in x_values]
    panels = []
    for neighbor_count in (1, 5, 15):
        boundary_model = OptimizedKNNScratch(
            n_neighbors=neighbor_count,
            p=2,
            weights="uniform",
            standardize=False,
        ).fit(model.X_train, y_train)
        flat_predictions = boundary_model.predict(grid)
        predictions = [
            flat_predictions[index : index + len(x_values)]
            for index in range(0, len(flat_predictions), len(x_values))
        ]
        panels.append({"k": neighbor_count, "predictions": predictions})

    boundaries = {
        "x_values": x_values,
        "y_values": y_values,
        "train_x": model.X_train,
        "train_y": list(y_train),
        "feature_names": feature_names,
        "panels": panels,
    }
    return neighborhood, boundaries


def _linear_space(start, stop, count):
    step = (stop - start) / (count - 1)
    return [start + index * step for index in range(count)]


def select_k_full_sort(distances, k):
    """对全部距离排序后取前 K 个。"""
    return sorted(distances)[:k]


def select_k_heap(distances, k):
    """使用大小约为 K 的堆选择最小的 K 个距离。"""
    return heapq.nsmallest(k, distances)


def _benchmark_search(X_train, X_test, k=5, repeats=3):
    # 先计算距离，再单独测量近邻选择阶段，避免 O(nd) 距离计算掩盖排序差异。
    reference_distances = [_distance(X_test[0], row) for row in X_train]
    train_sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
    selections = 20
    full_sort_ms = []
    heap_ms = []

    for train_size in train_sizes:
        distances = [
            reference_distances[index % len(reference_distances)]
            for index in range(train_size)
        ]

        def full_sort_search():
            for _ in range(selections):
                select_k_full_sort(distances, k)

        def heap_search():
            for _ in range(selections):
                select_k_heap(distances, k)

        full_times = []
        heap_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            full_sort_search()
            full_times.append((time.perf_counter() - start) * 1_000.0)
            start = time.perf_counter()
            heap_search()
            heap_times.append((time.perf_counter() - start) * 1_000.0)
        full_sort_ms.append(statistics.median(full_times))
        heap_ms.append(statistics.median(heap_times))

    return {
        "train_sizes": train_sizes,
        "full_sort_ms": full_sort_ms,
        "heap_ms": heap_ms,
        "selections": selections,
        "repeats": repeats,
    }


def run_experiment(data_path, seed=42, n_splits=5):
    """运行 KNN 六图所需的全部真实数据实验。"""
    features, targets = load_wdbc(data_path)
    X_train, X_test, y_train, y_test = stratified_split(features, targets, seed=seed)
    validation = validation_curves(
        X_train,
        y_train,
        k_values=[1, 3, 5, 7, 9, 11, 15, 19],
        n_splits=n_splits,
        seed=seed,
    )
    best_k = validation["best_k"]

    base_model = KNNScratch(n_neighbors=best_k).fit(X_train, y_train)
    optimized_model = OptimizedKNNScratch(
        n_neighbors=best_k,
        p=2,
        weights="distance",
        standardize=True,
    ).fit(X_train, y_train)
    base_predictions = base_model.predict(X_test)
    optimized_predictions = optimized_model.predict(X_test)
    base_metrics = classification_metrics(y_test, base_predictions)
    optimized_metrics = classification_metrics(y_test, optimized_predictions)
    base_scores = _positive_scores(base_model, X_test)
    optimized_scores = _positive_scores(optimized_model, X_test)

    scaled_uniform = OptimizedKNNScratch(
        n_neighbors=best_k, p=2, weights="uniform", standardize=True
    ).fit(X_train, y_train)
    manhattan = OptimizedKNNScratch(
        n_neighbors=best_k, p=1, weights="distance", standardize=True
    ).fit(X_train, y_train)
    optimization_models = [
        ("Base", base_model),
        ("+ Scale", scaled_uniform),
        ("+ Distance weight", optimized_model),
        ("Manhattan p=1", manhattan),
        ("Full heap", optimized_model),
    ]
    optimization = []
    for name, model in optimization_models:
        metrics = classification_metrics(y_test, model.predict(X_test))
        optimization.append(
            {"name": name, "accuracy": metrics["accuracy"], "f1": metrics["f1"]}
        )

    neighborhood, boundaries = _two_feature_report(
        X_train, X_test, y_train, y_test, best_k
    )
    runtime = _benchmark_search(
        optimized_model.X_train,
        optimized_model._transform(X_test),
        k=best_k,
    )
    return {
        "metadata": {
            "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "seed": seed,
            "folds": n_splits,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "best_k": best_k,
        },
        "neighborhood": neighborhood,
        "boundaries": boundaries,
        "validation": validation,
        "diagnostics": {
            "Base": _curve_report(y_test, base_scores),
            "Optimized": _curve_report(y_test, optimized_scores),
        },
        "confusion": {
            "Base": base_metrics["confusion"],
            "Optimized": optimized_metrics["confusion"],
        },
        "metrics": {"Base": base_metrics, "Optimized": optimized_metrics},
        "optimization": optimization,
        "runtime": runtime,
    }


def _style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.8, length=3)


def _save_png(figure, output_dir, stem):
    path = output_dir / f"{stem}.png"
    figure.savefig(path, format="png", dpi=300)
    plt.close(figure)
    return path


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _scatter_classes(axis, points, labels, colors, size=14, alpha=0.72):
    for label, name, color, marker in (
        (0, "Benign", colors[0], "o"),
        (1, "Malignant", colors[1], "^"),
    ):
        selected = [point for point, target in zip(points, labels) if target == label]
        axis.scatter(
            [point[0] for point in selected],
            [point[1] for point in selected],
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.35,
            alpha=alpha,
            label=name,
        )


def _plot_neighborhood(report):
    palette = FIGURE_PALETTES["neighborhood"]
    data = report["neighborhood"]
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    _style_axis(axis)
    _scatter_classes(axis, data["train_x"], data["train_y"], palette)
    neighbor_points = [data["train_x"][record["index"]] for record in data["neighbors"]]
    for point in neighbor_points:
        axis.plot(
            [data["query"][0], point[0]],
            [data["query"][1], point[1]],
            color=palette[3],
            linewidth=0.7,
            alpha=0.55,
            zorder=2,
        )
    radius = max(record["distance"] for record in data["neighbors"])
    axis.add_patch(
        plt.Circle(
            data["query"],
            radius,
            fill=False,
            color=palette[2],
            linestyle="--",
            linewidth=1.3,
        )
    )
    axis.scatter(
        [data["query"][0]],
        [data["query"][1]],
        s=95,
        marker="*",
        color=palette[2],
        edgecolor=palette[3],
        linewidth=0.8,
        label="Query",
        zorder=5,
    )
    axis.set_xlabel(data["feature_names"][0])
    axis.set_ylabel(data["feature_names"][1])
    axis.set_title("Local neighborhood and distance-weighted vote")
    axis.legend(ncols=3, loc="upper center")
    axis.set_aspect("equal", adjustable="datalim")
    return figure


def _plot_boundaries(report):
    palette = FIGURE_PALETTES["boundaries"]
    data = report["boundaries"]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True, sharey=True)
    for axis, panel in zip(axes, data["panels"]):
        axis.contourf(
            data["x_values"],
            data["y_values"],
            panel["predictions"],
            levels=[-0.5, 0.5, 1.5],
            colors=[palette[2], palette[3]],
            alpha=0.85,
        )
        _scatter_classes(axis, data["train_x"], data["train_y"], palette, size=8, alpha=0.72)
        _style_axis(axis)
        axis.set_title(f"K = {panel['k']}")
        axis.set_xlabel(data["feature_names"][0])
    axes[0].set_ylabel(data["feature_names"][1])
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("Decision-boundary sensitivity to K", y=1.10, fontsize=9)
    return figure


def _plot_validation(report):
    palette = FIGURE_PALETTES["validation"]
    data = report["validation"]
    figure, axis = plt.subplots(figsize=(4.8, 3.1))
    _style_axis(axis)
    styles = [
        ("Base accuracy", palette[0], "-", "o"),
        ("Base F1", palette[0], "--", "s"),
        ("Optimized accuracy", palette[1], "-", "o"),
        ("Optimized F1", palette[1], "--", "s"),
    ]
    for name, color, line_style, marker in styles:
        values = data["series"][name]
        axis.errorbar(
            data["k_values"],
            values["mean"],
            yerr=values["std"],
            color=color,
            linestyle=line_style,
            marker=marker,
            markersize=3.5,
            linewidth=1.3,
            capsize=2,
            label=name,
        )
    axis.axvline(data["best_k"], color=palette[2], linestyle=":", linewidth=1.1)
    axis.text(
        data["best_k"],
        axis.get_ylim()[0],
        f"  selected K = {data['best_k']}",
        color=palette[3],
        va="bottom",
        fontsize=7,
    )
    axis.set_xlabel("Number of neighbors, K")
    axis.set_ylabel("Five-fold CV score")
    axis.set_title("K selection with stratified cross-validation")
    axis.set_xticks(data["k_values"])
    axis.legend(ncols=2, loc="lower right")
    return figure


def _plot_diagnostics(report):
    palette = FIGURE_PALETTES["diagnostics"]
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
    for axis in axes:
        _style_axis(axis)
    for model_name, color, line_style in (
        ("Base", palette[0], "--"),
        ("Optimized", palette[1], "-"),
    ):
        diagnostics = report["diagnostics"][model_name]
        roc = diagnostics["roc"]
        pr = diagnostics["pr"]
        axes[0].plot(
            roc["fpr"], roc["tpr"], color=color, linestyle=line_style,
            linewidth=1.6, label=f"{model_name} (AUC = {roc['auc']:.3f})"
        )
        axes[1].step(
            pr["recall"], pr["precision"], where="post", color=color,
            linestyle=line_style, linewidth=1.6,
            label=f"{model_name} (AP = {pr['ap']:.3f})"
        )
    axes[0].plot([0, 1], [0, 1], color=palette[2], linestyle=":", linewidth=0.9)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].set_title("ROC curve")
    axes[1].set_title("Precision–recall curve")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")
    figure.suptitle("Full-feature test-set discrimination", y=1.02, fontsize=9)
    return figure


def _draw_confusion(axis, matrix, title, cmap, text_color):
    image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=max(max(row) for row in matrix))
    midpoint = max(max(row) for row in matrix) / 2.0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > midpoint else text_color,
            )
    axis.set_xticks([0, 1], ["Benign", "Malignant"])
    axis.set_yticks([0, 1], ["Benign", "Malignant"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    return image


def _plot_confusion(report):
    palette = FIGURE_PALETTES["confusion"]
    figure, axes = plt.subplots(1, 2, figsize=(6.2, 2.75))
    _draw_confusion(axes[0], report["confusion"]["Base"], "Base KNN", palette[0], palette[2])
    _draw_confusion(
        axes[1], report["confusion"]["Optimized"], "Optimized KNN", palette[1], palette[2]
    )
    figure.suptitle("Test-set error structure", y=1.03, fontsize=9)
    return figure


def _plot_optimization(report):
    palette = FIGURE_PALETTES["optimization"]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for axis in axes:
        _style_axis(axis)
    records = report["optimization"]
    positions = list(range(len(records)))
    width = 0.36
    accuracy_bars = axes[0].bar(
        [position - width / 2 for position in positions],
        [record["accuracy"] for record in records],
        width=width,
        color=palette[0],
        edgecolor="#333333",
        linewidth=0.5,
        label="Accuracy",
    )
    f1_bars = axes[0].bar(
        [position + width / 2 for position in positions],
        [record["f1"] for record in records],
        width=width,
        color=palette[1],
        edgecolor="#333333",
        linewidth=0.5,
        hatch="//",
        label="F1 score",
    )
    axes[0].bar_label(accuracy_bars, fmt="%.2f", padding=-13, fontsize=6, rotation=90)
    axes[0].bar_label(f1_bars, fmt="%.2f", padding=-13, fontsize=6, rotation=90)
    axes[0].set_xticks(positions, [record["name"] for record in records], rotation=24, ha="right")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Test-set score")
    axes[0].set_title("Predictive ablation")
    axes[0].legend(ncols=2, loc="upper center")

    runtime = report["runtime"]
    axes[1].plot(
        runtime["train_sizes"], runtime["full_sort_ms"], color=palette[2],
        marker="o", linewidth=1.5, markersize=4, label="Full sort"
    )
    axes[1].plot(
        runtime["train_sizes"], runtime["heap_ms"], color=palette[3],
        marker="s", linewidth=1.5, markersize=4, label="Heap selection"
    )
    axes[1].set_xlabel("Training-set size")
    axes[1].set_xscale("log")
    axes[1].set_ylabel(f"Selection time (ms, {runtime['selections']} runs)")
    axes[1].set_title("Neighbor-selection scaling")
    axes[1].legend(loc="upper left")
    figure.suptitle("Optimization evidence", y=1.02, fontsize=9)
    return figure


def render_six_figures(report, output_dir):
    """分别导出六张 300 DPI PNG 及六份对应源数据。"""
    output_dir = Path(output_dir)
    source_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("01_local_neighborhood", _plot_neighborhood),
        ("02_k_decision_boundaries", _plot_boundaries),
        ("03_k_validation_curve", _plot_validation),
        ("04_roc_pr_comparison", _plot_diagnostics),
        ("05_confusion_matrices", _plot_confusion),
        ("06_optimization_and_runtime", _plot_optimization),
    ]
    paths = [_save_png(plotter(report), output_dir, stem) for stem, plotter in plots]

    neighborhood = report["neighborhood"]
    neighbor_by_index = {record["index"]: record for record in neighborhood["neighbors"]}
    neighborhood_rows = []
    for index, (point, label) in enumerate(zip(neighborhood["train_x"], neighborhood["train_y"])):
        neighbor = neighbor_by_index.get(index, {})
        neighborhood_rows.append(
            ("train", index, point[0], point[1], label, neighbor.get("distance", ""), neighbor.get("weight", ""))
        )
    neighborhood_rows.append(
        ("query", "", neighborhood["query"][0], neighborhood["query"][1], neighborhood["query_label"], "", "")
    )
    _write_csv(
        source_dir / "01_local_neighborhood.csv",
        ["point_type", "index", "x", "y", "label", "distance_to_query", "vote_weight"],
        neighborhood_rows,
    )

    boundaries = report["boundaries"]
    boundary_rows = []
    for panel in boundaries["panels"]:
        for row_index, y_value in enumerate(boundaries["y_values"]):
            boundary_rows.extend(
                (panel["k"], x_value, y_value, panel["predictions"][row_index][column_index])
                for column_index, x_value in enumerate(boundaries["x_values"])
            )
    _write_csv(
        source_dir / "02_k_decision_boundaries.csv",
        ["k", "x", "y", "predicted_class"],
        boundary_rows,
    )

    validation = report["validation"]
    validation_rows = []
    for series_name, values in validation["series"].items():
        validation_rows.extend(
            (series_name, k, mean, std, validation["best_k"])
            for k, mean, std in zip(validation["k_values"], values["mean"], values["std"])
        )
    _write_csv(
        source_dir / "03_k_validation_curve.csv",
        ["series", "k", "mean", "std", "selected_k"],
        validation_rows,
    )

    diagnostic_rows = []
    for model_name, diagnostics in report["diagnostics"].items():
        diagnostic_rows.extend(
            (model_name, "ROC", x, y, diagnostics["roc"]["auc"])
            for x, y in zip(diagnostics["roc"]["fpr"], diagnostics["roc"]["tpr"])
        )
        diagnostic_rows.extend(
            (model_name, "PR", x, y, diagnostics["pr"]["ap"])
            for x, y in zip(diagnostics["pr"]["recall"], diagnostics["pr"]["precision"])
        )
    _write_csv(
        source_dir / "04_roc_pr_comparison.csv",
        ["model", "curve", "x", "y", "area"],
        diagnostic_rows,
    )

    confusion_rows = []
    for model_name, matrix in report["confusion"].items():
        for true_label, row in enumerate(matrix):
            confusion_rows.extend(
                (model_name, true_label, predicted_label, count)
                for predicted_label, count in enumerate(row)
            )
    _write_csv(
        source_dir / "05_confusion_matrices.csv",
        ["model", "true_class", "predicted_class", "count"],
        confusion_rows,
    )

    optimization_rows = [
        ("performance", record["name"], "accuracy", "", record["accuracy"])
        for record in report["optimization"]
    ]
    optimization_rows.extend(
        ("performance", record["name"], "f1", "", record["f1"])
        for record in report["optimization"]
    )
    runtime = report["runtime"]
    optimization_rows.extend(
        ("runtime", "Full sort", "milliseconds", size, value)
        for size, value in zip(runtime["train_sizes"], runtime["full_sort_ms"])
    )
    optimization_rows.extend(
        ("runtime", "Heap selection", "milliseconds", size, value)
        for size, value in zip(runtime["train_sizes"], runtime["heap_ms"])
    )
    _write_csv(
        source_dir / "06_optimization_and_runtime.csv",
        ["section", "series", "metric", "training_size", "value"],
        optimization_rows,
    )
    return paths


def _print_summary(report, output_dir):
    base = report["metrics"]["Base"]
    optimized = report["metrics"]["Optimized"]
    print(f"Output: {output_dir}")
    print(f"Selected K: {report['metadata']['best_k']}")
    print(f"Base: accuracy={base['accuracy']:.4f}, F1={base['f1']:.4f}")
    print(f"Optimized: accuracy={optimized['accuracy']:.4f}, F1={optimized['f1']:.4f}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate six KNN experiment figures.")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "classification" / "wdbc" / "wdbc.data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "figures" / "knn",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    report = run_experiment(args.data, seed=args.seed, n_splits=args.folds)
    render_six_figures(report, args.output)
    _print_summary(report, args.output)


if __name__ == "__main__":
    main()
