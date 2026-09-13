"""为手写逻辑回归生成六张可复现实验图。

模型训练与指标计算不依赖任何机器学习库；Matplotlib 仅用于绘图。
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Models.logistic_regression_optimized import OptimizedLogisticRegressionScratch


FIGURE_PALETTES = {
    "class_distribution": ["#4C956C", "#D1495B", "#2F3E46"],
    "loss": ["#7A7A7A", "#6C5CE7"],
    "confusion": ["YlGnBu", "#263238"],
    "roc": ["#008C95", "#A7A9AC"],
    "precision_recall": ["#CC6677", "#332288"],
    "ablation": ["#E69F00", "#56B4E9", "#2F2F2F"],
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
    """读取 WDBC：M 为正类（1），B 为负类（0）。"""
    features = []
    targets = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if not row:
                continue
            targets.append(1 if row[1] == "M" else 0)
            features.append([float(value) for value in row[2:]])
    return features, targets


def stratified_split(features, targets, test_size=0.2, seed=42):
    """按类别分别打乱并切分，保证训练集和测试集类别比例接近。"""
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
    X_train = [features[index] for index in train_indices]
    y_train = [targets[index] for index in train_indices]
    X_test = [features[index] for index in test_indices]
    y_test = [targets[index] for index in test_indices]
    return X_train, X_test, y_train, y_test


def classification_metrics(y_true, y_pred):
    """由二分类预测计算混淆矩阵和四项常用指标。"""
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(y_true, y_pred))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(y_true, y_pred))
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(y_true, y_pred))

    accuracy = (tp + tn) / len(y_true)
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
    """按分数降序组织样本；相同分数作为同一个阈值处理。"""
    ordered = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    groups = []
    for score, target in ordered:
        if not groups or score != groups[-1][0]:
            groups.append([score, 0, 0])
        groups[-1][1 if target == 1 else 2] += 1
    return groups


def roc_curve(y_true, scores):
    """手工计算 ROC 坐标及梯形积分 AUC。"""
    positive_count = sum(y_true)
    negative_count = len(y_true) - positive_count
    true_positive = 0
    false_positive = 0
    fpr = [0.0]
    tpr = [0.0]

    for _, group_positive, group_negative in _score_groups(y_true, scores):
        true_positive += group_positive
        false_positive += group_negative
        tpr.append(true_positive / positive_count)
        fpr.append(false_positive / negative_count)

    auc = sum(
        (fpr[index] - fpr[index - 1]) * (tpr[index] + tpr[index - 1]) / 2.0
        for index in range(1, len(fpr))
    )
    return fpr, tpr, auc


def precision_recall_curve(y_true, scores):
    """手工计算 PR 坐标与非插值平均精确率（AP）。"""
    positive_count = sum(y_true)
    true_positive = 0
    false_positive = 0
    recall = [0.0]
    precision = [1.0]
    average_precision = 0.0

    for _, group_positive, group_negative in _score_groups(y_true, scores):
        previous_recall = true_positive / positive_count
        true_positive += group_positive
        false_positive += group_negative
        current_recall = true_positive / positive_count
        current_precision = true_positive / (true_positive + false_positive)
        recall.append(current_recall)
        precision.append(current_precision)
        average_precision += (current_recall - previous_recall) * current_precision

    return recall, precision, average_precision


def _train_variant(X_train, y_train, X_test, y_test, settings, max_iter):
    model = OptimizedLogisticRegressionScratch(max_iter=max_iter, **settings)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = [pair[1] for pair in model.predict_proba(X_test)]
    metrics = classification_metrics(y_test, predictions)
    return model, metrics, scores


def run_experiment(data_path, seed=42, max_iter=600):
    """在同一分层划分上运行完整模型和累计消融实验。"""
    features, targets = load_wdbc(data_path)
    X_train, X_test, y_train, y_test = stratified_split(features, targets, seed=seed)

    variants = [
        (
            "Base",
            {
                "learning_rate": 1e-7,
                "l2": 0.0,
                "tol": 0.0,
                "standardize": False,
                "class_weight": None,
            },
        ),
        (
            "+ Standardization",
            {
                "learning_rate": 0.1,
                "l2": 0.0,
                "tol": 0.0,
                "standardize": True,
                "class_weight": None,
            },
        ),
        (
            "+ L2",
            {
                "learning_rate": 0.1,
                "l2": 0.01,
                "tol": 0.0,
                "standardize": True,
                "class_weight": None,
            },
        ),
        (
            "+ Class weight",
            {
                "learning_rate": 0.1,
                "l2": 0.01,
                "tol": 0.0,
                "standardize": True,
                "class_weight": "balanced",
            },
        ),
        (
            "Full (+ early stop)",
            {
                "learning_rate": 0.1,
                "l2": 0.01,
                "tol": 1e-5,
                "standardize": True,
                "class_weight": "balanced",
            },
        ),
    ]

    trained = []
    for name, settings in variants:
        model, metrics, scores = _train_variant(
            X_train, y_train, X_test, y_test, settings, max_iter
        )
        trained.append((name, model, metrics, scores))

    base_name, base_model, _, base_scores = trained[0]
    _, full_model, full_metrics, full_scores = trained[-1]
    fpr, tpr, auc = roc_curve(y_test, full_scores)
    recall, precision, average_precision = precision_recall_curve(y_test, full_scores)
    base_recall, base_precision, base_average_precision = precision_recall_curve(
        y_test, base_scores
    )

    return {
        "metadata": {
            "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "seed": seed,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "positive_class": "Malignant",
        },
        "class_counts": {"Benign": targets.count(0), "Malignant": targets.count(1)},
        "loss_series": {
            "Base configuration": base_model.loss_history,
            "Full optimized": full_model.loss_history,
        },
        "early_stop_iteration": full_model.n_iter,
        "confusion": full_metrics["confusion"],
        "metrics": full_metrics,
        "roc": {"fpr": fpr, "tpr": tpr, "auc": auc},
        "pr": {"recall": recall, "precision": precision, "ap": average_precision},
        "pr_comparison": {
            "Base configuration": {
                "recall": base_recall,
                "precision": base_precision,
                "ap": base_average_precision,
            },
            "Full optimized": {
                "recall": recall,
                "precision": precision,
                "ap": average_precision,
            },
        },
        "ablation": [
            {
                "name": name,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "iterations": model.n_iter,
            }
            for name, model, metrics, _ in trained
        ],
        "base_name": base_name,
    }


def _new_figure(figsize=(3.5, 2.7)):
    figure, axis = plt.subplots(figsize=figsize)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.8, length=3)
    return figure, axis


def _save_figure(figure, output_dir, stem):
    path = output_dir / f"{stem}.png"
    figure.savefig(path, format="png", dpi=300)
    plt.close(figure)
    return path


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _plot_class_distribution(report):
    figure, axis = _new_figure()
    benign, malignant, edge = FIGURE_PALETTES["class_distribution"]
    labels = list(report["class_counts"])
    counts = [report["class_counts"][label] for label in labels]
    bars = axis.bar(
        labels,
        counts,
        color=[benign, malignant],
        width=0.62,
        edgecolor=edge,
        linewidth=0.6,
    )
    axis.bar_label(bars, padding=3, fontsize=7)
    axis.set_ylabel("Number of samples")
    axis.set_title("WDBC class distribution")
    axis.set_ylim(0, max(counts) * 1.18)
    return figure


def _plot_loss(report):
    figure, axis = _new_figure((4.0, 2.8))
    base_color, optimized_color = FIGURE_PALETTES["loss"]
    styles = [
        (base_color, "--", "Base configuration"),
        (optimized_color, "-", "Full optimized"),
    ]
    for color, line_style, label in styles:
        values = report["loss_series"][label]
        axis.plot(
            range(1, len(values) + 1),
            values,
            color=color,
            linestyle=line_style,
            linewidth=1.5,
            label=label,
        )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Training objective")
    axis.set_title("Optimization convergence")
    axis.legend(loc="best")
    axis.set_xlim(left=1)
    return figure


def _plot_confusion(report):
    figure, axis = _new_figure((3.2, 2.8))
    color_map, text_color = FIGURE_PALETTES["confusion"]
    matrix = report["confusion"]
    image = axis.imshow(matrix, cmap=color_map, vmin=0)
    midpoint = max(max(row) for row in matrix) / 2.0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > midpoint else text_color,
                fontsize=9,
            )
    axis.set_xticks([0, 1], ["Benign", "Malignant"])
    axis.set_yticks([0, 1], ["Benign", "Malignant"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Test-set confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")
    return figure


def _plot_roc(report):
    figure, axis = _new_figure((3.3, 3.0))
    curve_color, reference_color = FIGURE_PALETTES["roc"]
    roc = report["roc"]
    axis.plot(
        roc["fpr"],
        roc["tpr"],
        color=curve_color,
        linewidth=1.7,
        label=f"Handwritten LR (AUC = {roc['auc']:.3f})",
    )
    axis.plot([0, 1], [0, 1], color=reference_color, linestyle="--", linewidth=0.9)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axis.set_title("ROC curve")
    axis.legend(loc="lower right")
    axis.set_aspect("equal", adjustable="box")
    return figure


def _plot_precision_recall(report):
    figure, axis = _new_figure((3.3, 3.0))
    base_color, optimized_color = FIGURE_PALETTES["precision_recall"]
    styles = [
        ("Base configuration", base_color, "--"),
        ("Full optimized", optimized_color, "-"),
    ]
    for label, color, line_style in styles:
        pr = report["pr_comparison"][label]
        axis.step(
            pr["recall"],
            pr["precision"],
            where="post",
            color=color,
            linestyle=line_style,
            linewidth=1.7,
            label=f"{label} (AP = {pr['ap']:.3f})",
        )
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.set_title("Precision–recall curve")
    axis.legend(loc="lower left")
    axis.set_aspect("equal", adjustable="box")
    return figure


def _plot_ablation(report):
    figure, axis = _new_figure((5.2, 3.0))
    accuracy_color, f1_color, edge_color = FIGURE_PALETTES["ablation"]
    records = report["ablation"]
    positions = list(range(len(records)))
    bar_width = 0.36
    accuracy_bars = axis.bar(
        [position - bar_width / 2 for position in positions],
        [record["accuracy"] for record in records],
        width=bar_width,
        color=accuracy_color,
        edgecolor=edge_color,
        linewidth=0.5,
        label="Accuracy",
    )
    f1_bars = axis.bar(
        [position + bar_width / 2 for position in positions],
        [record["f1"] for record in records],
        width=bar_width,
        color=f1_color,
        edgecolor=edge_color,
        linewidth=0.5,
        hatch="//",
        label="F1 score",
    )
    axis.bar_label(
        accuracy_bars,
        fmt="%.2f",
        padding=-14,
        fontsize=6,
        rotation=90,
        color=edge_color,
    )
    axis.bar_label(
        f1_bars,
        fmt="%.2f",
        padding=-14,
        fontsize=6,
        rotation=90,
        color=edge_color,
    )
    axis.set_xticks(positions, [record["name"] for record in records], rotation=18, ha="right")
    axis.set_ylabel("Test-set score")
    axis.set_ylim(0, 1.13)
    axis.set_title("Cumulative optimization ablation")
    axis.legend(ncols=2, loc="upper center")
    return figure


def render_six_figures(report, output_dir):
    """将六张 PNG 单图及其源数据分别写入输出目录。"""
    output_dir = Path(output_dir)
    source_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("01_class_distribution", _plot_class_distribution),
        ("02_loss_curve", _plot_loss),
        ("03_confusion_matrix", _plot_confusion),
        ("04_roc_curve", _plot_roc),
        ("05_precision_recall_curve", _plot_precision_recall),
        ("06_optimization_ablation", _plot_ablation),
    ]
    paths = [
        _save_figure(plotter(report), output_dir, stem)
        for stem, plotter in plots
    ]

    _write_csv(
        source_dir / "01_class_distribution.csv",
        ["class", "count"],
        report["class_counts"].items(),
    )
    loss_rows = []
    for series, values in report["loss_series"].items():
        loss_rows.extend((series, iteration, value) for iteration, value in enumerate(values, 1))
    _write_csv(source_dir / "02_loss_curve.csv", ["series", "iteration", "loss"], loss_rows)
    _write_csv(
        source_dir / "03_confusion_matrix.csv",
        ["true_class", "predicted_class", "count"],
        [
            ("Benign", "Benign", report["confusion"][0][0]),
            ("Benign", "Malignant", report["confusion"][0][1]),
            ("Malignant", "Benign", report["confusion"][1][0]),
            ("Malignant", "Malignant", report["confusion"][1][1]),
        ],
    )
    _write_csv(
        source_dir / "04_roc_curve.csv",
        ["false_positive_rate", "true_positive_rate", "auc"],
        zip(report["roc"]["fpr"], report["roc"]["tpr"], [report["roc"]["auc"]] * len(report["roc"]["fpr"])),
    )
    pr_rows = []
    for series, values in report["pr_comparison"].items():
        pr_rows.extend(
            (series, recall, precision, values["ap"])
            for recall, precision in zip(values["recall"], values["precision"])
        )
    _write_csv(
        source_dir / "05_precision_recall_curve.csv",
        ["series", "recall", "precision", "average_precision"],
        pr_rows,
    )
    _write_csv(
        source_dir / "06_optimization_ablation.csv",
        ["configuration", "accuracy", "f1", "iterations"],
        (
            (record["name"], record["accuracy"], record["f1"], record.get("iterations", ""))
            for record in report["ablation"]
        ),
    )
    return paths


def _print_summary(report, output_dir):
    metrics = report["metrics"]
    print(f"Output: {output_dir}")
    print(
        "Test metrics: "
        f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
        f"AUC={report['roc']['auc']:.4f}, AP={report['pr']['ap']:.4f}"
    )
    print(f"Iterations: {report['early_stop_iteration']}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate six logistic-regression figures.")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "classification" / "wdbc" / "wdbc.data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "figures" / "logistic_regression",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=600)
    args = parser.parse_args()

    report = run_experiment(args.data, seed=args.seed, max_iter=args.max_iter)
    render_six_figures(report, args.output)
    _print_summary(report, args.output)


if __name__ == "__main__":
    main()
