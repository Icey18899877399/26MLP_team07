"""One-Class SVM 六张独立科研级 PNG 图表。

仅使用 NumPy 读取 NPZ 和 Matplotlib 绘图；模型训练调用 Models 中的
纯 Python 手写实现，不依赖机器学习库。
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from Models.one_class_svm import OneClassSVMScratch
from Models.one_class_svm_optimized import OptimizedOneClassSVMScratch


FIGURE_FILENAMES = (
    "01_boundary_mechanism.png",
    "02_dual_kkt_structure.png",
    "03_optimization_convergence.png",
    "04_scores_roc_pr.png",
    "05_parameter_robustness.png",
    "06_cross_dataset_benchmark.png",
)

PALETTES = {
    FIGURE_FILENAMES[0]: ("#173F5F", "#2A9D8F", "#E76F51", "#F4A261", "#E9C46A"),
    FIGURE_FILENAMES[1]: ("#3D405B", "#81B29A", "#E07A5F", "#F2CC8F", "#B8B8D1"),
    FIGURE_FILENAMES[2]: ("#264653", "#E76F51", "#2A9D8F", "#E9C46A", "#8AB17D"),
    FIGURE_FILENAMES[3]: ("#4C2A85", "#1B998B", "#E84855", "#F9DC5C", "#C5CBE3"),
    FIGURE_FILENAMES[4]: ("#003F5C", "#58508D", "#BC5090", "#FF6361", "#FFA600"),
    FIGURE_FILENAMES[5]: ("#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 300,
    }
)


def load_npz_dataset(path):
    """读取 ODDS 风格的 X/y NPZ，转换为手写模型所需的列表。"""
    with np.load(Path(path), allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=float)
        y = np.asarray(data["y"], dtype=int).reshape(-1)
    return {"X": X.tolist(), "y": y.tolist()}


def stratified_split(labels, seed=17, train_ratio=0.6, validation_ratio=0.2):
    """按类别分层划分，返回可审计的原始行号。"""
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    rng = random.Random(seed)
    result = {"train": [], "validation": [], "test": []}
    for indices in groups.values():
        indices = list(indices)
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        n_validation = int(len(indices) * validation_ratio)
        result["train"].extend(indices[:n_train])
        result["validation"].extend(indices[n_train : n_train + n_validation])
        result["test"].extend(indices[n_train + n_validation :])
    for name in result:
        result[name].sort()
    return result


def fit_scale_only(X):
    """仅缩放、不中心化，避免破坏线性 OCSVM 的原点几何。"""
    feature_count = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(feature_count)]
    scales = []
    for j, mean in enumerate(means):
        variance = sum((row[j] - mean) ** 2 for row in X) / len(X)
        scales.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return scales


def transform_scale_only(X, scales):
    return [[value / scales[j] for j, value in enumerate(row)] for row in X]


def select_clean_reference(indices, labels, limit, seed):
    """从训练分区选取已知正常样本，用于 novelty-detection 协议。"""
    clean = [index for index in indices if int(labels[index]) == 0]
    rng = random.Random(seed)
    rng.shuffle(clean)
    return sorted(clean[:limit])


def support_categories(alphas, cap, tolerance=1e-8):
    categories = []
    for alpha in alphas:
        if alpha <= tolerance:
            categories.append("zero")
        elif alpha >= cap - tolerance:
            categories.append("capped")
        else:
            categories.append("free")
    return categories


def effective_support_fraction(alphas, cap, relative_tolerance=0.01):
    """忽略优化残留的数值尘埃，统计对核展开有实质贡献的系数。"""
    threshold = max(1e-12, float(cap) * float(relative_tolerance))
    return sum(alpha > threshold for alpha in alphas) / len(alphas)


def binary_ranking_curves(labels, anomaly_scores):
    """计算并列分数安全的 ROC/PR 曲线、AUROC 和 Average Precision。"""
    pairs = sorted(
        [(float(score), int(label)) for score, label in zip(anomaly_scores, labels)],
        key=lambda item: item[0], reverse=True,
    )
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("labels must contain both normal and anomaly samples")
    fpr, tpr, recall, precision = [0.0], [0.0], [0.0], [1.0]
    true_positive = false_positive = 0
    average_precision = 0.0
    cursor = 0
    while cursor < len(pairs):
        end = cursor + 1
        while end < len(pairs) and pairs[end][0] == pairs[cursor][0]:
            end += 1
        group = pairs[cursor:end]
        group_positive = sum(label for _, label in group)
        true_positive += group_positive
        false_positive += len(group) - group_positive
        new_recall = true_positive / positives
        new_precision = true_positive / (true_positive + false_positive)
        average_precision += (new_recall - recall[-1]) * new_precision
        fpr.append(false_positive / negatives)
        tpr.append(new_recall)
        recall.append(new_recall)
        precision.append(new_precision)
        cursor = end
    roc_auc = sum(
        (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
        for i in range(1, len(fpr))
    )
    return {
        "fpr": fpr, "tpr": tpr, "recall": recall, "precision": precision,
        "roc_auc": roc_auc, "average_precision": average_precision,
    }


def threshold_metrics(labels, decisions):
    """使用模型固有 decision=0 阈值，不从测试集选阈值。"""
    tn = fp = fn = tp = 0
    for label, decision in zip(labels, decisions):
        predicted_anomaly = float(decision) < 0.0
        if int(label) == 1 and predicted_anomaly:
            tp += 1
        elif int(label) == 1:
            fn += 1
        elif predicted_anomaly:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low, high = int(math.floor(position)), int(math.ceil(position))
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _summary(values, timing=False):
    values = [float(value) for value in values]
    if timing:
        return {"median": statistics.median(values), "q1": _quantile(values, 0.25), "q3": _quantile(values, 0.75), "values": values}
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.mean(values), "std": spread, "values": values}


def injection_count(clean_count, final_fraction):
    """计算需追加的异常样本数，使其占最终训练集的给定比例。"""
    if not 0.0 <= final_fraction < 1.0:
        raise ValueError("final_fraction must be in [0, 1)")
    return round(clean_count * final_fraction / max(1.0 - final_fraction, 1e-15))


def stratified_subsample(indices, labels, max_total, seed):
    """限额抽样时保留该分区的原始异常率。"""
    anomalies = [index for index in indices if labels[index] == 1]
    normals = [index for index in indices if labels[index] == 0]
    rng = random.Random(seed)
    rng.shuffle(anomalies)
    rng.shuffle(normals)
    target_total = min(int(max_total), len(indices))
    anomaly_limit = round(target_total * len(anomalies) / len(indices))
    if anomalies and target_total:
        anomaly_limit = max(1, anomaly_limit)
    anomaly_limit = min(len(anomalies), anomaly_limit)
    normal_limit = min(len(normals), target_total - anomaly_limit)
    anomaly_limit = min(len(anomalies), target_total - normal_limit)
    selected = anomalies[:anomaly_limit] + normals[:normal_limit]
    rng.shuffle(selected)
    return selected


def _rows(X, indices):
    return [list(X[index]) for index in indices]


def _prepare_dataset(path, seed, train_cap, evaluation_cap):
    loaded = load_npz_dataset(path)
    X, y = loaded["X"], loaded["y"]
    split = stratified_split(y, seed=seed)
    train_ids = select_clean_reference(split["train"], y, train_cap, seed + 101)
    validation_ids = stratified_subsample(split["validation"], y, evaluation_cap, seed + 211)
    test_ids = stratified_subsample(split["test"], y, evaluation_cap, seed + 307)
    train_raw = _rows(X, train_ids)
    validation_raw, test_raw = _rows(X, validation_ids), _rows(X, test_ids)
    scales = fit_scale_only(train_raw)
    return {
        "X": X, "y": y, "split": split,
        "train_ids": train_ids, "validation_ids": validation_ids, "test_ids": test_ids,
        "train_raw": train_raw, "validation_raw": validation_raw, "test_raw": test_raw,
        "validation_y": [y[index] for index in validation_ids],
        "test_y": [y[index] for index in test_ids], "scales": scales,
        "train_basic": transform_scale_only(train_raw, scales),
        "validation_basic": transform_scale_only(validation_raw, scales),
        "test_basic": transform_scale_only(test_raw, scales),
    }


def _new_basic(params, quick):
    return OneClassSVMScratch(
        nu=params["nu"], learning_rate=params["learning_rate"],
        max_iter=35 if quick else 90, tol=1e-5,
    )


def _new_optimized(params, quick):
    return OptimizedOneClassSVMScratch(
        nu=params["nu"], kernel="rbf", gamma=params["gamma"], standardize=True,
        max_iter=45 if quick else 120, tol=1e-5,
    )


def _evaluate(model, X, labels):
    decisions = model.decision_function(X)
    curves = binary_ranking_curves(labels, [-value for value in decisions])
    return {"decisions": decisions, "curves": curves, **threshold_metrics(labels, decisions)}


def _tune_basic(prepared, quick):
    candidates = [{"nu": 0.05, "learning_rate": 0.6}, {"nu": 0.15, "learning_rate": 0.6}]
    if not quick:
        candidates.append({"nu": 0.25, "learning_rate": 1.0})
    trials = []
    for params in candidates:
        model = _new_basic(params, quick).fit(prepared["train_basic"])
        result = _evaluate(model, prepared["validation_basic"], prepared["validation_y"])
        trials.append((result["curves"]["average_precision"], params))
    return max(trials, key=lambda item: (item[0], -item[1]["nu"]))[1]


def _tune_optimized(prepared, quick):
    nus = [0.1, 0.5] if quick else [0.1, 0.3, 0.5, 0.7]
    gammas = [0.25, 1.0] if quick else [0.25, 0.5, 1.0, 2.0]
    trials, heatmap = [], []
    for nu in nus:
        row = []
        for gamma in gammas:
            params = {"nu": nu, "gamma": gamma}
            model = _new_optimized(params, quick).fit(prepared["train_raw"])
            result = _evaluate(model, prepared["validation_raw"], prepared["validation_y"])
            ap = result["curves"]["average_precision"]
            trials.append((ap, params))
            row.append(ap)
        heatmap.append(row)
    best = max(trials, key=lambda item: (item[0], -item[1]["nu"], -item[1]["gamma"]))[1]
    return best, {"nus": nus, "gammas": gammas, "values": heatmap}


def _fit_and_time(factory, X_train, X_test, repeats):
    """预热后返回中位数所需的原始计时样本。"""
    factory().fit(X_train).decision_function(X_test[: min(8, len(X_test))])
    fit_times, score_times, models = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        model = factory().fit(X_train)
        fit_times.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        model.decision_function(X_test)
        score_times.append((time.perf_counter() - start) * 1000.0)
        models.append(model)
    return models[-1], fit_times, score_times


def _synthetic_data(seed=7, n_normal=72, n_anomaly=28):
    rng = random.Random(seed)
    normals = []
    for _ in range(n_normal):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.gauss(1.15, 0.16)
        normals.append([2.8 + radius * math.cos(angle), 2.8 + 0.78 * radius * math.sin(angle)])
    anomalies = []
    while len(anomalies) < n_anomaly:
        point = [rng.uniform(0.3, 5.3), rng.uniform(0.5, 5.1)]
        radius = math.sqrt((point[0] - 2.8) ** 2 + ((point[1] - 2.8) / 0.78) ** 2)
        if radius < 0.65 or radius > 1.75:
            anomalies.append(point)
    return normals, anomalies


def _mechanism_experiment(quick):
    normals, anomalies = _synthetic_data(n_normal=48 if quick else 72)
    scales = fit_scale_only(normals)
    basic = _new_basic({"nu": 0.12, "learning_rate": 0.8}, quick).fit(
        transform_scale_only(normals, scales)
    )
    optimized = _new_optimized({"nu": 0.4, "gamma": 1.0}, quick).fit(normals)
    resolution = 55 if quick else 95
    xs = np.linspace(0.1, 5.5, resolution)
    ys = np.linspace(0.3, 5.3, resolution)
    grid = [[float(x), float(y)] for y in ys for x in xs]
    training_scores = optimized.score_samples(normals)
    cap = 1.0 / (optimized.nu * len(normals))
    categories = support_categories(optimized.alphas_, cap, tolerance=cap * 0.01)
    return {
        "normals": normals, "anomalies": anomalies,
        "origin": [0.0, 0.0],
        "xs": xs.tolist(), "ys": ys.tolist(),
        "basic_decision": basic.decision_function(transform_scale_only(grid, scales)),
        "optimized_decision": optimized.decision_function(grid),
        "basic_support": basic.support_indices_,
        "optimized_support": [i for i, category in enumerate(categories) if category != "zero"],
        "alphas": optimized.alphas_, "cap": cap,
        "categories": categories,
        "training_decisions": [score - optimized.offset_ for score in training_scores],
        "basic_objective": basic.objective_history_,
        "optimized_objective": optimized.objective_history_,
    }


def _runtime_experiment(quick):
    sizes = [12, 18, 24] if quick else [24, 48, 72, 96]
    repeats = 2 if quick else 3
    basic_times, optimized_times, basic_residual, optimized_residual = [], [], [], []
    basic_q1_values, basic_q3_values = [], []
    optimized_q1_values, optimized_q3_values = [], []
    for size in sizes:
        normals, _ = _synthetic_data(seed=size, n_normal=size, n_anomaly=8)
        scales = fit_scale_only(normals)
        basic_X = transform_scale_only(normals, scales)
        b_values, o_values = [], []
        b_last = o_last = None
        # 两种方法分别预热，预热结果不进入计时统计。
        _new_basic({"nu": 0.12, "learning_rate": 0.8}, quick).fit(basic_X)
        _new_optimized({"nu": 0.12, "gamma": 1.0}, quick).fit(normals)
        for repeat in range(repeats):
            if repeat % 2 == 0:
                start = time.perf_counter()
                b_last = _new_basic({"nu": 0.12, "learning_rate": 0.8}, quick).fit(basic_X)
                b_values.append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                o_last = _new_optimized({"nu": 0.12, "gamma": 1.0}, quick).fit(normals)
                o_values.append((time.perf_counter() - start) * 1000.0)
            else:
                start = time.perf_counter()
                o_last = _new_optimized({"nu": 0.12, "gamma": 1.0}, quick).fit(normals)
                o_values.append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                b_last = _new_basic({"nu": 0.12, "learning_rate": 0.8}, quick).fit(basic_X)
                b_values.append((time.perf_counter() - start) * 1000.0)
        basic_times.append(statistics.median(b_values))
        optimized_times.append(statistics.median(o_values))
        basic_q1 = _quantile(b_values, 0.25)
        basic_q3 = _quantile(b_values, 0.75)
        optimized_q1 = _quantile(o_values, 0.25)
        optimized_q3 = _quantile(o_values, 0.75)
        basic_q1_values.append(basic_q1)
        basic_q3_values.append(basic_q3)
        optimized_q1_values.append(optimized_q1)
        optimized_q3_values.append(optimized_q3)
        basic_residual.append(abs(sum(b_last.alphas_) - 1.0))
        optimized_residual.append(abs(sum(o_last.alphas_) - 1.0))
    return {
        "sizes": sizes, "basic_time_ms": basic_times, "optimized_time_ms": optimized_times,
        "basic_time_q1_ms": basic_q1_values, "basic_time_q3_ms": basic_q3_values,
        "optimized_time_q1_ms": optimized_q1_values, "optimized_time_q3_ms": optimized_q3_values,
        "basic_residual": basic_residual, "optimized_residual": optimized_residual,
    }


def _benchmark_dataset(path, seeds, train_cap, evaluation_cap, quick):
    method_records = {"basic": [], "optimized": []}
    audit = first_evidence = heatmap = None
    for seed_index, seed in enumerate(seeds):
        prepared = _prepare_dataset(path, seed, train_cap, evaluation_cap)
        basic_params = _tune_basic(prepared, quick)
        optimized_params, current_heatmap = _tune_optimized(prepared, quick)
        repeats = 1 if quick else 2

        # 数据种子间交替两种方法的计时先后顺序。
        if seed_index % 2 == 0:
            basic, b_fit, b_score = _fit_and_time(
                lambda: _new_basic(basic_params, quick), prepared["train_basic"], prepared["test_basic"], repeats
            )
            optimized, o_fit, o_score = _fit_and_time(
                lambda: _new_optimized(optimized_params, quick), prepared["train_raw"], prepared["test_raw"], repeats
            )
        else:
            optimized, o_fit, o_score = _fit_and_time(
                lambda: _new_optimized(optimized_params, quick), prepared["train_raw"], prepared["test_raw"], repeats
            )
            basic, b_fit, b_score = _fit_and_time(
                lambda: _new_basic(basic_params, quick), prepared["train_basic"], prepared["test_basic"], repeats
            )

        b_eval = _evaluate(basic, prepared["test_basic"], prepared["test_y"])
        o_eval = _evaluate(optimized, prepared["test_raw"], prepared["test_y"])
        method_records["basic"].append(
            {
                "auprc": b_eval["curves"]["average_precision"],
                "auroc": b_eval["curves"]["roc_auc"], "f1": b_eval["f1"],
                "fit_time_ms": statistics.median(b_fit), "score_time_ms": statistics.median(b_score),
                "support_fraction": effective_support_fraction(
                    basic.alphas_, 1.0 / (basic.nu * len(prepared["train_basic"]))
                ),
            }
        )
        method_records["optimized"].append(
            {
                "auprc": o_eval["curves"]["average_precision"],
                "auroc": o_eval["curves"]["roc_auc"], "f1": o_eval["f1"],
                "fit_time_ms": statistics.median(o_fit), "score_time_ms": statistics.median(o_score),
                "support_fraction": effective_support_fraction(
                    optimized.alphas_, 1.0 / (optimized.nu * len(prepared["train_raw"]))
                ),
            }
        )
        if first_evidence is None:
            first_evidence = {
                "prepared": prepared, "basic": basic, "optimized": optimized,
                "basic_eval": b_eval, "optimized_eval": o_eval,
                "basic_params": basic_params, "optimized_params": optimized_params,
            }
            heatmap = current_heatmap
            audit = {
                "validation_ids": prepared["validation_ids"], "test_ids": prepared["test_ids"],
                "selected_train_ids": prepared["train_ids"],
                "selected_train_labels": [prepared["y"][i] for i in prepared["train_ids"]],
            }

    summary = {"n_seeds": len(seeds)}
    for method in ("basic", "optimized"):
        records = method_records[method]
        summary[method] = {
            "train_normal_cap": train_cap,
            "auprc": _summary([record["auprc"] for record in records]),
            "auroc": _summary([record["auroc"] for record in records]),
            "f1": _summary([record["f1"] for record in records]),
            "support_fraction": _summary([record["support_fraction"] for record in records]),
            "fit_time_ms": _summary([record["fit_time_ms"] for record in records], timing=True),
            "score_time_ms": _summary([record["score_time_ms"] for record in records], timing=True),
        }
    return summary, audit, first_evidence, heatmap


def _robustness_experiment(evidence, heatmap, quick):
    prepared = evidence["prepared"]
    params = evidence["optimized_params"]
    nu_values = [0.1, 0.3, 0.5, 0.7]
    effective_support_fractions, actual_support_fractions = [], []
    for nu in nu_values:
        model = _new_optimized({"nu": nu, "gamma": params["gamma"]}, quick).fit(prepared["train_raw"])
        cap = 1.0 / (nu * len(prepared["train_raw"]))
        effective_support_fractions.append(effective_support_fraction(model.alphas_, cap))
        actual_support_fractions.append(len(model.support_indices_) / len(prepared["train_raw"]))
    requested_contamination = [0.0, 0.02, 0.05, 0.10]
    anomaly_pool = [index for index in prepared["split"]["train"] if prepared["y"][index] == 1]
    rng = random.Random(991)
    rng.shuffle(anomaly_pool)
    contamination_ap, actual_contamination = [], []
    for fraction in requested_contamination:
        count = min(len(anomaly_pool), injection_count(len(prepared["train_raw"]), fraction))
        contaminated = list(prepared["train_raw"]) + _rows(prepared["X"], anomaly_pool[:count])
        actual_contamination.append(count / len(contaminated))
        model = _new_optimized(params, quick).fit(contaminated)
        result = _evaluate(model, prepared["test_raw"], prepared["test_y"])
        contamination_ap.append(result["curves"]["average_precision"])
    return {
        "heatmap": heatmap, "nu_values": nu_values,
        "effective_support_fractions": effective_support_fractions,
        "actual_support_fractions": actual_support_fractions,
        "requested_contamination": requested_contamination,
        "actual_contamination": actual_contamination,
        "contamination": actual_contamination, "contamination_ap": contamination_ap,
    }


def run_experiment(data_dir, quick=False):
    """执行全部可视化实验；quick 仅缩小样本和迭代数，不改变协议。"""
    data_dir = Path(data_dir)
    train_cap, evaluation_cap = (24, 90) if quick else (72, 720)
    seeds = [17, 29] if quick else [17, 29, 43]
    datasets = {"Cardio": data_dir / "6_cardio.npz", "Mammography": data_dir / "23_mammography.npz"}
    benchmark, audit, evidence = {}, {}, {}
    cardio_heatmap = None
    for name, path in datasets.items():
        summary, dataset_audit, first_evidence, heatmap = _benchmark_dataset(
            path, seeds, train_cap, evaluation_cap, quick
        )
        benchmark[name], audit[name], evidence[name] = summary, dataset_audit, first_evidence
        if name == "Cardio":
            cardio_heatmap = heatmap
    mechanism = _mechanism_experiment(quick)
    return {
        "protocol": {
            "backend": "Python", "output_format": "PNG only", "font": "Times New Roman",
            "threshold": "intrinsic decision = 0", "tuning_split": "validation",
            "test_used_for_tuning": False,
            "training_policy": "normal-only clean reference training; labels curate the reference set",
            "effective_support_threshold": "alpha > 0.01C",
            "evaluation_sampling": "stratified cap preserving validation/test prevalence",
            "timing": "warm-up, alternating method order, median and interquartile range",
            "complexity_note": "fixed equal normal-training cap because both handwritten solvers cache O(n^2) kernels",
        },
        "figures": {name: True for name in FIGURE_FILENAMES}, "palettes": PALETTES,
        "benchmark": benchmark, "audit": audit, "evidence": evidence,
        "mechanism": mechanism,
        "robustness": _robustness_experiment(evidence["Cardio"], cardio_heatmap, quick),
        "runtime": _runtime_experiment(quick), "quick": bool(quick),
    }


def _panel_label(ax, label):
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")


def _finish(fig, path):
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def relative_objective(history):
    """用初始对偶目标归一化，不把观测到的末点冒充理论最优值。"""
    values = [float(value) for value in history]
    scale = values[0]
    if abs(scale) < 1e-15:
        return [1.0 for _ in values]
    return [value / scale for value in values]


def _draw_boundary(report, output_path):
    data = report["mechanism"]
    colors = PALETTES[FIGURE_FILENAMES[0]]
    normals, anomalies = np.asarray(data["normals"]), np.asarray(data["anomalies"])
    xs, ys = np.asarray(data["xs"]), np.asarray(data["ys"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    specifications = (
        (axes[0], "Basic linear boundary", data["basic_decision"], data["basic_support"], colors[0]),
        (axes[1], "Optimized RBF boundary", data["optimized_decision"], data["optimized_support"], colors[1]),
    )
    for ax, title, decisions, supports, accent in specifications:
        field = np.asarray(decisions).reshape(len(ys), len(xs))
        low, high = min(field.min(), -1e-9), max(field.max(), 1e-9)
        ax.contourf(xs, ys, field, levels=[low, 0.0, high], colors=["#F6E8E4", "#E7F2EE"], alpha=0.75)
        ax.contour(xs, ys, field, levels=[0.0], colors=[accent], linewidths=2.0)
        ax.scatter(normals[:, 0], normals[:, 1], s=20, color=colors[0], alpha=0.72, label="Normal training")
        ax.scatter(anomalies[:, 0], anomalies[:, 1], s=28, marker="x", color=colors[2], linewidth=1.2, label="Held-out anomaly")
        if supports:
            sv = normals[supports]
            ax.scatter(sv[:, 0], sv[:, 1], s=64, facecolors="none", edgecolors=colors[3],
                       linewidth=1.2, label="Effective support vector")
        ax.scatter([data["origin"][0]], [data["origin"][1]], marker="*", s=90,
                   color=colors[4], edgecolor="black", linewidth=0.5, label="Origin")
        ax.set(xlabel="Feature 1", ylabel="Feature 2", title=title,
               xlim=(-0.18, 5.6), ylim=(-0.18, 5.4))
        ax.set_aspect("equal", adjustable="box")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=4)
    fig.suptitle("Nonlinear kernels recover a closed novelty boundary", y=1.12,
                 fontsize=12, fontweight="bold")
    _finish(fig, output_path)


def _draw_dual(report, output_path):
    data = report["mechanism"]
    colors = PALETTES[FIGURE_FILENAMES[1]]
    alphas, decisions = np.asarray(data["alphas"]), np.asarray(data["training_decisions"])
    cap = data["cap"]
    order = np.argsort(alphas)
    categories = np.asarray(data["categories"])[order]
    category_colors = {"zero": colors[4], "free": colors[1], "capped": colors[2]}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    axes[0].bar(np.arange(len(alphas)), alphas[order],
                color=[category_colors[value] for value in categories], width=0.9)
    axes[0].axhline(cap, color=colors[0], linestyle="--", linewidth=1.2)
    axes[0].set(xlabel="Training samples sorted by $\\alpha_i$", ylabel=r"Dual coefficient $\alpha_i$",
                title="Dual coefficient profile")
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=category_colors[key],
               markeredgecolor="none", label=label)
        for key, label in (("zero", r"$\alpha\approx0$"), ("free", "Effective free SV"), ("capped", r"$\alpha\approx C$"))
    ]
    legend.append(Line2D([0], [0], color=colors[0], linestyle="--", label="Box cap"))
    axes[0].legend(handles=legend, fontsize=7, loc="upper left")
    axes[0].text(0.98, 0.03, r"Effective threshold: $\alpha>0.01C$",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7,
                 color=colors[0])
    jitter = {"zero": -0.08, "free": 0.0, "capped": 0.08}
    axes[1].scatter([jitter[value] for value in data["categories"]], decisions,
                    c=[category_colors[value] for value in data["categories"]], s=32, alpha=0.8)
    axes[1].axhline(0.0, color=colors[0], linestyle="--", linewidth=1.2)
    axes[1].set_xticks([-0.08, 0.0, 0.08], [r"$\alpha\approx0$", "Effective free SV", r"$\alpha\approx C$"])
    axes[1].set(ylabel="Training decision value", title="KKT roles around the boundary")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.suptitle("Dual constraints organize boundary-related coefficient roles",
                 fontsize=12, fontweight="bold")
    _finish(fig, output_path)


def _draw_convergence(report, output_path):
    mechanism, runtime = report["mechanism"], report["runtime"]
    colors = PALETTES[FIGURE_FILENAMES[2]]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    for history, label, color in (
        (mechanism["basic_objective"], "Projected gradient", colors[0]),
        (mechanism["optimized_objective"], "Frank-Wolfe", colors[1]),
    ):
        values = np.maximum(relative_objective(history), 1e-15)
        axes[0].plot(range(len(values)), values, color=color, linewidth=2.0, label=label)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Iteration", ylabel="Objective / initial objective", title="Optimization convergence")
    axes[0].legend(fontsize=7)
    axes[1].plot(runtime["sizes"], runtime["basic_time_ms"], "o-", color=colors[0], linewidth=1.8, label="Basic")
    axes[1].plot(runtime["sizes"], runtime["optimized_time_ms"], "s-", color=colors[1], linewidth=1.8, label="Optimized")
    for method, color in (("basic", colors[0]), ("optimized", colors[1])):
        middle = np.asarray(runtime[f"{method}_time_ms"])
        low = middle - np.asarray(runtime[f"{method}_time_q1_ms"])
        high = np.asarray(runtime[f"{method}_time_q3_ms"]) - middle
        axes[1].errorbar(runtime["sizes"], middle, yerr=np.vstack([low, high]),
                         fmt="none", ecolor=color, capsize=3, linewidth=1.0)
    axes[1].set(xlabel="Normal training samples, n", ylabel="Median fit time (ms; IQR)",
                title="Runtime scaling")
    axes[1].legend(fontsize=7)
    floor = 1e-16
    axes[2].plot(runtime["sizes"], np.maximum(runtime["basic_residual"], floor), "o-", color=colors[2], label="Basic")
    axes[2].plot(runtime["sizes"], np.maximum(runtime["optimized_residual"], floor), "s-", color=colors[3], label="Optimized")
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Normal training samples, n", ylabel=r"$|\sum_i\alpha_i-1|$", title="Dual feasibility")
    axes[2].legend(fontsize=7)
    for label, ax in zip("abc", axes):
        _panel_label(ax, label)
    fig.suptitle("Both handwritten solvers preserve the dual constraint", fontsize=12, fontweight="bold")
    _finish(fig, output_path)


def _draw_scores(report, output_path):
    evidence = report["evidence"]["Cardio"]
    labels = np.asarray(evidence["prepared"]["test_y"])
    basic_eval, optimized_eval = evidence["basic_eval"], evidence["optimized_eval"]
    colors = PALETTES[FIGURE_FILENAMES[3]]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
    normal_scores = -np.asarray(optimized_eval["decisions"])[labels == 0]
    anomaly_scores = -np.asarray(optimized_eval["decisions"])[labels == 1]
    low, high = min(normal_scores.min(), anomaly_scores.min()), max(normal_scores.max(), anomaly_scores.max())
    bins = np.linspace(low, high if high > low else low + 1.0, 20)
    axes[0].hist(normal_scores, bins=bins, density=True, alpha=0.65, color=colors[1], label="Normal")
    axes[0].hist(anomaly_scores, bins=bins, density=True, alpha=0.65, color=colors[2], label="Anomaly")
    axes[0].axvline(0.0, color=colors[0], linestyle="--", linewidth=1.2, label="Intrinsic threshold")
    axes[0].set(xlabel="Anomaly score (-decision)", ylabel="Density", title="Optimized score separation")
    axes[0].legend(fontsize=7)
    for result, label, color, style in (
        (basic_eval, "Basic linear", colors[0], "--"),
        (optimized_eval, "Optimized RBF", colors[1], "-"),
    ):
        curve = result["curves"]
        axes[1].plot(curve["fpr"], curve["tpr"], linestyle=style, color=color,
                     linewidth=2.0, label=f"{label} (AUC={curve['roc_auc']:.3f})")
        axes[2].plot(curve["recall"], curve["precision"], linestyle=style, color=color,
                     linewidth=2.0, label=f"{label} (AP={curve['average_precision']:.3f})")
    axes[1].plot([0, 1], [0, 1], color="#AAAAAA", linestyle=":", linewidth=1.0)
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve",
                xlim=(0, 1), ylim=(0, 1.02))
    prevalence = labels.mean()
    axes[2].axhline(prevalence, color="#AAAAAA", linestyle=":", linewidth=1.0,
                    label=f"Prevalence={prevalence:.3f}")
    axes[2].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve",
                xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(fontsize=7, loc="lower right")
    axes[2].legend(fontsize=7, loc="upper right")
    for label, ax in zip("abc", axes):
        _panel_label(ax, label)
    fig.suptitle("RBF scoring improves rare-anomaly ranking on Cardio", fontsize=12, fontweight="bold")
    _finish(fig, output_path)


def _draw_robustness(report, output_path):
    data, heatmap = report["robustness"], report["robustness"]["heatmap"]
    colors = PALETTES[FIGURE_FILENAMES[4]]
    cmap = LinearSegmentedColormap.from_list("ocsvm_robustness", ["#F3EEF7", colors[2], colors[0]])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    matrix = np.asarray(heatmap["values"])
    image = axes[0].imshow(matrix, cmap=cmap,
                           vmin=max(0.0, matrix.min() - 0.03),
                           vmax=min(1.0, matrix.max() + 0.03), aspect="auto")
    axes[0].set_xticks(range(len(heatmap["gammas"])), [str(value) for value in heatmap["gammas"]])
    axes[0].set_yticks(range(len(heatmap["nus"])), [str(value) for value in heatmap["nus"]])
    axes[0].set(xlabel=r"RBF $\gamma$", ylabel=r"$\nu$", title="Validation AUPRC")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            text_color = "white" if matrix[row, column] > matrix.mean() else "black"
            axes[0].text(column, row, f"{matrix[row, column]:.2f}", ha="center",
                         va="center", color=text_color, fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].plot(data["nu_values"], data["actual_support_fractions"], "o-",
                 color=colors[0], linewidth=1.8, label=r"Actual SV ($\alpha>10^{-8}$)")
    axes[1].plot(data["nu_values"], data["effective_support_fractions"], "s-",
                 color=colors[1], linewidth=1.8, label=r"Effective SV ($\alpha>0.01C$)")
    axes[1].plot(data["nu_values"], data["nu_values"], "--", color=colors[3],
                 linewidth=1.4, label=r"Theoretical lower bound $\nu$")
    axes[1].set(xlabel=r"$\nu$", ylabel="Support-vector fraction",
                title="Model complexity control", ylim=(0, 1.03))
    axes[1].legend(fontsize=7)
    percentages = np.asarray(data["actual_contamination"]) * 100.0
    axes[2].plot(percentages, data["contamination_ap"], "o-", color=colors[4], linewidth=2.0)
    axes[2].fill_between(percentages, data["contamination_ap"], color=colors[4], alpha=0.13)
    axes[2].set(xlabel="Injected training contamination (%)", ylabel="Test AUPRC",
                title="Reference-set robustness", ylim=(0, 1.03))
    for label, ax in zip("abc", axes):
        _panel_label(ax, label)
    fig.suptitle("Hyperparameters trade ranking quality, sparsity and contamination tolerance",
                 fontsize=12, fontweight="bold")
    _finish(fig, output_path)


def _draw_benchmark(report, output_path):
    benchmark = report["benchmark"]
    colors = PALETTES[FIGURE_FILENAMES[5]]
    datasets = list(benchmark)
    methods, labels = ["basic", "optimized"], ["Basic linear", "Optimized RBF"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), constrained_layout=True)
    x, width = np.arange(len(datasets)), 0.34
    for method_index, (method, label, color) in enumerate(zip(methods, labels, colors[:2])):
        offset = (method_index - 0.5) * width
        ap = [benchmark[name][method]["auprc"]["mean"] for name in datasets]
        ap_error = [benchmark[name][method]["auprc"]["std"] for name in datasets]
        axes[0].bar(x + offset, ap, width, yerr=ap_error, capsize=3,
                    color=color, alpha=0.9, label=label)
        f1 = [benchmark[name][method]["f1"]["mean"] for name in datasets]
        f1_error = [benchmark[name][method]["f1"]["std"] for name in datasets]
        axes[1].bar(x + offset, f1, width, yerr=f1_error, capsize=3,
                    color=color, alpha=0.9, label=label)
    for ax, title, ylabel in (
        (axes[0], "Rare-event ranking", "AUPRC (mean ± SD)"),
        (axes[1], "Intrinsic-threshold detection", "F1 at decision=0 (mean ± SD)"),
    ):
        ax.set_xticks(x, datasets)
        ax.set_ylim(0, 1.03)
        ax.set(title=title, ylabel=ylabel)
    axes[0].legend(fontsize=7, loc="upper left")

    timing_x, timing_labels, timing_values, timing_errors, timing_colors = [], [], [], [], []
    cursor = 0.0
    for name in datasets:
        for method_index, method in enumerate(methods):
            stats = benchmark[name][method]["fit_time_ms"]
            timing_x.append(cursor)
            method_label = "Basic\nlinear" if method == "basic" else "Optimized\nRBF"
            timing_labels.append(f"{name}\n{method_label}")
            timing_values.append(stats["median"])
            timing_errors.append([stats["median"] - stats["q1"], stats["q3"] - stats["median"]])
            timing_colors.append(colors[method_index])
            cursor += 1.0
        cursor += 0.45
    asymmetric = np.asarray(timing_errors, dtype=float).T
    axes[2].bar(timing_x, timing_values, color=timing_colors, alpha=0.9)
    axes[2].errorbar(timing_x, timing_values, yerr=asymmetric, fmt="none",
                     ecolor="#333333", capsize=3, linewidth=1.0)
    axes[2].set_xticks(timing_x, timing_labels, fontsize=6.5)
    axes[2].set_yscale("log")
    axes[2].set(ylabel="Fit time (ms; median, IQR)", title="Handwritten solver cost")
    for label, ax in zip("abc", axes):
        _panel_label(ax, label)
    cap = benchmark[datasets[0]]["basic"]["train_normal_cap"]
    fig.suptitle(f"Cross-dataset benchmark under equal training cap (n={cap})",
                 fontsize=12, fontweight="bold")
    _finish(fig, output_path)


def render_all(report, output_dir):
    """渲染六张独立 PNG，不创建 SVG/PDF/TIFF 或中间图像。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    drawers = (_draw_boundary, _draw_dual, _draw_convergence,
               _draw_scores, _draw_robustness, _draw_benchmark)
    paths = []
    for filename, drawer in zip(FIGURE_FILENAMES, drawers):
        path = output_dir / filename
        drawer(report, path)
        paths.append(path)
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description="Render six One-Class SVM PNG figures.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "anomaly")
    parser.add_argument("--output-dir", type=Path, default=root / "figures" / "one_class_svm")
    parser.add_argument("--quick", action="store_true", help="Use the reduced test-size experiment.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_experiment(args.data_dir, quick=args.quick)
    render_all(report, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
