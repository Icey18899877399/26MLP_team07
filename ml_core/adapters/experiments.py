"""Actual scratch-model execution, held-out evaluation and ECharts payloads.

The ROC/PR design follows visualization/isolation_forest_figures.py: tied
scores are grouped before trapezoidal ROC integration and stepwise AP. The
one-class normal-only, bounded training protocol follows the SVM figure module.
No benchmark sweeps or matplotlib imports are needed by a web experiment.
"""

from collections import Counter
import inspect
import json
import math
import random
from uuid import uuid4

import numpy as np

from ..errors import ExperimentExecutionError, InvalidParameterError
from ..types import ExperimentResult
from .catalog import FAMILIES, estimator_class
from .classification import _classification_metrics, _stratified_split
from .clustering import adjusted_rand_index


def validate_params(params, family, sample_count):
    integer_limits = {
        "max_iter": (1, 2000), "n_neighbors": (1, 100), "max_depth": (1, 12),
        "min_samples_split": (2, 1000), "min_samples_leaf": (1, 1000),
        "n_estimators": (1, 100), "batch_size": (1, 256), "hidden_size": (1, 64),
        "n_clusters": (1, sample_count), "n_init": (1, 20), "min_samples": (1, sample_count),
        "leaf_size": (1, 1000), "max_samples": (2, 400), "n_split_candidates": (1, 10),
        "train_sample_limit": (10, 150 if family == "one_class_svm" else 1000),
    }
    enums = {"init": ("random", "k-means++"), "weights": ("uniform", "distance"),
             "class_weight": (None, "balanced"), "voting": ("soft", "hard"),
             "activation": ("relu", "tanh"), "algorithm": ("kd_tree", "brute"),
             "kernel": ("rbf", "linear")}
    for name, value in params.items():
        if name in integer_limits:
            low, high = integer_limits[name]
            valid = type(value) is int and low <= value <= high
        elif name in ("standardize", "oob_score"):
            valid = isinstance(value, bool)
        elif name in enums:
            valid = value in enums[name]
        elif name == "hidden_layer_sizes":
            valid = isinstance(value, (list, tuple)) and 1 <= len(value) <= 3 and all(
                type(width) is int and 1 <= width <= 64 for width in value)
        elif name == "max_features" and family == "random_forest":
            valid = value in ("sqrt", "log2")
        elif name == "gamma" and isinstance(value, str):
            valid = value in ("scale", "auto")
        else:
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            if valid:
                if name in ("l2", "tol", "l2_regularization", "ccp_alpha", "var_smoothing"):
                    valid = value >= 0
                elif name == "threshold":
                    valid = 0 <= value <= 1
                elif name in ("subsample", "nu", "max_features", "learning_rate"):
                    valid = 0 < value <= 1
                elif name == "contamination":
                    valid = 0 < value <= 0.5
                elif name == "p":
                    valid = value in (1, 2)
                else:
                    valid = value > 0
        if not valid:
            raise InvalidParameterError(f"Invalid value for {name}: {value!r}")


def chart(chart_id, title, description, series, x_name, y_name, categories=None):
    return {"id": chart_id, "title": title, "description": description, "option": {
        "tooltip": {"trigger": "item"}, "legend": {"type": "scroll"},
        "grid": {"left": 64, "right": 28, "top": 42, "bottom": 64, "containLabel": True},
        "xAxis": {"type": "category" if categories is not None else "value", "name": x_name,
                  **({"data": categories} if categories is not None else {})},
        "yAxis": {"type": "value", "name": y_name, "scale": True}, "series": series}}


def ranking_curves(targets, scores):
    pairs = sorted(zip(scores, targets), reverse=True)
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise ExperimentExecutionError("Ranking evaluation requires both held-out classes")
    tp = fp = 0
    roc, pr = [[0.0, 0.0]], [[0.0, 1.0]]
    auc = ap = 0.0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            tp += int(pairs[index][1] == 1)
            fp += int(pairs[index][1] == 0)
            index += 1
        recall, precision, fpr = tp / positives, tp / (tp + fp), fp / negatives
        auc += (fpr - roc[-1][0]) * (recall + roc[-1][1]) / 2
        ap += (recall - pr[-1][0]) * precision
        roc.append([fpr, recall])
        pr.append([recall, precision])
    return {"roc_auc": auc, "average_precision": ap}, roc, pr


def ranking_charts(roc, pr):
    return [chart("roc", "Held-out ROC", "Thresholds come from actual held-out scores; tied scores are grouped.",
                  [{"name": "ROC", "type": "line", "showSymbol": False, "data": roc}], "False positive rate", "True positive rate"),
            chart("precision_recall", "Held-out precision–recall", "Average precision uses recall increments at grouped score thresholds.",
                  [{"name": "Precision–recall", "type": "line", "step": "end", "showSymbol": False, "data": pr}], "Recall", "Precision")]


def projection_chart(train_x, evaluation_x, labels, description):
    # PCA is a display transform only; the estimator always sees all input features.
    center = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales[scales == 0] = 1
    standardized = (train_x - center) / scales
    _, _, axes = np.linalg.svd(standardized, full_matrices=False)
    axes = axes[:2].copy()
    for axis in axes:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    points = ((evaluation_x - center) / scales) @ axes.T
    series = [{"name": "Noise (-1)" if label == -1 else f"Label {label}", "type": "scatter", "symbolSize": 7,
               "data": points[np.asarray(labels) == label].tolist()} for label in sorted(set(labels))]
    return chart("projection", "2D PCA display projection", description + " PCA scaling and axes are fitted on training rows; all original features train the model.",
                 series, "Principal component 1", "Principal component 2")


def _standardize(train_x, test_x, scale_only=False):
    mean = np.zeros(train_x.shape[1]) if scale_only else train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1
    return (train_x - mean) / scale, (test_x - mean) / scale, mean, scale


def run_model(config, dataset, effective_params):
    family, variant = config.model.split(".")
    task = FAMILIES[family][1]
    params = dict(effective_params)
    validate_params(params, family, len(dataset.features))
    x = np.asarray(dataset.features, dtype=float)
    y = np.asarray(dataset.targets)
    generator = random.Random(config.random_state)
    test_size = 0.2 if config.test_size is None else float(config.test_size)
    if task == "clustering":
        train_ids = list(range(len(x)))
        test_ids = []
    elif task in ("classification", "anomaly_detection"):
        train_ids, test_ids = _stratified_split(dataset.targets, test_size, config.random_state)
    else:
        ids = list(range(len(x)))
        generator.shuffle(ids)
        count = max(1, min(len(x) - 1, round(len(x) * test_size)))
        test_ids, train_ids = ids[:count], ids[count:]
    available_train_count = len(train_ids)
    if family == "one_class_svm":
        train_ids = [index for index in train_ids if y[index] == 0]
    eligible_train_count = len(train_ids)
    limit = params.pop("train_sample_limit", len(train_ids))
    train_ids = train_ids[:limit]
    if len(train_ids) < 2:
        raise InvalidParameterError("The selected split leaves fewer than two eligible training samples")
    if params.get("n_neighbors", 1) > len(train_ids):
        raise InvalidParameterError("n_neighbors cannot exceed training sample count")
    train_x, train_y = x[train_ids], y[train_ids]
    eval_ids = train_ids if task == "clustering" else test_ids
    eval_x, eval_y = x[eval_ids], y[eval_ids]
    raw_train_x, raw_eval_x = train_x.copy(), eval_x.copy()
    cls = estimator_class(family, variant)
    supported = inspect.signature(cls).parameters
    metadata = {"feature_count": dataset.info.feature_count, "random_state": config.random_state,
                "source_sample_count": len(x), "train_sample_count": len(train_ids),
                "test_sample_count": len(test_ids), "train_sample_limit": limit,
                "available_train_sample_count": available_train_count,
                "eligible_train_sample_count": eligible_train_count,
                "sampling": "Seeded shuffle, split first, then cap training only; held-out rows are not capped.",
                "train_indices": train_ids, "test_indices": test_ids,
                "estimator": f"{cls.__module__}.{cls.__name__}",
                "preprocessing": "No feature scaling requested.",
                "evaluation_protocol": "All rows for unsupervised clustering; reference labels used only for external ARI." if task == "clustering" else "Single deterministic hold-out; evaluation rows excluded from fitting and preprocessing."}
    if task != "clustering":
        metadata["test_size"] = test_size
    scale_requested = params.get("standardize", False)
    if "standardize" not in supported:
        params.pop("standardize", None)
        if scale_requested:
            train_x, eval_x, mean, scale = _standardize(train_x, eval_x, family == "one_class_svm")
            metadata["preprocessing"] = "Training-only scale normalization without centering." if family == "one_class_svm" else "Training-only feature mean/std normalization."
            metadata["feature_means"] = mean.tolist()
            metadata["feature_scales"] = scale.tolist()
    elif scale_requested:
        metadata["preprocessing"] = "Scratch estimator fits feature scaling on its training partition (internal validation excluded where supported)."
    target_mean, target_scale = 0.0, 1.0
    if variant == "base" and family in ("linear_regression", "mlp_regression"):
        target_mean, target_scale = float(train_y.mean()), float(train_y.std()) or 1.0
        train_y = (train_y - target_mean) / target_scale
        metadata["target_preprocessing"] = "Training-only target mean/std; predictions restored to MPa."
        metadata["target_mean"], metadata["target_scale"] = target_mean, target_scale
    elif family in ("linear_regression", "mlp_regression"):
        metadata["target_preprocessing"] = "Scratch estimator fits target normalization on its training partition; predictions are in MPa."
    if "random_state" in supported:
        params["random_state"] = config.random_state
    model = cls(**params)
    if task in ("classification", "regression"):
        model.fit(train_x.tolist(), train_y.tolist())
    else:
        model.fit(train_x.tolist())
    charts = []
    if task == "classification":
        predictions = list(map(int, model.predict(eval_x.tolist())))
        metrics = _classification_metrics(eval_y.tolist(), predictions)
        probabilities = model.predict_proba(eval_x.tolist())
        if isinstance(probabilities[0], (list, tuple)):
            classes = list(getattr(model, "classes_", getattr(model, "classes", [0, 1])))
            probabilities = [row[classes.index(1)] for row in probabilities]
        ranking, roc, pr = ranking_curves(eval_y.tolist(), probabilities)
        metrics.update(ranking)
        confusion = chart("confusion_matrix", "Held-out confusion matrix", "Counts from actual held-out predictions; positive label 1 denotes malignant.",
                          [{"name": "Count", "type": "heatmap", "label": {"show": True}, "data": [[0, 0, metrics["tn"]], [1, 0, metrics["fp"]], [0, 1, metrics["fn"]], [1, 1, metrics["tp"]]]], "Predicted", "Actual", ["Benign (0)", "Malignant (1)"])
        confusion["option"]["yAxis"] = {"type": "category", "name": "Actual", "data": ["Benign (0)", "Malignant (1)"]}
        confusion["option"]["visualMap"] = {"min": 0, "max": max(metrics[key] for key in ("tn", "fp", "fn", "tp")), "calculable": True, "orient": "horizontal", "bottom": 0}
        charts = [confusion, *ranking_charts(roc, pr)]
        metadata["positive_label"] = 1
    elif task == "regression":
        predictions = np.asarray(model.predict(eval_x.tolist())) * target_scale + target_mean
        residual = eval_y - predictions
        mse = float(np.mean(residual ** 2))
        denominator = float(np.sum((eval_y - eval_y.mean()) ** 2))
        metrics = {"mse": mse, "rmse": math.sqrt(mse), "mae": float(np.mean(abs(residual))),
                   "r2": 1 - float(np.sum(residual ** 2)) / denominator if denominator else float(mse == 0)}
        charts = [chart("predicted_actual", "Predicted vs actual strength", "Every held-out row; predictions and targets are in original MPa units.",
                        [{"name": "Held-out predictions", "type": "scatter", "data": np.column_stack([eval_y, predictions]).tolist()}], "Actual (MPa)", "Predicted (MPa)"),
                  chart("residuals", "Held-out residuals", "Residual = actual − predicted, in MPa.",
                        [{"name": "Residual", "type": "scatter", "data": np.column_stack([predictions, residual]).tolist()}], "Predicted (MPa)", "Residual (MPa)")]
    elif task == "clustering":
        labels = list(map(int, model.labels_))
        counts = Counter(labels)
        metrics = {"adjusted_rand_index": adjusted_rand_index(dataset.targets, labels), "n_clusters": len(set(labels) - {-1})}
        if family == "kmeans":
            metrics["inertia"] = float(model.inertia_)
        else:
            metrics["noise_count"] = counts.get(-1, 0)
            metrics["noise_fraction"] = counts.get(-1, 0) / len(labels)
        metadata.update({"sample_count": len(labels), "cluster_sizes": {str(label): count for label, count in sorted(counts.items())}})
        charts = [projection_chart(raw_train_x, raw_eval_x, labels, "Colors are fitted cluster IDs; −1 is DBSCAN noise."),
                  chart("cluster_sizes", "Cluster membership counts", "Noise is shown separately as label −1 when present.",
                        [{"name": "Samples", "type": "bar", "data": [counts[label] for label in sorted(counts)]}], "Cluster", "Count", list(map(str, sorted(counts))))]
    else:
        svm = family == "one_class_svm"
        scores = [-float(value) for value in model.decision_function(eval_x.tolist())] if svm else list(map(float, model.score_samples(eval_x.tolist())))
        predictions = [int(value == -1) for value in model.predict(eval_x.tolist())]
        metrics = _classification_metrics(eval_y.tolist(), predictions)
        ranking, roc, pr = ranking_curves(eval_y.tolist(), scores)
        metrics.update(ranking)
        metadata["score_direction"] = "Higher means more anomalous; negative decision_function for SVM, score_samples for isolation forest."
        metadata["evaluation_protocol"] = ("Normal-only reference training selected with training labels; all held-out labels used only for evaluation." if svm else "Unsupervised mixed training; fixed contamination quantile fitted from training scores; held-out labels used only for evaluation.")
        metadata["threshold"] = 0.0 if svm else float(model.threshold_)
        charts = [projection_chart(raw_train_x, raw_eval_x, predictions, "Colors are predicted normal (0) or anomalous (1) on held-out rows."),
                  chart("anomaly_scores", "Held-out anomaly scores", "Held-out labels color actual scores; larger is more anomalous.",
                        [{"name": "Normal" if label == 0 else "Anomaly", "type": "scatter", "data": [[index, score] for index, (score, target) in enumerate(zip(scores, eval_y)) if target == label]} for label in (0, 1)], "Held-out row", "Anomaly score"),
                  *ranking_charts(roc, pr)]
    history = next((getattr(model, name) for name in ("loss_history", "loss_history_", "train_loss_") if len(getattr(model, name, []))), [])
    if history:
        metadata["training_history"] = list(map(float, history))
        loss_series = [{"name": "Training loss", "type": "line", "showSymbol": False, "data": [[i + 1, float(loss)] for i, loss in enumerate(history)]}]
        validation = getattr(model, "validation_loss_", [])
        if validation:
            loss_series.append({"name": "Internal validation loss", "type": "line", "showSymbol": False, "data": [[i + 1, float(loss)] for i, loss in enumerate(validation)]})
        charts.append(chart("training_loss", "Recorded training loss", "Actual scratch-estimator loss values; normalized target space for linear/MLP, original target space for GBDT. This is not held-out test loss.", loss_series, "Iteration", "Loss"))
    importance = getattr(model, "feature_importances_", [])
    weights = getattr(model, "weights", []) if family in ("logistic_regression", "linear_regression") else []
    if len(importance) or len(weights):
        values = importance if len(importance) else weights
        names = [f"Feature {i + 1}" for i in range(len(values))]
        charts.append(chart("feature_contribution", "Feature importance" if len(importance) else "Fitted feature coefficients",
                            "Actual normalized tree impurity gains." if len(importance) else "Signed coefficients in the model's fitted feature/target coordinate system; not causal effects.",
                            [{"name": "Importance" if len(importance) else "Coefficient", "type": "bar", "data": list(map(float, values))}], "Input feature", "Value", names))
    metadata["iteration_count"] = int(getattr(model, "n_iter_", getattr(model, "n_iter", len(history))))
    metadata["visualizations"] = charts
    result = ExperimentResult(str(uuid4()), config.model, config.dataset, task, dict(effective_params), metrics, metadata=metadata)
    try:
        json.dumps(result.to_dict(), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ExperimentExecutionError("Model produced non-finite or non-JSON output; try a smaller learning rate or enable scaling") from error
    return result
