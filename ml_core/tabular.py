"""Explicit numeric table selection and real fits independent of original pipelines."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import math
from uuid import uuid4

from .adapters import anomaly as ano, regression as reg
from .adapters._common import (stratified_split, require_non_negative_number,
                              require_positive_integer, require_optional_positive_integer_or_none)
from .adapters.charts import anomaly_charts
from .api import _validate_common_config
from .errors import InvalidConfigError, InvalidParameterError, MLCoreError, ExperimentExecutionError
from .interactive import list_interactive_experiments, _CLASSES
from .registry import get_model_spec
from .types import DatasetInfo, ExperimentConfig, ExperimentResult

OCSVM_MAX_ROWS = 512
TASKS = {"classification", "regression", "clustering", "anomaly_detection"}


@dataclass(frozen=True)
class TabularDataset:
    info: DatasetInfo
    features: tuple[tuple[float, ...], ...]
    targets: tuple
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    target_name: str | None

    @property
    def class_names(self):
        return self.label_names


def select_table(table, feature_columns, target_column, task):
    if task not in TASKS:
        raise InvalidConfigError("未知任务类型")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise InvalidConfigError("请选择至少一个不重复的特征列")
    unknown = set(feature_columns) - set(table.headers)
    if unknown or (target_column is not None and target_column not in table.headers):
        raise InvalidConfigError("选择包含不存在的列")
    if target_column in feature_columns:
        raise InvalidConfigError("目标列不能同时作为特征；这会造成标签泄漏")
    if task in ("classification", "regression") and target_column is None:
        raise InvalidConfigError("分类和回归必须选择目标列")
    if len(table.rows) < 4:
        raise InvalidConfigError("训练至少需要 4 行有效数据")
    indices = [table.headers.index(name) for name in feature_columns]
    features = []
    for row_number, row in enumerate(table.rows, start=1):
        selected = []
        for name, index in zip(feature_columns, indices):
            value = row[index]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InvalidConfigError(f"特征 {name} 第 {row_number} 行存在缺失、非数值或非有限值；未删除任何行。请选择完整数值列")
            selected.append(float(value))
        features.append(tuple(selected))
    targets, labels = (), ()
    if target_column is not None:
        index = table.headers.index(target_column)
        target_rows = (getattr(table, "raw_rows", ()) or table.rows) if task in ("classification", "clustering") else table.rows
        raw = [row[index] for row in target_rows]
        if any(value is None for value in raw):
            raise InvalidConfigError("目标列存在缺失值；未删除任何行")
        if task == "regression":
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in raw):
                raise InvalidConfigError("回归目标必须为完整有限数值")
            targets = tuple(float(v) for v in raw)
        elif task == "anomaly_detection":
            if any(v not in (0, 1) for v in raw):
                raise InvalidConfigError("异常参考标签必须采用 0=正常、1=异常")
            targets, labels = tuple(int(v) for v in raw), ("正常 (0)", "异常 (1)")
            if any(count < 2 for count in Counter(targets).values()):
                raise InvalidConfigError("异常参考标签每类至少需要 2 行以划分独立测试集")
        else:
            unique = sorted(set(raw), key=lambda v: (isinstance(v, str), v))
            if task == "classification" and (len(unique) != 2 or min(Counter(raw).values()) < 2):
                raise InvalidConfigError("当前分类指标仅支持二分类，且每类至少需要 2 行")
            mapping = {label: index for index, label in enumerate(unique)}
            targets, labels = tuple(mapping[value] for value in raw), tuple(str(v) for v in unique)
    info = DatasetInfo(table.id, table.name, task, len(features), len(indices), bool(targets))
    return TabularDataset(info, tuple(features), targets, tuple(feature_columns), labels, target_column)


def compatible_models(table, feature_columns, target_column, task):
    try:
        dataset = select_table(table, feature_columns, target_column, task)
    except InvalidConfigError as exc:
        return {"models": [], "issues": [str(exc)]}
    models = [item["model"] for item in list_interactive_experiments() if item["task"] == task]
    issues = []
    if task == "anomaly_detection" and len(dataset.features) > OCSVM_MAX_ROWS:
        models.remove("one_class_svm.optimized")
        issues.append(f"单类 SVM 核矩阵计算限制：最多 {OCSVM_MAX_ROWS} 行；当前 {len(dataset.features)} 行，不自动抽样")
    if task == "anomaly_detection" and dataset.targets and 0 not in dataset.targets:
        if "one_class_svm.optimized" in models:
            models.remove("one_class_svm.optimized")
        issues.append("单类 SVM 需要正常样本 (0) 训练")
    return {"models": models, "issues": [], "warnings": issues}


def _anomaly_fit(config, dataset, params):
    from .adapters.model_charts import model_charts
    labeled = bool(dataset.targets)
    if not labeled and config.test_size is not None:
        raise InvalidConfigError("无标签异常检测使用全量样本内诊断，不接受 test_size")
    if labeled:
        train_ids, test_ids = stratified_split(dataset.targets, config.test_size or 0.2, config.random_state)
        protocol = "独立分层留出测试；仅训练分区拟合预处理和模型；参考标签 0=正常、1=异常。"
    else:
        train_ids = test_ids = list(range(len(dataset.features)))
        protocol = "无标签全量训练与样本内诊断；没有真实标签，不计算分类效果指标。"
    train_targets = [dataset.targets[i] for i in train_ids] if labeled else []
    if config.model == "one_class_svm.optimized":
        if len(dataset.features) > OCSVM_MAX_ROWS:
            raise InvalidConfigError(f"单类 SVM 最多支持 {OCSVM_MAX_ROWS} 行，不自动抽样")
        if labeled:
            train_ids = [i for i in train_ids if dataset.targets[i] == 0]
            train_targets = [0] * len(train_ids)
        if len(train_ids) < 2:
            raise InvalidConfigError("单类 SVM 至少需要 2 个训练样本（有标签时仅使用正常样本）")
        ano._validate_one_class_svm_params(params)
        model = ano.OptimizedOneClassSVMScratch(**params, kernel="rbf")
    else:
        if params["contamination"] is None:
            # A label-free run cannot estimate contamination from ground truth.
            rate = sum(train_targets) / len(train_targets) if labeled else 0.1
            params["contamination"] = rate if 0 < rate <= 0.5 else 0.1
        ano._validate_isolation_forest_params(params)
        require_positive_integer(params, "n_split_candidates")
        model = ano.OptimizedIsolationForestScratch(**params, random_state=config.random_state)
    train_x = [dataset.features[i] for i in train_ids]
    test_x = [dataset.features[i] for i in test_ids]
    test_y = [dataset.targets[i] for i in test_ids] if labeled else []
    model.fit(train_x)
    if config.model == "one_class_svm.optimized":
        decisions = model.decision_function(test_x)
        scores = [-float(v) for v in decisions]
        predictions = [1 if value >= -1e-10 else -1 for value in decisions]
    else:
        scores = model.score_samples(test_x)
        predictions = [-1 if score > model.threshold_ else 1 for score in scores]
    metrics = ano._anomaly_metrics(test_y, predictions) if labeled else {}
    charts = anomaly_charts(test_y, scores, predictions, protocol=protocol, evaluation_label="独立测试集") if labeled else []
    charts += model_charts(config.model, model, train_features=train_x, evaluation_features=test_x,
                           train_targets=train_targets or None, evaluation_targets=test_y or None,
                           predictions=predictions, scores=scores, feature_names=dataset.feature_names,
                           class_names=dataset.label_names)
    return ExperimentResult(str(uuid4()), config.model, config.dataset, "anomaly_detection", dict(params), metrics,
        metadata={"train_sample_count": len(train_x), "test_sample_count": len(test_x) if labeled else 0,
                  "evaluation_sample_count": len(test_x), "train_test_overlap_count": len(set(train_ids) & set(test_ids)),
                  "train_anomaly_count": sum(train_targets) if labeled else None,
                  "train_sample_cap": None, "test_size": (config.test_size or 0.2) if labeled else None,
                  "evaluation_protocol": protocol, "visualizations": charts,
                  "contamination_policy": "null 使用训练标签异常比例 (0,0.5]；无标签或比例超出范围时显式采用 0.1，结果 effective_params 显示实际值"})


def run_tabular(config: ExperimentConfig, table, feature_columns, target_column, task=None):
    _validate_common_config(config)
    item = next((i for i in list_interactive_experiments() if i["model"] == config.model), None)
    if item is None:
        raise InvalidConfigError("未知模型")
    if task is not None and task != item["task"]:
        raise InvalidConfigError("选择的任务与模型不匹配")
    task = item["task"]
    dataset = select_table(table, feature_columns, target_column, task)
    availability = compatible_models(table, feature_columns, target_column, task)
    if config.model not in availability["models"]:
        raise InvalidConfigError("；".join(availability["issues"] + availability.get("warnings", [])))
    if task == "clustering" and config.test_size is not None:
        raise InvalidConfigError("聚类使用全量数据，不接受 test_size")
    unknown = set(config.params) - set(item["default_params"])
    if unknown:
        raise InvalidParameterError(f"未知参数: {', '.join(sorted(unknown))}")
    params = {**item["default_params"], **config.params}
    if "tol" in params:
        require_non_negative_number(params, "tol")
    if "class_weight" in params and params["class_weight"] not in (None, "balanced"):
        raise InvalidParameterError("class_weight 必须为 null 或 balanced")
    try:
        if task == "anomaly_detection":
            result = _anomaly_fit(config, dataset, params)
        elif task == "regression":
            name = item["id"]
            {"linear_regression": reg._validate_linear_params, "gbdt_regression": reg._validate_gbdt_params,
             "mlp_regression": reg._validate_mlp_params}[name](params)
            if name == "gbdt_regression":
                require_optional_positive_integer_or_none(params, "n_iter_no_change")
            if name == "mlp_regression":
                widths = params["hidden_layer_sizes"]
                if not isinstance(widths, list) or not widths or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in widths):
                    raise InvalidParameterError("hidden_layer_sizes 必须是非空正整数数组")
            result = reg._fit_evaluate_regressor(config, dataset, params, _CLASSES[name],
                                                extra_kwargs={"random_state": config.random_state}, train_cap=None)
        else:
            result = get_model_spec(config.model).runner(config, dataset, params)
        metadata = {**result.metadata, "execution_mode": "tabular_single_fit", "dataset_name": table.name,
                    "dataset_source": table.source, "sample_count": len(dataset.features), "feature_count": len(feature_columns),
                    "feature_names": list(feature_columns), "target_column": target_column,
                    "label_names": list(dataset.label_names), "random_state": config.random_state,
                    "full_pipeline_executed": False, "has_reference_labels": bool(dataset.targets)}
        metadata.setdefault("evaluation_sample_count", metadata.get("test_sample_count", len(dataset.features)))
        result = replace(result, metadata=metadata)
        json.dumps(result.to_dict(), allow_nan=False)
        return result
    except MLCoreError:
        raise
    except (ValueError, TypeError) as exc:
        raise InvalidParameterError(f"训练参数或数值无效: {exc}") from exc
    except Exception as exc:
        raise ExperimentExecutionError(f"数据集训练失败: {exc}") from exc
