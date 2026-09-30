"""Editable single fits on original project data, separate from archived pipelines.

Selection-dependent defaults are explicitly starting candidates, never advertised
as the results of the original cross-validation / benchmark pipeline.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from uuid import uuid4

from .api import _validate_common_config
from .adapters import classification as cls, regression as reg, anomaly as ano
from .adapters._common import (require_non_negative_number, require_positive_integer,
                              require_optional_positive_integer_or_none)
from .adapters.charts import anomaly_charts
from .adapters.model_charts import model_charts
from .datasets import LoadedDataset
from .errors import (MLCoreError, InvalidConfigError, InvalidParameterError,
                     IncompatibleDatasetError, UnknownModelError, ExperimentExecutionError)
from .original_catalog import list_original_experiments, original_root
from .registry import get_model_spec
from .types import DatasetInfo, ExperimentConfig, ExperimentResult


_CLASSES = {
    "cart_decision_tree": cls.OptimizedCARTClassifierScratch,
    "gaussian_naive_bayes": cls.OptimizedGaussianNaiveBayesScratch,
    "knn": cls.OptimizedKNNScratch,
    "logistic_regression": cls.OptimizedLogisticRegressionScratch,
    "random_forest": cls.OptimizedRandomForestClassifierScratch,
    "linear_regression": reg.OptimizedLinearRegressionScratch,
    "gbdt_regression": reg.OptimizedGBDTRegressorScratch,
    "mlp_regression": reg.OptimizedMLPRegressorScratch,
    "isolation_forest": ano.OptimizedIsolationForestScratch,
    "one_class_svm": ano.OptimizedOneClassSVMScratch,
}

# Fixed settings from the original figures factories. Tuned fields below start
# with the first original candidate; the catalog discloses this explicitly.
_OVERRIDES = {
    "cart_decision_tree": {"max_depth": 2, "class_weight": None},
    "logistic_regression": {"max_iter": 600, "l2": 0.01, "tol": 1e-5, "class_weight": "balanced"},
    "kmeans": {"max_iter": 200, "tol": 1e-5, "standardize": True},
    "linear_regression": {"tol": 1e-10},
    "gbdt_regression": {"n_estimators": 60, "learning_rate": 0.03, "max_depth": 1,
                        "min_samples_leaf": 3, "l2_regularization": 1.0, "subsample": 0.8,
                        "n_iter_no_change": None},
    "mlp_regression": {"hidden_layer_sizes": [8], "activation": "tanh", "max_iter": 300,
                       "batch_size": 64, "l2": 0.0005, "tol": 1e-5, "n_iter_no_change": 25},
    "one_class_svm": {"nu": 0.1, "gamma": 0.25, "max_iter": 120, "tol": 1e-5},
    "isolation_forest": {"n_estimators": 60, "max_samples": 64, "max_features": 0.35,
                         "n_split_candidates": 3, "contamination": None},
}
_EXTRA_DESCRIPTIONS = {
    "hidden_layer_sizes": "隐藏层宽度数组，例如 [8]、[16] 或 [16,8]；初值为原脚本首个候选",
    "tol": "收敛停止容差（非负有限数值）",
    "n_split_candidates": "每个节点尝试有效随机超平面的次数（正整数）",
    "class_weight": "类别权重（null 或 balanced）",
    "contamination": "异常比例；null 时按本次原训练分区的异常比例计算，与原脚本一致",
}


def list_interactive_experiments() -> list[dict]:
    """Return detached form definitions backed by original sources and factories."""
    items = []
    for original in list_original_experiments():
        name = original["id"]
        spec = get_model_spec(name + ".optimized")
        # Only names/descriptions come from the compatibility registry. Numeric
        # defaults come from original constructors and figures factories.
        if name in _CLASSES:
            signature = inspect.signature(_CLASSES[name])
            params = {key: signature.parameters[key].default for key in spec.info.default_params}
        else:
            from Models.kmeans_optimized import OptimizedKMeansScratch
            from Models.dbscan_optimized import OptimizedDBSCANScratch
            signature = inspect.signature(OptimizedKMeansScratch if name == "kmeans" else OptimizedDBSCANScratch)
            params = {key: signature.parameters[key].default for key in spec.info.default_params}
        params.update(deepcopy(_OVERRIDES.get(name, {})))
        descriptions = {key: _EXTRA_DESCRIPTIONS.get(key, spec.info.parameter_descriptions.get(key, key)) for key in params}
        if "max_iter" in descriptions:
            descriptions["max_iter"] = "最大训练迭代数（正整数）；初值取原模型/原图脚本配置"
        if name == "gbdt_regression":
            descriptions["n_iter_no_change"] = "早停耐心轮数：null 按原脚本关闭早停，正整数启用内部验证与早停"
            descriptions["validation_fraction"] = "内部验证比例（0~1），仅 n_iter_no_change 为正整数时生效"
        protocol = "单次优化模型训练；原脚本的数据源和固定配置，未执行完整交叉验证、选优、消融或多种子基准。"
        if name in {"mlp_regression", "gbdt_regression", "cart_decision_tree", "one_class_svm", "isolation_forest"}:
            protocol += "待调参数从原搜索候选起点开始，不代表原图最终最优参数。"
        if name in {"random_forest", "gaussian_naive_bayes", "knn", "dbscan"}:
            protocol += "待调参数采用原 Models 优化类构造默认值，不代表原图搜索结果。"
        if original["task"] == "classification":
            protocol += "分层留出测试，默认测试比例 0.2。"
        elif original["task"] == "regression":
            protocol += "随机留出测试，默认测试比例 0.2，完整使用训练分区，无额外训练截断。"
        elif original["task"] == "clustering":
            protocol += "全量原数据聚类，真实标签只用于外部评价。"
        elif name == "one_class_svm":
            protocol += "原分层 60/20/20 划分，正常训练上限 72，测试分层上限 720；验证分区保留但本次不选优。"
        else:
            protocol += "原分层 60/20/20 划分，仅训练分区拟合标准化；独立测试，验证分区保留但本次不选优。"
        items.append({"id": name, "model": name + ".optimized", "title": original["title"],
                      "task": original["task"], "dataset": original["datasets"][0]["id"],
                      "datasets": deepcopy(original["datasets"]), "default_params": params,
                      "parameter_descriptions": descriptions,
                      "random_state": 13 if name == "isolation_forest" else 17 if name == "one_class_svm" else 42,
                      "test_size": 0.2 if original["task"] in {"classification", "regression"} else None,
                      "protocol": protocol,
                      "source": f"Models/{name}_optimized.py；visualization/{name}_figures.py"})
    return items


def _load_original(item: dict, dataset_id: str) -> tuple[LoadedDataset, dict, str]:
    selected = next(d for d in item["datasets"] if d["id"] == dataset_id)
    path = original_root() / selected["path"]
    if path.suffix == ".npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            features = archive["X"].astype(float).tolist()
            targets = archive["y"].astype(int).reshape(-1).tolist()
    else:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        if dataset_id == "concrete":
            rows = [[float(v) for v in line.split(",")] for line in lines[1:] if line.strip()]
            features, targets = [r[:-1] for r in rows], [r[-1] for r in rows]
        elif dataset_id == "wdbc":
            rows = [line.split(",") for line in lines if line.strip()]
            features, targets = [[float(v) for v in r[2:]] for r in rows], [int(r[1] == "M") for r in rows]
        else:
            rows = [[float(v) for v in line.split()] for line in lines if line.strip()]
            features, targets = [r[:-1] for r in rows], [int(r[-1]) - 1 for r in rows]
    info = DatasetInfo(dataset_id, selected["name"], item["task"], len(features), len(features[0]), True)
    return LoadedDataset(info, tuple(tuple(r) for r in features), tuple(targets)), selected, hashlib.sha256(path.read_bytes()).hexdigest()


def _run_anomaly(config, item, selected, params):
    path = original_root() / selected["path"]
    if item["id"] == "one_class_svm":
        from visualization.one_class_svm_figures import _prepare_dataset
        prepared = _prepare_dataset(path, config.random_state, 72, 720)
        ano._validate_one_class_svm_params(params)
        train_x, test_x, test_y = prepared["train_raw"], prepared["test_raw"], prepared["test_y"]
        model = ano.OptimizedOneClassSVMScratch(**params, kernel="rbf").fit(train_x)
        decisions = model.decision_function(test_x)
        predictions = [1 if v >= -1e-10 else -1 for v in decisions]
        scores = [-v for v in decisions]
        extra = {"train_sample_cap": 72, "evaluation_sample_cap": 720,
                 "train_anomaly_count": 0, "validation_sample_count": len(prepared["validation_ids"]),
                 "train_test_overlap_count": len(set(prepared["train_ids"]) & set(prepared["test_ids"]))}
    else:
        from visualization.isolation_forest_figures import _prepare_dataset
        prepared = _prepare_dataset(path, config.random_state)
        train_x, test_x, test_y = prepared["train_X"], prepared["test_X"], prepared["test_y"]
        if params["contamination"] is None:
            params["contamination"] = sum(prepared["train_y"]) / len(prepared["train_y"])
        ano._validate_isolation_forest_params(params)
        require_positive_integer(params, "n_split_candidates")
        model = ano.OptimizedIsolationForestScratch(**params, random_state=config.random_state).fit(train_x)
        scores = model.score_samples(test_x)
        predictions = [(-1 if s > model.threshold_ else 1) for s in scores]
        extra = {"train_sample_cap": None, "validation_sample_count": len(prepared["validation_y"]),
                 "train_test_overlap_count": len(set(prepared["train_indices"]) & set(prepared["test_indices"]))}
    return ExperimentResult(str(uuid4()), config.model, config.dataset, "anomaly_detection", dict(params),
        ano._anomaly_metrics(test_y, predictions), metadata={
            "train_sample_count": len(train_x), "test_sample_count": len(test_x),
            "evaluation_sample_count": len(test_x), "feature_count": len(train_x[0]),
            "random_state": config.random_state, "test_size": 0.2,
            "evaluation_protocol": item["protocol"], **extra,
            "visualizations": anomaly_charts(test_y, scores, predictions, protocol=item["protocol"],
                                             evaluation_label="独立测试集") + model_charts(config.model, model,
                train_features=train_x, evaluation_features=test_x, evaluation_targets=test_y,
                predictions=predictions, scores=scores),
        })


def run_interactive_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Train once using original files, with honest evaluated counts and charts."""
    _validate_common_config(config)
    item = next((i for i in list_interactive_experiments() if i["model"] == config.model), None)
    if item is None:
        raise UnknownModelError(f"未知模型: {config.model}")
    if config.dataset not in [d["id"] for d in item["datasets"]]:
        raise IncompatibleDatasetError("交互训练仅支持该原实验使用的数据集")
    unknown = set(config.params) - set(item["default_params"])
    if unknown:
        raise InvalidParameterError(f"未知参数: {', '.join(sorted(unknown))}")
    if item["test_size"] is None and config.test_size is not None:
        raise InvalidConfigError("该实验采用原固定分区协议，不接受 test_size")
    params = {**item["default_params"], **config.params}
    if "tol" in params:
        require_non_negative_number(params, "tol")
    if "class_weight" in params and params["class_weight"] not in (None, "balanced"):
        raise InvalidParameterError("class_weight 必须是 null 或 balanced")
    try:
        dataset, selected, digest = _load_original(item, config.dataset)
        name = item["id"]
        if item["task"] == "anomaly_detection":
            result = _run_anomaly(config, item, selected, params)
        elif item["task"] == "regression":
            validator = {"linear_regression": reg._validate_linear_params,
                         "gbdt_regression": reg._validate_gbdt_params,
                         "mlp_regression": reg._validate_mlp_params}[name]
            validator(params)
            if name == "gbdt_regression":
                require_optional_positive_integer_or_none(params, "n_iter_no_change")
            if name == "mlp_regression":
                widths = params["hidden_layer_sizes"]
                if not isinstance(widths, list) or not widths or any(isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in widths):
                    raise InvalidParameterError("hidden_layer_sizes 必须是非空正整数数组")
            result = reg._fit_evaluate_regressor(config, dataset, params, _CLASSES[name],
                extra_kwargs={"random_state": config.random_state}, train_cap=None)
        else:
            result = get_model_spec(config.model).runner(config, dataset, params)
        metadata = {**result.metadata, "execution_mode": "interactive_single_fit",
                    "dataset_source": selected["path"], "dataset_sha256": digest,
                    "sample_count": len(dataset.features), "configuration_source": item["source"],
                    "single_fit_notice": item["protocol"], "full_pipeline_executed": False}
        metadata.setdefault("evaluation_sample_count", metadata.get("test_sample_count", len(dataset.features)))
        result = replace(result, metadata=metadata)
        json.dumps(result.to_dict(), allow_nan=False)
        return result
    except MLCoreError:
        raise
    except (ValueError, TypeError) as exc:
        raise InvalidParameterError(f"训练参数无效: {exc}") from exc
    except Exception as exc:
        raise ExperimentExecutionError(f"交互训练失败: {exc}") from exc
