"""Private registry connecting stable IDs to experiment runners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from .adapters.anomaly import run_isolation_forest, run_one_class_svm
from .adapters.classification import (
    run_cart_decision_tree,
    run_gaussian_naive_bayes,
    run_knn,
    run_logistic_regression,
    run_random_forest,
)
from .adapters.clustering import run_dbscan, run_kmeans
from .adapters.regression import (
    run_gbdt_regression,
    run_linear_regression,
    run_mlp_regression,
)
from .datasets import LoadedDataset
from .types import ExperimentConfig, ExperimentResult, JSONValue, ModelInfo


Runner = Callable[
    [ExperimentConfig, LoadedDataset, dict[str, JSONValue]],
    ExperimentResult,
]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    info: ModelInfo
    runner: Runner | None = None


_MODEL_SPECS = {
    "cart_decision_tree.optimized": ModelSpec(
        info=ModelInfo(
            id="cart_decision_tree.optimized",
            display_name="决策树",
            task="classification",
            compatible_datasets=("wdbc",),
            default_params={
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_impurity_decrease": 0.0,
                "ccp_alpha": 0.0,
            },
            parameter_descriptions={
                "max_depth": "最大树深（正整数，null 表示不限制）",
                "min_samples_split": "分裂所需最小样本数",
                "min_samples_leaf": "叶节点最小样本数",
                "min_impurity_decrease": "分裂最小不纯度下降",
                "ccp_alpha": "代价复杂度剪枝系数（非负）",
            },
        )
    ),
    "dbscan.optimized": ModelSpec(
        info=ModelInfo(
            id="dbscan.optimized",
            display_name="DBSCAN聚类",
            task="clustering",
            compatible_datasets=("seeds",),
            default_params={
                "eps": 0.5,
                "min_samples": 5,
                "standardize": True,
            },
            parameter_descriptions={
                "eps": "邻域半径（standardize=true 时作用于标准化空间）",
                "min_samples": "核心点所需最小邻居数（正整数）",
                "standardize": "是否对特征做标准化",
            },
        )
    ),
    "gaussian_naive_bayes.optimized": ModelSpec(
        info=ModelInfo(
            id="gaussian_naive_bayes.optimized",
            display_name="朴素贝叶斯",
            task="classification",
            compatible_datasets=("wdbc",),
            default_params={"var_smoothing": 1e-9},
            parameter_descriptions={
                "var_smoothing": "方差平滑项（正数，防止零概率）",
            },
        )
    ),
    "gbdt_regression.optimized": ModelSpec(
        info=ModelInfo(
            id="gbdt_regression.optimized",
            display_name="GBDT回归",
            task="regression",
            compatible_datasets=("concrete",),
            default_params={
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "subsample": 1.0,
                "validation_fraction": 0.1,
                "l2_regularization": 0.0,
            },
            parameter_descriptions={
                "n_estimators": "树的数量（正整数）",
                "learning_rate": "学习率（正数）",
                "max_depth": "树最大深度（正整数）",
                "min_samples_split": "分裂所需最小样本数",
                "min_samples_leaf": "叶节点最小样本数",
                "subsample": "每棵树样本采样比例（0~1）",
                "validation_fraction": "早停验证集比例（0~1）",
                "l2_regularization": "叶值 L2 正则化（非负）",
            },
        )
    ),
    "isolation_forest.optimized": ModelSpec(
        info=ModelInfo(
            id="isolation_forest.optimized",
            display_name="孤立森林",
            task="anomaly_detection",
            compatible_datasets=("6_cardio", "23_mammography"),
            default_params={
                "n_estimators": 100,
                "max_samples": 256,
                "contamination": 0.1,
                "max_features": 1.0,
            },
            parameter_descriptions={
                "n_estimators": "树的数量（正整数）",
                "max_samples": "每棵树采样数（正整数或 0~1 比例）",
                "contamination": "期望异常比例，决定判定阈值（心电图约 0.096，乳腺造影约 0.023）",
                "max_features": "每次切分使用的特征比例（正整数或 0~1 比例）",
            },
        )
    ),
    "kmeans.optimized": ModelSpec(
        info=ModelInfo(
            id="kmeans.optimized",
            display_name="K均值聚类",
            task="clustering",
            compatible_datasets=("seeds",),
            default_params={
                "n_clusters": 3,
                "init": "k-means++",
                "n_init": 10,
                "max_iter": 300,
                "tol": 1e-4,
                "standardize": True,
            },
            parameter_descriptions={
                "n_clusters": "簇数 k",
                "init": "初始化方式：'random' 或 'k-means++'",
                "n_init": "独立初始化次数，取惯性最低的一次",
                "max_iter": "每次初始化的最大迭代次数",
                "tol": "中心点移动的停止容差（非负）",
                "standardize": "聚类前是否对特征做标准化",
            },
        )
    ),
    "knn.optimized": ModelSpec(
        info=ModelInfo(
            id="knn.optimized",
            display_name="K近邻",
            task="classification",
            compatible_datasets=("wdbc",),
            default_params={
                "n_neighbors": 5,
                "p": 2,
                "weights": "distance",
                "standardize": True,
            },
            parameter_descriptions={
                "n_neighbors": "参与投票的最近邻居个数",
                "p": "Minkowski 距离阶数（p=1 曼哈顿，p=2 欧氏）",
                "weights": "投票权重（uniform/distance）",
                "standardize": "是否对特征做标准化",
            },
        )
    ),
    "linear_regression.optimized": ModelSpec(
        info=ModelInfo(
            id="linear_regression.optimized",
            display_name="线性回归",
            task="regression",
            compatible_datasets=("concrete", "california_housing"),
            default_params={
                "learning_rate": 0.01,
                "max_iter": 1000,
                "batch_size": 32,
                "l2": 0.0,
                "standardize": True,
                "standardize_target": True,
            },
            parameter_descriptions={
                "learning_rate": "学习率（正数）",
                "max_iter": "最大迭代次数（正整数）",
                "batch_size": "小批量大小（正整数）",
                "l2": "L2 正则化强度（非负）",
                "standardize": "是否对特征做标准化",
                "standardize_target": "是否对目标值做标准化（影响指标量纲）",
            },
        )
    ),
    "logistic_regression.optimized": ModelSpec(
        info=ModelInfo(
            id="logistic_regression.optimized",
            display_name="逻辑回归",
            task="classification",
            compatible_datasets=("wdbc",),
            default_params={
                "learning_rate": 0.1,
                "max_iter": 1000,
                "threshold": 0.5,
                "l2": 0.0,
                "tol": 1e-8,
                "standardize": True,
                "class_weight": None,
            },
            parameter_descriptions={
                "learning_rate": "梯度下降学习率（正数）",
                "max_iter": "最大迭代次数（正整数）",
                "threshold": "分类阈值（0~1）",
                "l2": "L2 正则化强度（非负）",
                "tol": "早停容差（非负）",
                "standardize": "是否对特征做标准化",
                "class_weight": "类别权重（null 或 'balanced'）",
            },
        )
    ),
    "mlp_regression.optimized": ModelSpec(
        info=ModelInfo(
            id="mlp_regression.optimized",
            display_name="MLP回归",
            task="regression",
            compatible_datasets=("concrete", "california_housing"),
            default_params={
                "activation": "relu",
                "learning_rate": 0.001,
                "max_iter": 100,
                "batch_size": 32,
                "l2": 0.0,
                "validation_fraction": 0.2,
                "n_iter_no_change": 20,
                "standardize": True,
                "standardize_target": True,
                "gradient_clip": 5.0,
            },
            parameter_descriptions={
                "activation": "激活函数（relu/tanh）",
                "learning_rate": "学习率（正数）",
                "max_iter": "最大迭代次数（正整数；交互默认 100，可调高进行更充分训练）",
                "batch_size": "小批量大小（正整数）",
                "l2": "L2 正则化强度（非负）",
                "validation_fraction": "早停验证集比例（0~1）",
                "n_iter_no_change": "早停耐心轮数（正整数）",
                "standardize": "是否对特征做标准化",
                "standardize_target": "是否对目标值做标准化",
                "gradient_clip": "梯度裁剪阈值（正数）",
            },
        )
    ),
    "one_class_svm.optimized": ModelSpec(
        info=ModelInfo(
            id="one_class_svm.optimized",
            display_name="单类SVM",
            task="anomaly_detection",
            compatible_datasets=("6_cardio", "23_mammography"),
            default_params={
                "nu": 0.1,
                "gamma": "scale",
                "max_iter": 500,
                "standardize": True,
            },
            parameter_descriptions={
                "nu": "训练误差上限（0~1，越小边界越紧）",
                "gamma": "RBF 核系数（'scale' 或正数）",
                "max_iter": "最大迭代次数（正整数）",
                "standardize": "是否对特征做标准化",
            },
        )
    ),
    "random_forest.optimized": ModelSpec(
        info=ModelInfo(
            id="random_forest.optimized",
            display_name="随机森林",
            task="classification",
            compatible_datasets=("wdbc",),
            default_params={
                "n_estimators": 50,
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "voting": "soft",
            },
            parameter_descriptions={
                "n_estimators": "树的数量（正整数）",
                "max_depth": "单棵树最大深度（正整数）",
                "min_samples_split": "分裂所需最小样本数",
                "min_samples_leaf": "叶节点最小样本数",
                "voting": "投票方式（soft 概率平均 / hard 多数投票）",
            },
        )
    ),
}


def model_catalog() -> tuple[ModelInfo, ...]:
    """Return detached metadata copies in stable identifier order."""

    return tuple(
        replace(
            _MODEL_SPECS[model_id].info,
            default_params=dict(_MODEL_SPECS[model_id].info.default_params),
            parameter_descriptions=dict(
                _MODEL_SPECS[model_id].info.parameter_descriptions
            ),
        )
        for model_id in sorted(_MODEL_SPECS)
    )


def get_model_spec(model_id: str) -> ModelSpec | None:
    return _MODEL_SPECS.get(model_id)


def register_runner(model_id: str, runner: Runner) -> None:
    """Attach an internal runner while preserving the public specification."""

    spec = _MODEL_SPECS[model_id]
    _MODEL_SPECS[model_id] = replace(spec, runner=runner)


register_runner("cart_decision_tree.optimized", run_cart_decision_tree)
register_runner("dbscan.optimized", run_dbscan)
register_runner("gaussian_naive_bayes.optimized", run_gaussian_naive_bayes)
register_runner("gbdt_regression.optimized", run_gbdt_regression)
register_runner("isolation_forest.optimized", run_isolation_forest)
register_runner("kmeans.optimized", run_kmeans)
register_runner("knn.optimized", run_knn)
register_runner("linear_regression.optimized", run_linear_regression)
register_runner("logistic_regression.optimized", run_logistic_regression)
register_runner("mlp_regression.optimized", run_mlp_regression)
register_runner("one_class_svm.optimized", run_one_class_svm)
register_runner("random_forest.optimized", run_random_forest)
