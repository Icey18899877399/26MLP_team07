"""Public metadata and allowlisted files for the twelve original figure workflows."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_WDBC = "data/classification/wdbc/wdbc.data"
_CONCRETE = "data/regression/concrete/concrete_data.csv"
_SEEDS = "data/clustering/seeds/seeds_dataset.txt"
_CARDIO = "data/anomaly/6_cardio.npz"
_MAMMOGRAPHY = "data/anomaly/23_mammography.npz"

# The names here are the archived output contract, not newly generated examples.
_SPECS = {
    "cart_decision_tree": ("CART 决策树", "classification", (_WDBC,), "01_split_gain_mechanism 02_tree_structure_comparison 03_decision_boundary_comparison 04_pruning_validation_curve 05_test_set_diagnostics 06_optimization_evidence", {"seed": 42, "folds": 5}),
    "gaussian_naive_bayes": ("高斯朴素贝叶斯", "classification", (_WDBC,), "01_gaussian_feature_fit 02_posterior_decision_surface 03_assumption_diagnostics 04_benchmark_performance 05_probability_diagnostics 06_optimization_analysis", {"seed": 42, "folds": 5}),
    "knn": ("K 近邻", "classification", (_WDBC,), "01_local_neighborhood 02_k_decision_boundaries 03_k_validation_curve 04_roc_pr_comparison 05_confusion_matrices 06_optimization_and_runtime", {"seed": 42, "folds": 5}),
    "logistic_regression": ("逻辑回归", "classification", (_WDBC,), "01_class_distribution 02_loss_curve 03_confusion_matrix 04_roc_curve 05_precision_recall_curve 06_optimization_ablation", {"seed": 42, "max_iter": 600}),
    "random_forest": ("随机森林", "classification", (_WDBC,), "01_bootstrap_oob_mechanism 02_tree_count_convergence 03_hyperparameter_response 04_tree_diversity_and_ensemble_gain 05_feature_importance_stability 06_final_benchmark_comparison", {"seed": 42, "folds": 5}),
    "gbdt_regression": ("GBDT 回归", "regression", (_CONCRETE,), "01_stagewise_residual_fitting 02_convergence_and_early_stopping 03_hyperparameter_landscape 04_feature_importance_and_dependence 05_prediction_and_residual_diagnostics 06_final_benchmark_comparison", {"seed": 42, "folds": 5, "max_estimators": 60}),
    "linear_regression": ("线性回归", "regression", (_CONCRETE,), "01_data_distribution_and_correlation 02_optimization_convergence 03_regularization_path 04_actual_vs_predicted 05_residual_diagnostics 06_final_benchmark_comparison", {"seed": 42, "folds": 5}),
    "mlp_regression": ("多层感知机回归", "regression", (_CONCRETE,), "01_network_and_nonlinear_fitting 02_training_dynamics_and_early_stopping 03_hyperparameter_performance 04_feature_interpretation 05_prediction_and_residual_diagnostics 06_final_benchmark_comparison", {"seed": 42, "folds": 5, "tuning_samples": 240, "tuning_max_iter": 80, "final_max_iter": 300, "seed_runs": 5, "timing_repeats": 3}),
    "dbscan": ("DBSCAN", "clustering", (_SEEDS,), "01_cluster_result 02_density_mechanism 03_k_distance_curve 04_parameter_sensitivity 05_standardization_comparison 06_benchmark_comparison", {"seed": 42, "min_samples": 5}),
    "kmeans": ("K-Means", "clustering", (_SEEDS,), "01_cluster_distribution 02_centroid_trajectory 03_convergence_comparison 04_cluster_number_selection 05_initialization_stability 06_benchmark_comparison", {"seed": 42, "clusters": 3, "stability_runs": 30, "benchmark_runs": 10}),
    "isolation_forest": ("孤立森林", "anomaly_detection", (_CARDIO, _MAMMOGRAPHY), "01_isolation_mechanism 02_score_landscape 03_score_distribution 04_roc_pr_curves 05_parameter_sensitivity 06_benchmark_comparison", {"seed": 13, "quick": False}),
    "one_class_svm": ("单类支持向量机", "anomaly_detection", (_CARDIO, _MAMMOGRAPHY), "01_boundary_mechanism 02_dual_kkt_structure 03_optimization_convergence 04_scores_roc_pr 05_parameter_robustness 06_cross_dataset_benchmark", {"quick": False}),
}

_DATASETS = {
    "wdbc": ("WDBC 乳腺癌诊断", (_WDBC,), False),
    "concrete": ("混凝土抗压强度", (_CONCRETE,), False),
    "seeds": ("小麦种子", (_SEEDS,), False),
    "6_cardio": ("Cardio 心电异常", (_CARDIO,), False),
    "23_mammography": ("Mammography 乳腺造影异常", (_MAMMOGRAPHY,), False),
    "adult": ("Adult 人口普查（存档）", ("data/classification/adult/adult.data", "data/classification/adult/adult.test"), True),
    "california_housing": ("California Housing（存档）", ("data/regression/california_housing/CaliforniaHousing/cal_housing.data",), True),
}

_PROTOCOL = {
    "cart_decision_tree": ("WDBC 原数据；5 折验证剪枝，并展示划分增益、决策边界与测试集诊断。", "同时比较基础与优化实现。"),
    "gaussian_naive_bayes": ("WDBC 原数据；5 折验证，检查高斯特征拟合、独立性假设与概率诊断。", "同时比较基础与优化实现。"),
    "knn": ("WDBC 原数据；5 折验证 K 值，输出 ROC/PR、混淆矩阵与运行时间。", "机制示意图由脚本生成演示样本。"),
    "logistic_regression": ("WDBC 原数据；随机种子 42、最大迭代 600，绘制损失与分类诊断。", "保留原优化消融；六张图均有对应 CSV 源数据。"),
    "random_forest": ("WDBC 原数据；5 折验证，包含 bootstrap/OOB、树数量收敛和特征重要性稳定性。", "同时比较基础与优化实现。"),
    "gbdt_regression": ("Concrete 原数据；5 折验证，最大基学习器数 60。", "展示逐步残差拟合、早停、超参数响应及最终基准比较。"),
    "linear_regression": ("Concrete 原数据；5 折验证，包含优化收敛与正则化路径。", "保留真实值/预测值和残差诊断。"),
    "mlp_regression": ("Concrete 原数据；5 折验证，调参样本 240、调参迭代 80、最终迭代 300。", "保留 5 次种子运行及 3 次计时重复。"),
    "dbscan": ("Seeds 原数据；随机种子 42，min_samples=5。", "绘制密度机制、k-distance、参数敏感性与标准化对比。"),
    "kmeans": ("Seeds 原数据；3 类，30 次初始化稳定性运行和 10 次基准运行。", "保留质心轨迹、收敛与类别数选择；六张图均有 CSV。"),
    "isolation_forest": ("Cardio 训练/验证/测试分层划分；只用 Cardio 验证集调参，迁移至 Mammography 基准。", "完整流程不启用 quick，原脚本随机种子 13；测试集不参与调参。"),
    "one_class_svm": ("Cardio 与 Mammography 各自分层划分；仅正常样本作干净参考训练。", "完整流程使用种子 17/29/43、正常训练上限 72、验证和测试各自分层上限 720；不启用 quick。"),
}


def original_root() -> Path:
    """Return original assets from checkout or from namespaced wheel resources."""
    checkout = _ROOT
    if (checkout / "pyproject.toml").is_file() and (checkout / "visualization").is_dir():
        return checkout
    return Path(__file__).resolve().parent / "resources" / "original"


def original_experiment_spec(experiment_id: str) -> tuple:
    try:
        return _SPECS[experiment_id]
    except KeyError:
        raise KeyError(f"Unknown original experiment: {experiment_id}") from None


def _title(stem: str) -> str:
    return stem.split("_", 1)[1].replace("_", " ").title()


def list_original_experiments() -> list[dict]:
    catalog = []
    for experiment_id, (title, task, paths, stems, parameters) in _SPECS.items():
        figure_stems = stems.split()
        source_data = [
            {"name": f"source_data/{stem}.csv", "url": f"/api/original-assets/{experiment_id}/source_data/{stem}.csv"}
            for stem in figure_stems
            if experiment_id in {"knn", "kmeans", "logistic_regression"}
        ]
        catalog.append({
            "id": experiment_id,
            "title": title,
            "task": task,
            "script": f"visualization.{experiment_id}_figures",
            "datasets": [
                {"id": next(key for key, item in _DATASETS.items() if path in item[1]), "name": next(item[0] for item in _DATASETS.values() if path in item[1]), "path": path}
                for path in paths
            ],
            "parameters": dict(parameters),
            "protocol": [
                *_PROTOCOL[experiment_id],
                "复现只传原始数据路径和独立输出目录，其余采用原脚本默认完整流程。",
            ],
            "figures": [
                {"name": f"{stem}.png", "title": _title(stem), "url": f"/api/original-assets/{experiment_id}/{stem}.png"}
                for stem in figure_stems
            ],
            "source_data": source_data,
        })
    return catalog


def list_original_datasets() -> list[dict]:
    return [
        {
            "id": dataset_id,
            "name": name,
            "paths": list(paths),
            "used_by": [experiment_id for experiment_id, spec in _SPECS.items() if any(path in spec[2] for path in paths)],
            "archived_only": archived_only,
        }
        for dataset_id, (name, paths, archived_only) in _DATASETS.items()
    ]


def resolve_original_asset(experiment_id: str, filename: str) -> Path:
    original_experiment_spec(experiment_id)
    item = next(item for item in list_original_experiments() if item["id"] == experiment_id)
    allowed = {asset["name"] for asset in item["figures"] + item["source_data"]}
    if filename not in allowed:
        raise ValueError("Unknown original asset")
    root = (original_root() / "figures" / experiment_id).resolve()
    target = (root / filename).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Original asset unavailable")
    return target
