# ML 公共包集成合同

## 已接入版本

本整合仓库合入模型 PR #1 的 `a72ea7f7f5e04cd0fe55d4a12d90eef67e872c22`，包含标准 `pyproject.toml`、`ml_core` 顶层入口、统一注册表、结构化类型、公共异常及 12 个算法；在此基础上扩展真实图表元数据。不再使用 external/ml-core 子模块。

安装：

```powershell
python -m pip install -e . -e "./backend[test]"
```

interaction 层禁止回退到 `Models.*`、adapter 实现或资源目录等私有路径。

## 顶层公共入口

```python
from ml_core import (
    ArtifactInfo,
    DatasetInfo,
    ExperimentConfig,
    ExperimentExecutionError,
    ExperimentResult,
    IncompatibleDatasetError,
    InvalidConfigError,
    InvalidParameterError,
    MLCoreError,
    ModelInfo,
    UnknownDatasetError,
    UnknownModelError,
    list_datasets,
    list_models,
    run_experiment,
)
```

函数合同：

```python
def list_models() -> tuple[ModelInfo, ...]: ...
def list_datasets() -> tuple[DatasetInfo, ...]: ...
def run_experiment(config: ExperimentConfig) -> ExperimentResult: ...
```

列表只广告确实可运行的项，且顺序稳定。当前模型 ID（按序）：
`cart_decision_tree.optimized`、`dbscan.optimized`、`gaussian_naive_bayes.optimized`、
`gbdt_regression.optimized`、`isolation_forest.optimized`、`kmeans.optimized`、
`knn.optimized`、`linear_regression.optimized`、`logistic_regression.optimized`、
`mlp_regression.optimized`、`one_class_svm.optimized`、`random_forest.optimized`；
数据集 ID（按序）：`23_mammography`、`6_cardio`、`california_housing`、
`concrete`、`seeds`、`wdbc`。

## 类型合同

- `ModelInfo`: `id`, `display_name`, `task`, `compatible_datasets`, `default_params`, `parameter_descriptions`
- `DatasetInfo`: `id`, `display_name`, `task`, `sample_count`, `feature_count`, `has_target`
- `ExperimentConfig`: `model`, `dataset`, `params`, `test_size`, `random_state`
- `ArtifactInfo`: `name`, `media_type`, `uri`
- `ExperimentResult`: `run_id`, `model`, `dataset`, `task`, `effective_params`, `metrics`, `artifacts`, `metadata`

`test_size` 默认为 `None`；分类与回归可以提供 0 到 1 之间的值，聚类不得提供，异常检测忽略该值。模型版本已经属于稳定模型 ID，不再有独立 `variant`。

所有返回值必须能递归转换为 JSON primitive/list/object。metrics 不允许 NaN/Infinity；不得返回 estimator、callable、Path、NumPy object、figure 或文件句柄。

## 异常映射

所有公共异常继承 `MLCoreError`：

- `UnknownModelError`
- `UnknownDatasetError`
- `IncompatibleDatasetError`
- `InvalidConfigError`
- `InvalidParameterError`
- `ExperimentExecutionError`

前五类是调用方可修正的领域错误，映射 HTTP 400；`ExperimentExecutionError` 及未预期执行异常映射为不泄露内部细节的 HTTP 502。包缺失、导出缺失或异常继承关系错误映射 HTTP 503。HTTP body 自身的 Pydantic 错误仍为 422。

## 责任边界

ML 包负责数据定位、解析、拆分、标准化、训练、预测、调参、评价、随机性与 artifact 生成。interaction 层只做：

1. 校验 HTTP JSON；
2. 构造公共 `ExperimentConfig`；
3. 调用三个顶层函数；
4. 重新校验返回对象；
5. 转换稳定错误语义。

上面的安装命令在整合仓库根目录执行。升级 ML 包时必须同时运行 ML 自身测试、interaction contract tests 和真实 HTTP smoke test。

## 剩余交付缺口

课程要求代码仓库不得包含数据集本体，而当前上游将数据文件跟踪在 `ml_core/resources/datasets` 并打入 wheel。最终交付前，ML 负责人需要移除数据集本体，提供来源、许可、下载/准备与校验机制，把缓存加入 `.gitignore`，并让缺失数据通过公共异常清晰表达。interaction 仓库不复制或代管数据文件。
