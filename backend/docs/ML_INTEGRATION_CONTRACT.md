# ML 公共包集成合同

> 历史资料：本文描述上游尚未接入 ML Core 时的目标合同。整合版已使用实际 ModelInfo/DatasetInfo API；请以 [当前整合说明](../../docs/FULL_STACK.md) 为准。

## 目的与现状

交互仓库需要一个小而稳定的 Python API，而不是依赖 `Models.logistic_regression` 等内部路径。外部仓库当前 commit `5f2ab0e7b522555cff7e9b6ae70a811bb17b4e0d`：

- 没有 `pyproject.toml`、`setup.py` 或 `setup.cfg`，不能作为标准 Python distribution 安装；
- `Models/__init__.py` 公开了 24 个 base/optimized 模型类，但这是算法类集合，不是 Web 实验 API；
- 监督学习模型通常提供 `fit(X, y)`/`predict(X)`，聚类与异常检测通常提供 `fit(X)`/`fit_predict(X)`，概率和 score 方法也因任务而异；
- 多个 `visualization/*_figures.py` 文件各自提供不同签名的 `run_experiment`，同时包含数据读取、评价与 Matplotlib 逻辑；
- 没有统一模型/数据集注册表，也没有 Dataset、ExperimentConfig 或 ExperimentResult 抽象。

因此本阶段只固定仓库版本并实现等待公共 API 的边界，不修改对方算法代码。

## 最小 package boundary

ML 仓库负责人应增加最小 packaging 层：

```text
pyproject.toml
ml_core/
  __init__.py
  api.py
  types.py
```

安装目标：

```powershell
python -m pip install -e external/ml-core
```

`ml_core/__init__.py` 必须稳定公开：

```python
from ml_core import (
    DatasetSpec,
    ExperimentConfig,
    ExperimentResult,
    ExperimentExecutionError,
    InvalidExperimentError,
    MLCoreError,
    ModelSpec,
    list_datasets,
    list_models,
    run_experiment,
)
```

内部仍可使用现有 `Models` 和实验代码，但公共调用方不应知道这些路径。

## 数据类型

以下字段是当前最小合同。类型可用 frozen dataclass、TypedDict 或 Pydantic model 实现，但对象必须能稳定转换为同形 JSON。

### ModelSpec

```json
{
  "id": "logistic_regression",
  "name": "Logistic Regression",
  "task_type": "classification",
  "variants": ["base", "optimized"],
  "parameters": {}
}
```

- `id` 是稳定、非空、适合 URL/JSON 的标识，不随显示文案改变。
- `task_type` 当前只使用 `classification`、`regression`、`clustering`、`anomaly_detection`。
- `variants` 使用显式字符串 `base` 和 `optimized`，不要把 variant 编进模型 id 或暴露具体类名。
- `parameters` 只描述前端需要知道的可配置参数；第一版允许为空，不要求一次完成参数表单 schema 系统。

### DatasetSpec

```json
{
  "id": "wdbc",
  "name": "Wisconsin Diagnostic Breast Cancer",
  "task_type": "classification"
}
```

数据集定位、下载、校验、解析和缓存全部由 ML 仓库负责。interaction 仓库不接受本地任意路径，也不复制数据集。

### ExperimentConfig

```json
{
  "model": "logistic_regression",
  "variant": "optimized",
  "dataset": "wdbc",
  "params": {},
  "test_size": 0.2,
  "random_state": 42
}
```

- `run_experiment` 接收 `ml_core.ExperimentConfig` 的单个实例。
- `params` 必须是 JSON 值组成的对象；ML 包负责验证模型特定参数。
- 数据划分、标准化、训练、预测、调参和评价都发生在 ML 包内。

### ExperimentResult

```json
{
  "model": "logistic_regression",
  "variant": "optimized",
  "dataset": "wdbc",
  "metrics": {"accuracy": 0.95, "f1": 0.94},
  "diagnostics": {}
}
```

- `metrics` 是名称到有限数值的映射；不得返回 NaN 或 Infinity。
- `diagnostics` 是可选的、JSON 可序列化的结构化数据，不是 Matplotlib figure、NumPy object 或磁盘图片路径。
- 第一版只返回已有可靠信息；不要为了填满 schema 伪造字段。

## Diagnostics 原则

Diagnostics 为 Web 可视化提供原始结构化数据，例如：

- 分类：confusion matrix、ROC/PR 点列、loss curve；
- 回归：prediction/residual 点列、loss curve；
- 聚类：二维投影点、cluster labels、centroids；
- 异常检测：scores、threshold、labels；
- 通用：feature names 与 feature importance。

每项应包含足够的标签、顺序和含义，且数组长度保持合理。前端消费者负责渲染；ML 仓库现有 Matplotlib 报告图不整体迁入交互仓库。

## 函数语义

```python
def list_models() -> list[ModelSpec]: ...
def list_datasets() -> list[DatasetSpec]: ...
def run_experiment(config: ExperimentConfig) -> ExperimentResult: ...
```

- 列表顺序应确定且稳定。
- `list_models` 只声明当前确实可运行的组合。
- `list_datasets` 只声明已能按交付规则准备的数据集。
- `run_experiment` 必须完成真实的 ML 流程；interaction 层不会补充分割、训练或评价逻辑。

## Exception contract

`ml_core` 顶层必须公开一个 `MLCoreError` 基类，以及两个最小子类：

- `InvalidExperimentError`：未知 id、任务类型不兼容、参数越界或数据不可用等调用方可修正的问题；
- `ExperimentExecutionError`：训练或评价过程失败。

异常消息必须适合记录且不得包含敏感路径。当前 interaction boundary 在公共异常类型落地前，会把 `ValueError`/Pydantic validation 映射为 HTTP 400，把其他执行异常映射为 HTTP 502；包缺失或合同不完整映射为 HTTP 503。

## JSON serialization

- 所有顶层公共对象应可转为普通 `dict`，值只含 JSON primitive、list 和 object。
- 数值数组在 API 返回前转换为普通 list；NumPy scalar 转为 Python 数值。
- 不返回 estimator、callable、Path、set、bytes、figure 或文件句柄。
- 不使用 NaN/Infinity。过大的诊断数据应先采样，并在结果中清楚说明采样规则。

## Compatibility 与版本

- `ml_core` 应在 `pyproject.toml` 中声明版本，并遵循语义化版本原则。
- 删除/重命名公共导出、字段、枚举值或改变语义属于不兼容变更。
- 新增可选 diagnostics 或 parameter metadata 可作为向后兼容变更。
- interaction 仓库用 submodule SHA 固定经过验证的实现；升级 SHA 时必须重新运行双方的 contract/smoke tests。
- 第一阶段不建立插件框架或每模型 Adapter。

## Interaction 禁止依赖的内部细节

- `Models.*` 的模块路径、类名、构造参数和实例属性；
- `visualization/*_figures.py` 的函数、报告 dict 细节和 Matplotlib 输出；
- ML 仓库中的 `data/`、`figures/`、`tests/` 或 `examples/` 路径；
- 数据划分、标准化、评价指标、调参和随机采样实现；
- base/optimized 具体类之间的差异。

## 数据集交付缺口

课程 PDF 要求代码仓库不得包含数据集本体，但当前 ML commit 的 `data/` 中存在压缩包和解压后的数据。最终交付前，ML 仓库负责人应：

1. 从 Git 跟踪中移除数据集本体并确认历史/发布策略；
2. 提供受许可约束的下载或准备脚本、来源说明、校验和与缓存目录；
3. 将缓存目录加入 `.gitignore`；
4. 让 `list_datasets` 反映准备状态，缺失数据时抛出清晰的公共异常。

该问题属于 ML 仓库的数据所有权范围，interaction 仓库不复制或代管这些文件。
