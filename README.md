# 第 07 组机器学习可视化平台

前端、后端与模型现已统一到本仓库。完整安装、启动、接口映射和限制见 [整合运行说明](docs/FULL_STACK.md)，代码来源见 [SOURCES](docs/SOURCES.md)。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . -e "./backend[test]"
npm --prefix frontend ci
python scripts/dev.py
```

打开 http://127.0.0.1:5173，按算法查看原有六图，或提交一次完整原实验复现。
界面布局主要参考 Geeker Admin；实验内容以本项目原始 `visualization/`、`data/`、`figures/` 为准，不使用模板 Mock 或队友测试配置。

- **原有成果**：12 个算法的 72 张原图，18 个已有绘图源数据 CSV，支持放大和下载。
- **重新训练**：调用对应原脚本，保留原数据、划分、种子、调参、交叉验证、消融和基础版/优化版比较；不是仅训练一个优化版模型。
- **结果隔离**：原图保持不动，新结果存入独立运行目录。历史成果不冒充本次训练结果。
- **数据来源**：原目录共有 7 组数据，其中 5 组用于当前原始六图流程；Adult 与 California Housing 保留为原项目存档，不自动替换实验输入。
- **真实状态**：后台串行任务、日志、运行记录与刷新恢复；不展示虚构的逐轮百分比。

运行说明、原实验 HTTP 接口和限制见 [FULL_STACK](docs/FULL_STACK.md)，原实验协议索引见 [ORIGINAL_EXPERIMENTS](docs/ORIGINAL_EXPERIMENTS.md)。

## ml-core 公共包

> 下方 `ExperimentConfig/run_experiment` 是保留兼容的单次训练 API，**不是当前前端的原实验复现入口**。原实验目录通过 `list_original_experiments()` 和 `list_original_datasets()` 获取；前端调用 `/api/original-*`。不要用下方示例参数覆盖原脚本的实验协议。

`ml-core` provides a stable experiment API in front of the repository's from-scratch machine-learning implementations. Application code imports from `ml_core` and does not depend on files inside `Models/`.

`list_models()` and `list_datasets()` are the source of truth for supported model variants, compatible datasets, defaults and parameter descriptions. The public API is stable while internal model adapters can evolve independently of the frontend.

## Installation

For development, install the local checkout in editable mode:

```bash
pip install -e .
```

Consumers should pin a release tag or commit instead of following the moving `main` branch:

```text
ml-core @ git+https://github.com/Icey18899877399/26MLP_team07.git@5c1bb06a52e9e542e20bb18a3f0f8d39b16f47ed
```

## Discovery

Discovery returns immutable metadata that an application can use to build model, dataset, and parameter selectors:

```python
from dataclasses import asdict
from ml_core import list_datasets, list_models

models = [asdict(item) for item in list_models()]
datasets = [asdict(item) for item in list_datasets()]
```

Only combinations implemented by the installed package are advertised.

## Run an experiment

```python
from ml_core import ExperimentConfig, run_experiment

result = run_experiment(
    ExperimentConfig(
        model="logistic_regression.optimized",
        dataset="wdbc",
        params={"max_iter": 1000},
        test_size=0.2,
        random_state=42,
    )
)

print(result.to_dict())
```

`ExperimentResult.to_dict()` contains JSON-compatible effective parameters, scalar metrics, artifact descriptors, and metadata. A fixed `random_state` produces reproducible splits or initializations; `run_id` is unique for each invocation.

Configuration mistakes use a public exception hierarchy:

```python
from ml_core import ExperimentConfig, InvalidParameterError, MLCoreError, run_experiment

config = ExperimentConfig(
    model="logistic_regression.optimized",
    dataset="wdbc",
    test_size=0.2,
)

try:
    result = run_experiment(config)
except InvalidParameterError as error:
    print(f"Invalid model parameter: {error}")
except MLCoreError as error:
    print(f"Experiment could not run: {error}")
```

## FastAPI integration

The core API is deliberately synchronous and has no dependency on FastAPI. Because training is CPU-bound, call it through FastAPI's worker-thread helper from an async route:

```python
from fastapi.concurrency import run_in_threadpool
from ml_core import ExperimentConfig, run_experiment


config = ExperimentConfig(
    model=request.model,
    dataset=request.dataset,
    params=request.params,
    test_size=request.test_size,
    random_state=request.random_state,
)
result = await run_in_threadpool(run_experiment, config)
return result.to_dict()
```

The integrated endpoints are `GET /api/models`, `GET /api/datasets`, and `POST /api/experiments`. Use the full model ID; do not send a separate `variant`. Malformed HTTP requests return 422, domain configuration errors return 400, unavailable packages return 503, and model execution failures return 502. See [HTTP contract](backend/docs/HTTP_API_CONTRACT.md).
