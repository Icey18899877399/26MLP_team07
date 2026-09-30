# 第 07 组机器学习可视化平台

前端、后端与模型现已统一到本仓库。完整安装、启动、接口映射和限制见 [整合运行说明](docs/FULL_STACK.md)，代码来源见 [SOURCES](docs/SOURCES.md)。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . -e "./backend[test]"
npm --prefix frontend ci
python scripts/dev.py
```

打开 http://127.0.0.1:5173，选择算法、数据集与参数，运行真实训练并查看指标和可视化。
页面流程参考此前提供的 MLTaskConfig 优秀案例视频；所有结果来自本仓库手写模型，不使用模拟训练数据。

## ml-core 公共包

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
