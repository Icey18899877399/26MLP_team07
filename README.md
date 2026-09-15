# ml-core

`ml-core` provides a stable experiment API in front of the repository's from-scratch machine-learning implementations. Application code imports from `ml_core` and does not depend on files inside `Models/`.

The first packaged release supports optimized logistic regression on WDBC and optimized K-Means on Seeds. `list_models()` and `list_datasets()` are the source of truth for the combinations available in an installed version.

## Installation

For development, install the local checkout in editable mode:

```bash
pip install -e .
```

Consumers should pin a release tag or commit instead of following the moving `main` branch:

```text
ml-core @ git+https://github.com/Icey18899877399/26MLP_team07.git@v0.1.0
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
from ml_core import InvalidParameterError, MLCoreError

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

Suggested transport endpoints are `GET /api/experiments/models`, `GET /api/experiments/datasets`, and `POST /api/experiments`. The web layer should map configuration errors to HTTP 422 and unexpected `ExperimentExecutionError` failures to HTTP 500.
