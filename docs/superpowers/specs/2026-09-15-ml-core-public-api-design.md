# ml_core Public API Design

## Goal

Turn the repository into an installable Python distribution named `ml-core` with a stable
`ml_core` import surface. Consumers such as the Zhiyuan FastAPI application must not import
from `Models.*`, depend on repository-relative paths, or interpret model-specific return values.

The first release supports two end-to-end experiment combinations:

- `logistic_regression.optimized` with the `wdbc` classification dataset.
- `kmeans.optimized` with the `seeds` clustering dataset.

Only combinations that are fully runnable appear in discovery results. Additional algorithms
can be registered later without changing the public API.

## Public API

The top-level package exports:

- `list_models() -> tuple[ModelInfo, ...]`
- `list_datasets() -> tuple[DatasetInfo, ...]`
- `run_experiment(config: ExperimentConfig) -> ExperimentResult`
- `ExperimentConfig`
- `ExperimentResult`
- `ModelInfo`
- `DatasetInfo`
- public `MLCoreError` subclasses used for caller-facing error handling

`ml_core.__init__` is the compatibility boundary. Internal modules, adapter names, model
classes, and dataset loaders are not part of the supported API.

## Package Layout

```text
pyproject.toml
ml_core/
  __init__.py
  api.py
  types.py
  errors.py
  registry.py
  datasets.py
  adapters/
    __init__.py
    classification.py
    clustering.py
  resources/
    __init__.py
    datasets/
      __init__.py
      wdbc.data
      seeds_dataset.txt
Models/                    # unchanged internal implementations
tests/
  test_public_api.py
  test_experiments.py
  test_wheel_install.py
```

The wheel contains both `ml_core` and the existing `Models` package. `ml_core` may import
`Models` internally, but consumers must not. A future release may move the implementations
under `ml_core._models` without affecting callers.

Built-in datasets live inside package resources and are located through `importlib.resources`.
No experiment depends on the process working directory or the source checkout layout.

## Types and Serialization Contract

`ExperimentConfig` is an immutable dataclass with these fields:

- `model: str`
- `dataset: str`
- `params: Mapping[str, JSONValue]`, default empty
- `test_size: float | None`, default `None`
- `random_state: int`, default `42`

`ExperimentResult` is an immutable dataclass containing:

- generated `run_id`
- stable model and dataset identifiers
- task type
- effective parameters after defaults are applied
- scalar metric dictionary
- JSON-compatible metadata
- artifact descriptors, empty in the first release

Every public result provides `to_dict()` and contains only JSON-compatible values. Model
instances, `Path` objects, and NumPy scalar or array objects never cross the public boundary.

`ModelInfo` describes the stable ID, display name, task, compatible datasets, default
parameters, and parameter descriptions. `DatasetInfo` describes the stable ID, display name,
task, sample count, feature count, and target availability. Discovery output is deterministic
and sorted by stable ID.

## Registry and Adapters

The registry maps each public model ID to an internal immutable specification containing its
metadata and runner callable. `run_experiment` performs this flow:

1. Validate the configuration and resolve the model specification.
2. Resolve and load the built-in dataset resource.
3. Reject an unsupported model/dataset pairing.
4. Merge model defaults with caller parameters and reject unknown parameters.
5. Invoke the model-specific adapter.
6. Normalize metrics and metadata into `ExperimentResult`.

Adapters isolate different estimator protocols. The classification adapter performs a
deterministic stratified train/test split, fits `OptimizedLogisticRegressionScratch`, and
returns accuracy, precision, recall, F1, and confusion-matrix counts. The clustering adapter
fits `OptimizedKMeansScratch` on the Seeds features and returns inertia, iteration count,
cluster count, and, because the dataset contains reference labels, adjusted Rand index.

Model IDs refer to public behavior rather than file names. Renaming or moving an implementation
only changes its registry adapter.

## Validation and Errors

Public exceptions derive from `MLCoreError`:

- `UnknownModelError`
- `UnknownDatasetError`
- `IncompatibleDatasetError`
- `InvalidConfigError`
- `InvalidParameterError`
- `ExperimentExecutionError`

Unknown identifiers and invalid client parameters remain distinguishable from unexpected model
execution failures. The core does not import FastAPI or choose HTTP status codes; the Zhiyuan
adapter maps configuration errors to HTTP 422 and unexpected execution failures to HTTP 500.

`test_size` must be between zero and one when supplied. It is used for supervised experiments.
Supplying it to the K-Means experiment is rejected instead of being silently ignored.

## Packaging and Dependencies

The project uses Hatchling, matching the Zhiyuan repository. Distribution metadata declares
Python 3.11 or later and every direct runtime dependency used by the public runners. Plotting
and benchmark-only dependencies are not required by the first public API release.

The package must pass all three installation modes:

- editable local install for development
- wheel build and clean-environment install
- VCS install pinned to a tag or commit

The interaction repository should pin a release tag, for example
`ml-core @ git+https://github.com/Icey18899877399/26MLP_team07.git@v0.1.0`, rather than depend on
the moving `main` branch.

## Zhiyuan Integration Boundary

This change does not modify the Zhiyuan repository. Its service layer can construct
`ExperimentConfig`, call `run_experiment`, and serialize `ExperimentResult.to_dict()`.
Because model execution is synchronous and CPU-bound, an async FastAPI route should call it in
a worker thread. Long-running job persistence and queues are outside this first release.

Recommended future endpoints are:

- `GET /api/experiments/models`
- `GET /api/experiments/datasets`
- `POST /api/experiments`

## Testing and Acceptance Criteria

The implementation is accepted when:

1. `from ml_core import ExperimentConfig, ExperimentResult, list_models, list_datasets,
   run_experiment` succeeds after installing the built wheel.
2. Discovery returns exactly the two supported model/dataset combinations with deterministic,
   JSON-serializable metadata.
3. Both supported experiments run from a directory outside the repository.
4. Repeating an experiment with the same configuration produces the same metrics.
5. Invalid IDs, invalid parameters, invalid splits, and incompatible pairs raise their documented
   public exception types.
6. Results round-trip through `json.dumps(result.to_dict())`.
7. Existing `Models` tests still pass without source changes.
8. A clean wheel-install smoke test proves that no repository-relative data path is required.

## Deferred Scope

The remaining ten algorithm families, baseline variants, plot generation, user-uploaded
datasets, asynchronous job storage, cancellation, and remote artifact serving are intentionally
deferred. They will be added behind the same registry and result contract after the first two
adapters establish a tested integration path.
