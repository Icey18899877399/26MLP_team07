# ml_core Public API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish an installable `ml-core` package that exposes stable discovery and experiment APIs for optimized logistic regression on WDBC and optimized K-Means on Seeds.

**Architecture:** A public dataclass contract and typed error hierarchy sit in front of a private registry. Dataset loaders use package resources, and task-specific adapters normalize the incompatible estimator protocols into one JSON-safe result type. Existing `Models` implementations remain unchanged and ship in the same wheel as internal dependencies.

**Tech Stack:** Python 3.11+, standard library, Hatchling, pytest, existing pure-Python `Models` implementations.

**Spec:** `docs/superpowers/specs/2026-09-15-ml-core-public-api-design.md`

## Global Constraints

- Distribution name is `ml-core`; import package is `ml_core`.
- Python requirement is `>=3.11`.
- `Models/` source files remain unchanged.
- Public output contains only JSON-compatible values.
- Only runnable models and datasets appear in discovery.
- Built-in datasets must load independently of the current working directory.
- The first release supports exactly `logistic_regression.optimized` + `wdbc` and `kmeans.optimized` + `seeds`.
- Model implementations, file paths, and adapter modules are private details.

---

### Task 1: Package metadata, public types, and errors

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `ml_core/__init__.py`
- Create: `ml_core/types.py`
- Create: `ml_core/errors.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Consumes: existing repository layout and Python 3.11 dataclasses.
- Produces: `ExperimentConfig`, `ExperimentResult`, `ArtifactInfo`, `ModelInfo`, `DatasetInfo`, `MLCoreError` subclasses, and a buildable package skeleton.

- [ ] **Step 1: Write failing public-import and serialization tests**

```python
import json


def test_top_level_exports_are_importable():
    from ml_core import ExperimentConfig, ExperimentResult

    config = ExperimentConfig(model="model", dataset="dataset")
    assert config.random_state == 42
    assert ExperimentResult


def test_result_is_json_serializable():
    from ml_core import ExperimentResult

    result = ExperimentResult(
        run_id="run-1",
        model="model",
        dataset="dataset",
        task="classification",
        effective_params={},
        metrics={"accuracy": 1.0},
        metadata={"samples": 2},
    )
    assert json.loads(json.dumps(result.to_dict()))["metrics"]["accuracy"] == 1.0
```

- [ ] **Step 2: Run the focused tests and verify the package is missing**

Run: `python -m pytest tests/test_public_api.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ml_core'`.

- [ ] **Step 3: Add immutable dataclasses and public exceptions**

Define in `ml_core/types.py`:

```python
@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: str
    dataset: str
    params: Mapping[str, JSONValue] = field(default_factory=dict)
    test_size: float | None = None
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    run_id: str
    model: str
    dataset: str
    task: str
    effective_params: dict[str, JSONValue]
    metrics: dict[str, int | float]
    artifacts: tuple[ArtifactInfo, ...] = ()
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return asdict(self)
```

Also define immutable `ArtifactInfo`, `ModelInfo`, and `DatasetInfo`. Define `MLCoreError`, `UnknownModelError`, `UnknownDatasetError`, `IncompatibleDatasetError`, `InvalidConfigError`, `InvalidParameterError`, and `ExperimentExecutionError` in `ml_core/errors.py`.

- [ ] **Step 4: Add Hatchling metadata and top-level re-exports**

Use this build configuration in `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "ml-core"
version = "0.1.0"
requires-python = ">=3.11"
readme = "README.md"

[project.optional-dependencies]
dev = ["pytest>=8.3"]

[tool.hatch.build.targets.wheel]
packages = ["ml_core", "Models"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Export the public types, errors, and API functions from `ml_core/__init__.py` through an explicit `__all__`. Add a README example that imports only from `ml_core`.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest tests/test_public_api.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the public contract**

```bash
git add pyproject.toml README.md ml_core tests/test_public_api.py
git commit -m "feat: add ml_core public contract"
```

---

### Task 2: Package-resource dataset catalog

**Files:**
- Create: `ml_core/resources/__init__.py`
- Create: `ml_core/resources/datasets/__init__.py`
- Copy: `data/classification/wdbc/wdbc.data` to `ml_core/resources/datasets/wdbc.data`
- Copy: `data/clustering/seeds/seeds_dataset.txt` to `ml_core/resources/datasets/seeds_dataset.txt`
- Create: `ml_core/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `DatasetInfo`, WDBC CSV rows, and tab-separated Seeds rows.
- Produces: private `load_dataset(dataset_id: str) -> LoadedDataset` plus public dataset metadata for `wdbc` and `seeds`.

- [ ] **Step 1: Write failing resource and parsing tests**

```python
def test_wdbc_loads_from_an_unrelated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = load_dataset("wdbc")
    assert len(dataset.features) == 569
    assert len(dataset.features[0]) == 30
    assert set(dataset.targets) == {0, 1}


def test_seeds_has_reference_labels():
    dataset = load_dataset("seeds")
    assert len(dataset.features) == 210
    assert len(dataset.features[0]) == 7
    assert set(dataset.targets) == {0, 1, 2}
```

- [ ] **Step 2: Run the focused tests and verify loader import failure**

Run: `python -m pytest tests/test_datasets.py -v`

Expected: FAIL because `ml_core.datasets` does not exist.

- [ ] **Step 3: Implement package-resource loaders**

Use `importlib.resources.files("ml_core.resources.datasets")`. Parse WDBC by discarding column zero, mapping `M` to `1` and `B` to `0`, and converting the remaining 30 fields to floats. Parse Seeds into seven floats and convert labels `1..3` to `0..2`. Expose immutable dataset metadata with exact sample and feature counts.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_datasets.py -v`

Expected: PASS from both the repository root and an unrelated temporary working directory.

- [ ] **Step 5: Commit resource-safe datasets**

```bash
git add ml_core/datasets.py ml_core/resources tests/test_datasets.py
git commit -m "feat: package built-in experiment datasets"
```

---

### Task 3: Registry and deterministic discovery

**Files:**
- Create: `ml_core/registry.py`
- Create: `ml_core/api.py`
- Modify: `ml_core/__init__.py`
- Modify: `tests/test_public_api.py`

**Interfaces:**
- Consumes: public info types and private `load_dataset()`.
- Produces: `list_models()`, `list_datasets()`, model lookup, compatibility checks, default merging, and public validation errors.

- [ ] **Step 1: Write failing discovery and validation tests**

```python
def test_discovery_is_stable_and_only_advertises_runnable_items():
    assert [item.id for item in list_models()] == [
        "kmeans.optimized",
        "logistic_regression.optimized",
    ]
    assert [item.id for item in list_datasets()] == ["seeds", "wdbc"]


def test_unknown_and_incompatible_identifiers_have_typed_errors():
    with pytest.raises(UnknownModelError):
        run_experiment(ExperimentConfig(model="missing", dataset="wdbc"))
    with pytest.raises(UnknownDatasetError):
        run_experiment(ExperimentConfig(model="logistic_regression.optimized", dataset="missing"))
    with pytest.raises(IncompatibleDatasetError):
        run_experiment(ExperimentConfig(model="logistic_regression.optimized", dataset="seeds"))
```

- [ ] **Step 2: Run focused discovery tests and verify failure**

Run: `python -m pytest tests/test_public_api.py -v`

Expected: FAIL because API functions and registry entries are not implemented.

- [ ] **Step 3: Implement immutable registry specifications**

Register stable metadata and allowed parameters for:

```python
"logistic_regression.optimized": {
    "datasets": ("wdbc",),
    "defaults": {
        "learning_rate": 0.1,
        "max_iter": 1000,
        "threshold": 0.5,
        "l2": 0.0,
        "tol": 1e-8,
        "standardize": True,
        "class_weight": None,
    },
}
"kmeans.optimized": {
    "datasets": ("seeds",),
    "defaults": {
        "n_clusters": 3,
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "standardize": True,
    },
}
```

Sort discovery by stable ID. Reject non-string IDs, non-mapping params, unknown parameter keys, non-integer random states, split values outside `(0, 1)`, and non-`None` `test_size` for clustering. Runners remain replaceable callables in the private registry.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_public_api.py -v`

Expected: discovery and typed-error tests PASS; experiment execution tests may still fail because adapters are not registered.

- [ ] **Step 5: Commit registry and discovery**

```bash
git add ml_core/api.py ml_core/registry.py ml_core/__init__.py tests/test_public_api.py
git commit -m "feat: add experiment discovery registry"
```

---

### Task 4: Optimized logistic-regression experiment adapter

**Files:**
- Create: `ml_core/adapters/__init__.py`
- Create: `ml_core/adapters/classification.py`
- Modify: `ml_core/registry.py`
- Test: `tests/test_experiments.py`

**Interfaces:**
- Consumes: `LoadedDataset`, effective logistic parameters, `test_size`, `random_state`, and `OptimizedLogisticRegressionScratch`.
- Produces: `run_logistic_regression(...) -> AdapterResult` with classification metrics and deterministic split metadata.

- [ ] **Step 1: Write failing deterministic experiment tests**

```python
def test_logistic_experiment_is_deterministic_and_json_safe():
    config = ExperimentConfig(
        model="logistic_regression.optimized",
        dataset="wdbc",
        params={"max_iter": 100},
        test_size=0.2,
        random_state=7,
    )
    first = run_experiment(config)
    second = run_experiment(config)
    assert first.metrics == second.metrics
    assert first.metadata == second.metadata
    assert 0.0 <= first.metrics["accuracy"] <= 1.0
    json.dumps(first.to_dict())
```

Also test that unknown parameters, non-positive `learning_rate`/`max_iter`, thresholds outside `[0, 1]`, negative `l2`/`tol`, invalid `standardize`, and invalid `class_weight` raise `InvalidParameterError`.

- [ ] **Step 2: Run the focused test and verify missing runner failure**

Run: `python -m pytest tests/test_experiments.py -k logistic -v`

Expected: FAIL because the logistic adapter is absent.

- [ ] **Step 3: Implement stratified split and normalized metrics**

Shuffle class-specific indices with `random.Random(random_state)`, assign at least one but fewer than all samples from each class to the test set, and compute `tn`, `fp`, `fn`, `tp`, accuracy, precision, recall, and F1 with zero-safe division. Instantiate the existing optimized estimator with effective parameters, fit on training rows, and return iteration count, train/test sample counts, and feature count as metadata.

- [ ] **Step 4: Run focused and existing logistic tests**

Run: `python -m pytest tests/test_experiments.py -k logistic -v`

Expected: PASS.

Run: `python -m pytest tests/test_logistic_regression.py tests/test_logistic_regression_optimized.py -v`

Expected: existing tests PASS without changes to `Models/`.

- [ ] **Step 5: Commit the classification adapter**

```bash
git add ml_core/adapters ml_core/registry.py tests/test_experiments.py
git commit -m "feat: expose logistic regression experiment"
```

---

### Task 5: Optimized K-Means experiment adapter

**Files:**
- Create: `ml_core/adapters/clustering.py`
- Modify: `ml_core/registry.py`
- Modify: `tests/test_experiments.py`

**Interfaces:**
- Consumes: `LoadedDataset`, effective K-Means parameters, `random_state`, and `OptimizedKMeansScratch`.
- Produces: `run_kmeans(...) -> AdapterResult` with inertia, convergence metadata, and adjusted Rand index.

- [ ] **Step 1: Write failing clustering tests**

```python
def test_kmeans_experiment_is_deterministic_and_scores_reference_labels():
    config = ExperimentConfig(
        model="kmeans.optimized",
        dataset="seeds",
        params={"n_init": 3, "max_iter": 100},
        random_state=11,
    )
    first = run_experiment(config)
    second = run_experiment(config)
    assert first.metrics == second.metrics
    assert first.metrics["inertia"] >= 0.0
    assert -1.0 <= first.metrics["adjusted_rand_index"] <= 1.0
    assert first.metadata["sample_count"] == 210
```

Also test that supplying `test_size`, invalid `n_clusters`, unsupported `init`, non-positive `n_init`/`max_iter`, negative `tol`, and non-boolean `standardize` raise the documented public validation exceptions.

- [ ] **Step 2: Run the focused test and verify missing runner failure**

Run: `python -m pytest tests/test_experiments.py -k kmeans -v`

Expected: FAIL because the clustering adapter is absent.

- [ ] **Step 3: Implement the adapter and dependency-free adjusted Rand index**

Build the contingency table for reference and predicted labels. Use `n * (n - 1) // 2` pair counts to compute the adjusted Rand index, returning `1.0` when both partitions have zero expected denominator and are identical. Fit `OptimizedKMeansScratch` with the top-level `random_state` and effective parameters. Return inertia, adjusted Rand index, cluster count, iteration count, sample count, feature count, and cluster sizes as JSON-safe values.

- [ ] **Step 4: Run focused and existing K-Means tests**

Run: `python -m pytest tests/test_experiments.py -k kmeans -v`

Expected: PASS.

Run: `python -m pytest tests/test_kmeans.py tests/test_kmeans_optimized.py -v`

Expected: existing tests PASS without changes to `Models/`.

- [ ] **Step 5: Commit the clustering adapter**

```bash
git add ml_core/adapters/clustering.py ml_core/registry.py tests/test_experiments.py
git commit -m "feat: expose kmeans experiment"
```

---

### Task 6: Complete contract, regression, and wheel-install verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_public_api.py`
- Create: `tests/test_wheel_install.py`

**Interfaces:**
- Consumes: completed public package and Hatchling metadata.
- Produces: clean-install evidence and copyable Zhiyuan integration documentation.

- [ ] **Step 1: Add contract and README examples**

Document editable installation, tag-pinned VCS installation, the exact `ExperimentConfig` call, `to_dict()` serialization, discovery functions, public error handling, and the FastAPI `run_in_threadpool` integration pattern.

Add a contract test asserting every `ModelInfo.compatible_datasets` item exists in `list_datasets()` and that each advertised default configuration runs successfully.

- [ ] **Step 2: Run the complete suite**

Run: `python -m pytest tests -q`

Expected: all new and existing tests PASS.

- [ ] **Step 3: Build a wheel**

Run: `python -m pip wheel . --no-deps --wheel-dir dist`

Expected: one `dist/ml_core-0.1.0-py3-none-any.whl` file and exit code 0.

- [ ] **Step 4: Verify clean installation outside the repository**

Create a temporary virtual environment outside the repository, install the wheel with `--no-deps`, change to the environment's temporary parent directory, and execute:

```python
import json
from ml_core import ExperimentConfig, list_datasets, list_models, run_experiment

assert len(list_models()) == 2
assert len(list_datasets()) == 2
result = run_experiment(
    ExperimentConfig(
        model="kmeans.optimized",
        dataset="seeds",
        params={"n_init": 2},
        random_state=42,
    )
)
json.dumps(result.to_dict())
```

Expected: exit code 0 without the source repository on `sys.path`.

- [ ] **Step 5: Check source integrity and package contents**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git diff 5f2ab0e -- Models`

Expected: no output, proving existing implementations are unchanged.

- [ ] **Step 6: Commit documentation and verification tests**

```bash
git add README.md tests/test_public_api.py tests/test_wheel_install.py
git commit -m "docs: document ml_core integration"
```

- [ ] **Step 7: Push the completed commits**

Run: `git push origin main`

Expected: remote `main` advances to the final verified commit.
