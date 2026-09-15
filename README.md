# ml-core

`ml-core` provides a stable experiment API in front of the repository's from-scratch machine-learning implementations. Application code imports from `ml_core` and does not depend on files inside `Models/`.

The first packaged release supports optimized logistic regression on WDBC and optimized K-Means on Seeds. See the public discovery functions for the combinations available in an installed version.

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

Development installation:

```bash
pip install -e .
```

Consumers should pin a release tag or commit:

```text
ml-core @ git+https://github.com/Icey18899877399/26MLP_team07.git@v0.1.0
```
