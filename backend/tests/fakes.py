from backend.contracts import (
    DatasetInfo,
    ExperimentConfig,
    ExperimentResult,
    MLBackendStatus,
    ModelInfo,
)


class FakeMLBackend:
    """Interaction-only test double. It is never used by production code."""

    def status(self) -> MLBackendStatus:
        return MLBackendStatus(
            available=True,
            package="fake_ml_core",
            detail="test double",
        )

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="logistic_regression.optimized",
                display_name="Logistic Regression (Optimized)",
                task="classification",
                compatible_datasets=["wdbc"],
                default_params={"max_iter": 1000},
                parameter_descriptions={"max_iter": "Maximum iterations"},
            )
        ]

    def list_datasets(self) -> list[DatasetInfo]:
        return [
            DatasetInfo(
                id="wdbc",
                display_name="Wisconsin Diagnostic Breast Cancer",
                task="classification",
                sample_count=569,
                feature_count=30,
                has_target=True,
            )
        ]

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        return ExperimentResult(
            run_id="test-run",
            model=config.model,
            dataset=config.dataset,
            task="classification",
            effective_params={"max_iter": 1000, **config.params},
            metrics={"accuracy": 0.9},
            artifacts=[],
            metadata={"source": "test-double"},
        )
