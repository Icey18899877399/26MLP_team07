from backend.contracts import (
    DatasetSpec,
    ExperimentConfig,
    ExperimentResult,
    MLBackendStatus,
    ModelSpec,
)


class FakeMLBackend:
    """Interaction-only test double. It is never used by production code."""

    def status(self) -> MLBackendStatus:
        return MLBackendStatus(
            available=True,
            package="fake_ml_core",
            detail="test double",
        )

    def list_models(self) -> list[ModelSpec]:
        return [
            ModelSpec(
                id="logistic_regression",
                name="Logistic Regression",
                task_type="classification",
                variants=["base", "optimized"],
                parameters={"learning_rate": {"type": "number"}},
            )
        ]

    def list_datasets(self) -> list[DatasetSpec]:
        return [
            DatasetSpec(
                id="wdbc",
                name="Wisconsin Diagnostic Breast Cancer",
                task_type="classification",
            )
        ]

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        return ExperimentResult(
            model=config.model,
            variant=config.variant,
            dataset=config.dataset,
            metrics={"accuracy": 0.9},
            diagnostics={"source": "test-double"},
        )
