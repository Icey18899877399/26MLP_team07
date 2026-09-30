from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue


TaskType = Literal[
    "classification",
    "regression",
    "clustering",
    "anomaly_detection",
]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MLBackendStatus(StrictContract):
    available: bool
    package: str
    detail: str


class ArtifactInfo(StrictContract):
    name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    uri: str = Field(min_length=1)


class ModelInfo(StrictContract):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    task: TaskType
    compatible_datasets: list[str] = Field(min_length=1)
    default_params: dict[str, JsonValue] = Field(default_factory=dict)
    parameter_descriptions: dict[str, str] = Field(default_factory=dict)


class DatasetInfo(StrictContract):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    task: TaskType
    sample_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    has_target: bool


class ExperimentConfig(StrictContract):
    model: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    test_size: float | None = Field(default=None, gt=0, lt=1)
    random_state: int = 42


class ExperimentResult(StrictContract):
    run_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    task: TaskType
    effective_params: dict[str, JsonValue] = Field(default_factory=dict)
    metrics: dict[str, int | FiniteFloat]
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
