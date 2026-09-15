"""Stable, JSON-friendly value objects exposed by :mod:`ml_core`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, TypeAlias


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration accepted by :func:`ml_core.run_experiment`."""

    model: str
    dataset: str
    params: Mapping[str, JSONValue] = field(default_factory=dict)
    test_size: float | None = None
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    """A generated artifact that a transport layer may choose to publish."""

    name: str
    media_type: str
    uri: str


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Normalized result returned for every supported experiment."""

    run_id: str
    model: str
    dataset: str
    task: str
    effective_params: dict[str, JSONValue]
    metrics: dict[str, int | float]
    artifacts: tuple[ArtifactInfo, ...] = ()
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a deep, JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata used by callers to build a model selector and parameter form."""

    id: str
    display_name: str
    task: str
    compatible_datasets: tuple[str, ...]
    default_params: dict[str, JSONValue]
    parameter_descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Metadata describing a built-in dataset."""

    id: str
    display_name: str
    task: str
    sample_count: int
    feature_count: int
    has_target: bool
