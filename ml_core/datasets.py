"""Private loaders and public metadata for packaged datasets."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
from importlib.resources import files

import numpy as np

from .types import DatasetInfo


_RESOURCE_PACKAGE = "ml_core.resources.datasets"


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Numeric rows and optional reference targets consumed by adapters."""

    info: DatasetInfo
    features: tuple[tuple[float, ...], ...]
    targets: tuple[int | float, ...]


_DATASETS = {
    "concrete": DatasetInfo("concrete", "Concrete Compressive Strength", "regression", 1030, 8, True),
    "cardio": DatasetInfo("cardio", "Cardiotocography Anomaly Detection", "anomaly_detection", 1831, 21, True),
    "seeds": DatasetInfo(
        id="seeds",
        display_name="Seeds",
        task="clustering",
        sample_count=210,
        feature_count=7,
        has_target=True,
    ),
    "wdbc": DatasetInfo(
        id="wdbc",
        display_name="Wisconsin Diagnostic Breast Cancer",
        task="classification",
        sample_count=569,
        feature_count=30,
        has_target=True,
    ),
}


def dataset_catalog() -> tuple[DatasetInfo, ...]:
    """Return deterministic metadata for every packaged dataset."""

    return tuple(_DATASETS[dataset_id] for dataset_id in sorted(_DATASETS))


def load_dataset(dataset_id: str) -> LoadedDataset:
    """Load a built-in dataset without relying on the process working directory."""

    info = _DATASETS[dataset_id]
    if dataset_id == "wdbc":
        return _load_wdbc(info)
    if dataset_id == "seeds":
        return _load_seeds(info)
    if dataset_id == "concrete":
        rows = list(csv.reader(io.StringIO(_resource_text("concrete_data.csv"))))[1:]
        return LoadedDataset(info, tuple(tuple(map(float, row[:-1])) for row in rows),
                             tuple(float(row[-1]) for row in rows))
    if dataset_id == "cardio":
        with files(_RESOURCE_PACKAGE).joinpath("6_cardio.npz").open("rb") as stream:
            with np.load(stream, allow_pickle=False) as data:
                return LoadedDataset(info, tuple(tuple(map(float, row)) for row in data["X"]),
                                     tuple(map(int, data["y"])))
    raise KeyError(dataset_id)


def _resource_text(filename: str) -> str:
    return files(_RESOURCE_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


def _load_wdbc(info: DatasetInfo) -> LoadedDataset:
    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    for line in _resource_text("wdbc.data").splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 32:
            raise ValueError("WDBC rows must contain an id, a label, and 30 features")
        targets.append(1 if fields[1] == "M" else 0)
        features.append(tuple(float(value) for value in fields[2:]))
    return LoadedDataset(info=info, features=tuple(features), targets=tuple(targets))


def _load_seeds(info: DatasetInfo) -> LoadedDataset:
    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    for line in _resource_text("seeds_dataset.txt").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Seeds rows must contain seven features and one label")
        features.append(tuple(float(value) for value in fields[:7]))
        targets.append(int(fields[7]) - 1)
    return LoadedDataset(info=info, features=tuple(features), targets=tuple(targets))
