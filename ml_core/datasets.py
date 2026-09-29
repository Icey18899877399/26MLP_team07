"""Private loaders and public metadata for packaged datasets."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from .types import DatasetInfo


_RESOURCE_PACKAGE = "ml_core.resources.datasets"


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Numeric rows and optional reference targets consumed by adapters."""

    info: DatasetInfo
    features: tuple[tuple[float, ...], ...]
    targets: tuple[int | float, ...]


_DATASETS = {
    "seeds": DatasetInfo(
        id="seeds",
        display_name="小麦种子",
        task="clustering",
        sample_count=210,
        feature_count=7,
        has_target=True,
    ),
    "wdbc": DatasetInfo(
        id="wdbc",
        display_name="乳腺癌诊断",
        task="classification",
        sample_count=569,
        feature_count=30,
        has_target=True,
    ),
    "concrete": DatasetInfo(
        id="concrete",
        display_name="混凝土强度",
        task="regression",
        sample_count=1030,
        feature_count=8,
        has_target=True,
    ),
    "california_housing": DatasetInfo(
        id="california_housing",
        display_name="加州房价",
        task="regression",
        sample_count=20640,
        feature_count=8,
        has_target=True,
    ),
    "6_cardio": DatasetInfo(
        id="6_cardio",
        display_name="心电图异常检测",
        task="anomaly_detection",
        sample_count=1831,
        feature_count=21,
        has_target=True,
    ),
    "23_mammography": DatasetInfo(
        id="23_mammography",
        display_name="乳腺造影异常检测",
        task="anomaly_detection",
        sample_count=11183,
        feature_count=6,
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
        return _load_concrete(info)
    if dataset_id == "california_housing":
        return _load_california_housing(info)
    if dataset_id == "6_cardio":
        return _load_anomaly_txt(info, "6_cardio.txt", feature_count=21)
    if dataset_id == "23_mammography":
        return _load_anomaly_txt(info, "23_mammography.txt", feature_count=6)
    raise KeyError(dataset_id)


def _resource_text(filename: str, encoding: str = "utf-8") -> str:
    return files(_RESOURCE_PACKAGE).joinpath(filename).read_text(encoding=encoding)


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


def _load_concrete(info: DatasetInfo) -> LoadedDataset:
    """Concrete: UTF-8 BOM + header row, 8 features, numeric target in the last column."""

    features: list[tuple[float, ...]] = []
    targets: list[float] = []
    lines = _resource_text("concrete_data.csv", encoding="utf-8-sig").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        fields = line.split(",")
        if index == 0:
            continue  # 英文表头
        if len(fields) != 9:
            raise ValueError("Concrete rows must contain eight features and one target")
        features.append(tuple(float(value) for value in fields[:8]))
        targets.append(float(fields[8]))
    return LoadedDataset(info=info, features=tuple(features), targets=tuple(targets))


def _load_california_housing(info: DatasetInfo) -> LoadedDataset:
    """California housing: no header, comma-separated floats with CRLF line endings."""

    features: list[tuple[float, ...]] = []
    targets: list[float] = []
    for line in _resource_text("cal_housing.data").splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("\r").split(",")
        if len(fields) != 9:
            raise ValueError(
                "California housing rows must contain eight features and one target"
            )
        features.append(tuple(float(value) for value in fields[:8]))
        targets.append(float(fields[8]))
    return LoadedDataset(info=info, features=tuple(features), targets=tuple(targets))


def _load_anomaly_txt(
    info: DatasetInfo,
    filename: str,
    feature_count: int,
) -> LoadedDataset:
    """Anomaly txt: whitespace-separated features plus a 0/1 label (1 = anomaly)."""

    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    expected = feature_count + 1
    for line in _resource_text(filename).splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != expected:
            raise ValueError(
                f"{filename} rows must contain {feature_count} features and one label"
            )
        features.append(tuple(float(value) for value in fields[:feature_count]))
        targets.append(int(fields[feature_count]))
    return LoadedDataset(info=info, features=tuple(features), targets=tuple(targets))
