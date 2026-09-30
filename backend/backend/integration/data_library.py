"""Allowlisted originals and bounded, persistent user tables (never executable)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import re
from uuid import uuid4

from ml_core.original_catalog import list_original_datasets, original_root

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 20000
MAX_COLUMNS = 128
_UPLOAD_ID = re.compile(r"upload_[a-f0-9]{32}\Z")


def _value(value):
    text = str(value).strip()
    if text in ("", "?"):
        return None
    try:
        number = float(text)
        # Preserve invalid numeric text for a visible selection error and valid JSON.
        return number if math.isfinite(number) else text
    except ValueError:
        return text


@dataclass(frozen=True)
class Table:
    id: str
    name: str
    source: str
    headers: tuple[str, ...]
    rows: tuple[tuple, ...]
    default_target: str | None = None
    paths: tuple[str, ...] = ()
    # Uploaded categorical tokens must not pass through a floating-point cast.
    # Keep a second view for numeric fitting and this lossless view for identity.
    raw_rows: tuple[tuple[str | None, ...], ...] = ()

    def summary(self):
        return {"id": self.id, "name": self.name, "source": self.source,
                "row_count": len(self.rows), "column_count": len(self.headers)}

    def detail(self):
        columns = []
        for index, name in enumerate(self.headers):
            values = [row[index] for row in self.rows]
            present = [value for value in values if value is not None]
            identities = [row[index] for row in (self.raw_rows or self.rows) if row[index] is not None]
            columns.append({"name": name, "dtype": "number" if present and all(isinstance(v, (int, float)) for v in present) else "string",
                            "missing_count": len(values) - len(present), "unique_count": len(set(identities))})
            if self.id == "wdbc" and name == "id":
                columns[-1]["suggested_role"] = "identifier"
        return {**self.summary(), "columns": columns,
                "preview": [dict(zip(self.headers, row)) for row in (self.raw_rows or self.rows)[:20]],
                "default_target": self.default_target, "paths": list(self.paths)}


def parse_upload(filename: str, content: str):
    if not filename or filename.strip() != filename or any(char in filename for char in '/\\:\x00') or any(ord(char) < 32 for char in filename):
        raise ValueError("文件名必须是无路径的 CSV/TSV 名称")
    suffix = Path(filename).suffix.lower()
    if suffix not in (".csv", ".tsv"):
        raise ValueError("仅支持带表头的 UTF-8 CSV/TSV 文件")
    if len(content.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise ValueError("文件内容超过 5 MiB 上限")
    if "\x00" in content:
        raise ValueError("表格不能包含空字节")
    reader = csv.reader(io.StringIO(content.lstrip("\ufeff")), delimiter="\t" if suffix == ".tsv" else ",", strict=True)
    try:
        headers = tuple(value.strip() for value in next(reader, []))
        if not headers or not all(headers) or len(set(headers)) != len(headers):
            raise ValueError("表头不能为空或重复")
        if len(headers) > MAX_COLUMNS:
            raise ValueError("表格超过 128 列上限")
        rows, raw_rows = [], []
        for line, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                raise ValueError(f"第 {line} 行列数与表头不一致；未删除任何行")
            rows.append(tuple(_value(value) for value in row))
            raw_rows.append(tuple(None if value.strip() in ("", "?") else value for value in row))
            if len(rows) > MAX_ROWS:
                raise ValueError("表格超过 20000 行上限")
        if not rows:
            raise ValueError("表格必须包含至少一行数据")
    except csv.Error as exc:
        raise ValueError(f"CSV/TSV 格式无效: {exc}") from exc
    return headers, tuple(rows), tuple(raw_rows)


@lru_cache(maxsize=7)
def _original(dataset_id: str) -> Table:
    item = next((item for item in list_original_datasets() if item["id"] == dataset_id), None)
    if item is None:
        raise KeyError(dataset_id)
    paths = [original_root() / path for path in item["paths"]]
    if dataset_id in ("6_cardio", "23_mammography"):
        import numpy as np
        with np.load(paths[0], allow_pickle=False) as data:
            x, y = data["X"], data["y"].reshape(-1)
            headers = tuple(f"feature_{i + 1}" for i in range(x.shape[1])) + ("label",)
            rows = tuple(tuple(float(v) for v in row) + (int(target),) for row, target in zip(x, y))
        target = "label"
    elif dataset_id == "wdbc":
        bases = ("radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension")
        headers = ("id", "diagnosis") + tuple(f"{name}_{suffix}" for suffix in ("mean", "se", "worst") for name in bases)
        rows = tuple(tuple(_value(v) for v in row) for row in csv.reader(paths[0].read_text(encoding="utf-8-sig").splitlines()) if row)
        target = "diagnosis"
    elif dataset_id == "concrete":
        parsed = list(csv.reader(paths[0].read_text(encoding="utf-8-sig").splitlines()))
        headers, rows = tuple(parsed[0]), tuple(tuple(_value(v) for v in row) for row in parsed[1:] if row)
        target = headers[-1]
    elif dataset_id == "seeds":
        headers = ("area", "perimeter", "compactness", "kernel_length", "kernel_width", "asymmetry", "groove_length", "class")
        rows = tuple(tuple(_value(v) for v in line.split()) for line in paths[0].read_text().splitlines() if line.strip())
        target = "class"
    elif dataset_id == "california_housing":
        headers = ("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "median_house_value")
        rows = tuple(tuple(_value(v) for v in row) for row in csv.reader(paths[0].read_text().splitlines()) if row)
        target = "median_house_value"
    else:
        headers = ("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")
        parsed = [row for path in paths for row in csv.reader(path.read_text(encoding="utf-8-sig").splitlines()) if row and not row[0].startswith("|")]
        rows = tuple(tuple(_value(v) for v in row[:-1]) + (row[-1].strip().rstrip("."),) for row in parsed)
        target = "income"
    return Table(dataset_id, item["name"], "original", headers, rows, target, tuple(item["paths"]))


class DataLibrary:
    def __init__(self, upload_dir: Path):
        self.upload_dir = Path(upload_dir).resolve()

    def get(self, dataset_id: str) -> Table:
        if not _UPLOAD_ID.fullmatch(dataset_id):
            return _original(dataset_id)
        path = self.upload_dir / f"{dataset_id}.json"
        if not path.is_file() or not path.resolve().is_relative_to(self.upload_dir):
            raise KeyError(dataset_id)
        saved = json.loads(path.read_text(encoding="utf-8"))
        headers, rows, raw_rows = parse_upload(saved["filename"], saved["content"])
        return Table(dataset_id, saved["filename"], "upload", headers, rows, raw_rows=raw_rows)

    def list(self):
        items = [_original(item["id"]).summary() for item in list_original_datasets()]
        for path in sorted(self.upload_dir.glob("upload_*.json")):
            if _UPLOAD_ID.fullmatch(path.stem):
                items.append(self.get(path.stem).summary())
        return items

    def upload(self, filename: str, content: str):
        headers, rows, raw_rows = parse_upload(filename, content)
        dataset_id = "upload_" + uuid4().hex
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        final = self.upload_dir / f"{dataset_id}.json"
        temporary = self.upload_dir / f"{dataset_id}.tmp"
        try:
            temporary.write_text(json.dumps({"filename": filename, "content": content}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(final)
        finally:
            temporary.unlink(missing_ok=True)
        return Table(dataset_id, filename, "upload", headers, rows, raw_rows=raw_rows)
