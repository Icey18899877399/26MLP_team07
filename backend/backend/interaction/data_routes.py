"""Local dataset library HTTP boundary."""
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.contracts import ExperimentResult
from ml_core.errors import MLCoreError, ExperimentExecutionError
from ml_core.interactive import list_interactive_experiments
from ml_core.tabular import compatible_models, run_tabular
from ml_core.types import ExperimentConfig

router = APIRouter(prefix="/api/data-library")


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    filename: str
    content: str


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_id: str
    feature_columns: list[str]
    target_column: str | None = None
    task: str


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_id: str
    feature_columns: list[str]
    target_column: str | None = None
    task: str | None = None
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    random_state: int | None = None
    test_size: float | None = None


def _table(request, dataset_id):
    try:
        return request.app.state.data_library.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"code": "unknown_dataset", "message": "数据集不存在"}) from exc


@router.get("")
def list_datasets(request: Request):
    return request.app.state.data_library.list()


@router.post("/uploads")
def upload_table(body: UploadRequest, request: Request):
    try:
        return request.app.state.data_library.upload(body.filename, body.content).detail()
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(400, detail={"code": "invalid_table", "message": str(exc)}) from exc


@router.post("/compatibility")
def compatibility(body: Selection, request: Request):
    return compatible_models(_table(request, body.dataset_id), body.feature_columns, body.target_column, body.task)


@router.post("/experiments", response_model=ExperimentResult)
def train(body: TrainingRequest, request: Request):
    table = _table(request, body.dataset_id)
    try:
        item = next((item for item in list_interactive_experiments() if item["model"] == body.model), None)
        seed = body.random_state if body.random_state is not None else item["random_state"] if item else 42
        config = ExperimentConfig(body.model, body.dataset_id, body.params, body.test_size, seed)
        return run_tabular(config, table, body.feature_columns, body.target_column, body.task).to_dict()
    except ExperimentExecutionError as exc:
        raise HTTPException(502, detail={"code": "ml_execution_failed", "message": str(exc)}) from exc
    except MLCoreError as exc:
        raise HTTPException(400, detail={"code": "ml_request_rejected", "message": str(exc)}) from exc


@router.get("/{dataset_id}")
def dataset_detail(dataset_id: str, request: Request):
    return _table(request, dataset_id).detail()
