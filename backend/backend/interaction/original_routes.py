"""HTTP contract for the original archived figures and full experiment reruns."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from ml_core import (
    list_original_datasets,
    list_original_experiments,
    resolve_original_asset,
    resolve_original_run_asset,
)
from backend.integration.original_jobs import OriginalJobManager


router = APIRouter(prefix="/api")


class OriginalRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str


def _jobs(request: Request) -> OriginalJobManager:
    app = request.app
    if app.state.original_jobs is None:
        with app.state.original_jobs_lock:
            if app.state.original_jobs is None:
                app.state.original_jobs = OriginalJobManager(app.state.original_jobs_dir)
    return app.state.original_jobs


@router.get("/original-experiments")
def original_experiments() -> list[dict]:
    return list_original_experiments()


@router.get("/original-datasets")
def original_datasets() -> list[dict]:
    return list_original_datasets()


@router.get("/original-assets/{experiment_id}/{filename:path}")
def original_asset(experiment_id: str, filename: str, download: bool = False) -> FileResponse:
    try:
        path = resolve_original_asset(experiment_id, filename)
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Original asset not found") from None
    return FileResponse(path, filename=path.name) if download else FileResponse(path)


@router.post("/original-runs", status_code=status.HTTP_202_ACCEPTED)
def start_original_run(body: OriginalRunRequest, request: Request) -> dict:
    try:
        return _jobs(request).submit(body.experiment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Original experiment not found") from None
    except OverflowError:
        raise HTTPException(status_code=409, detail="Original run queue is full") from None
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Original run service unavailable") from None


@router.get("/original-runs")
def original_runs(request: Request) -> list[dict]:
    return _jobs(request).list()


@router.get("/original-runs/{run_id}")
def original_run(run_id: str, request: Request) -> dict:
    try:
        return _jobs(request).get(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Original run not found") from None


@router.get("/original-runs/{run_id}/assets/{filename:path}")
def original_run_asset(run_id: str, filename: str, request: Request, download: bool = False) -> FileResponse:
    try:
        job = _jobs(request).get(run_id)
        if job["status"] != "completed":
            raise ValueError("Run not completed")
        path = resolve_original_run_asset(
            _jobs(request).state_dir / run_id / "output", filename
        )
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Run asset not found") from None
    return FileResponse(path, filename=path.name) if download else FileResponse(path)
