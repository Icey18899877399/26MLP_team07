"""Synchronous real training on the original experiment data."""
from fastapi import APIRouter, HTTPException

import ml_core
from backend.contracts import ExperimentConfig, ExperimentResult

router = APIRouter(prefix="/api")


@router.get("/interactive-experiments")
def interactive_experiments() -> list[dict]:
    return ml_core.list_interactive_experiments()


@router.post("/interactive-experiments", response_model=ExperimentResult)
def run_interactive_experiment(config: ExperimentConfig) -> dict:
    try:
        # Omitted seed uses the selected original experiment's seed (13/17/42).
        values = config.model_dump()
        if "random_state" not in config.model_fields_set:
            item = next((i for i in ml_core.list_interactive_experiments() if i["model"] == config.model), None)
            if item is not None:
                values["random_state"] = item["random_state"]
        return ml_core.run_interactive_experiment(ml_core.ExperimentConfig(**values)).to_dict()
    except ml_core.ExperimentExecutionError as exc:
        raise HTTPException(502, detail={"code": "ml_execution_failed", "message": str(exc)}) from exc
    except ml_core.MLCoreError as exc:
        raise HTTPException(400, detail={"code": "ml_request_rejected", "message": str(exc)}) from exc
