from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.contracts import MLBackendStatus


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictSchema):
    status: Literal["ok"] = "ok"
    ml_backend: MLBackendStatus


class ErrorDetail(StrictSchema):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ErrorResponse(StrictSchema):
    detail: ErrorDetail
