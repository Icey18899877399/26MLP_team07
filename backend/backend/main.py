import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from backend.integration.ml_backend import MLBackend, PackageMLBackend
from backend.interaction.routes import router
from backend.interaction.original_routes import router as original_router
from backend.integration.original_jobs import OriginalJobManager
from backend.interaction.validation import chinese_validation_error_handler
from backend.settings import Settings


def _default_original_jobs_dir() -> Path:
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "visualization").is_dir():
        return checkout / ".original-runs"
    user_data = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
    return user_data / "26MLP_team07" / "original-runs"


def create_app(
    ml_backend: MLBackend | None = None,
    settings: Settings | None = None,
    original_jobs_dir: Path | None = None,
    original_job_manager: OriginalJobManager | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    logging.getLogger("backend").setLevel(resolved_settings.log_level)

    jobs_dir = Path(original_jobs_dir).resolve() if original_jobs_dir is not None else _default_original_jobs_dir()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if _app.state.original_jobs is None:
            with _app.state.original_jobs_lock:
                if _app.state.original_jobs is None:
                    _app.state.original_jobs = OriginalJobManager(jobs_dir)
        try:
            yield
        finally:
            if _app.state.original_jobs is not None:
                _app.state.original_jobs.close()

    app = FastAPI(
        title="机器学习理论与实践交互平台",
        version="0.1.0",
        description="独立 ml_core 包之上的轻量 HTTP 交互层。",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.ml_backend = (
        ml_backend
        if ml_backend is not None
        else PackageMLBackend(package_name=resolved_settings.ml_package)
    )
    app.state.original_jobs = original_job_manager
    app.state.original_jobs_dir = jobs_dir
    app.state.original_jobs_lock = threading.Lock()
    app.add_exception_handler(RequestValidationError, chinese_validation_error_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    app.include_router(original_router)
    return app


app = create_app()
