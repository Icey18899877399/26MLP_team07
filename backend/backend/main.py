import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.integration.ml_backend import MLBackend, PackageMLBackend
from backend.interaction.routes import router
from backend.settings import Settings


def create_app(
    ml_backend: MLBackend | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    logging.getLogger("backend").setLevel(resolved_settings.log_level)

    app = FastAPI(
        title="机器学习理论与实践交互平台",
        version="0.1.0",
        description="Thin HTTP interaction layer over the independent ml_core package.",
    )
    app.state.settings = resolved_settings
    app.state.ml_backend = (
        ml_backend
        if ml_backend is not None
        else PackageMLBackend(package_name=resolved_settings.ml_package)
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
