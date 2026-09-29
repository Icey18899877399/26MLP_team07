import pytest
from pydantic import ValidationError

from backend.settings import Settings


def test_environment_overrides_cors_package_and_log_level(monkeypatch) -> None:
    monkeypatch.setenv(
        "MLP_CORS_ORIGINS",
        "https://ui.example.com/, http://localhost:4173,https://ui.example.com",
    )
    monkeypatch.setenv("MLP_ML_PACKAGE", "team07.ml_core")
    monkeypatch.setenv("MLP_LOG_LEVEL", "WARNING")

    settings = Settings(_env_file=None)

    assert settings.cors_origins == [
        "https://ui.example.com",
        "http://localhost:4173",
    ]
    assert settings.ml_package == "team07.ml_core"
    assert settings.log_level == "WARNING"


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "file:///tmp/ui",
        "http://localhost:not-a-port",
        "https://user:pass@example.com",
        "https://example.com/app",
    ],
)
def test_invalid_cors_origin_is_rejected(origin: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_origins=[origin])
