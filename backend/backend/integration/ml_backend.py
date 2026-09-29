from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Protocol, Sequence, cast

from pydantic import BaseModel, ValidationError

from backend.contracts import (
    DatasetSpec,
    ExperimentConfig,
    ExperimentResult,
    MLBackendStatus,
    ModelSpec,
)


logger = logging.getLogger(__name__)


class MLBackendError(RuntimeError):
    """Base error understood by the interaction layer."""


class MLBackendUnavailable(MLBackendError):
    """The external package is missing or does not satisfy the public contract."""


class MLRequestError(MLBackendError):
    """The ML package rejected a valid HTTP request for domain reasons."""


class MLExecutionError(MLBackendError):
    """The ML package failed while executing an experiment."""


class MLBackend(Protocol):
    def status(self) -> MLBackendStatus: ...

    def list_models(self) -> list[ModelSpec]: ...

    def list_datasets(self) -> list[DatasetSpec]: ...

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult: ...


class PackageMLBackend:
    """Call only the stable public API exported by the external ``ml_core`` package."""

    _required_exports = (
        "list_models",
        "list_datasets",
        "run_experiment",
        "ModelSpec",
        "DatasetSpec",
        "ExperimentConfig",
        "ExperimentResult",
        "MLCoreError",
        "InvalidExperimentError",
        "ExperimentExecutionError",
    )

    def __init__(
        self,
        package_name: str = "ml_core",
        importer: Callable[[str], ModuleType] = import_module,
    ) -> None:
        self._package_name = package_name
        self._module: ModuleType | None = None
        self._unavailable_reason: str | None = None
        self._invalid_experiment_error: type[Exception] | None = None
        self._native = False

        try:
            module = importer(package_name)
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "ML package import failed",
                extra={
                    "ml_package": package_name,
                    "ml_exception_type": type(exc).__name__,
                },
            )
            self._unavailable_reason = f"无法导入 {package_name}；请安装并检查其依赖"
            return

        self._native = hasattr(module, "ModelInfo") and hasattr(module, "DatasetInfo")
        if self._native:
            required = ("list_models", "list_datasets", "run_experiment", "ExperimentConfig",
                        "ExperimentResult", "MLCoreError", "ExperimentExecutionError")
            problems = [name for name in required if not callable(getattr(module, name, None))]
        else:
            problems = self._contract_problems(module)
        if problems:
            self._unavailable_reason = (
                f"{package_name} 公共合同不完整: {', '.join(sorted(problems))}"
            )
            return
        self._module = module
        self._invalid_experiment_error = cast(
            type[Exception], getattr(module, "MLCoreError" if self._native else "InvalidExperimentError")
        )

    def status(self) -> MLBackendStatus:
        if self._module is None:
            return MLBackendStatus(
                available=False,
                package=self._package_name,
                detail=self._unavailable_reason or "ML package 不可用",
            )
        return MLBackendStatus(
            available=True,
            package=self._package_name,
            detail="ML public package API 已连接",
        )

    def list_models(self) -> list[ModelSpec]:
        values = self._call("list_models")
        if self._native:
            try:
                return [ModelSpec(
                    id=item.id.rsplit(".", 1)[0], name=item.display_name,
                    task_type=item.task, variants=[item.id.rsplit(".", 1)[1]],
                    parameters=item.default_params,
                    compatible_datasets=list(item.compatible_datasets),
                    parameter_descriptions=item.parameter_descriptions,
                ) for item in values]
            except (AttributeError, TypeError, ValueError) as exc:
                raise MLExecutionError("Invalid model catalog") from exc
        return self._validate_sequence(values, ModelSpec, "model")

    def list_datasets(self) -> list[DatasetSpec]:
        values = self._call("list_datasets")
        if self._native:
            try:
                return [DatasetSpec(id=item.id, name=item.display_name, task_type=item.task,
                                    sample_count=item.sample_count, feature_count=item.feature_count)
                        for item in values]
            except (AttributeError, TypeError, ValueError) as exc:
                raise MLExecutionError("Invalid dataset catalog") from exc
        return self._validate_sequence(values, DatasetSpec, "dataset")

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        module = self._require_module()
        config_type = cast(Callable[..., Any], getattr(module, "ExperimentConfig"))
        try:
            values = config.model_dump(mode="python")
            if self._native:
                values.pop("variant")
                values["model"] = f"{config.model}.{config.variant}"
            package_config = config_type(**values)
        except Exception as exc:
            if isinstance(exc, (TypeError, ValueError, ValidationError)) or (
                self._invalid_experiment_error
                and isinstance(exc, self._invalid_experiment_error)
            ):
                raise MLRequestError(str(exc)) from exc
            logger.exception(
                "ML experiment configuration construction failed",
                extra={
                    "model": config.model,
                    "variant": config.variant,
                    "dataset": config.dataset,
                    "ml_exception_type": type(exc).__name__,
                },
            )
            raise MLExecutionError("ML experiment configuration failed") from exc

        try:
            value = cast(Callable[[Any], Any], getattr(module, "run_experiment"))(
                package_config
            )
        except Exception as exc:
            if self._native and isinstance(exc, module.ExperimentExecutionError):
                logger.exception("ML experiment execution failed")
                raise MLExecutionError("ML experiment execution failed") from exc
            if self._invalid_experiment_error and isinstance(
                exc, self._invalid_experiment_error
            ):
                raise MLRequestError(str(exc)) from exc
            logger.exception(
                "ML experiment execution failed",
                extra={
                    "model": config.model,
                    "variant": config.variant,
                    "dataset": config.dataset,
                    "ml_exception_type": type(exc).__name__,
                },
            )
            raise MLExecutionError("ML experiment execution failed") from exc

        try:
            if self._native:
                value = {
                    "model": config.model, "variant": config.variant, "dataset": value.dataset,
                    "metrics": value.metrics,
                    "diagnostics": {**value.metadata, "run_id": value.run_id,
                                    "effective_params": value.effective_params},
                }
            return ExperimentResult.model_validate(self._to_mapping(value))
        except (TypeError, ValueError, ValidationError) as exc:
            logger.exception(
                "ML package returned an invalid experiment result",
                extra={
                    "model": config.model,
                    "variant": config.variant,
                    "dataset": config.dataset,
                },
            )
            raise MLExecutionError(
                "ML package returned an invalid experiment result"
            ) from exc

    def _call(self, name: str) -> Any:
        module = self._require_module()
        try:
            return cast(Callable[[], Any], getattr(module, name))()
        except Exception as exc:
            logger.exception(
                "ML package call failed",
                extra={"ml_operation": name, "ml_exception_type": type(exc).__name__},
            )
            raise MLExecutionError(f"ML package operation failed: {name}") from exc

    def _require_module(self) -> ModuleType:
        if self._module is None:
            raise MLBackendUnavailable(
                self._unavailable_reason or f"{self._package_name} 不可用"
            )
        return self._module

    @classmethod
    def _contract_problems(cls, module: ModuleType) -> list[str]:
        problems = [name for name in cls._required_exports if not hasattr(module, name)]
        if problems:
            return problems

        for name in (
            "list_models",
            "list_datasets",
            "run_experiment",
            "ExperimentConfig",
        ):
            if not callable(getattr(module, name)):
                problems.append(f"{name} must be callable")

        exception_names = (
            "MLCoreError",
            "InvalidExperimentError",
            "ExperimentExecutionError",
        )
        exception_types: dict[str, type[Exception]] = {}
        for name in exception_names:
            value = getattr(module, name)
            if not isinstance(value, type) or not issubclass(value, Exception):
                problems.append(f"{name} must be an exception type")
            else:
                exception_types[name] = value

        base = exception_types.get("MLCoreError")
        if base:
            for name in ("InvalidExperimentError", "ExperimentExecutionError"):
                value = exception_types.get(name)
                if value and not issubclass(value, base):
                    problems.append(f"{name} must inherit MLCoreError")
        return problems

    @classmethod
    def _validate_sequence(
        cls,
        values: Any,
        schema: type[BaseModel],
        item_name: str,
    ) -> list[Any]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise MLExecutionError(f"ML package 返回的 {item_name} 列表格式无效")
        try:
            return [schema.model_validate(cls._to_mapping(item)) for item in values]
        except (TypeError, ValueError, ValidationError) as exc:
            raise MLExecutionError(
                f"ML package 返回的 {item_name} 数据不符合公共合同: {exc}"
            ) from exc

    @staticmethod
    def _to_mapping(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="python")
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        return value
