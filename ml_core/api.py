"""Implementation of the stable ml_core discovery and execution API."""

from __future__ import annotations

from collections.abc import Mapping

from .datasets import dataset_catalog, load_dataset
from .errors import (
    ExperimentExecutionError,
    IncompatibleDatasetError,
    InvalidConfigError,
    InvalidParameterError,
    MLCoreError,
    UnknownDatasetError,
    UnknownModelError,
)
from .registry import get_model_spec, model_catalog
from .types import DatasetInfo, ExperimentConfig, ExperimentResult, ModelInfo


def list_models() -> tuple[ModelInfo, ...]:
    """List runnable model contracts in stable identifier order."""

    return model_catalog()


def list_datasets() -> tuple[DatasetInfo, ...]:
    """List packaged dataset contracts in stable identifier order."""

    return dataset_catalog()


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Validate and run one experiment through its registered adapter."""

    _validate_common_config(config)
    spec = get_model_spec(config.model)
    if spec is None:
        raise UnknownModelError(f"未知模型: {config.model}")

    datasets = {dataset.id: dataset for dataset in dataset_catalog()}
    if config.dataset not in datasets:
        raise UnknownDatasetError(f"未知数据集: {config.dataset}")
    if config.dataset not in spec.info.compatible_datasets:
        raise IncompatibleDatasetError(
            f"模型 {config.model} 不支持数据集 {config.dataset}"
        )

    unknown_params = sorted(set(config.params) - set(spec.info.default_params))
    if unknown_params:
        raise InvalidParameterError(
            f"模型 {config.model} 存在未知参数: {', '.join(unknown_params)}"
        )
    if spec.info.task == "clustering" and config.test_size is not None:
        raise InvalidConfigError("聚类实验不使用 test_size")

    effective_params = dict(spec.info.default_params)
    effective_params.update(config.params)
    if spec.runner is None:
        raise ExperimentExecutionError(
            f"模型 {config.model} 的运行器在此构建中不可用"
        )

    try:
        dataset = load_dataset(config.dataset)
        return spec.runner(config, dataset, effective_params)
    except MLCoreError:
        raise
    except Exception as error:
        raise ExperimentExecutionError(
            f"实验 {config.model} 执行失败: {error}"
        ) from error


def _validate_common_config(config: ExperimentConfig) -> None:
    if not isinstance(config, ExperimentConfig):
        raise InvalidConfigError("config 必须是 ExperimentConfig 实例")
    if not isinstance(config.model, str) or not config.model:
        raise InvalidConfigError("model 必须是非空字符串")
    if not isinstance(config.dataset, str) or not config.dataset:
        raise InvalidConfigError("dataset 必须是非空字符串")
    if not isinstance(config.params, Mapping):
        raise InvalidConfigError("params 必须是映射")
    if any(not isinstance(name, str) or not name for name in config.params):
        raise InvalidParameterError("参数名必须是非空字符串")
    if isinstance(config.random_state, bool) or not isinstance(config.random_state, int):
        raise InvalidConfigError("random_state 必须是整数")
    if config.test_size is not None:
        if (
            isinstance(config.test_size, bool)
            or not isinstance(config.test_size, (int, float))
            or not 0.0 < config.test_size < 1.0
        ):
            raise InvalidConfigError("test_size 必须在 0 与 1 之间")
