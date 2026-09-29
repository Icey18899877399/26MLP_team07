"""Shared helpers for experiment adapters.

Validators raise InvalidParameterError with Chinese messages because the
interaction layer passes exception text verbatim to HTTP clients (400).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from ..errors import InvalidParameterError
from ..types import JSONValue


def require_positive_integer(params: dict[str, JSONValue], name: str) -> None:
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidParameterError(f"{name} 必须是正整数")


def require_positive_number(params: dict[str, JSONValue], name: str) -> None:
    value = params[name]
    if not _is_finite_number(value) or float(value) <= 0.0:
        raise InvalidParameterError(f"{name} 必须是正的有限数值")


def require_non_negative_number(params: dict[str, JSONValue], name: str) -> None:
    value = params[name]
    if not _is_finite_number(value) or float(value) < 0.0:
        raise InvalidParameterError(f"{name} 必须是非负有限数值")


def require_bounded_number(
    params: dict[str, JSONValue],
    name: str,
    minimum: float,
    maximum: float,
) -> None:
    value = params[name]
    if (
        not _is_finite_number(value)
        or float(value) < minimum
        or float(value) > maximum
    ):
        raise InvalidParameterError(
            f"{name} 必须是 {minimum} 到 {maximum} 之间的有限数值"
        )


def require_in_range(
    params: dict[str, JSONValue],
    name: str,
    minimum: float,
    maximum: float,
    *,
    left_open: bool = True,
    right_open: bool = False,
) -> None:
    """半开/开区间校验（如 subsample ∈ (0,1]、contamination ∈ (0,0.5]）。"""

    value = params[name]
    if not _is_finite_number(value):
        raise InvalidParameterError(f"{name} 必须是有限数值")
    number = float(value)
    low_ok = number > minimum if left_open else number >= minimum
    high_ok = number < maximum if right_open else number <= maximum
    if not (low_ok and high_ok):
        bounds = ("(", "[")[not left_open] + f"{minimum}, {maximum}" + (")", "]")[not right_open]
        raise InvalidParameterError(f"{name} 必须在 {bounds} 范围内")


def require_boolean(params: dict[str, JSONValue], name: str) -> None:
    if not isinstance(params[name], bool):
        raise InvalidParameterError(f"{name} 必须是布尔值")


def require_optional_positive_integer_or_none(
    params: dict[str, JSONValue],
    name: str,
) -> None:
    value = params[name]
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidParameterError(f"{name} 必须是正整数或 null")


def is_finite_number(value: JSONValue) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _is_finite_number(value: JSONValue) -> bool:
    return is_finite_number(value)


def stratified_split(
    targets: tuple[int, ...] | list[int],
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    """按类别分层切分（分类用）。"""

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[target].append(index)

    generator = random.Random(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for target in sorted(grouped):
        indices = list(grouped[target])
        generator.shuffle(indices)
        test_count = max(1, min(len(indices) - 1, round(len(indices) * test_size)))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    generator.shuffle(train_indices)
    generator.shuffle(test_indices)
    return train_indices, test_indices


def plain_split(
    sample_count: int,
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    """整体随机切分（回归用，不按类别分层）。"""

    indices = list(range(sample_count))
    generator = random.Random(random_state)
    generator.shuffle(indices)
    test_count = max(1, min(sample_count - 1, round(sample_count * test_size)))
    return indices[test_count:], indices[:test_count]


def regression_metrics(
    targets: list[float],
    predictions: list[float],
) -> dict[str, float]:
    """mse/rmse/mae/r2，全部为有限数值（ss_tot 为 0 时 r2 取 0.0）。"""

    count = len(targets)
    if count == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}
    squared_errors = sum(
        (target - prediction) ** 2
        for target, prediction in zip(targets, predictions)
    )
    mse = squared_errors / count
    mae = sum(
        abs(target - prediction)
        for target, prediction in zip(targets, predictions)
    ) / count
    mean_target = sum(targets) / count
    ss_tot = sum((target - mean_target) ** 2 for target in targets)
    r2 = 1.0 - squared_errors / ss_tot if ss_tot else 0.0
    return {
        "mse": round(mse, 8),
        "rmse": round(math.sqrt(mse), 8),
        "mae": round(mae, 8),
        "r2": round(r2, 8),
    }


def merge_model_kwargs(
    effective_params: dict[str, JSONValue],
    pinned: dict[str, JSONValue],
) -> dict[str, JSONValue]:
    """dict(effective_params) + 适配器注入的固定参数（如 n_jobs/hidden_layer_sizes）。"""

    merged = dict(effective_params)
    merged.update(pinned)
    return merged
