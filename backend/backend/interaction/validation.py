"""FastAPI 422 校验错误的中文翻译处理器。

返回体保持标准 422 形状（detail 数组含 loc/msg/type），只把 msg 翻译成中文；
loc 与 type 原样保留，保证前端错误解析与现有测试断言不受影响。
"""

from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _field(error: dict) -> str:
    loc = error.get("loc") or []
    return str(loc[-1]) if loc else "body"


_MESSAGE_TEMPLATES = {
    "json_invalid": "请求体不是合法的 JSON",
    "missing": "缺少必填字段 {field}",
    "extra_forbidden": "包含不允许的字段 {field}",
    "model_attributes_type": "请求体必须是 JSON 对象",
    "model_type": "请求体必须是 JSON 对象",
    "dict_type": "字段 {field} 必须是对象",
    "json_type": "字段 {field} 必须是合法的 JSON 值",
    "int_parsing": "字段 {field} 必须是整数",
    "int_type": "字段 {field} 必须是整数",
    "float_parsing": "字段 {field} 必须是数字",
    "float_type": "字段 {field} 必须是数字",
    "bool_parsing": "字段 {field} 必须是布尔值",
    "bool_type": "字段 {field} 必须是布尔值",
    "string_type": "字段 {field} 必须是字符串",
    "str_type": "字段 {field} 必须是字符串",
    "literal_error": "字段 {field} 的取值不在允许范围内",
    "greater_than": "字段 {field} 必须大于 {gt}",
    "greater_than_equal": "字段 {field} 必须大于等于 {ge}",
    "less_than": "字段 {field} 必须小于 {lt}",
    "less_than_equal": "字段 {field} 必须小于等于 {le}",
    "string_too_short": "字段 {field} 长度不能少于 {min_length}",
    "string_too_long": "字段 {field} 长度不能超过 {max_length}",
    "too_short": "字段 {field} 元素个数不能少于 {min_length}",
    "too_long": "字段 {field} 元素个数不能超过 {max_length}",
}


def _translate(error: dict) -> dict:
    translated = dict(error)
    error_type = str(error.get("type") or "")
    template = _MESSAGE_TEMPLATES.get(error_type)
    if template is None:
        return translated
    context = dict(error.get("ctx") or {})
    try:
        translated["msg"] = template.format(field=_field(error), **context)
    except KeyError:
        translated["msg"] = template
    return translated


def chinese_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    # The rejected input may contain NaN, bytes or exception objects. Never
    # echo it into JSONResponse: validation must remain a 422, not become 500.
    detail = [
        {key: translated[key] for key in ("loc", "msg", "type")}
        for error in exc.errors()
        for translated in [_translate(dict(error))]
    ]
    return JSONResponse(status_code=422, content={"detail": detail})
