# HTTP API 合同

> 上游合同存档。整合版允许 test_size=null（聚类必须为 null），并增加模型兼容数据集、参数说明与数据集大小字段；以运行服务的 /docs 和 [整合说明](../../docs/FULL_STACK.md) 为准。

## 边界与发现

前端无论采用本仓库目录还是独立 Git submodule，都只通过 HTTP/JSON 调用本服务，不读取 `backend/` 或 `external/ml-core/`。

- 本地基础地址：`http://127.0.0.1:8000`
- OpenAPI JSON：`GET /openapi.json`
- 交互式文档：`GET /docs`
- Content-Type：`application/json`

当前路径保持不带版本前缀的 `/api/*`。发生不兼容变更前，应先更新本文档和 OpenAPI，并通知前端负责人。

## 服务状态

### `GET /api/health`

即使 ML package 尚未接入，本端点也返回 HTTP 200，用于区分 interaction 服务与 ML backend 的状态。

```json
{
  "status": "ok",
  "ml_backend": {
    "available": false,
    "package": "ml_core",
    "detail": "无法导入 ml_core；请安装并检查其依赖"
  }
}
```

前端只能在 `ml_backend.available` 为 `true` 时开放真实实验操作，不得以 mock 结果替代。

## 模型与数据集

### `GET /api/models`

```json
[
  {
    "id": "logistic_regression",
    "name": "Logistic Regression",
    "task_type": "classification",
    "variants": ["base", "optimized"],
    "parameters": {}
  }
]
```

### `GET /api/datasets`

```json
[
  {
    "id": "wdbc",
    "name": "Wisconsin Diagnostic Breast Cancer",
    "task_type": "classification"
  }
]
```

两个端点在 ML package 缺失或公共合同不完整时返回 503；ML package 调用或返回校验失败时返回 502。

## 实验

### `POST /api/experiments`

请求：

```json
{
  "model": "logistic_regression",
  "variant": "optimized",
  "dataset": "wdbc",
  "params": {"learning_rate": 0.1},
  "test_size": 0.2,
  "random_state": 42
}
```

响应：

```json
{
  "model": "logistic_regression",
  "variant": "optimized",
  "dataset": "wdbc",
  "metrics": {"accuracy": 0.95},
  "diagnostics": {}
}
```

- `variant` 当前只允许 `base` 或 `optimized`。
- `test_size` 必须大于 0 且小于 1。
- 未声明字段会被拒绝。
- 模型特定参数由 ML package 负责验证。
- metrics 必须是有限数值，diagnostics 必须是 JSON-safe 数据。

## 错误

除 FastAPI 默认的 422 validation response 外，业务错误统一为：

```json
{
  "detail": {
    "code": "ml_backend_unavailable",
    "message": "human-readable message"
  }
}
```

| HTTP | code | 含义 |
| --- | --- | --- |
| 400 | `ml_request_rejected` | ML package 拒绝模型、数据集或参数组合 |
| 502 | `ml_execution_failed` | ML 执行失败或返回值违反公共合同 |
| 503 | `ml_backend_unavailable` | `ml_core` 缺失或公共导出不完整 |
| 422 | FastAPI validation detail | HTTP 请求结构不合法 |

502 只返回通用消息；内部异常、堆栈和本地文件路径不会暴露给前端。

## CORS

默认允许：

- `http://localhost:5173`
- `http://127.0.0.1:5173`

后端仅允许白名单 origin 使用 `GET`、`POST`、`OPTIONS` 和 `Content-Type` 请求头，且不启用 credentials。使用其他开发端口或部署域名时，通过逗号分隔的 `MLP_CORS_ORIGINS` 配置完整 origin，例如：

```dotenv
MLP_CORS_ORIGINS=http://localhost:4173,https://ml-ui.example.com
```

不支持通配符、URL path、query、fragment 或嵌入凭据。

## 源码交付模式

- 同仓库模式：前端源码直接位于根目录 `frontend/`。
- Submodule 模式：`git submodule add <frontend-repo-url> frontend`，父仓库固定经过联调的前端 SHA。

两种方式不能改变 HTTP wire contract。若采用 submodule，前端仓库不得反向包含本仓库。
