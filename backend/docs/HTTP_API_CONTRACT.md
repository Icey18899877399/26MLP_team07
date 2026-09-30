# HTTP API 合同

## 基本约定

- 本地地址：`http://127.0.0.1:8000`
- OpenAPI：`GET /openapi.json`
- 交互文档：`GET /docs`
- Content-Type：`application/json`

前端无论是普通目录还是 submodule，都只依赖本合同。当前路径不带版本前缀；不兼容变更必须先同步本文档、OpenAPI 和前端负责人。

## `GET /api/health`

interaction 服务可用时始终返回 200。已正确安装 ML 包的示例：

```json
{
  "status": "ok",
  "ml_backend": {
    "available": true,
    "package": "ml_core",
    "detail": "ML public package API 已连接"
  }
}
```

`available=false` 时前端不得开放真实实验操作。

## `GET /api/models`

```json
[
  {
    "id": "logistic_regression.optimized",
    "display_name": "Optimized Logistic Regression",
    "task": "classification",
    "compatible_datasets": ["wdbc"],
    "default_params": {
      "learning_rate": 0.1,
      "max_iter": 1000,
      "threshold": 0.5,
      "l2": 0.0,
      "tol": 1e-8,
      "standardize": true,
      "class_weight": null
    },
    "parameter_descriptions": {
      "max_iter": "Positive maximum training iterations."
    }
  }
]
```

`parameter_descriptions` 可用于表单帮助文本，但 ML 包仍是参数验证的最终权威。

## `GET /api/datasets`

```json
[
  {
    "id": "wdbc",
    "display_name": "Wisconsin Diagnostic Breast Cancer",
    "task": "classification",
    "sample_count": 569,
    "feature_count": 30,
    "has_target": true
  }
]
```

两个发现端点在包缺失/合同不完整时返回 503，在调用或返回校验失败时返回 502。

## `POST /api/experiments`

请求：

```json
{
  "model": "logistic_regression.optimized",
  "dataset": "wdbc",
  "params": {"max_iter": 500},
  "test_size": 0.2,
  "random_state": 42
}
```

响应形状：

```json
{
  "run_id": "8a40a70e-8a9f-4ccf-81be-0fe34f165fc7",
  "model": "logistic_regression.optimized",
  "dataset": "wdbc",
  "task": "classification",
  "effective_params": {"max_iter": 500},
  "metrics": {"accuracy": 0.95},
  "artifacts": [],
  "metadata": {"sample_count": 569}
}
```

- `test_size` 可省略或为 `null`；提供时必须在 0 与 1 之间。聚类实验必须省略它，异常检测（`anomaly_detection`）忽略该值。
- 未声明字段返回 422。
- 模型/数据集 ID 与参数来自发现端点，不存在独立 `variant` 字段。
- metrics 必须是有限数值；其他返回字段必须是 JSON-safe 数据。
- 本整合版扩展 `metadata.visualizations`：数组元素包含 `id`、`title`、`description`、`option`。`option` 为真实数据生成的 ECharts JSON 配置，前端可绘图或放大；不包含函数、文件系统路径或模拟训练过程。
- `metadata.evaluation_protocol`（如存在）说明评价口径。异常检测当前为数据内评价，不能将其指标解释为独立测试集成绩。
- artifact 只包含 `name`、`media_type`、`uri`，URI 的发布与访问策略后续另行约定。

## 错误

业务错误统一为：

```json
{"detail": {"code": "ml_request_rejected", "message": "human-readable message"}}
```

| HTTP | code | 含义 |
| --- | --- | --- |
| 400 | `ml_request_rejected` | 未知模型/数据集、不兼容组合或参数被 ML 包拒绝 |
| 502 | `ml_execution_failed` | 执行失败或 ML 返回值违反合同 |
| 503 | `ml_backend_unavailable` | `ml_core` 缺失或公共导出不完整 |
| 422 | FastAPI validation detail | HTTP 请求结构不合法 |

502 只返回通用消息，不暴露内部异常、堆栈或本地路径。

## CORS 与前端交付

默认允许 `http://localhost:5173`、`http://127.0.0.1:5173`，方法为 `GET/POST/OPTIONS`，请求头为 `Content-Type`，不启用 credentials。其他 origin 通过逗号分隔的 `MLP_CORS_ORIGINS` 设置。

- 同仓库：前端放在根目录 `frontend/`。
- Submodule：`git submodule add <frontend-repo-url> frontend`。

两种方式不能改变 wire contract，也不得形成循环 submodule。
