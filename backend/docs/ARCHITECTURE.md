# 交互平台架构

## 分层

```text
Frontend (owned separately)
      | HTTP / JSON
      v
FastAPI interaction layer
      | Python public package API
      v
ml_core package
      |
      v
Independent ML implementation
```

- `backend/interaction/` 负责 HTTP 路由、Pydantic 校验和错误语义。
- `backend/integration/` 只导入 `ml_core` 顶层公共入口、转换合同对象并隔离异常。
- `backend/contracts.py` 是不依赖 HTTP 路由的共享 wire contract。
- ML 仓库负责数据准备、训练、预测、评价、产物与元数据。
- 前端是独立 HTTP 消费者，可位于普通 `frontend/` 目录或同路径 submodule。

Git submodule 只解决源码版本固定；运行时边界是可安装的 Python package `ml_core`。

## 依赖方向

- 前端只访问 FastAPI，不读取 Python 或 ML 仓库文件。
- Route 只依赖 `MLBackend`，不导入具体模型类。
- `PackageMLBackend` 只依赖 `ml_core` 顶层公共导出。
- ML 仓库不依赖 FastAPI、后端 Pydantic 类型或前端技术。
- 数据拆分、标准化、训练、评价与随机策略完全留在 ML 包内。

## 请求流程

1. 前端读取 `/api/health`，确认 interaction 与 ML backend 状态。
2. 前端从 `/api/models` 和 `/api/datasets` 构建选择器与参数表单。
3. `/api/experiments` 校验 JSON 后构造 `ml_core.ExperimentConfig`。
4. `ml_core.run_experiment` 完成真实流程并返回 `ExperimentResult`。
5. 边界重新校验结果，FastAPI 返回 metrics、artifacts 和 metadata。

同步 ML 调用发生在 FastAPI 的同步路由中，因此由其 worker thread 执行，不阻塞事件循环。

## 当前实现

- health、models、datasets、experiments 四个端点。
- 严格请求/响应 schema、OpenAPI、CORS 与 400/502/503 错误映射。
- `ml_core` 真实模型发现、数据集发现与实验调用。
- 真实 `kmeans.optimized` 和 `logistic_regression.optimized` smoke test。
- ML submodule 固定到 `5c1bb06a52e9e542e20bb18a3f0f8d39b16f47ed`。
- 前端路径保持为空闲状态，兼容普通目录与 submodule。

WebSocket、长任务队列和前端可视化不属于当前阶段。ML 上游仍需在最终交付前解决数据集本体被 Git/package 跟踪的问题。
