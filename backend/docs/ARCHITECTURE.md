# 交互平台架构

> 上游独立仓库架构存档。当前已整合到单仓库并移除子模块，运行结构见 [整合说明](../../docs/FULL_STACK.md)。

## 目标与分层

本项目只负责课程大作业的 Web 交互平台，不重复实现机器学习算法。系统在逻辑上分为三层：

```text
Frontend (owned separately)
      |
      | HTTP / JSON
      v
FastAPI interaction layer
      |
      | Python public package API
      v
ml_core public API
      |
      v
Independent ML implementation
```

- 前端由另一位成员负责，可作为本仓库目录或独立 Git submodule 交付；运行时一律作为独立 HTTP 消费者。
- `backend/interaction/` 负责 HTTP 路由、Pydantic 校验和 HTTP 错误语义。
- `backend/integration/` 是很薄的运行时边界，只导入 `ml_core` 的公共入口、做轻量类型转换，并隔离包异常。
- ML layer 位于独立仓库 `26MLP_team07`，负责数据准备、训练、预测、评价和结构化 diagnostics。

Git submodule 只是把独立 ML 仓库固定到可复现的 commit；它不是运行时架构层。真正的程序依赖边界是 Python package `ml_core`。

## 依赖方向

- 前端只访问 FastAPI，不读取 ML 仓库或 Python 文件。
- FastAPI route 只依赖 `MLBackend`，不导入 `Models.*` 或具体模型类。
- `PackageMLBackend` 只依赖 `ml_core` 顶层公共导出。
- ML 仓库不依赖 FastAPI、Pydantic HTTP schema 或任何前端技术。
- 如果不同模型接口需要统一，优先由 ML 仓库在 `ml_core` 内解决；交互仓库不建立模型级 Adapter 集合。

## 请求流程

1. 前端请求 `GET /api/health`。
2. FastAPI 返回 interaction 服务状态和 `ml_core` 公共 API 可用性。
3. 仅当 ML API 可用时，前端获取模型与数据集列表并允许提交实验。
4. `POST /api/experiments` 经 Pydantic 校验后交给 `MLBackend.run_experiment`。
5. ML package 完成数据处理、训练、评价，并返回 JSON 可序列化的 `ExperimentResult`。
6. 前端展示 metrics，并以结构化 diagnostics 渲染图表。

## 当前实现

- FastAPI 的 health、models、datasets、experiments 四个端点。
- 严格 Pydantic 请求/响应 schema 和一致的 HTTP 错误映射。
- 可配置的显式 CORS 白名单与 OpenAPI 合同。
- `MLBackend` Protocol 与只访问公共包入口的 `PackageMLBackend`。
- ML 不可用时的明确状态和 503 行为。
- 用测试替身验证 interaction；测试替身不进入生产应用。
- 外部仓库以 `external/ml-core` submodule 固定版本。
- 前端源码当前未包含；`frontend/` 路径保留给未来同仓库目录或 Git submodule。

## 尚未实现

- 外部仓库还没有可安装的 `ml_core` 包，因此没有真实模型列表、数据集列表或实验调用。
- 没有复制 ML 仓库里的数据、预处理、训练、评价和 Matplotlib visualization 逻辑。
- WebSocket、前端可视化和长任务管理不属于本阶段。

只有外部仓库实现并发布 [ML integration contract](ML_INTEGRATION_CONTRACT.md) 后，才能把状态称为“真实 ML integration 已实现”。
