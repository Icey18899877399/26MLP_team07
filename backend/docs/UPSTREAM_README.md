# 26MLP_team07_backend

《机器学习理论与实践》第 07 组后端与集成交付仓库。

本仓库负责 FastAPI interaction layer、与独立 ML 仓库之间的稳定集成边界，以及可选的统一交付入口。前端由另一位成员负责；无论最终位于本仓库还是独立仓库，都只通过 HTTP/OpenAPI 使用本服务。算法由 [26MLP_team07](https://github.com/Icey18899877399/26MLP_team07) 独立维护，本仓库不复制 ML 实现。

```text
Frontend --HTTP/JSON--> FastAPI --Python public API--> ml_core
```

ML 源码以 Git submodule 位于 `external/ml-core`，父仓库固定经过联调的提交。详见 [架构说明](docs/ARCHITECTURE.md)、[HTTP API 合同](docs/HTTP_API_CONTRACT.md) 和 [ML 集成合同](docs/ML_INTEGRATION_CONTRACT.md)。

## 获取与运行

需要 Python 3.11+：

```powershell
git clone --recurse-submodules https://github.com/Moonia-Cherry/26MLP_team07_backend.git
cd 26MLP_team07_backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pip install -e external/ml-core
Copy-Item .env.example .env
python -m uvicorn backend.main:app --reload
```

已 clone 但没有 submodule 时，先运行：

```powershell
git submodule update --init --recursive
```

服务默认位于 `http://127.0.0.1:8000`，OpenAPI 文档位于 `/docs`。端点保持为：

- `GET /api/health`
- `GET /api/models`
- `GET /api/datasets`
- `POST /api/experiments`

安装 `ml_core` 后，health 中的 `ml_backend.available` 应为 `true`。当前公开的真实组合为 `kmeans.optimized + seeds` 和 `logistic_regression.optimized + wdbc`。

## 配置

默认允许 `http://localhost:5173` 和 `http://127.0.0.1:5173` 跨域访问：

```dotenv
MLP_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
MLP_ML_PACKAGE=ml_core
MLP_LOG_LEVEL=INFO
```

origin 白名单不接受 `*`、路径或带用户名/密码的 URL。

## Frontend ownership

本仓库当前不包含前端实现，也不创建空 `frontend/`：

- 同仓库：前端成员直接在 `frontend/` 开发并提交。
- 独立仓库：执行 `git submodule add <frontend-repo-url> frontend`，固定联调 SHA。

两种方式共享同一 HTTP 合同，不要求后端变化。前端仓库不得反向包含本仓库。

## 测试

```powershell
python -m pytest
```

测试既覆盖 HTTP/异常边界，也会在 `ml_core` 已安装时运行真实模型 smoke test；未安装时，真实集成测试会明确标记 skipped。

## 课程交付提醒

课程要求最终 PDF 提供可访问的代码仓库链接、运行说明、截图与架构设计，并要求代码仓库不得包含数据集本体。当前 ML 上游仍把数据集文件打包进 `ml_core/resources/datasets`，最终交付前需由 ML 负责人调整数据获取与缓存方案。
