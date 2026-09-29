# 26MLP_team07_backend

《机器学习理论与实践》第 07 组后端与集成交付仓库。

本仓库负责课程大作业的 FastAPI interaction layer、与独立 ML 仓库之间的稳定集成边界，以及可选的统一交付入口。前端由另一位成员负责；无论它最终位于本仓库还是独立仓库，都只通过 HTTP/OpenAPI 使用本服务。模型算法、训练、数据处理和评价仍由 [26MLP_team07](https://github.com/Icey18899877399/26MLP_team07) 独立维护，本仓库不会复制一套 ML Core。

## 仓库与运行时关系

```text
Frontend --HTTP/JSON--> FastAPI --Python API--> ml_core --internal--> Models
```

ML 源码以 Git submodule 位于 `external/ml-core`。Submodule 保留独立历史并让本仓库只记录一个 commit pointer；运行时边界则是未来可安装的 Python package `ml_core`。详见 [架构说明](docs/ARCHITECTURE.md)、[HTTP API 合同](docs/HTTP_API_CONTRACT.md) 和 [ML 集成合同](docs/ML_INTEGRATION_CONTRACT.md)。

## 获取代码

新 clone：

```powershell
git clone --recurse-submodules https://github.com/Moonia-Cherry/26MLP_team07_backend.git
cd 26MLP_team07_backend
```

已经 clone 但尚未获取 submodule：

```powershell
git submodule update --init --recursive
```

## Backend

需要 Python 3.11+：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
Copy-Item .env.example .env
python -m uvicorn backend.main:app --reload
```

服务默认位于 `http://127.0.0.1:8000`，OpenAPI 文档位于 `/docs`。当前端点：

- `GET /api/health`
- `GET /api/models`
- `GET /api/datasets`
- `POST /api/experiments`

默认配置允许 `http://localhost:5173` 和 `http://127.0.0.1:5173` 跨域访问。可在 `.env` 中设置：

```dotenv
MLP_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
MLP_ML_PACKAGE=ml_core
MLP_LOG_LEVEL=INFO
```

`MLP_CORS_ORIGINS` 是逗号分隔的显式 origin 白名单，不接受 `*`、路径或带用户名/密码的 URL。

## ML package

当前 pinned ML commit 还没有 packaging metadata 或 `ml_core` 公共入口，因此目前不要执行 editable install，也没有真实 ML 实验接入。ML 仓库负责人实现合同后，安装方式应为：

```powershell
python -m pip install -e external/ml-core
```

随后重启 FastAPI。`/api/health` 应从 `available: false` 变为 `available: true`。FastAPI 不会回退到 `Models.*` 私有 import。

## Frontend ownership

本仓库当前不包含前端实现，也不创建空的 `frontend/` 占位目录。未来两种源码管理方式都使用根目录的 `frontend/` 路径：

- 同仓库：前端成员直接在 `frontend/` 中开发并提交。
- 独立仓库：在前端仓库创建后执行 `git submodule add <frontend-repo-url> frontend`，由本仓库固定交付 SHA。

两种方式共享相同的 [HTTP API 合同](docs/HTTP_API_CONTRACT.md)，不要求后端代码变化。禁止前端仓库反向包含本仓库，以免形成循环 submodule。

## 测试

```powershell
python -m pytest
```

测试中的 `FakeMLBackend` 只验证 HTTP、schema 和依赖倒置；生产应用始终使用 `PackageMLBackend`。在 ML 仓库提供合同前，没有真实 integration smoke test，也不能把测试替身结果称为 ML 实验结果。

## 课程交付提醒

课程要求最终以一个 PDF 附件提交，并在其中提供可访问的代码仓库链接、运行说明、截图与架构设计；代码仓库不得包含数据集本体。当前 ML submodule 所指向的上游 commit 仍跟踪数据集，这是由 ML 仓库负责人在最终交付前处理的已知缺口。
