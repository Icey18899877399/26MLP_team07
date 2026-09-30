# 原实验平台启动与联调

本仓库包含 Vue 前端、FastAPI 后端、ml_core 公共 Python 包，以及用户自己的 Models、data、visualization、figures 原目录。前端以原实验为主，不使用队友测试适配器的默认数据和参数。

## 安装

需要 Python 3.11+、Node.js 22.12+。在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . -e "./backend[test]"
npm --prefix frontend ci
python scripts/dev.py
```

macOS/Linux 激活方式为 `source .venv/bin/activate`。

- 页面：http://127.0.0.1:5173
- API 文档：http://127.0.0.1:8000/docs
- Ctrl+C 关闭本地启动器。

分别启动可用：

```text
python -m uvicorn backend.main:app --app-dir backend --host 127.0.0.1 --port 8000
npm --prefix frontend run dev
```

前端开发服务器将 `/api` 代理至 8000。连接设置或 `frontend/.env` 中的 `VITE_API_BASE_URL` 填后端源地址，不含 `/api`；留空使用同源代理。图片与文件下载同样使用该地址。

## 页面使用

1. 选择算法，查看原数据路径、原脚本与只读协议参数。不同算法的实验不强行统一为一个随机种子或训练轮数。
2. “原有实验成果”展示仓库已有六张 PNG；可放大和下载。原来存在的 CSV 可下载，没有源数据文件的图不提供伪造 CSV。
3. 提交“完整复现”后，后台执行原脚本全部流程，包括其中定义的调参、重复实验与对比。前端展示真实排队/运行状态、耗时和日志，不伪造百分比。
4. 新结果与原图分开展示，运行记录由后端保存。页面刷新不等于取消训练。
5. 数据集页区分实际用于原图的五组数据与两个原项目存档数据集；不将存档数据替换进原实验。

原图可以直接浏览，不必先等待耗时的完整复现。图中的合成数据若来自原脚本的机制演示会保留，它们不是模板随机生成的替代结果。

## 原实验 HTTP 合同

| 请求 | 用途 |
| --- | --- |
| `GET /api/original-experiments` | 12 套原实验、数据、参数、协议与原图地址 |
| `GET /api/original-datasets` | 原项目数据目录、用途与存档标记 |
| `POST /api/original-runs` | 提交完整复现，返回 202 与运行记录 |
| `GET /api/original-runs` | 读取服务端历史，最新在前 |
| `GET /api/original-runs/{run_id}` | 读取状态、日志与该次产物 |
| `GET /api/original-assets/{experiment_id}/{filename}` | 下载白名单原图/已有 CSV |
| `GET /api/original-runs/{run_id}/assets/{filename}` | 下载该次运行产物 |

请求仅选择登记过的实验：

```json
{"experiment_id": "logistic_regression"}
```

不接受任意脚本、文件路径、输出目录或参数覆盖。未知实验 404，无效请求 422，队列容量限制 409。

目录元素字段：`id,title,task,script,datasets,parameters,protocol,figures,source_data`。图项为 `name,title,url`，源文件项为 `name,url`。`task` 包括 classification、regression、clustering、anomaly_detection。

运行记录字段：`run_id,experiment_id,status,created_at,started_at,finished_at,parameters,figures,source_data,log,error`。状态为 queued、running、completed、failed、interrupted。时间为 ISO 字符串或 null；日志为字符串；错误为字符串或 null。

## 原脚本执行和文件保护

模型层只向原模块 CLI 传原始数据路径及新输出目录，其他采用脚本默认设置。不会启用 `--quick`，不会附加后来引入的训练样本上限。原脚本自身的采样限制、调参子集与迭代上限保留并展示。

Matplotlib 与训练在独立子进程执行；任务串行运行，避免同时执行多套完整实验造成资源过载。输出写入根目录 `.original-runs/` 的唯一运行目录，已加入 Git 忽略；原 `figures/` 不被覆盖。此目录可能含较大的 PNG 和日志，请自行备份后再清理。不要在任务执行时移动它。

服务重启时，未完成任务显示 interrupted，而不是继续显示 running 或冒充 completed。历史不是分布式任务数据库；仅支持本地单后端进程，不要使用多个 Uvicorn worker 共享此目录。

## 安装包模式

`ml-core` 的依赖包含 Matplotlib 与 NumPy。wheel 同时包含原 `Models`、`visualization` 模块，原数据与原图收在 `ml_core/resources/original/data` 和 `ml_core/resources/original/figures`，不会向 site-packages 顶层写入通用名称的 data/figures 目录。

editable 开发安装使用仓库根目录的原文件；普通 wheel 安装使用上述包内资源。执行时明确传入相应原数据路径，其余仍采用原脚本默认配置。后端不向安装包资源写新图：源码模式历史位于仓库根 `.original-runs/`；安装模式在 Windows 使用 `%LOCALAPPDATA%/26MLP_team07/original-runs`，没有该环境变量时使用用户目录 `.local/share/26MLP_team07/original-runs`。测试或嵌入应用可以通过 `create_app(original_jobs_dir=...)` 指定独立目录。

下载原图和新产物时可加 `?download=true`，让服务器返回附件响应；图片展示不加此参数。这也支持前端与后端不同源的下载场景。

## 兼容边界

`GET /api/models`、`GET /api/datasets`、`POST /api/experiments` 与原 `ExperimentConfig/run_experiment` 保留兼容。它们属于旧单次训练 API，沿用此前适配器，不代表这里的原实验协议，新的主要页面不使用它们。

队友前后端仍贡献 HTTP 交互结构与 Vue 技术基础，但不决定本项目实验数据、参数或评价协议。此前“12 个优化版 + 通用 ECharts”的验收记录不视为本次原实验接入验收。

## 限制与验证

完整六图实验可能执行大量训练，耗时明显长于一次 fit；不承诺固定耗时，也没有模型内部逐轮实时进度。失败会保留错误与日志，不显示成功图表。模型的图中计时会随机器负载变化，不能要求复现 PNG 二进制一致。

本机课程用途。没有公网鉴权或用户隔离；不要直接暴露到公网。生产部署需同源 API 反向代理、Vue history 回退、鉴权和资源配额。`npm --prefix frontend run build` 生成 `frontend/dist`；直接双击 index.html 不能代替服务启动。

```powershell
python -m pytest tests/test_original_experiments.py backend/tests/test_original_experiments.py
python -m pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
git diff -- Models data visualization figures
```

最后一个命令应没有输出。浏览器验收包含全部 72 张原图和一次真实完整复现；只验证目录或构造命令不等于全部 12 套完整训练都已重跑。

启动服务后，可用 Playwright 与 Edge 执行 `node frontend/scripts/browser-smoke.mjs`（默认指向新的 original-browser-smoke）。没有本地 Playwright 依赖时，可用 `PLAYWRIGHT_MODULE` 指定已经安装的模块 file URL；浏览器可用 `BROWSER_CHANNEL` 配置。该测试会实际提交一次完整逻辑回归实验，保留其运行记录，不启动全部 12 套耗时训练。
