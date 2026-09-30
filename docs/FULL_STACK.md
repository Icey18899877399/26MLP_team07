# 完整平台启动与联调

代码统一位于 Icey18899877399/26MLP_team07。结构为 frontend/（Vue）、backend/backend/（FastAPI）、ml_core/（公共 Python 包）、Models/（算法实现）。无子模块，普通 clone 即可得到全部源码。

## 安装与启动

需要 Python 3.11+、Node.js 22.12+、npm。

在仓库根目录安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . -e "./backend[test]"
npm --prefix frontend ci
python scripts/dev.py
```

macOS/Linux 激活命令为 `source .venv/bin/activate`。统一启动器启动后端 8000 和前端 5173，Ctrl+C 结束。
浏览器打开 http://127.0.0.1:5173 。API 文档 http://127.0.0.1:8000/docs 。

也可以分别运行（在根目录）：

```text
python -m uvicorn backend.main:app --app-dir backend --host 127.0.0.1 --port 8000
npm --prefix frontend run dev
```

前端开发服务器把 /api 代理到 8000；无需配置跨域。可用 frontend/.env 设置
`VITE_API_BASE_URL=http://127.0.0.1:8000`，或在页面连接设置修改地址。该值是服务地址，不含 `/api`；留空则使用同源代理。
远程部署需把 /api 反向代理到后端，并为 Vue history 路由配置 index.html 回退。
`npm --prefix frontend run build` 生成 frontend/dist；构建产物无需上传源码仓库。
单独使用 Vite preview 不包含开发代理，需设置 API 地址并通过 `MLP_CORS_ORIGINS` 允许预览页来源，或配置生产同源反向代理。

## 已连通功能

- 动态模型与数据集列表、参数表单、真实实验执行与错误提示。
- 分类、回归、聚类、异常检测四类任务；模型与数据集以服务端公共目录为准。
- 同步队友 PR 后，公共目录提供 12 个算法的优化版（完整 ID 后缀 `.optimized`），不是 24 个可训练选项；基础算法源码仍保留在 Models/。
- 指标和图表来自当次真实实验；图表描述说明投影或抽样的含义。
- 数据集信息页、同一数据集历史结果对比、浏览器本地历史。
- 仅展示 ml_core 公共 API 实际支持的模型，不将模拟结果当作训练输出。

## 协议适配

前端使用队友 PR 的纯 HTTP JSON 架构，通过统一 API 客户端调用四个合同端点，不建立 WebSocket。
mock/ 仅保留上游模拟参考，不在统一启动流程中运行；其健康信息明确标为不可训练，避免把模拟结果当作真实结果。
新的设置/历史存储键与旧模拟记录隔离。

HTTP 与 ml_core 使用同一原生合同：`model` 为完整 ID，例如 `logistic_regression.optimized`。
不再接收单独的 `variant` 字段。目录返回 ModelInfo/DatasetInfo，不再转换为旧 ModelSpec/DatasetSpec。
ml_core 的请求类错误映射为 400，内部运行错误为 502（不暴露内部路径）；参数格式错误为 422；包不可用为 503。

请求示例：

```json
{
  "model": "logistic_regression.optimized",
  "dataset": "wdbc",
  "params": {"max_iter": 1000},
  "test_size": 0.2,
  "random_state": 42
}
```

向 POST /api/experiments 提交。聚类需 test_size 为 null 或省略；异常检测按上游合同不使用 test_size。
返回顶层 run_id、model、dataset、task、effective_params、metrics、artifacts、metadata。
`metadata.visualizations` 为图表数组，每项含 id、title、description、option；option 是可直接渲染的 ECharts JSON 配置。
前端耗时是完整请求时间，包含训练和评价。

## 当前限制

同步 HTTP 返回最终结果；暂无逐轮实时回调、服务端取消、跨刷新恢复、任务数据库。
如果模型提供训练历史，完成后才展示真实曲线；前端运行中状态不代表逐轮进度。
前端每次仅发起一项实验；保留页面直到返回结果。分类固定采用分层抽样。
原上游的数据预览与完整批量基准功能未实现；当前展示公开数据集元信息和同数据集历史结果对比。
历史只在本机浏览器保存。页面刷新后没有服务端任务恢复。
异常检测沿用队友的数据内评价：孤立森林在原数据拟合及评价，单类 SVM 用正常样本子集拟合后对原数据评价；这些指标不能视为独立测试集上的泛化成绩。页面和图表应保留协议说明。
本机/课程演示用途，部署到公网前需另行实现鉴权、限流与资源配额。
模型仓库既有数据文件仍保留；课程如要求代码仓库不得含数据，需另行设计下载/打包策略。

## 算法与数据集

| 任务 | 算法（优化版） | 可选数据集 |
| --- | --- | --- |
| 分类 | 逻辑回归、KNN、高斯朴素贝叶斯、CART、随机森林 | WDBC |
| 回归 | 线性回归、GBDT、MLP | Concrete；线性回归和 MLP 也支持 California Housing |
| 聚类 | K-Means、DBSCAN | Seeds |
| 异常检测 | 孤立森林、单类 SVM | Cardio、Mammography |

数据集 ID 以 `/api/datasets` 为准：`wdbc`、`seeds`、`concrete`、`california_housing`、`6_cardio`、`23_mammography`。
数据量、运行耗时和算法收敛情况会影响演示速度；不同任务/协议的指标不应直接横向排名。

为保证纯 Python 模型可用于交互演示，回归在随机划分后最多使用 2000 条训练样本，测试集不裁剪；单类 SVM 仍按上游协议最多取 300 条正常样本拟合。MLP 默认训练 100 轮，可在参数中修改。返回元数据记录实际样本数、上限和评价协议。
散点图最多显示 1000 个评价样本，图中说明显示数与总数；指标和分布统计仍用全部评价样本。真实损失曲线仅在模型提供训练记录时显示，并注明是否为标准化目标空间。

## 验证

```powershell
python -m pip install matplotlib
python -m unittest discover -s tests -q
Push-Location backend
python -m pytest
Pop-Location
npm --prefix frontend test
npm --prefix frontend run build
```

后端测试包括真实 ml_core 调用，不仅是 FakeMLBackend。
前端 tests/httpClient.test.js 检查协议转换；scripts/http-smoke.mjs 可在服务启动后执行：
`node frontend/scripts/http-smoke.mjs`，通过 Vite 代理运行真实实验。

2026-09-30 整合验收：358 项模型 unittest、5 项图表专项、56 项后端、14 项前端测试通过；生产构建及仓库外 wheel 安装通过。HTTP smoke 与浏览器 smoke 均实际运行全部 12 个算法；浏览器还验证放大/PNG 下载、JSON 导出、历史恢复、六数据集、对比页、手册和 390px 布局。
浏览器测试需 Playwright 与 Edge，可用 `PLAYWRIGHT_MODULE` 指定本地 Playwright 模块地址后执行 `node frontend/scripts/browser-smoke.mjs`。

来源提交与整合范围见 [SOURCES.md](SOURCES.md)。后端原生合同见 [HTTP_API_CONTRACT.md](../backend/docs/HTTP_API_CONTRACT.md)。
