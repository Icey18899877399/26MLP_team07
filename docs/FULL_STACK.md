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
`VITE_API_BASE_URL=http://127.0.0.1:8000/api`，或在页面连接设置修改地址。
远程部署需把 /api 反向代理到后端，并为 Vue history 路由配置 index.html 回退。
`npm --prefix frontend run build` 生成 frontend/dist；构建产物无需上传源码仓库。
单独使用 Vite preview 不包含开发代理，需设置 API 地址或配置生产反向代理。

## 已连通功能

- 动态模型与数据集列表、参数表单、真实实验执行与错误提示。
- 分类、回归、聚类、异常检测四类任务；模型与数据集以服务端公共目录为准。
- 基础版与优化版使用不同的完整模型 ID，可分别训练并保存结果。
- 指标和图表来自当次真实实验；图表描述说明投影或抽样的含义。
- 数据集信息页、同一数据集历史结果对比、浏览器本地历史。
- 仅展示 ml_core 公共 API 实际支持的模型，不将模拟结果当作训练输出。

## 协议适配

前端使用 HTTP JSON。保留原 Vue stores 的内部事件接口以复用界面，`wsClient.js` 仅为兼容导出，不建立 WebSocket。
原 mock/ 作为上游参考源码保留，不在启动流程中运行；它不是生产后端，也不作为真实结果来源。
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

向 POST /api/experiments 提交。聚类需 test_size 为 null 或省略。
返回顶层 run_id、model、dataset、task、effective_params、metrics、artifacts、metadata。
`metadata.visualizations` 为图表数组，每项含 id、title、description、option；option 是可直接渲染的 ECharts JSON 配置。
前端耗时是完整请求时间，包含训练和评价。

## 当前限制

同步 HTTP 返回最终结果；暂无逐轮实时回调、服务端取消、跨刷新恢复、任务数据库。
如果模型提供训练历史，完成后才展示真实曲线；前端运行中状态不代表逐轮进度。
前端每次仅发起一项实验；保留页面直到返回结果。分类固定采用分层抽样。
原上游的数据预览与完整批量基准功能未实现；当前展示公开数据集元信息和同数据集历史结果对比。
历史只在本机浏览器保存。页面刷新后没有服务端任务恢复。
本机/课程演示用途，部署到公网前需另行实现鉴权、限流与资源配额。
模型仓库既有数据文件仍保留；课程如要求代码仓库不得含数据，需另行设计下载/打包策略。

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

来源提交与整合范围见 [SOURCES.md](SOURCES.md)。后端原生合同见 [HTTP_API_CONTRACT.md](../backend/docs/HTTP_API_CONTRACT.md)。
