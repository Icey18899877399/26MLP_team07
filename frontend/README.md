# Vue 前端（整合版）

完整步骤见 [整合运行说明](../docs/FULL_STACK.md)。本目录执行 `npm ci`，再 `npm run dev`。
需先启动 Python 后端 8000；开发地址 http://127.0.0.1:5173，/api 自动代理到后端。

- `npm test`：HTTP 契约、超参数、连接切换、历史与对比测试。
- `npm run build`：生成 dist/。
- `node scripts/http-smoke.mjs`：服务启动后，通过真实 HTTP 验证全部 12 种算法与图表。
- `scripts/browser-smoke.mjs`：可选 Playwright 浏览器测试，需本地安装 Playwright 与 Edge。

基于队友前端 PR1 的 api / adapters / HTTP store 结构整合。默认使用同源 `/api` 代理；`VITE_API_BASE_URL` 留空，或设为不含 `/api` 的服务地址。可在右上角连接设置调整。

页面涵盖参数配置、真实请求状态、指标与图表画廊、历史参数复现、JSON 导出、数据集目录、同条件算法对比与使用手册。每张图可放大并下载 PNG。

后端不可用时展示明确标注的离线配置，禁止训练。`mock/` 仅保留上游历史参考，不参与运行或测试。新 HTTP 记录使用独立的本地存储键，不载入旧 WebSocket 演示历史；持久化最多保留最近 10 条，存储空间不足时保留会话结果并提示导出。
