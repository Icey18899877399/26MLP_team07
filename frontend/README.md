# Vue 前端（整合版）

完整步骤见 [整合运行说明](../docs/FULL_STACK.md)。本目录执行 `npm ci`，再 `npm run dev`。
需先启动 Python 后端 8000；开发地址 http://127.0.0.1:5173，/api 自动代理到后端。

- `npm test`：HTTP 请求/结果转换测试。
- `npm run build`：生成 dist/。
- `node scripts/http-smoke.mjs`：服务启动后，通过真实 HTTP 运行两种实验。
- `scripts/browser-smoke.mjs`：可选 Playwright 浏览器测试，需本地安装 Playwright 与 Edge。

已接入真实 HTTP。wsClient.js 仅保留 stores 的导入兼容名；mock/ 保留上游参考代码，不参与启动或测试。
当前只显示真实后端发布的模型，不混入旧 mock 数据。原说明保存在 UPSTREAM_README.md（历史资料）。
