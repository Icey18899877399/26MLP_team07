# 26MLP_team07_frontend

机器学习理论与实践课程大作业 —— 算法可视化平台前端（26MLP 第 07 组）。

后端算法仓库见 [26MLP_team07](https://github.com/Icey18899877399/26MLP_team07)。

## 技术栈

- Vue 3 + Vite 7（纯 JS，无 TypeScript）
- Element Plus（UI 组件库）
- ECharts 5（图表）
- Pinia（状态管理）+ Vue Router

## 运行

```bash
# 安装依赖（需要 Node.js ≥ 18）
npm install

# 同时启动前端（:5173）与内置 Mock 后端（:8765）
npm run start

# 仅启动前端
npm run dev

# 仅启动 Mock 后端
npm run mock
```

浏览器访问 http://localhost:5173

> 说明：开发阶段由 Node Mock 后端（`mock/`）模拟真实后端协议（随机数据）。
> 对接真实 Python 后端后，Mock 即退役。

## 目录结构

```
src/
├── views/        # 页面：实验工作台 / 算法对比 / 数据集 / 使用手册
├── components/   # 可复用组件（图表 / 实验面板 / 指标卡 / 通用件）
├── stores/       # Pinia 全局状态（连接注册表 / 训练运行 / 数据集）
├── services/     # wsClient：WebSocket 协议封装（心跳/重连/收发）
├── config/       # 全局配置 / 兜底注册表 / ECharts 主题
└── utils/        # 格式化、超参 schema、ID 生成等工具函数
mock/             # Node Mock 后端（模拟协议）
```

## 架构设计

- **配置驱动**：前端界面完全由后端推送的注册表（算法/数据集/指标/超参 schema）动态渲染，后端新增内容前端零改动；后端未连接时使用内置兜底注册表。
- **协议**：前后端通过消息协议通信（`docs/PROTOCOL.md`，待补），核心消息为 `config.registry` → `training.start` → `training.progress` → `training.result`。
- **断线恢复**：心跳超时自动重连，重连后对未结束的训练重新订阅进度。

## 当前状态

| 模块 | 状态 |
|------|------|
| 实验工作台（算法选择 / 超参表单 / 训练监控 / 图表） | ✅ 已完成 |
| 数据集页 | ⬜ 待开发 |
| 算法对比（Benchmark）页 | ⬜ 待开发 |
| 使用手册页 | ⬜ 待开发 |
| 协议文档 PROTOCOL.md | ⬜ 待补充 |
