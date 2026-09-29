/**
 * 前端配置。HTTP 地址可在右上角连接设置中修改。
 * defaultWsUrl 为兼容原 stores 保留的字段名，值为 HTTP API 地址。
 */
export const CONFIG = {
  // 开发时由 Vite 将 /api 代理到真实后端。
  defaultWsUrl: import.meta.env.VITE_API_BASE_URL || '/api',
  // localStorage 中保存设置的键名
  settingsKey: 'mlviz.http.settings.v1',
  // 心跳间隔（毫秒）
  heartbeatInterval: 15000,
  // 心跳应答超时（毫秒），超时判定连接已死
  heartbeatTimeout: 5000,
  // 重连退避：起始 1s，每次翻倍，上限 15s
  reconnectBase: 1000,
  reconnectMax: 15000,
  // 实时曲线每个序列最多保留的点数
  maxChartPoints: 500,
  // progress 消息渲染节流（毫秒）：距上一条小于该间隔则丢弃
  progressThrottleMs: 100,
  // 日志面板最多保留行数
  maxLogLines: 200,
  // 训练历史最多保留条数（localStorage）
  maxHistory: 50
}
