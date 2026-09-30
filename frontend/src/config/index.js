/**
 * 前端全局配置：HTTP 后端地址、轮询与超时、各种上限。
 * 后端地址可在界面右上角"连接设置"中修改，保存在 localStorage。
 *
 * 接口唯一合同见 HTTP_API_CONTRACT.md（仓库根目录）。
 */
export const CONFIG = {
  // 默认后端 HTTP 基地址（合同地址，真实 FastAPI 后端）
  defaultHttpUrl: import.meta.env?.VITE_API_BASE_URL || '',
  // localStorage 中保存设置的键名
  settingsKey: 'mlviz.http.v2.settings',
  // 健康检查轮询间隔（毫秒）
  healthPollIntervalMs: 15000,
  // health / 发现请求超时（毫秒）
  requestTimeoutMs: 10000,
  // 同步训练请求超时（毫秒，5 分钟）
  experimentTimeoutMs: 300000,
  // 运行中耗时刷新间隔（毫秒）
  elapsedTickMs: 1000,
  // 切分默认值（合同只定义 test_size/random_state 字段）
  defaultTestSize: 0.2,
  defaultSeed: 42,
  // 日志面板最多保留行数
  maxLogLines: 200,
  // 训练历史最多保留条数（localStorage）
  maxHistory: 50
}
