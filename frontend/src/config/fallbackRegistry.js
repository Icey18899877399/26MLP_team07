// Offline state never advertises unsupported models or simulated results.
export const fallbackRegistry = {
  server: { name: '尚未连接模型服务', version: '0.1.0' },
  taskTypes: [], algorithms: [], datasets: [], metrics: {}
}
