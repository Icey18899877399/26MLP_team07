/**
 * ID 生成工具。
 * runId 由前端生成，作为本地 UI 行键；服务器返回的 run_id 单独存储。
 */

/** 生成一次实验的本地 ID，如 r-a1b2c3 */
export function genRunId() {
  return `r-${Math.random().toString(36).slice(2, 8)}`
}
