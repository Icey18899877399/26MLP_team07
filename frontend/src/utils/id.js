/**
 * ID 生成工具。
 * runId 由前端生成（协议要求），消息 id 用于日志关联。
 */

let counter = 0

/** 生成一次训练的运行 ID，如 r-a1b2c3 */
export function genRunId() {
  return `r-${Math.random().toString(36).slice(2, 8)}`
}

/** 生成消息 ID，如 msg-001 */
export function genMsgId() {
  counter += 1
  return `msg-${counter}`
}
