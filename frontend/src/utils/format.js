/**
 * 格式化工具：数字、百分比、时长。
 */

/** 数值格式化：保留指定小数位，去掉多余的 0 */
export function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  const num = Number(value)
  if (Number.isInteger(num)) return String(num)
  return num.toFixed(digits).replace(/\.?0+$/, '')
}

/** 百分比格式化 */
export function formatPercent(value, digits = 2) {
  if (value === null || value === undefined) return '-'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

/** Only probability-like metrics have percentage units; counts and R² do not. */
export function formatMetric(id, value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—'
  const percentages = ['accuracy', 'precision', 'recall', 'f1', 'anomaly_precision', 'anomaly_recall', 'anomaly_f1']
  return percentages.includes(id) ? formatPercent(value) : formatNumber(value)
}

/** 毫秒时长格式化为易读文本 */
export function formatDuration(ms) {
  if (ms === null || ms === undefined) return '-'
  if (ms < 1000) return `${Math.round(ms)} ms`
  const totalSec = Math.round(ms / 1000)
  if (totalSec < 60) return `${totalSec}s`
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  if (min < 60) return `${min}m${sec}s`
  const hour = Math.floor(min / 60)
  return `${hour}h${min % 60}m`
}
