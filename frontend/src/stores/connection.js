/**
 * 连接状态 store：HTTP 后端健康轮询、注册表刷新、日志。
 *
 * 架构核心（可扩展性）：
 * - 后端可达时通过 GET /api/models + /api/datasets 拉取注册表，
 *   前端全部界面从注册表动态渲染，不硬编码任何算法/指标名
 * - 后端不可达时使用 src/config/fallbackRegistry.js 的静态兜底配置，
 *   界面照常渲染，但实验被 gating（合同要求）
 * - "已连接" = 最近一次 GET /api/health 返回 200 且
 *   ml_backend.available === true；available=false 时禁止一切实验
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api } from '../services/api.js'
import { httpClient } from '../services/httpClient.js'
import { buildRegistry } from '../services/adapters.js'
import { CONFIG } from '../config/index.js'
import { fallbackRegistry } from '../config/fallbackRegistry.js'

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(CONFIG.settingsKey) || '{}')
  } catch {
    return {}
  }
}

function saveSettings(settings) {
  try {localStorage.setItem(CONFIG.settingsKey, JSON.stringify(settings))} catch { /* settings remain in memory */ }
}

/** 迁移旧设置：WS 时代的 wsUrl → baseUrl（仅协议方案替换，端口/路径保留） */
function migrateSettings() {
  const stored = loadSettings()
  if (typeof stored.baseUrl === 'string' && stored.baseUrl) return stored.baseUrl
  if (typeof stored.wsUrl === 'string' && stored.wsUrl) {
    const migrated = stored.wsUrl.replace(/^ws:\/\//i, 'http://').replace(/^wss:\/\//i, 'https://')
    return migrated
  }
  return CONFIG.defaultHttpUrl
}

export const useConnectionStore = defineStore('connection', () => {
  // ---------- 状态 ----------
  const status = ref('connecting') // 'connecting' | 'connected' | 'unavailable' | 'offline'
  const health = ref(null) // 最近一次 /api/health 响应体
  const registry = ref(fallbackRegistry) // 初始兜底；只在刷新成功时替换
  const lastError = ref(null)
  const logLines = ref([]) // [{ ts, level, message }]
  const settings = ref({ baseUrl: migrateSettings() })

  // 模块级非响应式状态
  let pollTimer = null
  let healthInFlight = null
  let endpointGeneration = 0
  let originalOnly = false

  // ---------- 计算属性 ----------
  const isUsingFallback = computed(() => registry.value.server.name === fallbackRegistry.server.name)
  const baseUrl = computed(() => settings.value.baseUrl || CONFIG.defaultHttpUrl)
  const mlBackendAvailable = computed(() => health.value?.ml_backend?.available === true)
  /** 唯一 gating：全部实验操作的前置条件 */
  const canRunExperiments = computed(() => status.value === 'connected' && mlBackendAvailable.value && !isUsingFallback.value)

  // ---------- 内部 ----------
  function addLog(level, message) {
    logLines.value.push({ ts: Date.now(), level, message })
    if (logLines.value.length > CONFIG.maxLogLines) {
      logLines.value.splice(0, logLines.value.length - CONFIG.maxLogLines)
    }
  }

  /** 状态转换时记录日志（同状态连续失败不刷屏） */
  function transitionTo(next, log) {
    if (status.value === next) return
    status.value = next
    if (log) addLog(log.level, log.message)
  }

  // ---------- 动作 ----------
  /** 应用启动时调用一次：读设置、启动健康轮询 */
  function init(options = {}) {
    originalOnly = options.originalOnly === true
    if (pollTimer) clearInterval(pollTimer)
    httpClient.setBaseUrl(baseUrl.value)
    checkHealth()
    pollTimer = setInterval(checkHealth, CONFIG.healthPollIntervalMs)
  }

  /** 健康检查：200+available → connected；200+!available → unavailable；其余 → offline */
  async function checkHealth() {
    const generation = endpointGeneration
    if (healthInFlight === generation) return
    healthInFlight = generation
    try {
      const resp = await api.checkHealth()
      if (generation !== endpointGeneration) return
      health.value = resp
      lastError.value = null
      if (resp?.ml_backend?.available === true) {
        const wasConnected = status.value === 'connected'
        transitionTo('connected', { level: 'info', message: '已连接后端服务器' })
        if (!originalOnly && (!wasConnected || isUsingFallback.value)) await refreshRegistry(generation)
      } else {
        const detail = resp?.ml_backend?.detail
        transitionTo('unavailable', {
          level: 'warn',
          message: `后端服务可达，但 ML 包不可用${detail ? `（${detail}）` : ''}`
        })
      }
    } catch (err) {
      if (generation !== endpointGeneration) return
      health.value = null
      lastError.value = err.message
      transitionTo('offline', {
        level: 'error',
        message: `无法连接后端服务器（${err.code || 'network_error'}），界面使用静态兜底配置`
      })
    } finally {
      if (healthInFlight === generation) healthInFlight = null
    }
  }

  /** 拉取模型/数据集并重建注册表；失败保持现状（不降级到兜底） */
  async function refreshRegistry(generation = endpointGeneration) {
    const requestedUrl = baseUrl.value
    try {
      const [models, datasets] = await Promise.all([api.fetchModels(), api.fetchDatasets()])
      if (generation !== endpointGeneration) return
      registry.value = buildRegistry({ models, datasets, baseUrl: requestedUrl })
      lastError.value = null
      addLog('info', `已获取后端注册表：${models.length} 个模型，${datasets.length} 个数据集`)
    } catch (err) {
      if (generation !== endpointGeneration) return
      registry.value = fallbackRegistry
      lastError.value = `注册表刷新失败：${err.message}`
      addLog('error', `注册表刷新失败：${err.message}`)
    }
  }

  /** 修改后端地址：持久化 + 立即重新检查 */
  function setBaseUrl(url) {
    endpointGeneration += 1
    registry.value = fallbackRegistry
    health.value = null
    lastError.value = null
    settings.value.baseUrl = url
    saveSettings({ baseUrl: url })
    httpClient.setBaseUrl(url)
    status.value = 'connecting'
    addLog('info', `后端地址已更新为 ${url}，正在重新连接…`)
    return checkHealth()
  }

  /** 手动刷新（badge 刷新按钮）：健康检查，成功时连带刷新注册表 */
  async function refreshAll() {
    await checkHealth()
    if (!originalOnly && status.value === 'connected') await refreshRegistry()
  }

  return {
    status,
    health,
    registry,
    lastError,
    logLines,
    settings,
    isUsingFallback,
    baseUrl,
    mlBackendAvailable,
    canRunExperiments,
    init,
    checkHealth,
    refreshRegistry,
    setBaseUrl,
    refreshAll,
    addLog
  }
})
