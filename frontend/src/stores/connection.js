/**
 * 连接状态 store：HTTP 健康状态、后端注册表、日志。
 *
 * 架构核心（可扩展性）：
 * - HTTP 客户端读取真实模型/数据集，再分发本地 config.registry 事件
 * - 前端全部界面都从这个注册表动态渲染，不硬编码任何算法/指标名
 * - 未连接后端时显示空目录并禁用训练
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { wsClient } from '../services/wsClient'
import { CONFIG } from '../config'
import { fallbackRegistry } from '../config/fallbackRegistry'
import { genMsgId } from '../utils/id'
import { useTrainingStore } from './training'
import { useDatasetStore } from './datasets'

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(CONFIG.settingsKey) || '{}')
  } catch {
    return {}
  }
}

function saveSettings(settings) {
  localStorage.setItem(CONFIG.settingsKey, JSON.stringify(settings))
}

export const useConnectionStore = defineStore('connection', () => {
  // ---------- 状态 ----------
  const status = ref('offline') // 'connecting' | 'connected' | 'offline'
  const registry = ref(fallbackRegistry)
  const lastError = ref(null)
  const logLines = ref([]) // [{ ts, level, message }]
  const settings = ref({ wsUrl: CONFIG.defaultWsUrl, ...loadSettings() })

  // ---------- 计算属性 ----------
  const isUsingFallback = computed(() => registry.value.server.name === fallbackRegistry.server.name)
  const wsUrl = computed(() => settings.value.wsUrl || CONFIG.defaultWsUrl)

  // ---------- 内部 ----------
  function addLog(level, message) {
    logLines.value.push({ ts: Date.now(), level, message })
    if (logLines.value.length > CONFIG.maxLogLines) {
      logLines.value.splice(0, logLines.value.length - CONFIG.maxLogLines)
    }
  }

  /** 应用后端推送的注册表 */
  function applyRegistry(payload) {
    if (!payload || !Array.isArray(payload.algorithms)) {
      addLog('warn', '收到格式不正确的 config.registry，已忽略')
      return
    }
    registry.value = payload
    lastError.value = null
    addLog('info', `已接收后端注册表：${payload.algorithms.length} 个算法，${(payload.datasets || []).length} 个数据集`)
  }

  /** 消息分发器：config.registry 归本 store，training.* 交给训练 store，未知类型忽略（前向兼容） */
  function dispatchMessage(msg) {
    const { type, payload } = msg
    if (type === 'config.registry') {
      applyRegistry(payload)
    } else if (type === 'log') {
      addLog(payload?.level || 'info', payload?.message || String(payload))
    } else if (type === 'error') {
      lastError.value = payload?.message || '后端返回错误'
      addLog('error', `[${payload?.code || 'ERROR'}] ${lastError.value}`)
    } else if (type === 'pong') {
      // wsClient 内部处理心跳
    } else if (typeof type === 'string' && type.startsWith('training.')) {
      useTrainingStore().handleMessage(msg)
    } else if (typeof type === 'string' && type.startsWith('dataset.')) {
      useDatasetStore().handleMessage(msg)
    } else {
      // 协议前向兼容：未知消息类型一律忽略，不崩溃
      console.warn('[connection] 忽略未知消息类型:', type)
    }
  }

  // ---------- 动作 ----------
  /** 应用启动时调用一次：读设置、接线、连接 */
  function init() {
    wsClient.onMessage = dispatchMessage
    wsClient.onStatusChange = (s) => {
      status.value = s
      if (s === 'connected') {
        addLog('info', '已连接后端服务器')
        // 连接后可主动刷新注册表（后端通常也会自动推送）
        refreshRegistry()
        // 断线重连恢复：订阅未结束训练的当前状态
        useTrainingStore().onReconnected()
      } else if (s === 'offline') {
        addLog('warn', '服务未连接，请检查后端并点击重连')
      } else {
        addLog('info', '正在连接后端服务器…')
      }
    }
    connect()
  }

  function connect(url) {
    if (url) settings.value.wsUrl = url
    wsClient.connect(settings.value.wsUrl)
  }

  function disconnect() {
    wsClient.disconnect()
  }

  function setWsUrl(url) {
    settings.value.wsUrl = url
    saveSettings({ wsUrl: url })
    // 重新连接生效
    wsClient.disconnect()
    setTimeout(() => wsClient.connect(url), 300)
  }

  function refreshRegistry() {
    wsClient.send({ type: 'config.get', id: genMsgId(), payload: {} })
  }

  return {
    status,
    registry,
    lastError,
    logLines,
    settings,
    isUsingFallback,
    wsUrl,
    init,
    connect,
    disconnect,
    setWsUrl,
    refreshRegistry,
    dispatchMessage,
    addLog
  }
})
