/**
 * 训练 store：管理所有训练运行（run）的状态与消息归约。
 *
 * 协议要点（见 docs/PROTOCOL.md）：
 * - runId 由前端在发起 training.start 时生成，之后所有相关消息都带 runId，
 *   前端始终按 runId 查找对应的 run，绝不假设"最新消息属于某个 run"
 * - 断线恢复：重连成功后调用 onReconnected()，对所有未结束的 run 发送
 *   training.subscribe，后端回复当前状态（仍在训练 → 继续推 progress；
 *   已结束 → 直接推 result；找不到 → training.error NOT_FOUND，标记"已丢失"）
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { wsClient } from '../services/wsClient'
import { CONFIG } from '../config'
import { genMsgId, genRunId } from '../utils/id'
import { useConnectionStore } from './connection'

const HISTORY_KEY = 'mlviz.real.history.v1'

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
  } catch {
    return []
  }
}

function saveHistory(history) {
  try {
    const trimmed = history.slice(0, CONFIG.maxHistory)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed))
  } catch (err) {
    console.warn('[training] 历史记录保存失败:', err.message)
  }
}

/** 新建 run 对象 */
function createRun({ runId, algorithmId, datasetId, hyperparams, split, algorithmName, datasetName }) {
  return {
    runId,
    algorithmId,
    datasetId,
    algorithmName,
    datasetName,
    hyperparams: hyperparams || {},
    split: split || null,
    status: 'queued', // queued | running | done | error | cancelled | lost
    progress: { epoch: 0, totalEpochs: 0, elapsedMs: 0, stage: '', series: {} }, // series: { loss: [..], val_loss: [..] }
    result: null, // training.result 的 payload
    error: null,
    startedAt: Date.now(),
    finishedAt: null,
    lastProgressAt: 0 // progress 节流用
  }
}

export const useTrainingStore = defineStore('training', () => {
  // ---------- 状态 ----------
  /** 所有 run，按 runId 索引（Vue 3 响应式支持 Map） */
  const runs = ref(new Map())
  /** 历史完成的 run（含 session 内的 + localStorage 恢复的），供 benchmark 复用 */
  const history = ref(loadHistory())

  // ---------- 计算属性 ----------
  const runList = computed(() => [...runs.value.values()])
  const activeRuns = computed(() => runList.value.filter((r) => ['queued', 'running'].includes(r.status)))
  const finishedRuns = computed(() =>
    runList.value
      .filter((r) => r.status === 'done')
      .sort((a, b) => b.finishedAt - a.finishedAt)
  )

  // ---------- 内部 ----------
  function getRun(runId) {
    return runs.value.get(runId)
  }

  /** progress 节流：距上一条 <100ms 直接丢弃（渲染性能保护） */
  function shouldDropProgress(run) {
    const now = Date.now()
    if (now - run.lastProgressAt < CONFIG.progressThrottleMs) return true
    run.lastProgressAt = now
    return false
  }

  function finishRun(run) {
    run.finishedAt = Date.now()
    if (run.status !== 'done') return
    history.value.unshift({
      runId: run.runId,
      algorithmId: run.algorithmId,
      algorithmName: run.algorithmName,
      datasetId: run.datasetId,
      datasetName: run.datasetName,
      hyperparams: run.hyperparams,
      split: run.split,
      result: run.result,
      finishedAt: run.finishedAt
    })
    history.value = history.value.slice(0, CONFIG.maxHistory)
    saveHistory(history.value)
  }

  // ---------- 消息归约 ----------
  function handleMessage(msg) {
    const { type, payload } = msg
    const run = payload?.runId ? getRun(payload.runId) : null

    switch (type) {
      case 'training.accepted': {
        if (run) {
          run.status = 'running'
          run.error = null
        }
        break
      }
      case 'training.progress': {
        if (!run) return
        if (shouldDropProgress(run)) return
        run.progress.epoch = payload.epoch ?? run.progress.epoch
        run.progress.totalEpochs = payload.totalEpochs ?? run.progress.totalEpochs
        run.progress.elapsedMs = payload.elapsedMs ?? run.progress.elapsedMs
        run.progress.stage = payload.stage ?? run.progress.stage
        // metrics 里的任意数值键都画成曲线（loss/val_loss 等），动态发现序列名
        const series = run.progress.series
        for (const [key, value] of Object.entries(payload.metrics || {})) {
          if (typeof value !== 'number') continue
          if (!series[key]) series[key] = []
          series[key].push(value)
          if (series[key].length > CONFIG.maxChartPoints) series[key].shift()
        }
        break
      }
      case 'training.result': {
        if (run) {
          run.status = 'done'
          run.result = payload
          run.progress.elapsedMs = payload.durations?.train_ms ?? run.progress.elapsedMs
          finishRun(run)
        }
        break
      }
      case 'training.error': {
        if (run) {
            run.status = payload?.code === 'NOT_FOUND' ? 'lost' : 'error'
          run.error = payload?.message || payload?.code || '未知错误'
          if (run.status === 'error') finishRun(run)
        }
        break
      }
      case 'training.cancelled': {
        if (run) {
          run.status = 'cancelled'
        }
        break
      }
      default:
        // 前向兼容：未知训练消息忽略
        break
    }
  }

  // ---------- 动作 ----------
  /** 发起训练。返回 runId；未连接后端时返回 null 并报错 */
  function startRun({ algorithmId, datasetId, hyperparams, split, algorithmName, datasetName }) {
    if (!wsClient.connected) {
      useConnectionStore().addLog('error', '未连接后端服务器，无法开始训练')
      return null
    }
    const runId = genRunId()
    const run = createRun({ runId, algorithmId, datasetId, hyperparams, split, algorithmName, datasetName })
    runs.value.set(runId, run)
    const ok = wsClient.send({
      type: 'training.start',
      id: genMsgId(),
      payload: { runId, algorithmId, datasetId, hyperparams, split }
    })
    if (!ok) {
      run.status = 'error'
      run.error = '消息发送失败，连接已断开'
    }
    return runId
  }

  function cancelRun(runId) {
    wsClient.send({ type: 'training.cancel', id: genMsgId(), payload: { runId } })
  }

  /** 重连成功后调用：订阅所有未结束的 run 的当前状态 */
  function onReconnected() {
    for (const run of runs.value.values()) {
      if (['queued', 'running'].includes(run.status)) {
        wsClient.send({ type: 'training.subscribe', id: genMsgId(), payload: { runId: run.runId } })
      }
    }
  }

  function removeRun(runId) {
    runs.value.delete(runId)
  }

  function clearRuns() {
    for (const [id, run] of runs.value) {
      if (!['queued', 'running'].includes(run.status)) runs.value.delete(id)
    }
  }

  /** 查找某数据集下已完成的 run（benchmark 复用历史结果，避免重复训练） */
  function doneRunsForDataset(datasetId) {
    return history.value.filter((h) => h.datasetId === datasetId && h.status !== 'error')
  }

  return {
    runs,
    history,
    runList,
    activeRuns,
    finishedRuns,
    getRun,
    handleMessage,
    startRun,
    cancelRun,
    onReconnected,
    removeRun,
    clearRuns,
    doneRunsForDataset
  }
})
