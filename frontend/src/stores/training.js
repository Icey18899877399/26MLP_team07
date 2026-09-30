/**
 * 训练 store：管理实验运行（run）的状态。
 *
 * 合同（HTTP_API_CONTRACT.md）下训练是同步请求：
 * POST /api/experiments 直接返回最终结果（run_id + metrics），
 * 没有进度流/取消/订阅。运行中只显示等待态与耗时。
 *
 * - 本地 id 由前端生成（UI 行键，Map 键）；服务器 run_id 完成后存入 run
 * - 失败 run 不写入历史（不可用于对比），保留在 errorRuns 直到清空
 */
import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'
import { api } from '../services/api.js'
import { buildExperimentRequest } from '../services/adapters.js'
import { CONFIG } from '../config/index.js'
import { genRunId } from '../utils/id.js'
import { useConnectionStore } from './connection.js'

const HISTORY_KEY = 'mlviz.http.v2.history'

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    if (!Array.isArray(parsed)) return []
    return parsed.filter(h => h.schemaVersion === 2 && h.status === 'done' &&
      h.result?.run_id && h.result?.model === h.modelId && h.result?.dataset === h.datasetId)
  } catch {
    return []
  }
}

function saveHistory(history) {
  const trimmed = history.slice(0, Math.min(CONFIG.maxHistory, 10))
  while (true) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed))
      if (trimmed.length < Math.min(history.length, 10)) useConnectionStore().addLog('warn', '浏览器存储空间不足，仅保存最近部分实验；本次会话完整结果仍可导出。')
      return
    } catch {
      if (trimmed.length <= 1) {
        useConnectionStore().addLog('warn', '历史记录未能写入浏览器存储；现有缓存与本次训练结果仍保留，请导出重要结果。')
        return
      }
      trimmed.pop()
    }
  }
}

/** 新建 run 对象 */
function createRun({ modelId, modelName, datasetId, datasetName, taskType, params, testSize, randomState }) {
  return {
    id: genRunId(),
    runId: null, // 服务器 run_id，成功时设置
    modelId,
    modelName,
    datasetId,
    datasetName,
    taskType,
    params: params || {},
    testSize,
    randomState,
    status: 'running', // 'running' | 'done' | 'error'
    startedAt: Date.now(),
    finishedAt: null,
    elapsedMs: 0,
    result: null, // 完整合同响应（run_id/model/dataset/task/effective_params/metrics/artifacts/metadata）
    error: null // { code, message }
  }
}

export const useTrainingStore = defineStore('training', () => {
  // ---------- 状态 ----------
  /** 所有 run，按本地 id 索引（Vue 3 响应式支持 Map） */
  const runs = ref(new Map())
  /** 历史完成的 run（含 session 内的 + localStorage 恢复的），供 benchmark 复用 */
  const history = ref(loadHistory())

  // ---------- 计算属性 ----------
  const runList = computed(() => [...runs.value.values()])
  const activeRuns = computed(() => runList.value.filter((r) => r.status === 'running'))
  const errorRuns = computed(() => runList.value.filter((r) => r.status === 'error'))
  const finishedRuns = computed(() =>
    runList.value
      .filter((r) => r.status === 'done')
      .sort((a, b) => b.finishedAt - a.finishedAt)
  )

  // ---------- 内部 ----------
  function getRun(id) {
    return runs.value.get(id)
  }

  function finishRun(run) {
    history.value.unshift({
      schemaVersion: 2,
      id: run.id,
      runId: run.runId || run.id,
      modelId: run.modelId,
      modelName: run.modelName,
      datasetId: run.datasetId,
      datasetName: run.datasetName,
      taskType: run.taskType,
      params: run.params,
      testSize: run.testSize,
      randomState: run.randomState,
      result: run.result,
      finishedAt: run.finishedAt,
      elapsedMs: run.elapsedMs,
      status: 'done'
    })
    history.value = history.value.slice(0, CONFIG.maxHistory)
    saveHistory(history.value)
  }

  // ---------- 动作 ----------
  /**
   * 发起同步训练。返回本地 id；后端不可用时返回 null。
   * 不 await 调用方也不需要结果——状态经 run 对象流转。
   */
  function startRun({ modelId, modelName, datasetId, datasetName, taskType, params, testSize, randomState }) {
    const conn = useConnectionStore()
    if (!conn.canRunExperiments) {
      conn.addLog('error', '后端不可用或未连接，无法开始训练')
      return null
    }
    // reactive 包裹：run 的字段在异步流程中被直接赋值，
    // 原始对象无法触发 Vue 响应式（Map 只对 get 返回值代理）
    const run = reactive(createRun({ modelId, modelName, datasetId, datasetName, taskType, params, testSize, randomState }))
    runs.value.set(run.id, run)

    // 等待态走秒计时
    const timer = setInterval(() => {
      run.elapsedMs = Date.now() - run.startedAt
    }, CONFIG.elapsedTickMs)

    conn.addLog('info', `提交训练：${modelName || modelId} × ${datasetName || datasetId}；随机种子 ${randomState}`)
    const body = buildExperimentRequest({
      model: modelId,
      dataset: datasetId,
      params,
      testSize,
      randomState,
      taskType
    })

    ;(async () => {
      try {
        const resp = await api.runExperiment(body)
        run.runId = resp.run_id
        run.result = resp
        if (!run.result.metrics || typeof run.result.metrics !== 'object') run.result.metrics = {}
        run.status = 'done'
        run.finishedAt = Date.now()
        run.elapsedMs = run.finishedAt - run.startedAt
        finishRun(run)
        conn.addLog('info', `训练完成：${modelName} × ${datasetName}（${resp.run_id}）`)
      } catch (err) {
        run.status = 'error'
        run.finishedAt = Date.now()
        run.elapsedMs = run.finishedAt - run.startedAt
        run.error = { code: err.code || 'unknown', message: err.message }
        if (err.code === 'timeout') {
          run.error.message = `训练请求超时（超过 ${Math.round(CONFIG.experimentTimeoutMs / 60000)} 分钟），后端可能仍在执行`
        }
        conn.addLog('error', `[${run.error.code}] ${run.error.message}`)
      } finally {
        clearInterval(timer)
      }
    })()

    return run.id
  }

  function removeRun(id) {
    if (runs.value.get(id)?.status !== 'running') runs.value.delete(id)
  }

  function clearRuns() {
    for (const [id, run] of runs.value) {
      if (run.status !== 'running') runs.value.delete(id)
    }
    history.value = []
    saveHistory([])
  }

  /** 查找某数据集下已完成的 run（benchmark 复用历史结果，避免重复训练） */
  function doneRunsForDataset(datasetId) {
    return history.value.filter((h) => h.datasetId === datasetId && h.status === 'done')
  }

  return {
    runs,
    history,
    runList,
    activeRuns,
    errorRuns,
    finishedRuns,
    getRun,
    startRun,
    removeRun,
    clearRuns,
    doneRunsForDataset
  }
})
