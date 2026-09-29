/**
 * Mock 后端服务器 —— 用 Node 模拟 Python 后端，协议与 docs/PROTOCOL.md 完全一致。
 * 前端开发/演示期间可完全依赖它：算法注册、数据集详情、训练全流程（含断线继续训练）。
 *
 * 启动：
 *   node mock/server.js                # 默认 127.0.0.1:8765
 *   node mock/server.js --port 9000    # 换端口
 *   node mock/server.js --fast         # 训练只跑 5 个 epoch（快速演示）
 *
 * 与真实后端的差异：训练过程是模拟的（假 loss 曲线 + 性能表加噪声），
 * 后端同学照 PROTOCOL.md 实现真实训练即可无缝替换。
 */
import { WebSocketServer } from 'ws'
import { ALGORITHMS, DATASETS, METRICS, PERFORMANCE, LOSS_CURVES } from './registry.js'

// ---------- 命令行参数 ----------
const args = process.argv.slice(2)
const portIdx = args.indexOf('--port')
const PORT = portIdx >= 0 ? parseInt(args[portIdx + 1], 10) : 8765
const FAST = args.includes('--fast')
const EPOCH_INTERVAL = FAST ? 60 : 300
const TOTAL_EPOCHS_RANGE = FAST ? [5, 5] : [30, 90]

// ---------- 工具 ----------
/** mulberry32 种子随机数：同 runId 结果可复现 */
function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hashStr(str) {
  let h = 0
  for (let i = 0; i < str.length; i++) {
    h = (Math.imul(31, h) + str.charCodeAt(i)) | 0
  }
  return h >>> 0
}

const log = (...msg) => console.log('[MOCK]', ...msg)

/** 发送：JSON 序列化失败时保护 */
function send(ws, msg) {
  try {
    ws.send(JSON.stringify(msg))
  } catch (err) {
    log('发送失败:', err.message)
  }
}

// ---------- 全局运行状态（训练跨连接持续，见协议"断线继续训练"） ----------
const runs = new Map() // runId -> runState

function buildRegistryPayload() {
  return {
    server: { name: 'mock-backend', version: '0.1.0', protocolVersion: 1 },
    taskTypes: ['classification', 'regression', 'clustering'],
    algorithms: ALGORITHMS,
    datasets: DATASETS.map((d) => ({
      id: d.id,
      name: d.name,
      taskType: d.taskType,
      nSamples: d.nSamples,
      nFeatures: d.nFeatures,
      nClasses: d.nClasses,
      description: d.description,
      split: d.split
    })),
    metrics: METRICS
  }
}

// ---------- 模拟训练 ----------
function makeProgress(run) {
  const curve = LOSS_CURVES[run.curve] || LOSS_CURVES.fast
  const base = curve.baseLoss * Math.exp(-curve.decay * run.epoch)
  let loss
  if (curve.stepwise) {
    // 决策树：每 5 个 epoch 阶梯式下降
    loss = curve.baseLoss * Math.exp(-curve.decay * Math.floor(run.epoch / 5) * 5)
  } else if (curve.floor) {
    loss = Math.max(curve.floor, base)
  } else {
    loss = base
  }
  loss += (run.rand() - 0.5) * loss * 0.1 // ±5% 噪声

  const metrics = { loss: round4(loss) }
  // 分类任务附带验证集 loss 与训练准确率
  if (run.taskType === 'classification') {
    const acc = Math.min(1, 0.2 + 0.8 * (1 - Math.exp(-curve.decay * run.epoch * 2)))
    metrics.val_loss = round4(loss * (1.05 + run.rand() * 0.15))
    metrics.accuracy = round4(acc + (run.rand() - 0.5) * 0.03)
  }
  return metrics
}

function round4(v) {
  return Math.round(v * 10000) / 10000
}

/** 生成混淆矩阵（近似对角线分布） */
function makeConfusionMatrix(dataset, accuracy, rand) {
  const n = dataset.nClasses
  const testRatio = dataset.split.defaultTestRatio
  const total = Math.round(dataset.nSamples * testRatio)
  const perClass = dataset.classDistribution.map((c) => Math.round((c / dataset.nSamples) * total))

  const matrix = Array.from({ length: n }, () => Array(n).fill(0))
  for (let c = 0; c < n; c++) {
    const correct = Math.round(perClass[c] * accuracy)
    matrix[c][c] = correct
    let errors = perClass[c] - correct
    for (let j = 0; j < n && errors > 0; j++) {
      if (j === c) continue
      const take = j === n - 1 || (c + 1) % n === j ? errors : Math.floor(errors * rand())
      const assigned = Math.min(take, errors)
      matrix[c][j] += assigned
      errors -= assigned
    }
  }
  return matrix
}

/** 生成 ROC 曲线数据（近似形状） */
function makeRocCurves(dataset, rand) {
  const n = dataset.nClasses
  const curves = []
  for (let c = 0; c < n; c++) {
    const auc = 0.85 + rand() * 0.14 // 0.85~0.99
    const fpr = [0, rand() * 0.2, 1]
    const tpr = [0, 0.85 + rand() * 0.1, 1]
    curves.push({ label: dataset.classNames[c], fpr, tpr, auc: round4(auc) })
  }
  const avgAuc = round4(curves.reduce((s, c) => s + c.auc, 0) / n)
  return { auc: avgAuc, curves }
}

function finishRun(run) {
  const dataset = DATASETS.find((d) => d.id === run.datasetId)
  const perf = PERFORMANCE[`${run.algoId}@${run.datasetId}`]
  const metrics = {}

  if (run.taskType === 'classification') {
    const noise = (run.rand() - 0.5) * 0.03 // ±1.5% 噪声
    metrics.accuracy = round4(Math.min(1, (perf?.accuracy ?? 0.9) + noise))
    metrics.precision = round4(Math.min(1, (perf?.precision ?? 0.9) + noise))
    metrics.recall = round4(Math.min(1, (perf?.recall ?? 0.9) + noise))
    metrics.f1 = round4(Math.min(1, (perf?.f1 ?? 0.9) + noise))
    metrics.confusion_matrix = makeConfusionMatrix(dataset, metrics.accuracy, run.rand)
    metrics.class_names = dataset.classNames
    metrics.roc_auc = makeRocCurves(dataset, run.rand)
  } else if (run.taskType === 'regression') {
    metrics.mse = round4((perf?.mse ?? 20) * (1 + (run.rand() - 0.5) * 0.1))
    metrics.rmse = round4(Math.sqrt(metrics.mse))
    metrics.mae = round4((perf?.mae ?? 3) * (1 + (run.rand() - 0.5) * 0.1))
    metrics.r2 = round4(Math.min(0.99, (perf?.r2 ?? 0.7) * (1 + (run.rand() - 0.5) * 0.06)))
  } else if (run.taskType === 'clustering') {
    metrics.silhouette = round4(Math.max(0, (perf?.silhouette ?? 0.7) + (run.rand() - 0.5) * 0.08))
    metrics.davies_bouldin = round4((perf?.davies_bouldin ?? 0.6) * (1 + (run.rand() - 0.5) * 0.2))
    metrics.inertia = Math.round((perf?.inertia ?? 4000) * (1 + (run.rand() - 0.5) * 0.1))
  }

  metrics.durations = { train_ms: perf?.train_ms ?? 50, eval_ms: perf?.eval_ms ?? 30 }
  return metrics
}

function startTraining(payload) {
  const algo = ALGORITHMS.find((a) => a.id === payload.algorithmId)
  const dataset = DATASETS.find((d) => d.id === payload.datasetId)
  const perf = PERFORMANCE[`${payload.algorithmId}@${payload.datasetId}`]

  const [minEp, maxEp] = TOTAL_EPOCHS_RANGE
  const run = {
    id: payload.runId,
    algoId: payload.algorithmId,
    datasetId: payload.datasetId,
    taskType: dataset.taskType,
    hyperparams: payload.hyperparams || {},
    split: payload.split || null,
    curve: perf?.curve || 'fast',
    epoch: 0,
    totalEpochs: minEp + Math.floor(hashStr(payload.runId) % (maxEp - minEp + 1)),
    rand: mulberry32(hashStr(payload.runId)),
    status: 'running',
    subscribers: new Set(),
    interval: null,
    result: null,
    startedAt: Date.now()
  }
  runs.set(run.id, run)

  run.interval = setInterval(() => {
    run.epoch += 1
    const progress = {
      type: 'training.progress',
      payload: {
        runId: run.id,
        epoch: run.epoch,
        totalEpochs: run.totalEpochs,
        elapsedMs: Date.now() - run.startedAt,
        stage: 'train',
        metrics: makeProgress(run)
      }
    }
    for (const ws of run.subscribers) send(ws, progress)
    log(`progress ${run.id} epoch ${run.epoch}/${run.totalEpochs}`)

    if (run.epoch >= run.totalEpochs) {
      clearInterval(run.interval)
      run.interval = null
      run.status = 'done'
      run.result = {
        runId: run.id,
        algorithmId: run.algoId,
        datasetId: run.datasetId,
        status: 'done',
        hyperparams: run.hyperparams,
        metrics: finishRun(run)
      }
      const resultMsg = { type: 'training.result', payload: run.result }
      for (const ws of run.subscribers) send(ws, resultMsg)
      log(`result ${run.id} 完成`)
    }
  }, EPOCH_INTERVAL)
}

function handleMessage(ws, msg) {
  const { type, id, payload = {} } = msg
  log('←', type, payload.runId || payload.datasetId || '')

  switch (type) {
    case 'ping': {
      send(ws, { type: 'pong', replyTo: id, payload: { t: payload.t } })
      break
    }

    case 'config.get': {
      send(ws, { type: 'config.registry', replyTo: id, payload: buildRegistryPayload() })
      break
    }

    case 'dataset.load': {
      const ds = DATASETS.find((d) => d.id === payload.datasetId)
      if (!ds) {
        send(ws, {
          type: 'dataset.error',
          replyTo: id,
          payload: { id: payload.datasetId, code: 'UNKNOWN_DATASET', message: `数据集 ${payload.datasetId} 不存在` }
        })
        break
      }
      const info = {
        ...ds,
        sampleRows: ds.sampleRows.slice(0, 50), // 协议约定最多 50 行
        splitPreview: {
          train: Math.round(ds.nSamples * (1 - ds.split.defaultTestRatio)),
          test: Math.round(ds.nSamples * ds.split.defaultTestRatio),
          seed: ds.split.defaultSeed
        }
      }
      send(ws, { type: 'dataset.info', replyTo: id, payload: info })
      break
    }

    case 'training.start': {
      const algoOk = ALGORITHMS.some((a) => a.id === payload.algorithmId)
      const dsOk = DATASETS.some((d) => d.id === payload.datasetId)
      if (!algoOk || !dsOk) {
        send(ws, {
          type: 'training.error',
          replyTo: id,
          payload: {
            runId: payload.runId,
            code: !algoOk ? 'UNKNOWN_ALGORITHM' : 'UNKNOWN_DATASET',
            message: `未知的${!algoOk ? '算法' : '数据集'}: ${!algoOk ? payload.algorithmId : payload.datasetId}`
          }
        })
        break
      }
      // 同名 run 重复 start → 复用已有 run（幂等）
      const existing = runs.get(payload.runId)
      if (existing) {
        existing.subscribers.add(ws)
        send(ws, { type: 'training.accepted', replyTo: id, payload: { runId: payload.runId } })
        break
      }
      startTraining(payload)
      runs.get(payload.runId).subscribers.add(ws)
      send(ws, { type: 'training.accepted', replyTo: id, payload: { runId: payload.runId } })
      break
    }

    case 'training.cancel': {
      const run = runs.get(payload.runId)
      if (run && run.interval) {
        clearInterval(run.interval)
        run.interval = null
        run.status = 'cancelled'
        for (const sub of run.subscribers) {
          send(sub, { type: 'training.cancelled', replyTo: id, payload: { runId: payload.runId } })
        }
      }
      break
    }

    case 'training.subscribe': {
      const run = runs.get(payload.runId)
      if (!run) {
        send(ws, {
          type: 'training.error',
          replyTo: id,
          payload: { runId: payload.runId, code: 'NOT_FOUND', message: '训练不存在（后端已重启，运行已丢失）' }
        })
        break
      }
      if (run.status === 'running') {
        run.subscribers.add(ws)
        send(ws, { type: 'training.accepted', replyTo: id, payload: { runId: payload.runId } })
      } else if (run.status === 'done') {
        send(ws, { type: 'training.result', replyTo: id, payload: run.result })
      } else if (run.status === 'cancelled') {
        send(ws, { type: 'training.cancelled', replyTo: id, payload: { runId: payload.runId } })
      }
      break
    }

    default: {
      // 协议前向兼容：忽略未知消息
      log('未知消息类型，忽略:', type)
    }
  }
}

// ---------- 服务器 ----------
const wss = new WebSocketServer({ port: PORT, host: '127.0.0.1' })

wss.on('connection', (ws) => {
  log(`客户端连接（当前 ${wss.clients.size} 个）`)

  // 连接即推送注册表（协议：无需客户端请求）
  send(ws, { type: 'config.registry', payload: buildRegistryPayload() })

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString())
      handleMessage(ws, msg)
    } catch (err) {
      log('消息处理出错:', err.message)
    }
  })

  ws.on('close', () => {
    // 从所有 run 的订阅者中移除该连接；训练本身继续运行（协议要求）
    for (const run of runs.values()) {
      run.subscribers.delete(ws)
    }
    log(`客户端断开（剩余 ${wss.clients.size} 个）`)
  })
})

log(`Mock 后端已启动: ws://127.0.0.1:${PORT}${FAST ? '（fast 模式）' : ''}`)
