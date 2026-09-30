/**
 * Mock 后端服务器 —— 按 HTTP_API_CONTRACT.md 实现的替身后端（node:http，零依赖）。
 * 前端开发/演示期间可完全依赖它：模型/数据集发现、健康检查、同步实验。
 *
 * 启动：
 *   node mock/server.js                 # 默认 http://127.0.0.1:8765
 *   node mock/server.js --port 9000     # 换端口
 *   node mock/server.js --no-ml         # 模拟 ml_core 缺失（health available=false）
 *   node mock/server.js --delay 0       # 覆盖实验响应延迟（默认 1000~3000ms）
 *
 * 与真实后端的差异：训练是模拟的（性能表加噪声，以 run_id 为种子可复现）。
 * 后端同学按 HTTP_API_CONTRACT.md 实现真实 FastAPI 服务即可无缝替换。
 */
import http from 'node:http'
import crypto from 'node:crypto'
import { MODELS, DATASETS, PERFORMANCE } from './registry.js'

// ---------- 命令行参数 ----------
const args = process.argv.slice(2)
const portIdx = args.indexOf('--port')
const PORT = portIdx >= 0 ? parseInt(args[portIdx + 1], 10) : 8765
const NO_ML = args.includes('--no-ml')
const delayIdx = args.indexOf('--delay')
const DELAY_OVERRIDE = delayIdx >= 0 ? parseInt(args[delayIdx + 1], 10) : null

const ALLOWED_ORIGINS = new Set(['http://localhost:5173', 'http://127.0.0.1:5173'])
const ALLOWED_FIELDS = new Set(['model', 'dataset', 'params', 'test_size', 'random_state'])

// ---------- 工具 ----------
/** mulberry32 种子随机数：同 run_id 结果可复现 */
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

function round4(v) {
  return Math.round(v * 10000) / 10000
}

const log = (...msg) => console.log('[MOCK]', ...msg)

// ---------- HTTP 工具 ----------
function sendJson(res, status, body) {
  const data = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(data)
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = ''
    req.on('data', (chunk) => {
      raw += chunk
      if (raw.length > 1e6) reject(new Error('body too large'))
    })
    req.on('end', () => resolve(raw))
    req.on('error', reject)
  })
}

function withCors(req, res) {
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  const origin = req.headers.origin
  if (origin && ALLOWED_ORIGINS.has(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin)
  }
}

// ---------- 模拟实验 ----------
/**
 * 生成实验 metrics。键集合与真实后端各任务适配器输出一致：
 * 分类 8 键（accuracy/precision/recall/f1/tn/fp/fn/tp）、回归 4 键、
 * 聚类分 kmeans（inertia/ARI/n_clusters）与 dbscan（n_clusters/noise_points/ARI）两套、
 * 异常 5 键（true/detected/anomaly_recall/anomaly_precision/anomaly_f1）。
 * 全部为有限数值（合同要求）。
 */
function makeMetrics(model, dataset, rand) {
  const perf = PERFORMANCE[`${model.id}@${dataset.id}`]
  const metrics = {}

  if (dataset.task === 'classification') {
    const noise = (rand() - 0.5) * 0.03 // ±1.5% 噪声
    metrics.accuracy = round4(Math.min(1, (perf?.accuracy ?? 0.9) + noise))
    metrics.precision = round4(Math.min(1, (perf?.precision ?? 0.9) + noise))
    metrics.recall = round4(Math.min(1, (perf?.recall ?? 0.9) + noise))
    metrics.f1 = round4(Math.min(1, (perf?.f1 ?? 0.9) + noise))
    metrics.tn = Math.round(perf?.tn ?? 70)
    metrics.fp = Math.round(perf?.fp ?? 3)
    metrics.fn = Math.round(perf?.fn ?? 2)
    metrics.tp = Math.round(perf?.tp ?? 39)
  } else if (dataset.task === 'regression') {
    metrics.mse = round4((perf?.mse ?? 20) * (1 + (rand() - 0.5) * 0.1))
    metrics.rmse = round4(Math.sqrt(metrics.mse))
    metrics.mae = round4((perf?.mae ?? 3) * (1 + (rand() - 0.5) * 0.1))
    metrics.r2 = round4(Math.min(0.99, (perf?.r2 ?? 0.7) * (1 + (rand() - 0.5) * 0.06)))
  } else if (dataset.task === 'clustering') {
    if (model.id === 'dbscan.optimized') {
      metrics.n_clusters = Math.round(perf?.n_clusters ?? 3)
      metrics.noise_points = Math.round(perf?.noise_points ?? 12)
      metrics.adjusted_rand_index = round4(Math.min(1, Math.max(-1, (perf?.adjusted_rand_index ?? 0.65) + (rand() - 0.5) * 0.08)))
    } else {
      metrics.inertia = round4((perf?.inertia ?? 180) * (1 + (rand() - 0.5) * 0.1))
      metrics.adjusted_rand_index = round4(Math.min(1, Math.max(-1, (perf?.adjusted_rand_index ?? 0.72) + (rand() - 0.5) * 0.06)))
      metrics.n_clusters = Math.round(perf?.n_clusters ?? 3)
    }
  } else if (dataset.task === 'anomaly_detection') {
    metrics.true_anomalies = Math.round(perf?.true_anomalies ?? 176)
    metrics.detected_anomalies = Math.round(perf?.detected_anomalies ?? 190)
    metrics.anomaly_recall = round4(Math.min(1, Math.max(0, (perf?.anomaly_recall ?? 0.8) + (rand() - 0.5) * 0.06)))
    metrics.anomaly_precision = round4(Math.min(1, Math.max(0, (perf?.anomaly_precision ?? 0.75) + (rand() - 0.5) * 0.06)))
    metrics.anomaly_f1 = round4(Math.min(1, Math.max(0, (perf?.anomaly_f1 ?? 0.78) + (rand() - 0.5) * 0.06)))
  }
  return metrics
}

// ---------- 路由 ----------
async function handleRequest(req, res) {
  withCors(req, res)
  const url = new URL(req.url, `http://${req.headers.host || '127.0.0.1'}`)
  const path = url.pathname

  if (req.method === 'OPTIONS') {
    res.writeHead(204)
    res.end()
    return
  }

  // 健康检查：服务可用时恒 200；available 取决于 --no-ml
  if (path === '/api/health' && req.method === 'GET') {
    sendJson(res, 200, {
      status: 'ok',
      ml_backend: {
        available: false, // Reference demo only: never enable real training in the application.
        package: 'offline_demo',
        detail: NO_ML ? 'Mock 模式：模拟 ml_core 缺失' : 'Mock ML 包已连接（模拟数据）'
      }
    })
    return
  }

  if (path === '/api/models' && req.method === 'GET') {
    sendJson(res, 200, MODELS)
    return
  }

  if (path === '/api/datasets' && req.method === 'GET') {
    sendJson(res, 200, DATASETS)
    return
  }

  if (path === '/api/experiments' && req.method === 'POST') {
    await handleExperiment(req, res)
    return
  }

  sendJson(res, 404, { detail: { code: 'not_found', message: '接口不存在' } })
}

async function handleExperiment(req, res) {
  let raw
  try {
    raw = await readBody(req)
  } catch (err) {
    sendJson(res, 413, { detail: { code: 'payload_too_large', message: '请求体过大' } })
    return
  }

  let body
  try {
    body = JSON.parse(raw)
  } catch (err) {
    sendJson(res, 422, {
      detail: [{ loc: ['body'], msg: 'Invalid JSON', type: 'value_error.jsondecode' }]
    })
    return
  }

  // 未知顶层字段 → 422（合同：未声明字段返回 422）
  const extraFields = Object.keys(body).filter((k) => !ALLOWED_FIELDS.has(k))
  if (extraFields.length > 0) {
    sendJson(res, 422, {
      detail: extraFields.map((field) => ({
        loc: ['body', field],
        msg: 'extra fields not permitted',
        type: 'value_error.extra'
      }))
    })
    return
  }

  // 未知模型/数据集 → 400
  const model = MODELS.find((m) => m.id === body.model)
  const dataset = DATASETS.find((d) => d.id === body.dataset)
  if (!model) {
    sendJson(res, 400, { detail: { code: 'ml_request_rejected', message: `未知模型: ${body.model}` } })
    return
  }
  if (!dataset) {
    sendJson(res, 400, { detail: { code: 'ml_request_rejected', message: `未知数据集: ${body.dataset}` } })
    return
  }

  // 不兼容组合 → 400
  if (!model.compatible_datasets.includes(dataset.id)) {
    sendJson(res, 400, {
      detail: { code: 'ml_request_rejected', message: `模型与数据集不兼容: ${model.id} × ${dataset.id}` }
    })
    return
  }

  // test_size 校验（合同：聚类必须省略；提供时必须在 0 与 1 之间）
  if (body.test_size !== undefined && body.test_size !== null) {
    if (dataset.task === 'clustering') {
      sendJson(res, 400, {
        detail: { code: 'ml_request_rejected', message: '聚类实验不得指定 test_size' }
      })
      return
    }
    if (typeof body.test_size !== 'number' || !Number.isFinite(body.test_size) || body.test_size <= 0 || body.test_size >= 1) {
      sendJson(res, 422, {
        detail: [
          {
            loc: ['body', 'test_size'],
            msg: 'test_size 必须在 0 与 1 之间',
            type: 'value_error.number.not_gt'
          }
        ]
      })
      return
    }
  }

  if (body.params !== undefined && (typeof body.params !== 'object' || body.params === null || Array.isArray(body.params))) {
    sendJson(res, 422, {
      detail: [{ loc: ['body', 'params'], msg: 'params 必须是对象', type: 'type_error.dict' }]
    })
    return
  }

  // 模拟同步训练：延迟后返回最终结果
  const delay = DELAY_OVERRIDE ?? 1000 + Math.random() * 2000
  const startedAt = Date.now()
  await new Promise((resolve) => setTimeout(resolve, delay))
  const elapsed = Date.now() - startedAt

  const runId = crypto.randomUUID()
  const rand = mulberry32(hashStr(runId))
  const perf = PERFORMANCE[`${model.id}@${dataset.id}`]

  const resp = {
    run_id: runId,
    model: model.id,
    dataset: dataset.id,
    task: dataset.task,
    effective_params: body.params || {},
    metrics: makeMetrics(model, dataset, rand),
    artifacts: [],
    metadata: {
      sample_count: dataset.sample_count,
      feature_count: dataset.feature_count,
      elapsed_ms: elapsed,
      train_ms: perf?.train_ms ?? 50,
      eval_ms: perf?.eval_ms ?? 30
    }
  }
  log(`实验完成 ${model.id} × ${dataset.id} → ${runId}（${elapsed}ms）`)
  sendJson(res, 200, resp)
}

// ---------- 服务器 ----------
const server = http.createServer((req, res) => {
  handleRequest(req, res).catch((err) => {
    log('请求处理出错:', err.message)
    if (!res.headersSent) {
      sendJson(res, 500, { detail: { code: 'internal_error', message: '服务器内部错误' } })
    } else {
      res.end()
    }
  })
})

server.listen(PORT, '127.0.0.1', () => {
  log(`Mock 后端已启动: http://127.0.0.1:${PORT}${NO_ML ? '（--no-ml 模式）' : ''}`)
})
