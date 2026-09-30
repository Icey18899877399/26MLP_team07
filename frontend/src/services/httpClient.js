import { algorithmNotes, metricDefinition } from '../config/teaching.js'
/** Native HTTP transport; local events preserve the Vue store interface. */
export function makeRegistry(models, datasets) {
  const hyperparam = (name, value, hints) => {
    const field = { name, label: name, default: value, hint: hints[name], group: '模型参数' }
    if (name === 'init') return { ...field, type: 'choice', options: ['k-means++', 'random'] }
    if (value === null || Array.isArray(value) || typeof value === 'object') return { ...field, type: 'json' }
    if (typeof value === 'boolean') return { ...field, type: 'bool' }
    if (typeof value === 'number') {
      const integer = Number.isInteger(value) && !/rate|tol|alpha|lambda|gamma|weight|ratio|contamination|^C$/.test(name)
      return { ...field, type: integer ? 'int' : 'float', step: integer ? 1 : 0.001 }
    }
    return { ...field, type: 'text' }
  }
  const metric = (id, name, higherIsBetter = true) => ({ id, name, type: 'scalar', higherIsBetter })
  return {
    server: { name: 'ML Core 真实实验服务', version: '0.1.0', protocolVersion: 1 },
    taskTypes: [...new Set(models.map(m => m.task))],
    algorithms: models.map(m => ({
      id: m.id, model: m.id, name: m.display_name,
      family: m.id.split('.')[0], taskTypes: [m.task], compatibleDatasets: m.compatible_datasets,
      ...algorithmNotes(m.id),
      hyperparams: Object.entries(m.default_params || {}).map(([key, value]) => hyperparam(key, value, m.parameter_descriptions || {}))
    })),
    datasets: datasets.map(d => ({
      id: d.id, name: d.display_name, taskType: d.task, nSamples: d.sample_count, nFeatures: d.feature_count, hasTarget: d.has_target,
      split: { defaultTestRatio: 0.2, defaultSeed: 42, stratifySupported: d.task === 'classification' }
    })),
    metrics: {
      classification: [
        metric('accuracy', '准确率'), metric('precision', '精确率'),
        metric('recall', '召回率'), metric('f1', 'F1'),
        { id: 'confusion_matrix', name: '混淆矩阵', type: 'matrix', chart: 'heatmap' }
      ],
      clustering: [metric('inertia', '簇内平方和', false), metric('adjusted_rand_index', '调整兰德指数'), metric('n_clusters', '簇数')],
      regression: ['mse', 'rmse', 'mae', 'r2'].map(metricDefinition),
      anomaly_detection: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'].map(metricDefinition)
    }
  }
}
export function experimentRequest(payload, registry) {
  const algorithm = registry.algorithms.find(a => a.id === payload.algorithmId)
  if (!algorithm) throw new Error('算法不在当前服务的可用列表中')
  if (algorithm.compatibleDatasets?.length && !algorithm.compatibleDatasets.includes(payload.datasetId)) {
    throw new Error('该算法不支持所选数据集')
  }
  const params = { ...payload.hyperparams }
  if (params.class_weight === 'none') params.class_weight = null
  return {
    model: algorithm.id, dataset: payload.datasetId,
    params, random_state: payload.split?.seed ?? 42,
    test_size: algorithm.taskTypes.includes('clustering') ? null : (payload.split?.testRatio ?? 0.2)
  }
}
export function resultPayload(request, result, elapsedMs) {
  const metrics = { ...result.metrics }
  if (['tn', 'fp', 'fn', 'tp'].every(k => Number.isFinite(metrics[k]))) {
    metrics.confusion_matrix = [[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]]
    metrics.class_names = result.metadata?.class_names || ['类别 0', '类别 1']
  }
  return {
    runId: request.runId, algorithmId: request.algorithmId, datasetId: request.datasetId,
    serverRunId: result.run_id, task: result.task,
    status: 'done', hyperparams: result.effective_params ?? request.hyperparams,
    metrics, diagnostics: result.metadata || {}, artifacts: result.artifacts,
    visualizations: result.metadata?.visualizations || [], durations: { train_ms: elapsedMs }
  }
}
export class HttpClient {
  constructor({ fetchImpl = (...args) => fetch(...args), defaultUrl = '/api' } = {}) {
    this.fetch = fetchImpl
    this.url = defaultUrl
    this.status = 'offline'
    this.onMessage = null
    this.onStatusChange = null
    this.registry = null
    this.generation = 0
    this.pending = new Set()
  }
  get connected() { return this.status === 'connected' }
  emit(type, payload) { this.onMessage?.({ type, payload }) }
  setStatus(status) { this.status = status; this.onStatusChange?.(status) }
  async request(path, options = {}) {
    const response = await this.fetch(this.url.replace(/\/$/, '') + path, {
      ...options, headers: { 'Content-Type': 'application/json', ...options.headers }
    })
    let data
    try { data = await response.json() } catch { throw new Error('服务未返回 JSON，请检查 API 地址') }
    if (!response.ok) {
      const detail = data.detail
      throw new Error(Array.isArray(detail) ? detail.map(e => e.loc.join('.') + ': ' + e.msg).join('; ')
        : detail?.message || detail || 'HTTP ' + response.status)
    }
    return data
  }
  async connect(url) {
    const generation = ++this.generation
    if (url) this.url = url
    this.setStatus('connecting')
    try {
      const health = await this.request('/health')
      if (!health.ml_backend?.available) throw new Error(health.ml_backend?.detail || '模型服务不可用')
      const [models, datasets] = await Promise.all([this.request('/models'), this.request('/datasets')])
      if (generation !== this.generation) return
      this.registry = makeRegistry(models, datasets)
      this.emit('config.registry', this.registry)
      this.setStatus('connected')
    } catch (error) {
      if (generation !== this.generation) return
      this.setStatus('offline')
      this.emit('error', { message: error.message })
    }
  }
  disconnect() { ++this.generation; this.setStatus('offline') }
  send(message) {
    if (!this.connected) return false
    const { type, payload = {} } = message
    if (type === 'config.get') this.emit('config.registry', this.registry)
    if (type === 'dataset.load') {
      const dataset = this.registry.datasets.find(d => d.id === payload.datasetId)
      this.emit(dataset ? 'dataset.info' : 'dataset.error', dataset || { id: payload.datasetId })
    }
    if (type === 'training.start') void this.run(payload)
    if (type === 'training.subscribe' && !this.pending.has(payload.runId)) {
      this.emit('training.error', { runId: payload.runId, code: 'NOT_FOUND', message: 'HTTP 服务不保存跨页面训练任务，请重新运行。' })
    }
    return true
  }
  async run(payload) {
    if (this.pending.has(payload.runId)) return
    this.pending.add(payload.runId)
    const started = Date.now()
    try {
      const body = experimentRequest(payload, this.registry)
      this.emit('training.accepted', { runId: payload.runId })
      const result = await this.request('/experiments', { method: 'POST', body: JSON.stringify(body) })
      this.emit('training.result', resultPayload(payload, result, Date.now() - started))
    } catch (error) {
      this.emit('training.error', { runId: payload.runId, message: error.message })
    } finally { this.pending.delete(payload.runId) }
  }
}
