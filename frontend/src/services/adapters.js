/**
 * 合同形状 ↔ 前端内部形状 的纯函数适配器。
 *
 * 合同（HTTP_API_CONTRACT.md）是前端唯一依赖，但现有视图/表单组件
 * 消费的是历史内部形状（algorithms[].taskTypes、hyperparams[] 等）。
 * 这里把合同数据转换成内部形状，让组件层零感知传输差异。
 *
 * 超参表单元数据从合同推断（用户已拍板：无静态元数据表）：
 *   default_params 的值类型 → 控件类型；
 *   parameter_descriptions[key] → 提示文字。
 * 合同不提供 min/max/options，因此推断出的控件无范围约束。
 */
import { METRICS } from '../config/metrics.js'
import { algorithmNotes } from '../config/teaching.js'

/**
 * 从合同参数值推断表单条目。
 * @param {string} key 参数键
 * @param {*} value default_params 中的默认值
 * @param {object} [descriptions] parameter_descriptions（参数键 → 帮助文字）
 * @returns {{name, label, type, default, hint?, group}|null} null 值返回 null（跳过）
 */
export function inferHyperparam(key, value, descriptions = {}) {
  if (value === undefined) return null
  let type
  let defaultValue = value
  if (value === null || typeof value === 'object') {
    type = 'json'
  } else if (typeof value === 'boolean') {
    type = 'bool'
  } else if (typeof value === 'number') {
    type = Number.isInteger(value) ? 'int' : 'float'
  } else if (typeof value === 'string') {
    type = 'text'
  } else {
    // 对象/数组等边界情况：防御性降级为文本，默认值序列化
    type = 'text'
    defaultValue = JSON.stringify(value)
  }
  const entry = { name: key, label: key, type, default: defaultValue, group: '其他参数' }
  const hint = descriptions[key]
  if (hint !== undefined && hint !== null) entry.hint = String(hint)
  return entry
}

/** 合同模型 → 内部 algorithm */
export function modelToAlgorithm(model) {
  return {
    id: model.id,
    name: model.display_name,
    taskTypes: [model.task],
    description: algorithmNotes(model.id).description,
    compatibleDatasets: model.compatible_datasets || [],
    hyperparams: Object.entries(model.default_params || {})
      .map(([key, value]) => inferHyperparam(key, value, model.parameter_descriptions))
      .filter((entry) => entry !== null)
  }
}

/** 合同数据集 → 内部 dataset */
export function datasetToDataset(ds) {
  return {
    id: ds.id,
    name: ds.display_name,
    taskType: ds.task,
    nSamples: ds.sample_count,
    nFeatures: ds.feature_count,
    nClasses: 0, // 合同无类别数信息
    description: ds.description || '',
    hasTarget: ds.has_target,
    split: ds.split || null
  }
}

/**
 * 构造 POST /api/experiments 请求体。
 * 合同：聚类必须省略 test_size（异常检测忽略该值）；未知字段会 422 → 只发送已知字段。
 */
export function buildExperimentRequest({ model, dataset, params, testSize, randomState, taskType }) {
  const body = { model, dataset, params }
  const skipTestSize = ['clustering', 'anomaly_detection'].includes(taskType)
  if (!skipTestSize && testSize !== null && testSize !== undefined) {
    body.test_size = testSize
  }
  if (randomState !== null && randomState !== undefined) {
    body.random_state = randomState
  }
  return body
}

/**
 * 组装内部注册表形状（等价于旧 WS 的 config.registry payload）。
 * server.name 设为 baseUrl —— fallback 的 server.name 是字面量，两者
 * 永不相等，connection store 的 isUsingFallback 判断机制得以保留。
 */
export function buildRegistry({ models, datasets, baseUrl }) {
  const algorithms = (models || []).map(modelToAlgorithm)
  const taskTypes = [...new Set(algorithms.flatMap((a) => a.taskTypes))]
  return {
    server: { name: baseUrl },
    taskTypes,
    algorithms,
    datasets: (datasets || []).map(datasetToDataset),
    metrics: METRICS
  }
}

/** Compare only results evaluated with the same dataset and split protocol. */
export function areComparable(a, b) {
  if (!a || !b || a.datasetId !== b.datasetId || a.taskType !== b.taskType || a.randomState !== b.randomState) return false
  if (['classification', 'regression'].includes(a.taskType) && a.testSize !== b.testSize) return false
  const left = a.result?.metadata || {}
  const right = b.result?.metadata || {}
  return left.evaluation_protocol === right.evaluation_protocol &&
    left.train_sample_cap === right.train_sample_cap && left.test_sample_count === right.test_sample_count
}
