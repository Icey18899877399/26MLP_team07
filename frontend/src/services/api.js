/**
 * 合同端点层（HTTP_API_CONTRACT.md 是唯一依据）。
 * 全项目只有这个文件出现 /api/* 路径与请求形状。
 */
import { httpClient } from './httpClient.js'
import { CONFIG } from '../config/index.js'

export const api = {
  /** GET /api/health → { status, ml_backend: { available, package, detail } } */
  checkHealth: () => httpClient.get('/api/health'),

  /** GET /api/models → 合同模型数组 */
  fetchModels: () => httpClient.get('/api/models'),

  /** GET /api/datasets → 合同数据集数组 */
  fetchDatasets: () => httpClient.get('/api/datasets'),

  /** POST /api/experiments → 同步训练，返回 { run_id, metrics, ... } */
  runExperiment: (body) =>
    httpClient.post('/api/experiments', body, { timeoutMs: CONFIG.experimentTimeoutMs })
}
