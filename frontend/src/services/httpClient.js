/**
 * HTTP 客户端封装（零依赖，原生 fetch）。
 *
 * 职责：
 * - 持有可变 baseUrl（由 connection store 在 init/setBaseUrl 时设置）
 * - get/post JSON 请求，AbortController 超时
 * - 错误归一化：所有失败统一抛 HttpError { status, code, message, detail }
 *
 * 错误归一化规则（对应 HTTP_API_CONTRACT.md 的错误约定）：
 * - detail 为对象且含 code/message → 业务错误（400 ml_request_rejected、
 *   502 ml_execution_failed、503 ml_backend_unavailable）
 * - detail 为数组 → FastAPI 422 校验错误，拼出可读信息，code 统一
 *   validation_error
 * - 请求超时 → code 'timeout'；网络错误 → code 'network_error'
 * - 其余非 2xx → code 'http_<status>'
 */
import { CONFIG } from '../config/index.js'

export class HttpError extends Error {
  constructor({ status, code, message, detail }) {
    super(message)
    this.name = 'HttpError'
    this.status = status
    this.code = code
    this.detail = detail
  }
}

function normalizeErrorBody(body, status) {
  // FastAPI 422：detail 为校验错误数组
  if (Array.isArray(body?.detail)) {
    const message = body.detail
      .map((item) => `${item.loc?.join('.') || 'body'}: ${item.msg}`)
      .join('；')
    return new HttpError({ status, code: 'validation_error', message, detail: body.detail })
  }
  // 合同业务错误：{ detail: { code, message } }
  if (body?.detail && typeof body.detail === 'object' && body.detail.code) {
    return new HttpError({
      status,
      code: body.detail.code,
      message: body.detail.message || body.detail.code,
      detail: body.detail
    })
  }
  return new HttpError({ status, code: `http_${status}`, message: `请求失败（HTTP ${status}）`, detail: body })
}

export class HttpClient {
  constructor() {
    this.baseUrl = CONFIG.defaultHttpUrl
  }

  /** 设置基地址（自动去除末尾 '/'） */
  setBaseUrl(url) {
    this.baseUrl = String(url || '').replace(/\/+$/, '')
  }

  get(path, options = {}) {
    return this._request('GET', path, null, options)
  }

  post(path, body, options = {}) {
    return this._request('POST', path, body, options)
  }

  async _request(method, path, body, { timeoutMs = CONFIG.requestTimeoutMs } = {}) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    let res
    try {
      res = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body === null ? undefined : JSON.stringify(body),
        signal: controller.signal
      })
      let data
      try { data = await res.json() } catch (err) {
        if (err.name === 'AbortError') throw err
        throw new HttpError({status: res.status, code: 'invalid_response', message: '后端返回了无效 JSON，请检查服务地址与代理配置'})
      }
      if (!res.ok) throw normalizeErrorBody(data, res.status)
      if (data === null || typeof data !== 'object') throw new HttpError({status: res.status, code: 'invalid_response', message: '后端响应格式不符合接口约定'})
      return data
    } catch (err) {
      if (err instanceof HttpError) throw err
      if (err.name === 'AbortError') {
        throw new HttpError({ status: null, code: 'timeout', message: '请求超时' })
      }
      throw new HttpError({ status: null, code: 'network_error', message: '无法连接后端服务器' })
    } finally {
      clearTimeout(timer)
    }

  }
}

/** 全局单例：整个应用共用一个 HTTP 客户端 */
export const httpClient = new HttpClient()
