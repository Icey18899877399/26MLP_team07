import { httpClient } from './httpClient.js'
import { CONFIG } from '../config/index.js'

export const interactiveApi = {
  list: () => httpClient.get('/api/interactive-experiments'),
  run: body => httpClient.post('/api/interactive-experiments', body, { timeoutMs: CONFIG.experimentTimeoutMs })
}
