import { httpClient } from './httpClient.js'
import { CONFIG } from '../config/index.js'
export const dataLibraryApi = {
  list: () => httpClient.get('/api/data-library'),
  detail: id => httpClient.get(`/api/data-library/${encodeURIComponent(id)}`),
  upload: (filename, content) => httpClient.post('/api/data-library/uploads', { filename, content }),
  compatibility: selection => httpClient.post('/api/data-library/compatibility', selection),
  run: body => httpClient.post('/api/data-library/experiments', body, { timeoutMs: CONFIG.experimentTimeoutMs })
}
