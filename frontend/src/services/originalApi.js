import { httpClient } from './httpClient.js'

export function createOriginalApi(client) {
  return {
    listExperiments: () => client.get('/api/original-experiments'),
    listDatasets: () => client.get('/api/original-datasets'),
    listRuns: () => client.get('/api/original-runs'),
    getRun: runId => client.get(`/api/original-runs/${encodeURIComponent(runId)}`),
    startRun: experimentId => client.post('/api/original-runs', { experiment_id: experimentId }),
    assetUrl: path => {
      if (!path?.startsWith('/api/')) return ''
      return `${client.baseUrl || ''}${path}`
    },
    downloadUrl: path => {
      if (!path?.startsWith('/api/')) return ''
      return `${client.baseUrl || ''}${path}${path.includes('?') ? '&' : '?'}download=true`
    }
  }
}

export const originalApi = createOriginalApi(httpClient)
