import test from 'node:test'
import assert from 'node:assert/strict'
import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
import { createOriginalApi, originalApi } from '../src/services/originalApi.js'
import { useOriginalExperimentsStore } from '../src/stores/originalExperiments.js'
import { useConnectionStore } from '../src/stores/connection.js'
import { api as legacyApi } from '../src/services/api.js'
import { originalFigureTitle } from '../src/config/originalTeaching.js'
import { createRunLinkResolver } from '../src/utils/originalSelection.js'

globalThis.localStorage = { getItem: () => null, setItem: () => {} }

test('original API sends only an experiment id and uses dedicated endpoints', async () => {
  const calls = []
  const client = {
    baseUrl: 'http://localhost:8000',
    get: async path => { calls.push(['GET', path]); return [] },
    post: async (path, body) => { calls.push(['POST', path, body]); return { run_id: 'run-1' } }
  }
  const api = createOriginalApi(client)
  await api.listExperiments()
  await api.listDatasets()
  await api.listRuns()
  await api.getRun('run-1')
  await api.startRun('logistic_regression')
  assert.deepEqual(calls, [
    ['GET', '/api/original-experiments'],
    ['GET', '/api/original-datasets'],
    ['GET', '/api/original-runs'],
    ['GET', '/api/original-runs/run-1'],
    ['POST', '/api/original-runs', { experiment_id: 'logistic_regression' }]
  ])
})

test('asset URLs follow the configured backend origin', () => {
  const api = createOriginalApi({ baseUrl: 'https://lab.example:8000' })
  assert.equal(api.assetUrl('/api/original-assets/knn/01.png'), 'https://lab.example:8000/api/original-assets/knn/01.png')
  assert.equal(api.assetUrl('/api/original-runs/r1/assets/02.png'), 'https://lab.example:8000/api/original-runs/r1/assets/02.png')
  assert.equal(api.downloadUrl('/api/original-assets/knn/01.png'), 'https://lab.example:8000/api/original-assets/knn/01.png?download=true')
})

test('original figure cards use model-specific Chinese chart labels', () => {
  assert.equal(originalFigureTitle('random_forest', { name: '06_final_benchmark_comparison.png', title: 'Final Benchmark Comparison' }), '最终基准对比')
  assert.equal(originalFigureTitle('dbscan', { name: '02_density_mechanism.png', title: 'Density Mechanism' }), '密度聚类机制')
  assert.equal(originalFigureTitle('future_model', { name: '01_new.png', title: 'New Chart' }), 'New Chart')
})

test('completed run stops polling and keeps server artifacts separate from archive', async () => {
  setActivePinia(createPinia())
  const conn = useConnectionStore()
  conn.status = 'connected'; conn.health = { ml_backend: { available: true } }
  const store = useOriginalExperimentsStore()
  const originals = { ...originalApi }
  const archived = { name: 'old.png', title: '原图', url: '/api/original-assets/knn/old.png' }
  const generated = { name: 'new.png', title: '新图', url: '/api/original-runs/r1/assets/new.png' }
  originalApi.listExperiments = async () => [{ id: 'knn', title: 'KNN', task: 'classification', datasets: [], parameters: {}, protocol: [], figures: [archived], source_data: [] }]
  originalApi.listRuns = async () => [{ run_id: 'r1', experiment_id: 'knn', status: 'running', figures: [], source_data: [], log: '', error: null }]
  originalApi.getRun = async () => ({ run_id: 'r1', experiment_id: 'knn', status: 'completed', figures: [generated], source_data: [], log: 'done', error: null })
  try {
    await store.loadCatalog()
    await store.loadRuns()
    assert.equal(store.activeRunIds.length, 1)
    assert.equal(await store.refreshRun('r1'), false)
    assert.equal(store.activeRunIds.length, 0)
    assert.deepEqual(store.selectedExperiment.figures, [archived])
    assert.deepEqual(store.runs[0].figures, [generated])
  } finally { Object.assign(originalApi, originals); store.stopPolling() }
})

test('a run cannot start before server history is available', async () => {
  setActivePinia(createPinia())
  const conn = useConnectionStore()
  conn.status = 'connected'; conn.health = { ml_backend: { available: true } }
  const store = useOriginalExperimentsStore()
  const saved = originalApi.listExperiments
  originalApi.listExperiments = async () => [{ id: 'knn', title: 'KNN', task: 'classification', figures: [], datasets: [], protocol: [], parameters: {}, source_data: [] }]
  try {
    await store.loadCatalog()
    assert.equal(store.canRun, true)
    store.runsLoading = true
    assert.equal(store.canRun, false)
    store.runsLoading = false
    store.runsError = 'history unavailable'
    assert.equal(store.canRun, false)
  } finally { originalApi.listExperiments = saved; store.stopPolling() }
})

test('a delayed history snapshot cannot downgrade a refreshed terminal run', async () => {
  setActivePinia(createPinia())
  const store = useOriginalExperimentsStore()
  const saved = { ...originalApi }
  let releaseList
  originalApi.listRuns = () => new Promise(resolve => { releaseList = resolve })
  originalApi.getRun = async () => ({ run_id: 'r1', experiment_id: 'knn', status: 'completed', figures: [], source_data: [], log: 'done', error: null })
  store.runs = [{ run_id: 'r1', experiment_id: 'knn', status: 'running', figures: [], source_data: [], log: '', error: null }]
  try {
    const listing = store.loadRuns()
    await store.refreshRun('r1')
    releaseList([{ run_id: 'r1', experiment_id: 'knn', status: 'running', figures: [], source_data: [], log: '', error: null }])
    await listing
    assert.equal(store.runs[0].status, 'completed')
    assert.deepEqual(store.activeRunIds, [])
  } finally { Object.assign(originalApi, saved); store.stopPolling() }
})

test('a delayed earlier run refresh cannot downgrade the newer response', async () => {
  setActivePinia(createPinia())
  const store = useOriginalExperimentsStore()
  const saved = originalApi.getRun
  let releaseOld
  let calls = 0
  originalApi.getRun = () => ++calls === 1
    ? new Promise(resolve => { releaseOld = resolve })
    : Promise.resolve({ run_id: 'r1', experiment_id: 'knn', status: 'completed', figures: [], source_data: [], log: 'done', error: null })
  try {
    const old = store.refreshRun('r1')
    await store.refreshRun('r1')
    releaseOld({ run_id: 'r1', experiment_id: 'knn', status: 'running', figures: [], source_data: [], log: '', error: null })
    await old
    assert.equal(store.runs[0].status, 'completed')
  } finally { originalApi.getRun = saved; store.stopPolling() }
})

test('history run link is consumed once and cannot override later manual selection', () => {
  const resolver = createRunLinkResolver()
  resolver.setPending('old-run')
  assert.equal(resolver.take([]), null)
  assert.deepEqual(resolver.take([{ run_id: 'old-run', experiment_id: 'knn' }]), { run_id: 'old-run', experiment_id: 'knn' })
  assert.equal(resolver.take([{ run_id: 'old-run', experiment_id: 'knn' }]), null)
  resolver.setPending('another-run')
  resolver.cancel()
  assert.equal(resolver.take([{ run_id: 'another-run', experiment_id: 'dbscan' }]), null)
})

test('switching backend clears stale catalog, runs, and selection', async () => {
  setActivePinia(createPinia())
  const conn = useConnectionStore()
  const store = useOriginalExperimentsStore()
  const originals = { ...originalApi }
  originalApi.listExperiments = async () => [{ id: 'knn', title: 'KNN', task: 'classification', datasets: [], parameters: {}, protocol: [], figures: [], source_data: [] }]
  originalApi.listRuns = async () => [{ run_id: 'old', experiment_id: 'knn', status: 'completed', figures: [], source_data: [] }]
  try {
    await store.loadCatalog()
    await store.loadRuns()
    assert.equal(store.selectedExperiment?.id, 'knn')
    conn.settings.baseUrl = 'http://different-backend'
    await nextTick()
    assert.equal(store.selectedExperiment, null)
    assert.deepEqual(store.experiments, [])
    assert.deepEqual(store.runs, [])
  } finally { Object.assign(originalApi, originals); store.stopPolling() }
})

test('original-only connection checks health without fetching legacy model registry', async () => {
  setActivePinia(createPinia())
  const conn = useConnectionStore()
  const saved = { ...legacyApi }
  const oldInterval = globalThis.setInterval
  let legacyCalls = 0
  legacyApi.checkHealth = async () => ({ ml_backend: { available: true } })
  legacyApi.fetchModels = async () => { legacyCalls++; return [] }
  legacyApi.fetchDatasets = async () => { legacyCalls++; return [] }
  globalThis.setInterval = () => 0
  try {
    conn.init({ originalOnly: true })
    await new Promise(resolve => setTimeout(resolve, 0))
    assert.equal(conn.status, 'connected')
    assert.equal(legacyCalls, 0)
    await conn.refreshAll()
    assert.equal(legacyCalls, 0)
  } finally { Object.assign(legacyApi, saved); globalThis.setInterval = oldInterval }
})
