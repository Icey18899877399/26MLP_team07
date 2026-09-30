import test from 'node:test'
import assert from 'node:assert/strict'
import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
import { useInteractiveStore } from '../src/stores/interactive.js'
import { interactiveApi } from '../src/services/interactiveApi.js'
import { useConnectionStore } from '../src/stores/connection.js'

globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const item = { id: 'knn', model: 'knn.optimized', dataset: 'wdbc', datasets: [{ id: 'wdbc', name: 'WDBC', path: 'data/classification/wdbc/wdbc.data' }], default_params: { n_neighbors: 5 }, parameter_descriptions: {}, random_state: 42, test_size: 0.2, protocol: 'original holdout', source: 'visualization.knn_figures' }
const result = { run_id: 'r1', model: 'knn.optimized', dataset: 'wdbc', task: 'classification', effective_params: { n_neighbors: 3 }, metrics: { accuracy: 0.9 }, metadata: { visualizations: [{ id: 'roc', option: { series: [] } }] } }
function setup() {
  setActivePinia(createPinia())
  const conn = useConnectionStore()
  conn.status = 'connected'; conn.health = { ml_backend: { available: true } }
  return { store: useInteractiveStore(), conn }
}

test('editable training sends chosen parameters and keeps result tied to its algorithm', async () => {
  const saved = { ...interactiveApi }
  const { store } = setup()
  let finish
  let payload
  interactiveApi.list = async () => [item]
  interactiveApi.run = body => { payload = body; return new Promise(resolve => { finish = resolve }) }
  try {
    await store.loadCatalog()
    const pending = store.run('knn', { n_neighbors: 3 }, 'wdbc', 0.25, 7)
    assert.equal(store.busy, true)
    assert.equal(store.records.knn.status, 'running')
    assert.deepEqual(payload, { model: 'knn.optimized', dataset: 'wdbc', params: { n_neighbors: 3 }, test_size: 0.25, random_state: 7 })
    assert.equal(await store.run('knn', {}, 'wdbc', 0.2, 42), null)
    finish(result); await pending
    assert.equal(store.records.knn.status, 'completed')
    assert.equal(store.records.knn.result.run_id, 'r1')
    assert.equal(store.records.dbscan, undefined)
    assert.equal(store.busy, false)
  } finally { Object.assign(interactiveApi, saved) }
})

test('failed rerun never displays an earlier result as the current result', async () => {
  const saved = { ...interactiveApi }
  const { store } = setup()
  interactiveApi.list = async () => [item]
  interactiveApi.run = async () => result
  try {
    await store.loadCatalog(); await store.run('knn', {}, 'wdbc', 0.2, 42)
    interactiveApi.run = async () => { throw new Error('invalid n_neighbors') }
    await store.run('knn', { n_neighbors: -1 }, 'wdbc', 0.2, 42)
    assert.equal(store.records.knn.status, 'failed')
    assert.equal(store.records.knn.result, null)
    assert.match(store.records.knn.error, /invalid n_neighbors/)
  } finally { Object.assign(interactiveApi, saved) }
})

test('endpoint changes invalidate catalog and ignore late training responses', async () => {
  const saved = { ...interactiveApi }
  const { store, conn } = setup()
  let finish
  interactiveApi.list = async () => [item]
  interactiveApi.run = () => new Promise(resolve => { finish = resolve })
  try {
    await store.loadCatalog()
    const pending = store.run('knn', {}, 'wdbc', 0.2, 42)
    conn.settings.baseUrl = 'http://new-endpoint'
    await nextTick()
    assert.deepEqual(store.catalog, [])
    assert.deepEqual(store.records, {})
    finish(result); await pending
    assert.deepEqual(store.records, {})
  } finally { Object.assign(interactiveApi, saved) }
})

test('catalog failure or offline backend disables runs without test defaults', async () => {
  const saved = { ...interactiveApi }
  const { store, conn } = setup()
  interactiveApi.list = async () => { throw new Error('catalog unavailable') }
  try {
    await store.loadCatalog()
    assert.equal(store.canRun, false)
    assert.deepEqual(store.catalog, [])
    assert.equal(await store.run('knn', {}, 'wdbc', 0.2, 42), null)
    interactiveApi.list = async () => [item]
    await store.loadCatalog()
    assert.equal(store.canRun, true)
    conn.status = 'offline'
    assert.equal(store.canRun, false)
  } finally { Object.assign(interactiveApi, saved) }
})
