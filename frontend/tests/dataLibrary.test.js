import test from 'node:test'
import assert from 'node:assert/strict'
import { createPinia, setActivePinia } from 'pinia'
import { useDataLibraryStore } from '../src/stores/dataLibrary.js'
import { dataLibraryApi } from '../src/services/dataLibraryApi.js'
import { useConnectionStore } from '../src/stores/connection.js'
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const detail = id => ({ id, name: id, source: 'upload', row_count: 20, column_count: 3, columns: [{ name: 'x', dtype: 'number', missing_count: 0, unique_count: 20 }, { name: 'z', dtype: 'number', missing_count: 0, unique_count: 20 }, { name: 'label', dtype: 'string', missing_count: 0, unique_count: 2 }], preview: [{ x: 1, z: 2, label: 'yes' }], default_target: 'label', paths: [] })
function setup() { setActivePinia(createPinia()); const conn = useConnectionStore(); conn.status = 'connected'; conn.health = { ml_backend: { available: true } }; return { store: useDataLibraryStore(), conn } }

test('reading a file cannot upload it to a newly selected endpoint or after navigation', async () => {
  const saved = { ...dataLibraryApi }; const { store, conn } = setup(); let finish, calls = 0, active = true
  dataLibraryApi.upload = async () => { calls++; return detail('uploaded') }
  const file = { name: 'example.csv', arrayBuffer: () => new Promise(resolve => { finish = resolve }) }
  try {
    const pending = store.uploadFile(file, () => active); conn.settings.baseUrl = 'http://changed'; finish(new TextEncoder().encode('x,label\n1,a')); assert.equal(await pending, null)
    const next = store.uploadFile(file, () => active); active = false; finish(new TextEncoder().encode('x,label\n1,a')); assert.equal(await next, null); assert.equal(calls, 0)
  } finally { Object.assign(dataLibraryApi, saved) }
})

test('endpoint change during post-upload refresh cannot navigate to the old upload', async () => {
  const saved = { ...dataLibraryApi }; const { store, conn } = setup(); let finish
  dataLibraryApi.upload = async () => detail('upload_old')
  dataLibraryApi.list = () => new Promise(resolve => { finish = resolve })
  try { const pending = store.upload('test.csv', 'x,label\n1,a'); await Promise.resolve(); conn.settings.baseUrl = 'http://changed'; finish([]); assert.equal(await pending, null) } finally { Object.assign(dataLibraryApi, saved) }
})

test('late dataset detail cannot replace a newer selected dataset', async () => {
  const saved = { ...dataLibraryApi }; const { store } = setup(); let finish
  dataLibraryApi.detail = id => id === 'old' ? new Promise(resolve => { finish = resolve }) : Promise.resolve(detail(id))
  try { const pending = store.open('old'); await store.open('new'); finish(detail('old')); await pending; assert.equal(store.detail.id, 'new'); assert.deepEqual(store.selection.feature_columns, ['x', 'z']) } finally { Object.assign(dataLibraryApi, saved) }
})

test('changed feature selection invalidates compatibility and late compatibility responses', async () => {
  const saved = { ...dataLibraryApi }; const { store } = setup(); let finish
  dataLibraryApi.detail = async id => detail(id)
  dataLibraryApi.compatibility = () => new Promise(resolve => { finish = resolve })
  try { await store.open('data'); const pending = store.checkCompatibility(); store.selection.feature_columns = ['x']; finish({ models: ['knn.optimized'], issues: [] }); await pending; assert.deepEqual(store.compatibleModels, []); assert.equal(store.canTrain, false) } finally { Object.assign(dataLibraryApi, saved) }
})

test('training sends frozen column selection and new dataset identity', async () => {
  const saved = { ...dataLibraryApi }; const { store } = setup(); let request
  dataLibraryApi.detail = async id => detail(id)
  dataLibraryApi.compatibility = async () => ({ models: ['knn.optimized'], issues: [] })
  dataLibraryApi.run = async body => { request = body; return { run_id: 'r1', model: body.model, dataset: body.dataset_id, metrics: { accuracy: 1 }, metadata: { visualizations: [{ id: 'knn_neighborhood', option: { series: [] } }] } } }
  try { await store.open('upload_demo'); await store.checkCompatibility(); store.model = 'knn.optimized'; await store.train({ n_neighbors: 3 }, 42, 0.2); assert.deepEqual(request.feature_columns, ['x', 'z']); assert.equal(request.target_column, 'label'); assert.equal(request.dataset_id, 'upload_demo'); assert.equal(store.run.status, 'completed'); store.selection.task = 'clustering'; assert.equal(store.run, null); assert.equal(store.canTrain, false) } finally { Object.assign(dataLibraryApi, saved) }
})

test('endpoint switch clears data and prevents late training result from reappearing', async () => {
  const saved = { ...dataLibraryApi }; const { store, conn } = setup(); let finish
  dataLibraryApi.detail = async id => detail(id)
  dataLibraryApi.compatibility = async () => ({ models: ['knn.optimized'], issues: [] })
  dataLibraryApi.run = () => new Promise(resolve => { finish = resolve })
  try { await store.open('upload_demo'); await store.checkCompatibility(); store.model = 'knn.optimized'; const pending = store.train({}, 42, 0.2); conn.settings.baseUrl = 'http://changed'; finish({ run_id: 'old', model: 'knn.optimized', dataset: 'upload_demo', metadata: { visualizations: [] } }); await pending; assert.equal(store.detail, null); assert.equal(store.run, null); assert.equal(store.canTrain, false) } finally { Object.assign(dataLibraryApi, saved) }
})
