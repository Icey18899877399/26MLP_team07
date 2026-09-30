import test from 'node:test'
import assert from 'node:assert/strict'
import { createPinia, setActivePinia } from 'pinia'
import { useTrainingStore } from '../src/stores/training.js'
import { useConnectionStore } from '../src/stores/connection.js'
import { api } from '../src/services/api.js'

const storage = new Map()
globalThis.localStorage = {getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value)}
test('old websocket history must not appear as successful native experiments', () => {
  storage.set('mlviz.history', JSON.stringify([{algorithmId: 'fake', result: {metrics: {accuracy: 1}}}]))
  setActivePinia(createPinia())
  assert.equal(useTrainingStore().history.length, 0)
})
test('clearing completed history preserves the pending request and its eventual real result', async () => {
  storage.clear()
  setActivePinia(createPinia())
  const conn = useConnectionStore()
  conn.status = 'connected'
  conn.health = {ml_backend: {available: true}}
  conn.registry = {server: {name: 'live'}, algorithms: [], datasets: []}
  const training = useTrainingStore()
  const original = api.runExperiment
  let finish
  api.runExperiment = () => new Promise(resolve => { finish = resolve })
  try {
    const id = training.startRun({modelId: 'knn.optimized', datasetId: 'wdbc', taskType: 'classification', params: {n_neighbors: 5}, testSize: .2, randomState: 42})
    training.clearRuns()
    assert.equal(training.activeRuns.length, 1)
    finish({run_id: 'real-run', model: 'knn.optimized', dataset: 'wdbc', task: 'classification', effective_params: {n_neighbors: 5}, metrics: {accuracy: .9}, metadata: {visualizations: []}, artifacts: []})
    await new Promise(resolve => setTimeout(resolve, 0))
    assert.equal(training.getRun(id).status, 'done')
    assert.equal(training.history[0].runId, 'real-run')
    training.clearRuns()
    assert.equal(training.history.length, 0)
  } finally {
    if (finish) finish({run_id: 'cleanup', metrics: {}})
    api.runExperiment = original
  }
})

test('storage quota failures preserve the successful result and previous cached records', async () => {
  setActivePinia(createPinia())
  const conn=useConnectionStore()
  conn.status='connected';conn.health={ml_backend:{available:true}};conn.registry={server:{name:'live'},algorithms:[],datasets:[]}
  const training=useTrainingStore()
  const original=api.runExperiment
  const originalSet=localStorage.setItem
  storage.set('mlviz.http.v2.history','previous record')
  localStorage.setItem=()=>{throw new DOMException('quota','QuotaExceededError')}
  api.runExperiment=async()=>({run_id:'quota-run',model:'knn.optimized',dataset:'wdbc',task:'classification',metrics:{accuracy:.9},metadata:{},effective_params:{},artifacts:[]})
  try {
    const id=training.startRun({modelId:'knn.optimized',datasetId:'wdbc',taskType:'classification',params:{}})
    await new Promise(resolve=>setTimeout(resolve,0))
    assert.equal(training.getRun(id).status,'done')
    assert.equal(training.history[0].runId,'quota-run')
    assert.equal(storage.get('mlviz.http.v2.history'),'previous record')
    assert.ok(conn.logLines.some(line=>line.level==='warn'))
  } finally {api.runExperiment=original;localStorage.setItem=originalSet}
})
