import test from 'node:test'
import assert from 'node:assert/strict'
import { makeRegistry, experimentRequest, resultPayload, HttpClient } from '../src/services/httpClient.js'
import { buildFormModel, extractHyperparams } from '../src/utils/hyperparams.js'
const models = [
  { id: 'kmeans.optimized', display_name: 'K-Means', task: 'clustering', default_params: {n_clusters: 3}, compatible_datasets: ['seeds'] },
  { id: 'logistic_regression.optimized', display_name: 'LR', task: 'classification', default_params: {class_weight: null}, compatible_datasets: ['wdbc'] },
  { id: 'linear_regression.optimized', display_name: 'Linear', task: 'regression', default_params: {}, compatible_datasets: ['housing'] },
  { id: 'isolation_forest.optimized', display_name: 'Isolation', task: 'anomaly_detection', default_params: {}, compatible_datasets: ['anomaly'] }
]
const datasets = [{id: 'seeds', display_name: 'Seeds', task: 'clustering', sample_count: 210, feature_count: 7, has_target: true}]
test('native catalog preserves full model IDs, four tasks and dataset metadata', () => {
  const registry = makeRegistry(models, datasets)
  assert.equal(registry.algorithms.length, 4)
  assert.equal(registry.algorithms[0].id, 'kmeans.optimized')
  assert.equal(registry.datasets[0].hasTarget, true)
  assert.equal(registry.taskTypes.length, 4)
})
test('native request never sends variant and only clustering disables splitting', () => {
  const registry = makeRegistry(models, datasets)
  for (const [algorithmId, datasetId, split] of [['kmeans.optimized', 'seeds', null], ['logistic_regression.optimized', 'wdbc', .3], ['linear_regression.optimized', 'housing', .3], ['isolation_forest.optimized', 'anomaly', .3]]) {
    const request = experimentRequest({algorithmId, datasetId, split: {testRatio: .3, seed: 7}}, registry)
    assert.equal(request.model, algorithmId)
    assert.equal(request.test_size, split)
    assert.equal(request.random_state, 7)
    assert.equal('variant' in request, false)
  }
  assert.throws(() => experimentRequest({algorithmId: 'kmeans.optimized', datasetId: 'wdbc'}, registry))
})
test('optional and list parameters retain their types', () => {
  const registry = makeRegistry([{...models[0], default_params: {max_depth: null, hidden_layers: [8, 4], enabled: true, learning_rate: .02}}], [])
  const schema = registry.algorithms[0].hyperparams
  const form = buildFormModel(schema)
  assert.deepEqual(extractHyperparams(form, schema), {max_depth: null, hidden_layers: [8, 4], enabled: true, learning_rate: .02})
  form.max_depth = '5'
  assert.equal(extractHyperparams(form, schema).max_depth, 5)
  form.hidden_layers = '[invalid]'
  assert.throws(() => extractHyperparams(form, schema), /hidden_layers/)
})
test('native results retain charts, effective parameters and server run identity', () => {
  const charts = [{id: 'actual', title: '实际结果', option: {series: [{type: 'scatter', data: [[1,2]]}]}}]
  const result = resultPayload({runId: 'local', hyperparams: {max_iter: 1}}, {run_id: 'server', task: 'classification', effective_params: {max_iter: 20}, metrics: {tn: 3, fp: 1, fn: 2, tp: 4}, artifacts: [], metadata: {visualizations: charts}}, 10)
  assert.deepEqual(result.metrics.confusion_matrix, [[3,1],[2,4]])
  assert.equal(result.serverRunId, 'server')
  assert.deepEqual(result.visualizations, charts)
  assert.equal(result.hyperparams.max_iter, 20)
  assert.equal(result.durations.train_ms, 10)
})
test('HTTP errors reach the correct run and release pending state', async () => {
  const client = new HttpClient({fetchImpl: async () => ({ok: false, json: async () => ({detail: {message: 'invalid parameter'}})})})
  client.registry = makeRegistry(models, datasets)
  const messages = []
  client.onMessage = m => messages.push(m)
  await client.run({runId: 'r', algorithmId: 'kmeans.optimized', datasetId: 'seeds'})
  assert.equal(messages.at(-1).type, 'training.error')
  assert.equal(messages.at(-1).payload.runId, 'r')
  assert.match(messages.at(-1).payload.message, /invalid parameter/)
  assert.equal(client.pending.size, 0)
})
