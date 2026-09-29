import test from 'node:test'
import assert from 'node:assert/strict'
import { makeRegistry, experimentRequest, resultPayload, HttpClient } from '../src/services/httpClient.js'
const registry = makeRegistry([
  { id: 'kmeans', name: 'K-Means', task_type: 'clustering', variants: ['optimized'], parameters: {n_clusters: 3}, compatible_datasets: ['seeds'] },
  { id: 'logistic_regression', name: 'LR', task_type: 'classification', variants: ['optimized'], parameters: {class_weight: null}, compatible_datasets: ['wdbc'] }
], [{id: 'seeds', name: 'Seeds', task_type: 'clustering', sample_count: 210, feature_count: 7}])
test('catalog and clustering split', () => {
  assert.equal(registry.algorithms.length, 2)
  const request = experimentRequest({algorithmId: 'kmeans.optimized', datasetId: 'seeds', split: {testRatio: .3, seed: 7}}, registry)
  assert.equal(request.test_size, null)
  assert.equal(request.variant, 'optimized')
  assert.equal(request.random_state, 7)
  assert.throws(() => experimentRequest({algorithmId: 'kmeans.optimized', datasetId: 'wdbc'}, registry))
})
test('classification split and null class weight', () => {
  const request = experimentRequest({algorithmId: 'logistic_regression.optimized', datasetId: 'wdbc', hyperparams: {class_weight: 'none'}, split: {testRatio: .3}}, registry)
  assert.equal(request.test_size, .3)
  assert.equal(request.params.class_weight, null)
})
test('confusion matrix comes from actual counts', () => {
  const result = resultPayload({runId: 'r'}, {metrics: {tn: 3, fp: 1, fn: 2, tp: 4}}, 10)
  assert.deepEqual(result.metrics.confusion_matrix, [[3,1],[2,4]])
  assert.equal(result.durations.train_ms, 10)
})
test('HTTP errors reach the correct run', async () => {
  const client = new HttpClient({fetchImpl: async () => ({ok: false, json: async () => ({detail: {message: 'invalid parameter'}})})})
  client.registry = registry
  const messages = []
  client.onMessage = m => messages.push(m)
  await client.run({runId: 'r', algorithmId: 'kmeans.optimized', datasetId: 'seeds'})
  assert.equal(messages.at(-1).type, 'training.error')
  assert.equal(messages.at(-1).payload.runId, 'r')
  assert.match(messages.at(-1).payload.message, /invalid parameter/)
  assert.equal(client.pending.size, 0)
})
