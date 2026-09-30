import test from 'node:test'
import assert from 'node:assert/strict'
import { buildRegistry, buildExperimentRequest } from '../src/services/adapters.js'
import { buildFormModel, extractHyperparams } from '../src/utils/hyperparams.js'
import * as adapters from '../src/services/adapters.js'
import * as format from '../src/utils/format.js'

test('optional and array model parameters survive the editable form and request', () => {
  const registry = buildRegistry({ models: [{id: 'mlp_regression.optimized', task: 'regression', display_name: 'MLP', default_params: {max_depth: null, hidden_layers: [16, 8], l2: 0, enabled: true}, compatible_datasets: ['concrete']}], datasets: [], baseUrl: '' })
  const schema = registry.algorithms[0].hyperparams
  const form = buildFormModel(schema)
  assert.deepEqual(extractHyperparams(form, schema), {max_depth: null, hidden_layers: [16, 8], l2: 0, enabled: true})
  form.max_depth = '5'
  assert.equal(extractHyperparams(form, schema).max_depth, 5)
  form.hidden_layers = '[bad]'
  assert.throws(() => extractHyperparams(form, schema), /hidden_layers/)
})

test('registry preserves compatibility and never truncates native model identities', () => {
  const registry = buildRegistry({models: [{id: 'kmeans.optimized', task: 'clustering', display_name: 'K-Means', default_params: {}, compatible_datasets: ['seeds']}], datasets: [{id: 'seeds', task: 'clustering', display_name: '种子', sample_count: 210, feature_count: 7, has_target: true}], baseUrl: ''})
  assert.equal(registry.algorithms[0].id, 'kmeans.optimized')
  assert.deepEqual(registry.algorithms[0].compatibleDatasets, ['seeds'])
  assert.equal(registry.datasets[0].hasTarget, true)
})

test('requests use native contract with no variant and omit unsupervised test_size', () => {
  for (const taskType of ['classification', 'regression', 'clustering', 'anomaly_detection']) {
    const body = buildExperimentRequest({model: 'native.optimized', dataset: 'ds', params: {max_depth: null}, randomState: 7, testSize: .3, taskType})
    assert.equal(body.model, 'native.optimized')
    assert.equal(body.random_state, 7)
    assert.equal('variant' in body, false)
    assert.equal(body.test_size, ['classification', 'regression'].includes(taskType) ? .3 : undefined)
  }
})

test('comparison excludes a different seed, split or evaluation protocol', () => {
  const run = {datasetId: 'wdbc', taskType: 'classification', randomState: 42, testSize: .2, result: {metadata: {evaluation_protocol: 'stratified_holdout'}}}
  assert.equal(adapters.areComparable?.(run, {...run, modelId: 'different'}), true)
  assert.equal(adapters.areComparable?.(run, {...run, randomState: 7}), false)
  assert.equal(adapters.areComparable?.(run, {...run, testSize: .3}), false)
  assert.equal(adapters.areComparable?.(run, {...run, result: {metadata: {evaluation_protocol: 'in_sample'}}}), false)
})
test('gamma accepts positive numeric input without converting the scale keyword', () => {
  const schema = [{name: 'gamma', type: 'text', default: 'scale'}]
  assert.equal(extractHyperparams({gamma: 'scale'}, schema).gamma, 'scale')
  assert.equal(extractHyperparams({gamma: '0.1'}, schema).gamma, .1)
})
test('counts and signed fit scores never acquire percentage units', () => {
  assert.equal(format.formatMetric?.('n_clusters', 1), '1')
  assert.equal(format.formatMetric?.('fp', 0), '0')
  assert.equal(format.formatMetric?.('r2', .85), '0.85')
  assert.equal(format.formatMetric?.('accuracy', .9), '90.00%')
})
