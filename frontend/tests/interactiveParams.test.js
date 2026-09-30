import test from 'node:test'
import assert from 'node:assert/strict'
import { interactiveAlgorithm } from '../src/utils/interactiveParams.js'
import { buildFormModel, extractHyperparams } from '../src/utils/hyperparams.js'

test('nullable class weight can be changed from balanced to null', () => {
  const { hyperparams } = interactiveAlgorithm({ model: 'logistic_regression.optimized', default_params: { class_weight: 'balanced' }, parameter_descriptions: {} })
  const form = buildFormModel(hyperparams)
  assert.deepEqual(extractHyperparams(form, hyperparams), { class_weight: 'balanced' })
  form.class_weight = 'null'
  assert.deepEqual(extractHyperparams(form, hyperparams), { class_weight: null })
})

test('gamma supports both original numeric start and automatic scale', () => {
  const { hyperparams } = interactiveAlgorithm({ model: 'one_class_svm.optimized', default_params: { gamma: 0.25 }, parameter_descriptions: {} })
  const form = buildFormModel(hyperparams)
  assert.equal(typeof form.gamma, 'string')
  assert.deepEqual(extractHyperparams(form, hyperparams), { gamma: 0.25 })
  form.gamma = 'scale'
  assert.deepEqual(extractHyperparams(form, hyperparams), { gamma: 'scale' })
})
