import assert from 'node:assert/strict'
import { HttpClient } from '../src/services/httpClient.js'

const client = new HttpClient({ defaultUrl: process.env.API_URL || 'http://127.0.0.1:5173/api' })
const messages = []
client.onMessage = message => messages.push(message)
await client.connect()
assert.equal(client.connected, true, JSON.stringify(messages))
for (const [algorithmId, datasetId, metric] of [
  ['kmeans.optimized', 'seeds', 'adjusted_rand_index'],
  ['logistic_regression.optimized', 'wdbc', 'accuracy']
]) {
  await client.run({ runId: algorithmId, algorithmId, datasetId, hyperparams: {}, split: {seed: 42, testRatio: .2} })
  const message = messages.at(-1)
  assert.equal(message.type, 'training.result', JSON.stringify(message))
  assert.equal(typeof message.payload.metrics[metric], 'number')
  assert.ok(message.payload.diagnostics.run_id)
  console.log(algorithmId, JSON.stringify(message.payload.metrics))
}
client.disconnect()
console.log('Real frontend -> proxy -> FastAPI -> ml_core smoke passed')
