import assert from 'node:assert/strict'
import {api} from '../src/services/api.js'
import {httpClient} from '../src/services/httpClient.js'
import {buildExperimentRequest} from '../src/services/adapters.js'
httpClient.setBaseUrl((process.env.API_URL || 'http://127.0.0.1:5173').replace(/\/api\/?$/, ''))
assert.equal((await api.checkHealth()).ml_backend.available, true)
const models=await api.fetchModels()
const datasets=await api.fetchDatasets()
assert.equal(models.length,12)
assert.equal(datasets.length,6)
for (const model of models) {
  const dataset=model.compatible_datasets?.[0] || datasets.find(ds=>ds.task===model.task)?.id
  const body=buildExperimentRequest({model:model.id,dataset,params:{},testSize:.2,randomState:42,taskType:model.task})
  assert.equal('variant' in body,false)
  const result=await api.runExperiment(body)
  assert.equal(result.model,model.id)
  assert.ok(result.run_id)
  assert.ok(Object.values(result.metrics).some(value=>typeof value==='number'))
  const charts=result.metadata?.visualizations
  assert.ok(charts?.length>0, `${model.id} missing charts`)
  for (const chart of charts) {
    assert.ok(chart.id && chart.title && chart.description)
    assert.ok(chart.option?.series?.length>0)
    assert.doesNotThrow(()=>JSON.parse(JSON.stringify(chart.option)))
  }
  console.log(`${model.id} / ${dataset}: ${charts.map(chart=>chart.id).join(', ')}`)
}
console.log('PASS: 12 algorithms through frontend native HTTP client -> Vite proxy -> real ML results and charts')
