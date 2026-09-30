import test from 'node:test'
import assert from 'node:assert/strict'
import {createPinia, setActivePinia} from 'pinia'
import {useConnectionStore} from '../src/stores/connection.js'
import {api} from '../src/services/api.js'
globalThis.localStorage = {getItem:()=>null,setItem:()=>{}}

test('changing endpoint immediately prevents training with previous server registry', async () => {
  setActivePinia(createPinia())
  const conn=useConnectionStore()
  const originals={...api}
  conn.status='connected';conn.health={ml_backend:{available:true}};conn.registry={server:{name:'old'},algorithms:[{id:'old'}],datasets:[]}
  api.checkHealth=async()=>({ml_backend:{available:true}})
  api.fetchModels=async()=>{throw new Error('catalog unavailable')}
  api.fetchDatasets=async()=>[]
  try {
    await conn.setBaseUrl('http://new')
    await new Promise(resolve=>setTimeout(resolve,0))
    assert.equal(conn.canRunExperiments,false)
    assert.equal(conn.isUsingFallback,true)
  } finally {Object.assign(api,originals)}
})
test('late health reply from previous endpoint cannot overwrite the current unavailable state', async () => {
  setActivePinia(createPinia())
  const conn=useConnectionStore()
  const originals={...api}
  let finishOld
  api.checkHealth=()=>new Promise(resolve=>{finishOld=resolve})
  const pending=conn.checkHealth()
  api.checkHealth=async()=>({ml_backend:{available:false,detail:'unavailable'}})
  api.fetchModels=async()=>[];api.fetchDatasets=async()=>[]
  try {
    await conn.setBaseUrl('http://new')
    finishOld({ml_backend:{available:true}})
    await pending
    await new Promise(resolve=>setTimeout(resolve,0))
    assert.equal(conn.status,'unavailable')
    assert.equal(conn.canRunExperiments,false)
  } finally {Object.assign(api,originals)}
})
