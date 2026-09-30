import test from 'node:test'
import assert from 'node:assert/strict'
import { HttpClient } from '../src/services/httpClient.js'
test('a successful non-JSON response is a protocol error, not a completed experiment', async () => {
  const original=globalThis.fetch
  globalThis.fetch=async()=>new Response('<html>proxy error</html>',{status:200})
  try {await assert.rejects(new HttpClient().get('/api/models'), error=>error.code==='invalid_response')} finally {globalThis.fetch=original}
})
test('timeout covers response body reading after response headers arrive', async () => {
  const original=globalThis.fetch
  globalThis.fetch=async(url,{signal})=>({ok:true,status:200,json:()=>new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>resolve({late:true}),60)
    signal.addEventListener('abort',()=>{clearTimeout(timer);reject(new DOMException('aborted','AbortError'))})
  })})
  try {await assert.rejects(new HttpClient().get('/api/models',{timeoutMs:5}), error=>error.code==='timeout')} finally {globalThis.fetch=original}
})
test('HTTP validation errors retain field and business error codes', async () => {
  const original=globalThis.fetch
  try {
    globalThis.fetch=async()=>new Response(JSON.stringify({detail:[{loc:['body','params','gamma'],msg:'must be positive'}]}),{status:422})
    await assert.rejects(new HttpClient().post('/api/experiments',{}),error=>error.code==='validation_error' && error.message.includes('gamma'))
    globalThis.fetch=async()=>new Response(JSON.stringify({detail:{code:'ml_request_rejected',message:'不支持该数据集'}}),{status:400})
    await assert.rejects(new HttpClient().post('/api/experiments',{}),error=>error.code==='ml_request_rejected' && error.message==='不支持该数据集')
  } finally {globalThis.fetch=original}
})
