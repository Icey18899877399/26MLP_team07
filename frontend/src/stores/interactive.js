import { computed, ref, watch } from 'vue'
import { defineStore } from 'pinia'
import { interactiveApi } from '../services/interactiveApi.js'
import { useConnectionStore } from './connection.js'

// Separate from immutable original galleries and full-script reproduction jobs.
export const useInteractiveStore = defineStore('interactive', () => {
  const conn = useConnectionStore()
  const catalog = ref([])
  const loading = ref(false)
  const error = ref('')
  const records = ref({})
  const busy = ref(false)
  let generation = 0
  let catalogRequest = 0
  const canRun = computed(() => conn.status === 'connected' && conn.mlBackendAvailable && !busy.value && !loading.value && !error.value && catalog.value.length > 0)

  watch(() => conn.baseUrl, () => {
    generation += 1
    catalogRequest += 1
    catalog.value = []; records.value = {}; error.value = ''; loading.value = false; busy.value = false
  }, { flush: 'sync' })

  async function loadCatalog() {
    const token = ++catalogRequest
    const epoch = generation
    loading.value = true; error.value = ''
    try {
      const data = await interactiveApi.list()
      if (epoch !== generation || token !== catalogRequest) return
      if (!Array.isArray(data) || data.some(item => !item.id || !item.model || !item.default_params || !Array.isArray(item.datasets))) throw new Error('调参目录格式不正确')
      catalog.value = data
    } catch (err) {
      if (epoch !== generation || token !== catalogRequest) return
      catalog.value = []; error.value = err.message
    } finally {
      if (epoch === generation && token === catalogRequest) loading.value = false
    }
  }

  async function run(id, params, dataset, testSize, randomState) {
    const item = catalog.value.find(entry => entry.id === id)
    if (!canRun.value || !item) return null
    const epoch = generation
    const request = { model: item.model, dataset, params: structuredClone(params), random_state: randomState }
    if (item.test_size != null) request.test_size = testSize
    const record = { status: 'running', startedAt: Date.now(), finishedAt: null, request, result: null, error: '' }
    records.value[id] = record
    busy.value = true
    try {
      const result = await interactiveApi.run(request)
      if (epoch !== generation) return null
      if (result?.model !== request.model || result?.dataset !== dataset || !result?.run_id || !Array.isArray(result?.metadata?.visualizations)) throw new Error('训练结果与本次请求不匹配或缺少图表')
      records.value[id] = { ...record, status: 'completed', finishedAt: Date.now(), result }
      return result
    } catch (err) {
      if (epoch !== generation) return null
      const uncertain = ['timeout', 'network_error'].includes(err.code)
      records.value[id] = { ...record, status: uncertain ? 'unknown' : 'failed', finishedAt: Date.now(), error: uncertain ? `${err.message}；无法确认服务端是否仍在计算。请勿连续重复提交，稍后确认服务状态。` : err.message }
      return null
    } finally {
      if (epoch === generation) busy.value = false
    }
  }
  return { catalog, loading, error, records, busy, canRun, loadCatalog, run }
})
