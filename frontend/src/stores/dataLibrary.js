import { computed, ref, watch } from 'vue'
import { defineStore } from 'pinia'
import { dataLibraryApi } from '../services/dataLibraryApi.js'
import { useConnectionStore } from './connection.js'

export const useDataLibraryStore = defineStore('dataLibrary', () => {
  const conn = useConnectionStore()
  const items = ref([]), detail = ref(null)
  const loading = ref(false), detailLoading = ref(false), uploading = ref(false)
  const error = ref(''), detailError = ref(''), uploadError = ref('')
  const selection = ref({ task: 'classification', feature_columns: [], target_column: null })
  const compatibleModels = ref([]), issues = ref([]), warnings = ref([]), checking = ref(false), model = ref('')
  const busy = ref(false), run = ref(null)
  let epoch = 0, listSeq = 0, detailSeq = 0, checkSeq = 0, runSeq = 0
  const selectionKey = computed(() => JSON.stringify([detail.value?.id, selection.value]))
  let checkedKey = ''
  const canTrain = computed(() => conn.status === 'connected' && conn.mlBackendAvailable && !busy.value && !checking.value && !detailLoading.value && !!detail.value && checkedKey === selectionKey.value && compatibleModels.value.includes(model.value) && !issues.value.length)
  function invalidateSelection() { checkSeq++; checking.value = false; compatibleModels.value = []; issues.value = []; warnings.value = []; checkedKey = ''; model.value = ''; runSeq++; run.value = null }
  watch(selectionKey, invalidateSelection, { flush: 'sync' })
  watch(model, () => { runSeq++; run.value = null }, { flush: 'sync' })
  watch(() => conn.baseUrl, () => {
    epoch++; listSeq++; detailSeq++; runSeq++
    items.value = []; detail.value = null; loading.value = false; detailLoading.value = false; uploading.value = false
    error.value = ''; detailError.value = ''; uploadError.value = ''; busy.value = false; run.value = null; invalidateSelection()
  }, { flush: 'sync' })
  async function load() {
    const token = ++listSeq, generation = epoch
    loading.value = true; error.value = ''
    try { const data = await dataLibraryApi.list(); if (generation === epoch && token === listSeq) { if (!Array.isArray(data)) throw new Error('数据目录格式错误'); items.value = data } }
    catch (err) { if (generation === epoch && token === listSeq) { items.value = []; error.value = err.message } }
    finally { if (generation === epoch && token === listSeq) loading.value = false }
  }
  async function open(id) {
    const token = ++detailSeq, generation = epoch
    detail.value = null; detailError.value = ''; detailLoading.value = true; run.value = null
    try {
      const data = await dataLibraryApi.detail(id)
      if (generation !== epoch || token !== detailSeq) return
      if (data.id !== id || !Array.isArray(data.columns) || !Array.isArray(data.preview)) throw new Error('数据集详情格式错误')
      detail.value = data
      selection.value = { task: 'classification', target_column: data.default_target || null, feature_columns: data.columns.filter(c => c.dtype === 'number' && c.name !== data.default_target && c.name.toLowerCase() !== 'id' && c.suggested_role !== 'identifier').map(c => c.name) }
    } catch (err) { if (generation === epoch && token === detailSeq) detailError.value = err.message }
    finally { if (generation === epoch && token === detailSeq) detailLoading.value = false }
  }
  async function upload(filename, content) {
    if (uploading.value) return null
    const generation = epoch
    uploading.value = true; uploadError.value = ''
    try { const data = await dataLibraryApi.upload(filename, content); if (generation !== epoch) return null; await load(); return generation === epoch ? data : null }
    catch (err) { if (generation === epoch) uploadError.value = err.message; return null }
    finally { if (generation === epoch) uploading.value = false }
  }
  async function uploadFile(file, isActive = () => true) {
    const generation = epoch
    const content = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer())
    if (generation !== epoch || !isActive()) return null
    return upload(file.name, content)
  }
  function selectedBody() { return { dataset_id: detail.value.id, task: selection.value.task, feature_columns: [...selection.value.feature_columns], target_column: selection.value.target_column || null } }
  async function checkCompatibility() {
    if (!detail.value) return
    const token = ++checkSeq, generation = epoch, key = selectionKey.value
    checking.value = true; compatibleModels.value = []; issues.value = []; warnings.value = []; model.value = ''; checkedKey = ''
    try {
      const data = await dataLibraryApi.compatibility(selectedBody())
      if (generation !== epoch || token !== checkSeq || key !== selectionKey.value) return
      if (!Array.isArray(data.models) || !Array.isArray(data.issues)) throw new Error('算法兼容性响应格式错误')
      compatibleModels.value = data.models; issues.value = data.issues; warnings.value = data.warnings || []; checkedKey = key
      model.value = data.models[0] || ''
    } catch (err) { if (generation === epoch && token === checkSeq) issues.value = [err.message] }
    finally { if (generation === epoch && token === checkSeq) checking.value = false }
  }
  async function train(params, randomState, testSize) {
    if (!canTrain.value) return null
    const token = ++runSeq, generation = epoch
    const request = { ...selectedBody(), model: model.value, params: JSON.parse(JSON.stringify(params)), random_state: randomState }
    if (['classification', 'regression'].includes(selection.value.task)) request.test_size = testSize
    const pending = { status: 'running', request, startedAt: Date.now(), finishedAt: null, result: null, error: '' }
    run.value = pending; busy.value = true
    try {
      const result = await dataLibraryApi.run(request)
      if (generation !== epoch || token !== runSeq) return null
      if (result.dataset !== request.dataset_id || result.model !== request.model || !result.run_id || !Array.isArray(result.metadata?.visualizations)) throw new Error('训练结果与本次数据或模型不匹配')
      run.value = { ...pending, status: 'completed', finishedAt: Date.now(), result }
      return result
    } catch (err) {
      if (generation === epoch && token === runSeq) run.value = { ...pending, status: 'failed', finishedAt: Date.now(), error: ['timeout', 'network_error'].includes(err.code) ? `${err.message}；服务端可能仍在计算，请勿重复提交。` : err.message }
      return null
    } finally { if (generation === epoch) busy.value = false }
  }
  return { items, detail, loading, detailLoading, uploading, error, detailError, uploadError, selection, compatibleModels, issues, warnings, checking, model, busy, run, canTrain, load, open, upload, uploadFile, checkCompatibility, train }
})
