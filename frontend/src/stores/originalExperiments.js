import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useConnectionStore } from './connection.js'
import { originalApi } from '../services/originalApi.js'

const ACTIVE = new Set(['queued', 'running'])

export const useOriginalExperimentsStore = defineStore('originalExperiments', () => {
  const conn = useConnectionStore()
  const experiments = ref([])
  const datasets = ref([])
  const runs = ref([])
  const selectedId = ref('')
  const catalogLoading = ref(false)
  const datasetsLoading = ref(false)
  const runsLoading = ref(false)
  const submitting = ref(false)
  const catalogError = ref('')
  const datasetsError = ref('')
  const runsError = ref('')
  const runError = ref('')
  let generation = 0
  let pollTimer = null
  let runsRevision = 0
  let listRequest = 0
  const runRequests = new Map()

  const selectedExperiment = computed(() => experiments.value.find(item => item.id === selectedId.value) || null)
  const activeRunIds = computed(() => runs.value.filter(run => ACTIVE.has(run.status)).map(run => run.run_id))
  const canRun = computed(() => conn.status === 'connected' && conn.mlBackendAvailable &&
    !!selectedExperiment.value && !catalogError.value && !runsLoading.value && !runsError.value &&
    !submitting.value && !activeRunIds.value.length)

  function stopPolling() {
    if (pollTimer !== null) clearTimeout(pollTimer)
    pollTimer = null
  }

  function schedulePolling() {
    stopPolling()
    if (!activeRunIds.value.length) return
    pollTimer = setTimeout(async () => {
      pollTimer = null
      await Promise.all(activeRunIds.value.map(id => refreshRun(id)))
      if (activeRunIds.value.length) schedulePolling()
    }, 2500)
  }

  function reset() {
    generation++
    runsRevision++
    listRequest++
    runRequests.clear()
    stopPolling()
    experiments.value = []
    datasets.value = []
    runs.value = []
    selectedId.value = ''
    catalogError.value = ''
    datasetsError.value = ''
    runsError.value = ''
    runError.value = ''
    catalogLoading.value = false
    datasetsLoading.value = false
    runsLoading.value = false
    submitting.value = false
  }

  watch(() => conn.baseUrl, reset)

  async function loadCatalog() {
    const current = generation
    catalogLoading.value = true
    catalogError.value = ''
    try {
      const result = await originalApi.listExperiments()
      if (current !== generation) return
      experiments.value = Array.isArray(result) ? result : []
      if (!experiments.value.some(item => item.id === selectedId.value)) selectedId.value = experiments.value[0]?.id || ''
    } catch (error) {
      if (current !== generation) return
      experiments.value = []
      selectedId.value = ''
      catalogError.value = error.message
    } finally {
      if (current === generation) catalogLoading.value = false
    }
  }

  async function loadDatasets() {
    const current = generation
    datasetsLoading.value = true
    datasetsError.value = ''
    try {
      const result = await originalApi.listDatasets()
      if (current !== generation) return
      datasets.value = Array.isArray(result) ? result : []
    } catch (error) {
      if (current !== generation) return
      datasets.value = []
      datasetsError.value = error.message
    } finally {
      if (current === generation) datasetsLoading.value = false
    }
  }

  async function loadRuns() {
    const current = generation
    const revision = runsRevision
    const request = ++listRequest
    runsLoading.value = true
    runsError.value = ''
    try {
      const result = await originalApi.listRuns()
      if (current !== generation || request !== listRequest) return
      const listing = Array.isArray(result) ? result : []
      if (revision === runsRevision) runs.value = listing
      else {
        const known = new Set(runs.value.map(run => run.run_id))
        runs.value = [...runs.value, ...listing.filter(run => !known.has(run.run_id))]
      }
      schedulePolling()
    } catch (error) {
      if (current !== generation || request !== listRequest) return
      runsError.value = error.message
    } finally {
      if (current === generation && request === listRequest) runsLoading.value = false
    }
  }

  async function refreshRun(runId) {
    const current = generation
    const request = (runRequests.get(runId) || 0) + 1
    runRequests.set(runId, request)
    try {
      const run = await originalApi.getRun(runId)
      if (current !== generation || request !== runRequests.get(runId)) return false
      const index = runs.value.findIndex(item => item.run_id === runId)
      if (index >= 0) runs.value.splice(index, 1, run)
      else runs.value.unshift(run)
      runsRevision++
      runsError.value = ''
      if (!activeRunIds.value.length) stopPolling()
      return ACTIVE.has(run.status)
    } catch (error) {
      if (current === generation && request === runRequests.get(runId)) runsError.value = error.message
      return false
    }
  }

  async function startRun() {
    if (!canRun.value) return null
    const current = generation
    submitting.value = true
    runError.value = ''
    try {
      const run = await originalApi.startRun(selectedId.value)
      if (current !== generation) return null
      runs.value.unshift(run)
      runsRevision++
      schedulePolling()
      return run
    } catch (error) {
      if (current === generation) runError.value = error.message
      return null
    } finally {
      if (current === generation) submitting.value = false
    }
  }

  return { experiments, datasets, runs, selectedId, selectedExperiment, activeRunIds, canRun,
    catalogLoading, datasetsLoading, runsLoading, submitting, catalogError, datasetsError, runsError, runError,
    loadCatalog, loadDatasets, loadRuns, refreshRun, startRun, stopPolling, reset }
})
