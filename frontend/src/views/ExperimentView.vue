<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useConnectionStore } from '../stores/connection.js'
import { useOriginalExperimentsStore } from '../stores/originalExperiments.js'
import { originalTaskNames, originalTaskOrder, originalModelNotes } from '../config/originalTeaching.js'
import OriginalFigureGallery from '../components/original/OriginalFigureGallery.vue'
import OriginalRunCard from '../components/original/OriginalRunCard.vue'
import InteractiveExperiment from '../components/original/InteractiveExperiment.vue'
import { createRunLinkResolver } from '../utils/originalSelection.js'

const route = useRoute()
const router = useRouter()
const conn = useConnectionStore()
const store = useOriginalExperimentsStore()
const task = ref('classification')
const resultTab = ref('archive')
const selectedRunId = ref('')
const runLink = createRunLinkResolver()
let pendingExperimentId = ''
const grouped = computed(() => store.experiments.filter(item => item.task === task.value))
const currentRuns = computed(() => store.runs.filter(run => run.experiment_id === store.selectedId))
const selectedRun = computed(() => currentRuns.value.find(run => run.run_id === selectedRunId.value) || currentRuns.value[0] || null)
const note = computed(() => originalModelNotes[store.selectedId] || '查看原实验中的模型行为、评价结果及绘图过程。')
const canRunMessage = computed(() => {
  if (store.catalogError) return '无法读取原实验目录，请检查后端服务后重试。'
  if (conn.status === 'offline') return '后端未连接，暂不能运行完整原实验。'
  if (conn.status === 'unavailable') return '后端算法模块不可用，暂不能运行完整原实验。'
  if (conn.status !== 'connected') return '正在连接后端服务…'
  if (store.activeRunIds.length) return '已有原实验在排队或运行；后台按顺序执行。'
  return ''
})

function clearRouteSelection() {
  runLink.cancel()
  pendingExperimentId = ''
  if (route.query.run || route.query.experiment) router.replace({ query: { ...route.query, run: undefined, experiment: undefined } })
}
function selectExperiment(item) {
  clearRouteSelection()
  store.selectedId = item.id
  selectedRunId.value = ''
  resultTab.value = 'archive'
}
function selectTask(key) {
  task.value = key
  const first = store.experiments.find(item => item.task === key)
  if (first) selectExperiment(first)
}
async function startRun() {
  const run = await store.startRun()
  if (run) { clearRouteSelection(); selectedRunId.value = run.run_id; resultTab.value = 'run' }
}
async function reload() { await Promise.all([store.loadCatalog(), store.loadRuns()]) }
watch(() => store.selectedExperiment?.task, value => { if (value) task.value = value })
watch(() => conn.baseUrl, async () => { await nextTick(); reload() })
function resolveRouteSelection() {
  const run = runLink.take(store.runs)
  if (run) { store.selectedId = run.experiment_id; selectedRunId.value = run.run_id; resultTab.value = 'run'; pendingExperimentId = '' }
  else if (pendingExperimentId && store.experiments.some(item => item.id === pendingExperimentId)) {
    store.selectedId = pendingExperimentId
    selectedRunId.value = ''
    resultTab.value = 'archive'
    pendingExperimentId = ''
  }
}
watch([() => route.query.run, () => route.query.experiment], ([runId, experimentId]) => {
  runLink.setPending(runId)
  pendingExperimentId = typeof experimentId === 'string' ? experimentId : ''
  resolveRouteSelection()
}, { immediate: true })
watch(() => store.runs.map(run => run.run_id).join(','), resolveRouteSelection)
watch(() => store.experiments, resolveRouteSelection)
onMounted(reload)
</script>

<template>
  <section class="page-intro"><div><div class="eyebrow">ORIGINAL EXPERIMENTS / FIGURE ATLAS</div><h1>原实验图谱</h1><p>按原绘图脚本浏览 12 套实验与 72 张历史原图；完整复现会生成独立的新产物。</p></div><div class="intro-counts"><div><strong>{{ store.experiments.length }}</strong><span>原始算法</span></div><div><strong>{{ store.experiments.reduce((sum, item) => sum + (item.figures?.length || 0), 0) }}</strong><span>历史图像</span></div></div></section>
  <el-alert v-if="store.catalogError" :title="store.catalogError" description="目录加载失败，确认后端地址与服务状态后刷新。" type="error" :closable="false" show-icon class="page-alert"><el-button size="small" @click="reload">重新加载</el-button></el-alert>
  <div v-if="store.catalogLoading && !store.experiments.length" v-loading="true" class="page-card loading-card">正在读取原实验目录…</div>
  <div v-else-if="store.experiments.length" class="experiment-shell">
    <aside class="experiment-selector page-card"><div class="selector-heading"><strong>实验目录</strong><span>选择任务与算法</span></div><div class="task-selector"><button v-for="key in originalTaskOrder" :key="key" type="button" :class="{ active: task === key }" @click="selectTask(key)">{{ originalTaskNames[key] }}<span>{{ store.experiments.filter(item => item.task === key).length }}</span></button></div><div class="selector-divider" /><button v-for="item in grouped" :key="item.id" class="model-option" :data-experiment-id="item.id" type="button" :class="{ active: store.selectedId === item.id }" @click="selectExperiment(item)"><span class="model-option-mark" /><span>{{ item.title }}</span><small>{{ item.figures?.length || 0 }} 图</small></button></aside>
    <div v-if="store.selectedExperiment" class="experiment-content">
      <section class="page-card model-overview"><div class="overview-top"><div><div class="eyebrow">{{ originalTaskNames[store.selectedExperiment.task] }} · {{ store.selectedExperiment.id }}</div><h2>{{ store.selectedExperiment.title }}</h2><p>{{ note }}</p></div><span class="script-chip">{{ store.selectedExperiment.script }}</span></div><div class="overview-meta"><div><span>原始数据来源</span><div v-if="store.selectedExperiment.datasets?.length" class="dataset-pills"><span v-for="dataset in store.selectedExperiment.datasets" :key="dataset.id" :title="dataset.path">{{ dataset.name }}</span></div><strong v-else>原脚本生成的合成数据 / 机制演示</strong></div><div><span>历史原图</span><strong>{{ store.selectedExperiment.figures?.length || 0 }} 张 PNG</strong></div><div><span>本实验历史</span><strong>{{ currentRuns.length }} 次复现</strong></div></div></section>
      <section class="page-card gallery-section"><div class="card-title-row"><div><div class="eyebrow">01 / ORIGINAL FIGURES</div><h2>原始实验可视化 · 六张原图</h2></div><span class="section-side-note">点击任一图片可查看原尺寸</span></div><el-tabs v-model="resultTab" class="result-tabs"><el-tab-pane label="历史归档原图" name="archive"><p class="archive-banner">以下图像是项目已有的原始成果，并非本次启动复现所得。</p><OriginalFigureGallery :figures="store.selectedExperiment.figures || []" :source-data="store.selectedExperiment.source_data || []" :experiment-id="store.selectedId" archived /></el-tab-pane><el-tab-pane label="本次复现产物" name="run"><div v-if="currentRuns.length" class="run-picker"><span>选择运行</span><el-select v-model="selectedRunId" placeholder="最近一次运行" clearable><el-option v-for="run in currentRuns" :key="run.run_id" :label="`${run.run_id} · ${run.status}`" :value="run.run_id" /></el-select></div><div v-if="selectedRun"><OriginalRunCard :run="selectedRun" /><p v-if="selectedRun.status !== 'completed'" class="section-hint">本次运行结束后，服务端返回的图像与 CSV 将显示在下方。</p><OriginalFigureGallery v-if="selectedRun.figures?.length" :figures="selectedRun.figures" :source-data="selectedRun.source_data || []" :experiment-id="store.selectedId" /></div><el-empty v-else description="尚无本算法复现记录；启动原实验后可在这里查看新产物" :image-size="80" /></el-tab-pane></el-tabs></section>
      <InteractiveExperiment :key="store.selectedId" :experiment-id="store.selectedId" />
      <details class="original-reproduction-details">
        <summary>完整原实验协议与复现（高级）</summary>
      <section class="page-card protocol-card"><div class="card-title-row"><div><div class="eyebrow">EXPERIMENT PROTOCOL</div><h2>原实验协议</h2></div><span class="read-only-pill">只读 · 源脚本默认值</span></div><p class="section-hint">复现仅选择实验，不传测试比例、随机种子或替代数据；具体流程由原绘图脚本决定。</p><div class="protocol-columns"><div><h3>脚本参数</h3><dl v-if="Object.keys(store.selectedExperiment.parameters || {}).length" class="parameter-list"><div v-for="(value, key) in store.selectedExperiment.parameters" :key="key"><dt>{{ key }}</dt><dd>{{ typeof value === 'object' ? JSON.stringify(value) : String(value) }}</dd></div></dl><p v-else class="run-muted">采用脚本内定义的默认参数。</p></div><div><h3>实验步骤</h3><ol v-if="store.selectedExperiment.protocol?.length" class="protocol-list"><li v-for="line in store.selectedExperiment.protocol" :key="line">{{ line }}</li></ol><p v-else class="run-muted">请参阅原绘图脚本。</p></div></div></section>
      <section class="page-card run-launch-card"><div class="card-title-row"><div><div class="eyebrow">FULL REPRODUCTION</div><h2>完整复现</h2></div><el-button type="primary" :loading="store.submitting" :disabled="!store.canRun" @click="startRun">启动原实验</el-button></div><p class="section-hint">后台串行执行完整原脚本。新图写入本次运行的独立目录，原目录和历史原图不会被覆盖。</p><el-alert v-if="canRunMessage" :title="canRunMessage" type="warning" :closable="false" class="run-alert" /><el-alert v-if="store.runError" :title="store.runError" type="error" :closable="false" class="run-alert" /><el-alert v-if="store.runsError" :title="`运行状态刷新失败：${store.runsError}`" type="error" :closable="false" class="run-alert" /><OriginalRunCard v-if="selectedRun" :run="selectedRun" compact /></section>
      </details>
    </div>
  </div>
  <el-empty v-else-if="!store.catalogError" description="后端尚未提供原实验目录" />
</template>
