<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useConnectionStore } from '../stores/connection'
import { useTrainingStore } from '../stores/training'
import { CONFIG } from '../config'
import { formatDuration } from '../utils/format'
import { algorithmNotes, metricDefinition, taskNames } from '../config/teaching'
import { downloadJson } from '../utils/download'
import AlgorithmList from '../components/experiment/AlgorithmList.vue'
import HyperparamForm from '../components/experiment/HyperparamForm.vue'
import RunMonitor from '../components/experiment/RunMonitor.vue'
import MetricCard from '../components/metrics/MetricCard.vue'
import ChartGallery from '../components/charts/ChartGallery.vue'
import JsonFallback from '../components/metrics/JsonFallback.vue'

const conn = useConnectionStore()
const training = useTrainingStore()
const selectedAlgo = ref(null)
const selectedDatasetId = ref('')
const initialValues = ref(null)
const split = ref({testSize: CONFIG.defaultTestSize, seed: CONFIG.defaultSeed})
const algorithms = computed(() => conn.registry.algorithms || [])
const datasets = computed(() => conn.registry.datasets || [])
const availableDatasets = computed(() => datasets.value.filter(d => selectedAlgo.value?.taskTypes?.includes(d.taskType) &&
  (!selectedAlgo.value.compatibleDatasets?.length || selectedAlgo.value.compatibleDatasets.includes(d.id))))
const selectedDataset = computed(() => datasets.value.find(d => d.id === selectedDatasetId.value))
const notes = computed(() => algorithmNotes(selectedAlgo.value?.id || ''))
watch(algorithms, list => { selectedAlgo.value = list.find(a => a.id === selectedAlgo.value?.id) || list[0] || null }, {immediate: true})
watch(() => selectedAlgo.value?.id, () => {
  initialValues.value = null
  selectedDatasetId.value = availableDatasets.value[0]?.id || ''
  split.value = {testSize: CONFIG.defaultTestSize, seed: CONFIG.defaultSeed}
}, {immediate: true})
watch(availableDatasets, list => {
  if (!list.some(d => d.id === selectedDatasetId.value)) selectedDatasetId.value = list[0]?.id || ''
})
function handleTrain(params) {
  if (!selectedAlgo.value || !selectedDataset.value || training.activeRuns.length) return
  training.startRun({modelId: selectedAlgo.value.id, modelName: selectedAlgo.value.name,
    datasetId: selectedDataset.value.id, datasetName: selectedDataset.value.name,
    taskType: selectedDataset.value.taskType, params, testSize: split.value.testSize, randomState: split.value.seed})
}
const selectedRunId = ref(training.history[0]?.runId || '')
const selectedRun = computed(() => training.history.find(run => run.runId === selectedRunId.value))
watch(() => training.history[0]?.runId, id => { selectedRunId.value = id || '' })
const visibleMetrics = computed(() => Object.entries(selectedRun.value?.result?.metrics || {})
  .filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
  .map(([id]) => conn.registry.metrics?.[selectedRun.value.taskType]?.find(metric => metric.id === id) || metricDefinition(id)))
const resultMetadata = computed(() => {
  const {visualizations, ...rest} = selectedRun.value?.result?.metadata || {}
  return rest
})
const gatingMessage = computed(() => conn.lastError || ({offline: '后端未连接，当前为离线配置预览。启动后端后点击右上角刷新。', unavailable: '后端可达，但算法模块尚不可用。', connecting: '正在连接后端并读取模型与数据集…'})[conn.status] || '正在读取真实模型配置…')
async function reproduce(run) {
  const algorithm = algorithms.value.find(a => a.id === run.modelId)
  if (!algorithm) { conn.addLog('warn', '当前后端未提供该历史模型，无法载入配置'); return }
  selectedAlgo.value = algorithm
  await nextTick()
  if (!availableDatasets.value.some(d => d.id === run.datasetId)) { conn.addLog('warn', '当前模型已不支持该历史数据集'); return }
  selectedDatasetId.value = run.datasetId
  split.value = {testSize: run.testSize ?? CONFIG.defaultTestSize, seed: run.randomState ?? CONFIG.defaultSeed}
  initialValues.value = {...run.params}
  document.getElementById('configuration')?.scrollIntoView({behavior: 'smooth', block: 'start'})
}
</script>

<template>
  <section class="workspace-hero">
    <div><div class="eyebrow">TEAM 07 · MACHINE LEARNING LAB</div><h1>让每一次实验，都看得见</h1><p>配置算法，运行真实训练，从指标与图表理解模型的行为。</p></div>
    <div class="hero-stats"><div><strong>{{ algorithms.length }}</strong><span>手写算法</span></div><div><strong>{{ datasets.length }}</strong><span>真实数据集</span></div><div><strong>4</strong><span>学习任务</span></div></div>
  </section>
  <div class="experiment-layout">
    <aside class="panel-left page-card"><div class="section-heading"><h3>算法库</h3><span class="eyebrow">SELECT MODEL</span></div>
      <AlgorithmList :algorithms="algorithms" :active-id="selectedAlgo?.id" @select="selectedAlgo = $event" />
      <div class="sidebar-footnote">所有结果由后端算法实际计算。离线模式仅供浏览配置。</div>
    </aside>
    <div class="panel-right">
      <section id="configuration" class="page-card configuration-card">
        <div class="section-heading"><div><span class="section-number">01</span><h2>参数配置</h2></div><el-tag effect="plain">{{ taskNames[selectedDataset?.taskType] || '选择任务' }}</el-tag></div>
        <div class="algorithm-intro"><h3>{{ selectedAlgo?.name || '选择算法' }}</h3><p>{{ notes.description }}</p></div>
        <el-alert v-if="!conn.canRunExperiments" :title="gatingMessage" type="warning" :closable="false" show-icon class="gating-alert" />
        <el-form label-position="top" class="dataset-form"><div class="dataset-row">
          <el-form-item label="实验数据集" class="dataset-select"><el-select v-model="selectedDatasetId" aria-label="实验数据集"><el-option v-for="d in availableDatasets" :key="d.id" :label="d.name" :value="d.id" /></el-select></el-form-item>
          <el-form-item v-if="selectedDataset && ['classification','regression'].includes(selectedDataset.taskType)" label="测试集比例"><el-select v-model="split.testSize" aria-label="测试集比例"><el-option v-for="ratio in [.1,.2,.3,.4,.5]" :key="ratio" :label="`${ratio * 100}% 测试 / ${100-ratio*100}% 训练`" :value="ratio" /></el-select></el-form-item>
          <el-form-item label="切分随机种子"><el-input-number v-model="split.seed" :min="0" :max="99999" controls-position="right" /></el-form-item>
        </div></el-form>
        <div v-if="selectedDataset" class="dataset-facts"><span>{{ selectedDataset.nSamples.toLocaleString() }} 个样本</span><span>{{ selectedDataset.nFeatures }} 个特征</span><span>{{ selectedDataset.taskType === 'classification' ? '分层随机切分' : selectedDataset.taskType === 'regression' ? '随机留出测试' : selectedDataset.taskType === 'clustering' ? '全量无监督聚类' : '拟合样本包含于评估集' }}</span></div>
        <div class="subsection-label">模型超参数 <span>悬停 ⓘ 查看参数含义；null 表示使用自动策略</span></div>
        <HyperparamForm :algorithm="selectedAlgo" :initial-values="initialValues" :disabled="!conn.canRunExperiments || !selectedDatasetId || training.activeRuns.length > 0" @submit="handleTrain" />
        <p class="learning-note"><el-icon><Opportunity /></el-icon>{{ notes.insight }}</p>
      </section>

      <section class="page-card">
        <div class="section-heading"><div><span class="section-number">02</span><h2>运行状态</h2></div><el-tag :type="training.activeRuns.length ? 'warning' : 'info'" effect="plain">{{ training.activeRuns.length ? '正在计算' : '等待实验' }}</el-tag></div>
        <RunMonitor v-for="run in [...training.activeRuns, ...training.errorRuns]" :key="run.id" :run="run" />
        <p v-if="!training.activeRuns.length && !training.errorRuns.length" class="muted">{{ selectedRun ? '最近一次实验已完成。可调整参数继续训练，或查看下方结果。' : '准备好参数后点击「开始训练」。运行期间显示实际请求状态与耗时。' }}</p>
        <el-collapse><el-collapse-item title="运行日志（连接与请求事件）" name="logs"><div class="run-log"><div v-for="(line, index) in conn.logLines.slice(-15)" :key="index" :class="`log-${line.level}`"><time>{{ new Date(line.ts).toLocaleTimeString('zh-CN', {hour12:false}) }}</time>{{ line.message }}</div><span v-if="!conn.logLines.length">暂无日志</span></div></el-collapse-item></el-collapse>
      </section>

      <section class="page-card result-section">
        <div class="section-heading"><div><span class="section-number">03</span><h2>实验结果</h2></div><el-button v-if="selectedRun" size="small" @click="downloadJson(selectedRun, `experiment-${selectedRun.runId}.json`)">导出完整结果</el-button></div>
        <div v-if="selectedRun">
          <div class="result-meta"><el-tag type="success" size="small">已完成</el-tag><strong>{{ selectedRun.modelName }}</strong><span>{{ selectedRun.datasetName }}</span><span>请求耗时 {{ formatDuration(selectedRun.elapsedMs || 0) }}</span></div>
          <p class="run-identity">{{ selectedRun.runId }}</p>
          <el-alert v-if="resultMetadata.evaluation_protocol" :title="resultMetadata.evaluation_protocol" :type="selectedRun.taskType === 'anomaly_detection' ? 'warning' : 'info'" :closable="false" class="protocol-alert" />
          <div class="metric-grid"><MetricCard v-for="metric in visibleMetrics" :key="metric.id" :metric="metric" :value="selectedRun.result.metrics[metric.id]" /></div>
          <div class="gallery-heading"><h3>可视化分析</h3><span class="muted">图表来自本次运行的真实数据 · 支持放大与图片下载</span></div>
          <ChartGallery :visualizations="selectedRun.result.metadata?.visualizations || []" />
          <el-collapse class="result-details"><el-collapse-item title="实际生效参数与评估说明"><div class="metadata-grid"><div><h4>实际生效参数</h4><JsonFallback :value="selectedRun.result.effective_params" /></div><div><h4>评估与数据元信息</h4><JsonFallback :value="resultMetadata" /></div></div></el-collapse-item><el-collapse-item title="全部评价指标"><JsonFallback :value="selectedRun.result.metrics" /></el-collapse-item></el-collapse>
        </div>
        <el-empty v-else description="完成第一次训练后，在这里查看指标与图表" :image-size="80" />
      </section>

      <section class="page-card">
        <div class="section-heading"><div><span class="section-number">04</span><h2>实验历史</h2></div><div><el-button v-if="training.history.length" text size="small" @click="downloadJson(training.history, 'experiment-history.json')">导出历史</el-button><el-popconfirm title="清除已完成的实验记录？正在运行的实验会保留。" @confirm="training.clearRuns()"><template #reference><el-button :disabled="!training.history.length && !training.errorRuns.length" text size="small">清空历史</el-button></template></el-popconfirm></div></div>
        <el-table v-if="training.history.length" :data="training.history" row-key="runId" size="small" class="history-table"><el-table-column label="算法 / 数据集" min-width="180"><template #default="{row}"><strong>{{ row.modelName }}</strong><div class="muted">{{ row.datasetName }}</div></template></el-table-column><el-table-column label="完成时间" min-width="130"><template #default="{row}">{{ new Date(row.finishedAt).toLocaleString('zh-CN',{hour12:false}) }}</template></el-table-column><el-table-column label="种子" prop="randomState" width="70" /><el-table-column label="操作" width="165"><template #default="{row}"><el-button text size="small" type="primary" @click="selectedRunId = row.runId">查看</el-button><el-button text size="small" @click="reproduce(row)">载入参数</el-button></template></el-table-column></el-table>
        <p v-else class="muted">暂无完成的实验。历史仅在当前浏览器保存，可导出 JSON 留存。</p>
      </section>
    </div>
  </div>
</template>

<style scoped>
.experiment-layout{display:grid;grid-template-columns:245px minmax(0,1fr);gap:22px;align-items:start}
.panel-left{position:sticky;top:18px;max-height:calc(100vh - 110px);overflow:auto;padding:18px 12px}
.panel-right{min-width:0}.panel-left .section-heading{padding:0 8px}.sidebar-footnote{font-size:11px;color:#8a9b91;line-height:1.8;margin:16px 10px 0;border-top:1px solid #edf1ee;padding-top:12px}
.algorithm-intro h3{margin:0 0 5px;font-size:20px}.algorithm-intro p{font-size:13px;color:#718378;line-height:1.7;margin:0 0 22px}
.dataset-row{display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:18px}.dataset-row .el-select,.dataset-row .el-input-number{width:100%}
.dataset-facts{display:flex;gap:18px;font-size:12px;color:#658071;background:#f4f9f6;padding:11px 13px;border-radius:7px;margin:-3px 0 20px;flex-wrap:wrap}
.subsection-label{font-size:13px;font-weight:600;border-top:1px solid #eaf0ec;padding-top:18px;margin-bottom:18px}.subsection-label span{font-size:11px;font-weight:400;color:#94a298;margin-left:12px}
.learning-note{display:flex;align-items:flex-start;gap:8px;color:#688675;background:#f3f9f5;line-height:1.8;font-size:12px;padding:12px;border-radius:8px;margin:18px 0 0}
.learning-note .el-icon{margin-top:4px;flex-shrink:0}.gating-alert,.protocol-alert{margin-bottom:18px}
.run-log{background:#f7faf8;border:1px solid #e8efea;border-radius:8px;padding:12px;font-family:Consolas,monospace;font-size:12px;line-height:2;max-height:230px;overflow:auto}.run-log time{color:#8c9d91;margin-right:15px}.log-error{color:#b6524c}.log-warn{color:#a27b2e}
.result-meta{display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:13px}.result-meta span{color:#7f8d83}.run-identity{font-family:monospace;color:#9ca99f;font-size:11px;margin:12px 0}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:18px 0 28px}.gallery-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:15px;flex-wrap:wrap}.gallery-heading h3{margin:0;font-size:16px}
.result-details{margin-top:18px}.metadata-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.metadata-grid>div{min-width:0}
@media(max-width:1050px){.experiment-layout{grid-template-columns:215px minmax(0,1fr)}.dataset-row{grid-template-columns:1fr 1fr}.dataset-select{grid-column:1/-1}}
@media(max-width:760px){.experiment-layout{grid-template-columns:1fr}.panel-left{position:static;max-height:280px}.dataset-row,.metadata-grid{grid-template-columns:1fr}.dataset-select{grid-column:auto}}
</style>
