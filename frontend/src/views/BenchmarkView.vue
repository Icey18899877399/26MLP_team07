<script setup>
import { computed, ref, watch } from 'vue'
import { useTrainingStore } from '../stores/training'
import { useConnectionStore } from '../stores/connection'
import { areComparable } from '../services/adapters'
import { metricDefinition } from '../config/teaching'
import { downloadJson } from '../utils/download'
import BaseChart from '../components/charts/BaseChart.vue'
const training = useTrainingStore()
const conn = useConnectionStore()
const datasetId = ref('')
const referenceId = ref('')
const metricId = ref('')
const datasets = computed(() => [...new Set(training.history.map(run => run.datasetId))])
const datasetRuns = computed(() => training.history.filter(run => run.datasetId === datasetId.value))
const reference = computed(() => datasetRuns.value.find(run => run.runId === referenceId.value))
const comparable = computed(() => datasetRuns.value.filter(run => areComparable(run, reference.value)))
const metrics = computed(() => Object.keys(reference.value?.result?.metrics || {}).filter(id => typeof reference.value.result.metrics[id] === 'number'))
watch(datasets, ids => {if (!ids.includes(datasetId.value)) datasetId.value=ids[0] || ''}, {immediate:true})
watch(datasetRuns, runs => {referenceId.value=runs[0]?.runId || ''}, {immediate:true})
watch(metrics, ids => {metricId.value=ids.includes(metricId.value) ? metricId.value : ids[0] || ''}, {immediate:true})
function label(id) { return conn.registry.metrics?.[reference.value?.taskType]?.find(metric => metric.id === id)?.name || metricDefinition(id).name }
function number(value) {return typeof value === 'number' ? Number(value.toPrecision(5)).toLocaleString('zh-CN', {maximumFractionDigits:5}) : '—'}
const chart = computed(() => ({tooltip:{trigger:'axis'},grid:{left:80,right:30,bottom:90,top:30,containLabel:true},xAxis:{type:'category',data:comparable.value.map((run,i)=>`${run.modelName}\n#${i+1}`),axisLabel:{interval:0,rotate:20}},yAxis:{type:'value',name:label(metricId.value)},series:[{name:label(metricId.value),type:'bar',data:comparable.value.map(run=>run.result.metrics[metricId.value] ?? null),barMaxWidth:55,itemStyle:{color:'#39996e',borderRadius:[5,5,0,0]}}]}))
</script>

<template>
  <section class="workspace-hero"><div><div class="eyebrow">EXPERIMENT COMPARISON</div><h1>在相同条件下，比较模型</h1><p>仅比较真实完成的实验。数据集、随机种子、切分比例与评价协议必须一致。</p></div></section>
  <section class="page-card"><div class="compare-controls"><el-select v-model="datasetId" placeholder="选择已运行的数据集" aria-label="选择已运行的数据集"><el-option v-for="id in datasets" :key="id" :value="id" :label="conn.registry.datasets.find(ds=>ds.id===id)?.name || id" /></el-select><el-select v-model="referenceId" placeholder="选择基准实验" aria-label="选择基准实验"><el-option v-for="run in datasetRuns" :key="run.runId" :value="run.runId" :label="`${run.modelName} · 种子 ${run.randomState} · ${run.runId.slice(-8)}`" /></el-select><el-select v-model="metricId" placeholder="比较指标" aria-label="比较指标"><el-option v-for="id in metrics" :key="id" :value="id" :label="label(id)" /></el-select></div>
    <template v-if="reference"><el-alert :title="`符合相同条件的实验 ${comparable.length} 条，排除条件不同的实验 ${datasetRuns.length-comparable.length} 条。`" type="info" :closable="false" /><p class="muted">{{ reference.result.metadata?.evaluation_protocol }} · 种子 {{ reference.randomState }}<template v-if="['classification','regression'].includes(reference.taskType)"> · 测试比例 {{ reference.testSize*100 }}%</template></p><BaseChart :option="chart" height="380px" /><el-table :data="comparable"><el-table-column label="算法" prop="modelName" min-width="150"/><el-table-column v-for="id in metrics" :key="id" :label="label(id)" min-width="120"><template #default="{row}">{{ number(row.result.metrics[id]) }}</template></el-table-column><el-table-column label="参数" min-width="180"><template #default="{row}"><code>{{ JSON.stringify(row.result.effective_params || row.params) }}</code></template></el-table-column></el-table><el-button class="export-button" @click="downloadJson(comparable,'comparable-experiments.json')">导出本组对比结果</el-button></template>
    <el-empty v-else description="先在实验工作台完成训练，再来比较结果" :image-size="100" />
  </section>
</template>

<style scoped>
.compare-controls{display:grid;grid-template-columns:1fr 1.5fr 1fr;gap:16px;margin-bottom:20px}.export-button{margin-top:20px}code{font-size:11px;word-break:break-all}@media(max-width:700px){.compare-controls{grid-template-columns:1fr}}
</style>
