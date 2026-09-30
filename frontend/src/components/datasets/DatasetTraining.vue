<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useDataLibraryStore } from '../../stores/dataLibrary.js'
import { useInteractiveStore } from '../../stores/interactive.js'
import { useConnectionStore } from '../../stores/connection.js'
import { interactiveAlgorithm } from '../../utils/interactiveParams.js'
import HyperparamForm from '../experiment/HyperparamForm.vue'
import ChartGallery from '../charts/ChartGallery.vue'
import { METRICS } from '../../config/metrics.js'
const store = useDataLibraryStore(), catalog = useInteractiveStore(), conn = useConnectionStore()
const seed = ref(42), testSize = ref(0.2), now = ref(Date.now())
const entry = computed(() => catalog.catalog.find(item => item.model === store.model))
const form = computed(() => interactiveAlgorithm(entry.value))
const elapsed = computed(() => store.run ? Math.max(0, Math.floor(((store.run.finishedAt || now.value) - store.run.startedAt) / 1000)) : 0)
const metricNames = Object.fromEntries(Object.values(METRICS).flat().map(item => [item.id, item.name]))
const supervised = computed(() => ['classification', 'regression'].includes(store.selection.task))
watch(entry, value => { if (value) seed.value = value.random_state })
watch(() => store.selection.target_column, target => { store.selection.feature_columns = store.selection.feature_columns.filter(name => name !== target) })
let timer
onMounted(() => { if (!catalog.catalog.length && !catalog.loading) catalog.loadCatalog(); timer = setInterval(() => { now.value = Date.now() }, 1000) })
onUnmounted(() => clearInterval(timer))
function start(params) { now.value = Date.now(); store.train(params, seed.value, testSize.value) }
function name(id) { return catalog.catalog.find(item => item.model === id)?.title || id }
</script>
<template>
  <section class="page-card dataset-training">
    <div class="card-title-row"><div><div class="eyebrow">BUILD AN EXPERIMENT</div><h2>使用此数据集训练</h2></div><span class="section-side-note">选择字段 → 检查兼容性 → 调参</span></div>
    <p class="section-hint">特征必须是有限数值。缺失值和文本特征会明确提示，不会被自动删除或替换。分类暂支持二分类；异常标签如提供，须用 0 表示正常、1 表示异常。</p>
    <fieldset :disabled="store.busy" class="dataset-fieldset">
      <div class="dataset-selection-grid">
        <label>任务类型<select v-model="store.selection.task" aria-label="任务类型"><option value="classification">分类</option><option value="regression">回归</option><option value="clustering">聚类</option><option value="anomaly_detection">异常检测</option></select></label>
        <label>{{ supervised ? '标签列（必选）' : '参考标签（可选）' }}<select v-model="store.selection.target_column" aria-label="标签列"><option :value="null">不使用标签</option><option v-for="column in store.detail.columns" :key="column.name" :value="column.name">{{ column.name }}</option></select></label>
      </div>
      <div class="feature-heading"><strong>特征列</strong><span>已选择 {{ store.selection.feature_columns.length }} 列；请手动取消 ID 等非预测字段</span></div>
      <div class="feature-options"><label v-for="column in store.detail.columns" :key="column.name" :class="{ 'field-unavailable': column.dtype !== 'number' || column.name === store.selection.target_column }"><input v-model="store.selection.feature_columns" type="checkbox" :value="column.name" :disabled="column.dtype !== 'number' || column.name === store.selection.target_column" :aria-label="`特征 ${column.name}`"><span>{{ column.name }}</span><small v-if="column.name === store.selection.target_column">标签</small><small v-else-if="column.dtype !== 'number'">非数值</small><small v-else-if="column.missing_count">缺失 {{ column.missing_count }}</small></label></div>
    </fieldset>
    <el-button :loading="store.checking" :disabled="store.busy || conn.status !== 'connected'" @click="store.checkCompatibility">检查可用算法</el-button>
    <el-alert v-for="issue in store.issues" :key="issue" :title="issue" type="warning" :closable="false" show-icon class="run-alert" />
    <el-alert v-for="warning in store.warnings" :key="warning" :title="warning" type="info" :closable="false" show-icon class="run-alert" />
    <p v-if="conn.status !== 'connected'" class="section-hint">后端未连接，暂不能校验或训练。</p>
    <div v-if="store.compatibleModels.length" class="compatible-config">
      <div class="dataset-selection-grid">
        <label>可用算法<select v-model="store.model" aria-label="可用算法" :disabled="store.busy"><option v-for="id in store.compatibleModels" :key="id" :value="id">{{ name(id) }}</option></select></label>
        <label>随机种子<el-input-number v-model="seed" aria-label="训练随机种子" :min="0" :max="4294967295" :disabled="store.busy" /></label>
        <label v-if="supervised">测试集比例<el-input-number v-model="testSize" aria-label="训练测试集比例" :min="0.05" :max="0.5" :step="0.05" :disabled="store.busy" /></label>
      </div>
      <p class="section-hint">采用算法起始参数进行单次训练；本数据集的字段和划分以本次提交为准，不改变原图实验。</p>
      <HyperparamForm v-if="form" :key="store.model" :algorithm="form" :disabled="!store.canTrain || !Number.isInteger(seed) || (supervised && !Number.isFinite(testSize))" @submit="start" />
      <el-alert v-else-if="catalog.error" :title="catalog.error" type="error" :closable="false"><el-button @click="catalog.loadCatalog">重载参数</el-button></el-alert>
    </div>
    <div v-if="store.run" class="dataset-run-status" role="status" aria-live="polite"><el-tag :type="store.run.status === 'completed' ? 'success' : store.run.status === 'running' ? 'warning' : 'danger'">{{ {running:'真实训练中',completed:'训练完成',failed:'训练未完成'}[store.run.status] }}</el-tag><span>{{ elapsed }} 秒</span></div>
    <el-alert v-if="store.run?.error" :title="store.run.error" type="error" :closable="false" />
    <div v-if="store.run?.result" :key="store.run.result.run_id" class="dataset-training-result">
      <h3>本次训练结果</h3><p class="section-hint">图表对应本次提交的参数。调整字段或算法后会清空旧结果，避免混淆。</p>
      <details class="dataset-result-config"><summary>实际参数、字段与评估协议</summary><pre>{{ JSON.stringify({ ...store.run.request, params: store.run.result.effective_params, protocol: store.run.result.metadata.evaluation_protocol, train_samples: store.run.result.metadata.train_sample_count, test_samples: store.run.result.metadata.test_sample_count, labels: store.run.result.metadata.label_names }, null, 2) }}</pre></details>
      <div class="dataset-metrics"><div v-for="(value,key) in store.run.result.metrics" :key="key"><span>{{ metricNames[key] || key }}</span><strong>{{ typeof value === 'number' ? Number(value.toPrecision(5)) : value }}</strong></div></div>
      <ChartGallery :visualizations="store.run.result.metadata.visualizations" />
    </div>
  </section>
</template>
<style scoped>
.dataset-fieldset{border:0;padding:0;margin:0 0 18px;min-width:0}.dataset-selection-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:14px 0 22px}.dataset-selection-grid>label{display:flex;flex-direction:column;gap:9px;font-size:12px;color:#61736f}.dataset-selection-grid select{width:100%;min-width:0;padding:10px;border:1px solid #dbe5e2;border-radius:6px;background:white;color:#364e46}.dataset-selection-grid .el-input-number{width:100%}.feature-heading{display:flex;align-items:baseline;flex-wrap:wrap;gap:12px;font-size:12px;margin-bottom:12px}.feature-heading span{font-size:11px;color:#8a9894}.feature-options{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:8px;max-height:235px;overflow:auto;border:1px solid #e6edeb;padding:12px;border-radius:8px}.feature-options label{display:flex;align-items:center;gap:6px;min-width:0;font-size:11px;color:#526b62}.feature-options span{overflow-wrap:anywhere}.feature-options small{color:#97a49f;white-space:nowrap}.field-unavailable{opacity:.6}.compatible-config{margin-top:22px;border-top:1px solid #e6edeb;padding-top:8px}.dataset-run-status{display:flex;gap:12px;align-items:center;font-size:12px;margin:22px 0;color:#7e8c87}.dataset-training-result h3{font-size:17px}.dataset-result-config{font-size:12px;color:#607a6d}.dataset-result-config summary{cursor:pointer}.dataset-result-config pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f6;padding:14px;border-radius:8px}.dataset-metrics{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}.dataset-metrics>div{background:#f0f7f4;border-radius:7px;padding:12px 18px;min-width:110px;display:flex;flex-direction:column;gap:8px}.dataset-metrics span{font-size:11px;color:#81938b}.dataset-metrics strong{font-size:20px;color:#315447}
</style>
