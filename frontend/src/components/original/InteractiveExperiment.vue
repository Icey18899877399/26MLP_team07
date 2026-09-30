<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useInteractiveStore } from '../../stores/interactive.js'
import { useConnectionStore } from '../../stores/connection.js'
import { interactiveAlgorithm } from '../../utils/interactiveParams.js'
import HyperparamForm from '../experiment/HyperparamForm.vue'
import ChartGallery from '../charts/ChartGallery.vue'

const props = defineProps({ experimentId: { type: String, required: true } })
const store = useInteractiveStore()
const conn = useConnectionStore()
const entry = computed(() => store.catalog.find(item => item.id === props.experimentId))
const record = computed(() => store.records[props.experimentId])
const dataset = ref('')
const testSize = ref(null)
const seed = ref(42)
const now = ref(Date.now())
const elapsed = computed(() => record.value ? Math.max(0, Math.floor(((record.value.finishedAt || now.value) - record.value.startedAt) / 1000)) : 0)
const algorithm = computed(() => interactiveAlgorithm(entry.value))
const labels = { accuracy: '准确率', precision: '精确率', recall: '召回率', f1: 'F1', roc_auc: 'ROC AUC', pr_auc: 'PR AUC', rmse: 'RMSE', mse: 'MSE', mae: 'MAE', r2: 'R²', silhouette: '轮廓系数', silhouette_score: '轮廓系数', inertia: '簇内平方和', ari: '调整兰德指数', n_clusters: '簇数量', noise_ratio: '噪声比例' }
const statusText = { running: '真实训练中', completed: '训练完成', failed: '训练失败', unknown: '运行状态待确认' }
watch(entry, value => {
  if (!value) return
  dataset.value = value.dataset
  testSize.value = value.test_size
  seed.value = value.random_state
}, { immediate: true })
watch(() => conn.baseUrl, () => store.loadCatalog())
let clock
onMounted(() => {
  if (!store.catalog.length && !store.loading) store.loadCatalog()
  clock = setInterval(() => { now.value = Date.now() }, 1000)
})
onUnmounted(() => clearInterval(clock))
function start(params) { now.value = Date.now(); store.run(props.experimentId, params, dataset.value, testSize.value, seed.value) }
function metric(value) { return typeof value === 'number' ? new Intl.NumberFormat('zh-CN', { maximumSignificantDigits: 5 }).format(value) : String(value ?? '—') }
</script>

<template>
  <section class="page-card interactive-panel">
    <div class="card-title-row"><div><div class="eyebrow">02 / PARAMETER LAB</div><h2>调参训练与动态可视化</h2></div><span class="read-only-pill">参数可编辑 · 真实计算</span></div>
    <p class="section-hint">修改参数后点击“开始训练”，下方指标与图表会在计算完成后更新。此区是单次训练探索，不替代上方 6 张原图，也不等同于完整交叉验证与消融实验。</p>
    <el-alert v-if="store.error" :title="store.error" type="error" :closable="false" show-icon><el-button @click="store.loadCatalog">重新加载参数</el-button></el-alert>
    <p v-else-if="store.loading" role="status">正在读取原始实验的可调参数…</p>
    <template v-else-if="entry">
      <div class="interactive-provenance"><strong>数据与参数来源</strong><code>{{ entry.source }}</code><p>{{ entry.protocol }}</p><code v-for="source in entry.datasets" :key="source.id">{{ source.name }} · {{ source.path }}</code></div>
      <el-form label-position="top" class="interactive-settings">
        <el-form-item label="原始数据集"><el-select v-model="dataset" :disabled="store.busy || entry.datasets.length === 1" aria-label="原始数据集"><el-option v-for="source in entry.datasets" :key="source.id" :label="source.name" :value="source.id" /></el-select></el-form-item>
        <el-form-item v-if="entry.test_size != null" label="测试集比例"><el-input-number v-model="testSize" aria-label="测试集比例" :min="0.05" :max="0.5" :step="0.05" :disabled="store.busy" /></el-form-item>
        <el-form-item label="随机种子"><el-input-number v-model="seed" aria-label="随机种子" :min="0" :max="4294967295" :step="1" :disabled="store.busy" /></el-form-item>
      </el-form>
      <HyperparamForm :algorithm="algorithm" :disabled="!store.canRun || !dataset || !Number.isInteger(seed) || (entry.test_size != null && !Number.isFinite(testSize))" @submit="start" />
      <p v-if="conn.status !== 'connected'" class="section-hint">后端未连接，暂不能训练；不会使用模拟结果替代。</p>
      <p v-else-if="store.busy && record?.status !== 'running'" class="section-hint">另一个算法正在训练，请等待它结束后再提交。</p>
      <div v-if="record" class="interactive-run-state" role="status" aria-live="polite"><el-tag :type="record.status === 'completed' ? 'success' : record.status === 'running' ? 'warning' : 'danger'">{{ statusText[record.status] }}</el-tag><span>耗时 {{ elapsed }} 秒</span><span v-if="record.status === 'running'">正在根据提交参数计算，请稍候；不会显示虚构进度。</span></div>
      <el-alert v-if="record?.error" :title="record.error" type="error" :closable="false" />
      <div v-if="record?.status === 'completed' && record.result" :key="record.result.run_id" class="interactive-result">
        <div class="dynamic-result-heading"><h3>本次训练结果</h3><code>{{ record.result.run_id }}</code></div>
        <p class="section-hint">以下图表对应本次已提交的参数；继续编辑表单不会改变本次结果，需再次点击训练。</p>
        <details class="submitted-config"><summary>查看本次实际参数与评估协议</summary><pre>{{ JSON.stringify({ dataset: record.result.dataset, random_state: record.request.random_state, test_size: record.request.test_size, params: record.result.effective_params, protocol: record.result.metadata.evaluation_protocol, train_samples: record.result.metadata.train_sample_count, test_samples: record.result.metadata.test_sample_count }, null, 2) }}</pre></details>
        <div class="dynamic-metrics"><div v-for="(value, key) in record.result.metrics" :key="key"><span>{{ labels[key] || key }}</span><strong>{{ metric(value) }}</strong></div></div>
        <ChartGallery :visualizations="record.result.metadata.visualizations" />
      </div>
      <el-empty v-else-if="!record" description="调整参数并开始训练后，这里将显示本算法的真实指标与动态图表" :image-size="60" />
    </template>
    <el-empty v-else description="未读取到本算法的调参配置，请重试"><el-button @click="store.loadCatalog">重新加载参数</el-button></el-empty>
  </section>
</template>

<style scoped>
.interactive-provenance{background:#f5f9f8;border:1px solid #e4eeeb;border-radius:7px;padding:14px;margin-bottom:20px;font-size:12px;color:#667b79;line-height:1.7}
.interactive-provenance strong,.interactive-provenance code{display:block;overflow-wrap:anywhere}.interactive-provenance code{font-size:11px}.interactive-provenance p{margin:8px 0}
.interactive-settings{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px}.interactive-settings .el-input-number{width:100%}
.interactive-run-state{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:20px 0;font-size:12px;color:#72828b}.dynamic-result-heading{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;margin-top:24px}.dynamic-result-heading h3{font-size:16px}.dynamic-result-heading code{font-size:11px;color:#85959c;overflow-wrap:anywhere}
.dynamic-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin:20px 0}.dynamic-metrics>div{padding:14px;background:#f3f8f7;border-radius:7px;display:flex;flex-direction:column;gap:8px}.dynamic-metrics span{font-size:11px;color:#7a8c8a}.dynamic-metrics strong{font-size:21px;color:#284941;overflow-wrap:anywhere}.submitted-config{font-size:12px;color:#687d7a}.submitted-config summary{cursor:pointer}.submitted-config pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8f9;padding:12px;border-radius:6px}
</style>
