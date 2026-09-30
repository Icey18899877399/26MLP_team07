<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useConnectionStore } from '../stores/connection.js'
import { useOriginalExperimentsStore } from '../stores/originalExperiments.js'
import { originalApi } from '../services/originalApi.js'
import OriginalRunCard from '../components/original/OriginalRunCard.vue'
import OriginalFigureGallery from '../components/original/OriginalFigureGallery.vue'

const router = useRouter()
const conn = useConnectionStore()
const store = useOriginalExperimentsStore()
const selectedId = ref('')
const selected = computed(() => store.runs.find(run => run.run_id === selectedId.value) || null)
const name = id => store.experiments.find(item => item.id === id)?.title || id
const statusText = { queued: '排队中', running: '运行中', completed: '已完成', failed: '失败', interrupted: '已中断' }
const statusType = { queued: 'info', running: 'warning', completed: 'success', failed: 'danger', interrupted: 'danger' }
watch(() => store.runs, list => { if (!list.some(run => run.run_id === selectedId.value)) selectedId.value = list[0]?.run_id || '' }, { deep: true })
async function reload() { await Promise.all([store.loadCatalog(), store.loadRuns()]) }
watch(() => conn.baseUrl, async () => { await nextTick(); reload() })
onMounted(reload)
</script>

<template>
  <section class="page-intro"><div><div class="eyebrow">REPRODUCTION HISTORY</div><h1>复现历史</h1><p>记录保存在后端，刷新浏览器后仍可继续查看运行状态、日志与新产物。</p></div><el-button @click="reload"><el-icon><Refresh /></el-icon>刷新记录</el-button></section>
  <el-alert v-if="store.runsError" :title="store.runsError" description="无法读取服务端运行记录，请检查连接后重试。" type="error" :closable="false" class="page-alert" />
  <section class="page-card history-card" v-loading="store.runsLoading"><div class="card-title-row"><div><div class="eyebrow">SERVER RECORDS</div><h2>运行列表</h2></div><span class="section-side-note">{{ store.runs.length }} 条服务端记录</span></div><div v-if="store.runs.length" class="history-table-wrap"><table class="history-table"><thead><tr><th>实验</th><th>状态</th><th>开始时间</th><th>产物</th><th>操作</th></tr></thead><tbody><tr v-for="run in store.runs" :key="run.run_id" :class="{ selected: selectedId === run.run_id }"><td><strong>{{ name(run.experiment_id) }}</strong><small>{{ run.run_id }}</small></td><td><el-tag :type="statusType[run.status] || 'info'" effect="light">{{ statusText[run.status] || run.status }}</el-tag></td><td>{{ run.created_at ? new Date(run.created_at).toLocaleString('zh-CN', { hour12: false }) : '—' }}</td><td>{{ run.figures?.length || 0 }} 图 · {{ run.source_data?.length || 0 }} CSV</td><td><el-button text type="primary" @click="selectedId = run.run_id">查看</el-button></td></tr></tbody></table></div><el-empty v-else description="服务端暂无原实验复现记录" :image-size="90" /></section>
  <section v-if="selected" class="page-card history-detail"><div class="card-title-row"><div><div class="eyebrow">RUN DETAIL</div><h2>{{ name(selected.experiment_id) }} · 运行详情</h2></div><el-button text type="primary" @click="router.push({ path: '/experiment', query: { run: selected.run_id } })">返回实验图谱</el-button></div><OriginalRunCard :run="selected" /><div v-if="Object.keys(selected.parameters || {}).length" class="history-params"><h3>本次记录的原脚本参数</h3><code>{{ JSON.stringify(selected.parameters, null, 2) }}</code></div><div v-if="selected.figures?.length" class="history-gallery"><h3>本次运行生成的图像</h3><OriginalFigureGallery :figures="selected.figures" :source-data="selected.source_data || []" :experiment-id="selected.experiment_id" /></div><p v-else class="section-hint">该运行尚无新 PNG 产物；若运行失败，请查看上方错误与日志。</p></section>
</template>
