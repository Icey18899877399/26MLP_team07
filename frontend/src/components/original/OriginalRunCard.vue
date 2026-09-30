<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useOriginalExperimentsStore } from '../../stores/originalExperiments.js'

const props = defineProps({ run: { type: Object, required: true }, compact: Boolean })
const store = useOriginalExperimentsStore()
const now = ref(Date.now())
let clock
onMounted(() => { clock = setInterval(() => { now.value = Date.now() }, 1000) })
onUnmounted(() => clearInterval(clock))
const labels = { queued: '排队中', running: '运行中', completed: '已完成', failed: '失败', interrupted: '已中断' }
const types = { queued: 'info', running: 'warning', completed: 'success', failed: 'danger', interrupted: 'danger' }
const elapsed = computed(() => {
  const start = Date.parse(props.run.started_at || props.run.created_at || '')
  const end = props.run.finished_at ? Date.parse(props.run.finished_at) : now.value
  if (!Number.isFinite(start) || !Number.isFinite(end)) return '—'
  const seconds = Math.max(0, Math.floor((end - start) / 1000))
  return `${Math.floor(seconds / 3600).toString().padStart(2, '0')}:${Math.floor((seconds % 3600) / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`
})
</script>

<template>
  <div class="original-run-card" :class="{ compact }">
    <div class="run-card-header"><div><el-tag :type="types[run.status] || 'info'" effect="light">{{ labels[run.status] || run.status }}</el-tag><code>{{ run.run_id }}</code></div><el-button text size="small" @click="store.refreshRun(run.run_id)"><el-icon><Refresh /></el-icon>刷新状态</el-button></div>
    <div class="run-card-facts"><span>创建于 {{ run.created_at ? new Date(run.created_at).toLocaleString('zh-CN', { hour12: false }) : '—' }}</span><span>实际耗时 {{ elapsed }}</span><span v-if="run.finished_at">结束于 {{ new Date(run.finished_at).toLocaleString('zh-CN', { hour12: false }) }}</span></div>
    <el-alert v-if="run.error" :title="run.error" type="error" :closable="false" class="run-error" />
    <el-collapse v-if="run.log || run.error" class="run-log-collapse"><el-collapse-item :title="`运行日志${run.log ? ` · ${run.log.length} 字符` : ''}`" name="log"><pre class="original-run-log">{{ run.log || run.error }}</pre></el-collapse-item></el-collapse>
    <p v-else-if="['queued', 'running'].includes(run.status)" class="run-muted">后台正在执行原脚本；此处显示服务端状态，不估算完成百分比。</p>
  </div>
</template>
