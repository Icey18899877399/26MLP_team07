<script setup>
import { computed, ref } from 'vue'
import BaseChart from './BaseChart.vue'

const props = defineProps({ visualizations: {type: Array, default: () => []} })
const charts = computed(() => props.visualizations.filter(chart => chart?.option && typeof chart.option === 'object'))
const selected = ref(null)
const expanded = ref(false)
const largeChart = ref(null)
const chartRefs = new Map()
function enlarge(chart) { selected.value = chart; expanded.value = true }
function download(chart, instance) {
  if (!instance) return
  const link = document.createElement('a')
  link.download = `${chart.id || 'visualization'}.png`
  link.href = instance.getDataURL({type: 'png', pixelRatio: 2, backgroundColor: '#fff'})
  link.click()
}
</script>

<template>
  <div class="chart-gallery">
    <article v-for="(chart, index) in charts" :key="chart.id" class="chart-card" :data-chart-id="chart.id">
      <header><div><span class="chart-index">{{ String(index + 1).padStart(2, '0') }}</span><h4>{{ chart.title }}</h4></div>
        <el-button text size="small" @click="enlarge(chart)" :aria-label="`放大${chart.title}`"><el-icon><FullScreen /></el-icon></el-button>
      </header>
      <BaseChart :ref="el => el ? chartRefs.set(chart.id, el) : chartRefs.delete(chart.id)" :option="chart.option" height="320px" />
      <p>{{ chart.description }}</p>
      <footer><el-button text size="small" @click="enlarge(chart)">点击放大查看</el-button><el-button text size="small" @click="download(chart, chartRefs.get(chart.id)?.getInstance())">下载 PNG</el-button></footer>
    </article>
  </div>
  <el-empty v-if="!charts.length" description="本次结果未提供可视化图表" :image-size="70" />
  <el-dialog v-model="expanded" :title="selected?.title" width="min(1100px, 94vw)" destroy-on-close class="chart-dialog">
    <BaseChart v-if="expanded && selected" ref="largeChart" :option="selected.option" height="60vh" />
    <p class="muted">{{ selected?.description }}</p>
    <template #footer><el-button @click="expanded = false">关闭</el-button><el-button type="primary" @click="download(selected, largeChart?.getInstance())">下载高清 PNG</el-button></template>
  </el-dialog>
</template>

<style scoped>
.chart-gallery {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.chart-card {min-width:0;border:1px solid #e0ebe6;border-radius:12px;background:#fff;padding:16px 12px 6px;overflow:hidden}
header,header>div,footer{display:flex;align-items:center;justify-content:space-between;gap:10px}
header {padding:0 4px 8px} h4{margin:0;font-size:15px;font-weight:600}
.chart-index{font-size:11px;color:#29966c;background:#edf7f1;border-radius:5px;padding:4px 6px;font-family:monospace}
p{font-size:12px;line-height:1.7;color:#75857d;min-height:38px;margin:8px 8px}
footer{border-top:1px solid #edf2ef;padding-top:4px}
@media(max-width:1050px){.chart-gallery{grid-template-columns:1fr}}
</style>
