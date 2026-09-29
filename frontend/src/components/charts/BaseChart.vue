<script setup>
import { markRaw, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { chartTheme } from '../../config/chartTheme'

/**
 * 通用 ECharts 封装 —— 项目中唯一直接接触 echarts 的地方，所有图表都基于它：
 * - 统一注册 mlviz 主题（配色/字体/提示框）
 * - 自动初始化与销毁（onMounted/onBeforeUnmount，防内存泄漏）
 * - markRaw 防止 Vue 把 echarts 实例转成响应式（性能关键）
 * - ResizeObserver + 120ms 防抖：容器尺寸变化（如侧边栏折叠动画）时自动 resize
 * - 父组件传入完整 option，watch 变化后 setOption(notMerge) 全量替换，避免合并 bug
 */
echarts.registerTheme('mlviz', chartTheme)

const props = defineProps({
  option: { type: Object, default: null },
  height: { type: String, default: '320px' },
  loading: { type: Boolean, default: false }
})

const container = ref(null)
let chart = null
let resizeObserver = null
let resizeTimer = null

function resize() {
  if (chart) chart.resize()
}

function debouncedResize() {
  if (resizeTimer) clearTimeout(resizeTimer)
  resizeTimer = setTimeout(resize, 120)
}

onMounted(() => {
  chart = markRaw(echarts.init(container.value, 'mlviz', { renderer: 'canvas' }))
  if (props.option) chart.setOption(props.option)

  resizeObserver = new ResizeObserver(debouncedResize)
  resizeObserver.observe(container.value)
  window.addEventListener('resize', debouncedResize)
})

onBeforeUnmount(() => {
  if (resizeTimer) clearTimeout(resizeTimer)
  if (resizeObserver) resizeObserver.disconnect()
  window.removeEventListener('resize', debouncedResize)
  if (chart) {
    chart.dispose()
    chart = null
  }
})

watch(
  () => props.option,
  (opt) => {
    if (chart && opt) {
      // notMerge: true —— 每次全量替换配置，避免残留旧 series
      chart.setOption(opt, { notMerge: true })
    }
  },
  { deep: true }
)

watch(
  () => props.loading,
  (loading) => {
    if (!chart) return
    if (loading) chart.showLoading('default', { text: '加载中…' })
    else chart.hideLoading()
  }
)

/** 供父组件直接操作实例（如导出图片） */
function getInstance() {
  return chart
}

defineExpose({ getInstance })
</script>

<template>
  <div ref="container" class="base-chart" :style="{ height, width: '100%' }" />
</template>

<style scoped>
.base-chart {
  background: transparent;
}
</style>
