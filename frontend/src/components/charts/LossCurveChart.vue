<script setup>
import { computed } from 'vue'
import BaseChart from './BaseChart.vue'

/**
 * 实时训练曲线：
 * - 序列名由 progress.metrics 的键动态发现（loss、val_loss、accuracy…任意数值键）
 * - 键名含 loss/err 的走左轴，其余走右轴（双轴启发式）
 * - 流式更新期间 animation:false 防抖动；点数上限由 store 截断（maxChartPoints）
 */
const props = defineProps({
  series: { type: Object, required: true }, // { loss: [0.8, 0.5, ...], val_loss: [...] }
  height: { type: String, default: '220px' }
})

const option = computed(() => {
  const keys = Object.keys(props.series)
  if (keys.length === 0) return null

  const lossKeys = keys.filter((k) => /loss|err/i.test(k))
  const otherKeys = keys.filter((k) => !lossKeys.includes(k))
  const useDualAxis = lossKeys.length > 0 && otherKeys.length > 0

  const seriesArr = keys.map((key) => {
    const values = props.series[key]
    const isLoss = lossKeys.includes(key)
    return {
      name: key,
      type: 'line',
      showSymbol: false,
      // 流式更新关闭动画，避免每次 setOption 重新动画造成抖动
      animation: false,
      smooth: values.length > 30,
      lineStyle: { width: 2 },
      yAxisIndex: useDualAxis && !isLoss ? 1 : 0,
      data: values.map((v, i) => [i + 1, v])
    }
  })

  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    grid: { left: 48, right: useDualAxis ? 48 : 24, top: 36, bottom: 28 },
    xAxis: { type: 'value', name: 'epoch', minInterval: 1, nameLocation: 'middle', nameGap: 24 },
    yAxis: [
      { type: 'value', name: lossKeys.length ? 'loss' : '', scale: true },
      ...(useDualAxis ? [{ type: 'value', name: '其它指标', scale: true, splitLine: { show: false } }] : [])
    ],
    series: seriesArr
  }
})
</script>

<template>
  <BaseChart v-if="option" :option="option" :height="height" />
  <el-empty v-else description="暂无训练曲线" :image-size="40" />
</template>
