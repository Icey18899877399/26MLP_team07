<script setup>
import { computed } from 'vue'
import BaseChart from './BaseChart.vue'

/**
 * 混淆矩阵热力图：行=真实类别，列=预测类别。
 * 类名从 value 旁的同级 class_names 取（协议约定），缺省时用数字下标。
 */
const props = defineProps({
  matrix: { type: Array, required: true }, // number[][]
  classNames: { type: Array, default: () => [] },
  height: { type: String, default: '300px' }
})

const option = computed(() => {
  if (!props.matrix || props.matrix.length === 0) return null
  const n = props.matrix.length
  const labels =
    props.classNames.length === n
      ? props.classNames
      : Array.from({ length: n }, (_, i) => String(i))

  // 计算最大值用于 visualMap（至少为 1，避免全 0 时异常）
  const maxVal = Math.max(1, ...props.matrix.flat())

  const data = []
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      data.push([j, i, props.matrix[i][j]])
    }
  }

  return {
    tooltip: {
      position: 'top',
      formatter: (params) =>
        `真实: ${labels[params.value[1]]}<br/>预测: ${labels[params.value[0]]}<br/>样本数: ${params.value[2]}`
    },
    grid: { left: 80, right: 24, top: 20, bottom: 70 },
    xAxis: {
      type: 'category',
      data: labels,
      name: '预测类别',
      nameLocation: 'middle',
      nameGap: 45,
      splitArea: { show: true },
      axisLabel: n > 10 ? { rotate: 45, interval: 0 } : {}
    },
    yAxis: {
      type: 'category',
      data: labels,
      name: '真实类别',
      nameLocation: 'middle',
      nameGap: 60,
      splitArea: { show: true }
    },
    visualMap: {
      min: 0,
      max: maxVal,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: { color: ['#f0f5ff', '#409eff'] }
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: { show: true, formatter: (p) => (p.value[2] === 0 ? '' : p.value[2]) },
        emphasis: { itemStyle: { shadowBlur: 4, shadowColor: 'rgba(0,0,0,.3)' } }
      }
    ]
  }
})
</script>

<template>
  <BaseChart v-if="option" :option="option" :height="height" />
  <el-empty v-else description="无混淆矩阵数据" :image-size="40" />
</template>
