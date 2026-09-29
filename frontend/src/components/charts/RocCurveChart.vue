<script setup>
import { computed } from 'vue'
import BaseChart from './BaseChart.vue'

/**
 * ROC 曲线：每条曲线一个系列，含对角线参考线，图例中显示 AUC。
 * 数据来自结果指标的 roc_auc: { auc, curves: [{ label, fpr[], tpr[] }] }
 */
const props = defineProps({
  roc: { type: Object, required: true },
  height: { type: String, default: '300px' }
})

const option = computed(() => {
  const curves = props.roc?.curves || []
  if (curves.length === 0) return null

  const series = curves.map((c) => ({
    name: c.label,
    type: 'line',
    showSymbol: false,
    lineStyle: { width: 2 },
    data: c.fpr.map((fpr, i) => [fpr, c.tpr[i]])
  }))

  // 对角线参考（随机分类器）
  series.push({
    name: '随机分类器',
    type: 'line',
    showSymbol: false,
    lineStyle: { type: 'dashed', color: '#909399', width: 1 },
    data: [
      [0, 0],
      [1, 1]
    ],
    tooltip: { show: false }
  })

  const aucText = props.roc?.auc != null ? `AUC = ${Number(props.roc.auc).toFixed(4)}` : ''

  return {
    title: aucText ? { text: aucText, right: 16, top: 4, textStyle: { fontSize: 13, color: '#606266' } } : undefined,
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    grid: { left: 48, right: 24, top: 40, bottom: 32 },
    xAxis: { type: 'value', name: 'FPR（假正例率）', min: 0, max: 1 },
    yAxis: { type: 'value', name: 'TPR（真正例率）', min: 0, max: 1 },
    series
  }
})
</script>

<template>
  <BaseChart v-if="option" :option="option" :height="height" />
  <el-empty v-else description="无 ROC 数据" :image-size="40" />
</template>
