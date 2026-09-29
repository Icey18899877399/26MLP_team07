<script setup>
import { computed } from 'vue'
import { formatNumber } from '../../utils/format'

/**
 * 数值指标卡片：大数字展示。
 * 颜色规则：指标定义中 higherIsBetter=true → 高值绿色；false → 低值绿色。
 * 数值超出 [0,1] 且指标名含 error/mse/mae 等时按原值展示，否则尝试百分比显示。
 */
const props = defineProps({
  metric: { type: Object, required: true },
  value: { type: [Number, String], default: null }
})

const hasValue = computed(() => props.value !== null && props.value !== undefined)

const displayText = computed(() => {
  if (!hasValue.value) return '-'
  const v = Number(props.value)
  const id = props.metric.id || ''
  // 误差类指标量纲不定，直接格式化数值
  if (/error|mse|rmse|mae|loss|inertia|davies/i.test(id)) {
    return formatNumber(v, 4)
  }
  // [0,1] 范围内的指标按百分比展示更直观
  if (v >= 0 && v <= 1) {
    return `${(v * 100).toFixed(2)}%`
  }
  return formatNumber(v, 4)
})

const levelClass = computed(() => {
  if (!hasValue.value) return 'neutral'
  // 只有同时知道好坏方向才能上色
  const v = Number(props.value)
  if (v >= 0 && v <= 1 && props.metric.higherIsBetter != null) {
    const good = props.metric.higherIsBetter ? v >= 0.8 : v <= 0.2
    return good ? 'good' : 'bad'
  }
  return 'neutral'
})
</script>

<template>
  <div class="scalar-metric">
    <div class="value" :class="levelClass">{{ displayText }}</div>
    <div v-if="!hasValue" class="empty-tip">无数据</div>
  </div>
</template>

<style scoped>
.scalar-metric {
  text-align: left;
}

.value {
  font-size: 28px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.value.good {
  color: var(--el-color-success);
}

.value.bad {
  color: var(--el-color-danger);
}

.value.neutral {
  color: var(--el-text-color-primary);
}

.empty-tip {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
