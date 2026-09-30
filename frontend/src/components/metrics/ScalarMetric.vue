<script setup>
import { computed } from 'vue'
import { formatMetric } from '../../utils/format'

/**
 * 数值指标卡片：大数字展示。
 * 百分比仅用于概率类指标；计数、误差和 R² 保留原始量纲。
 */
const props = defineProps({
  metric: { type: Object, required: true },
  value: { type: [Number, String], default: null }
})

const hasValue = computed(() => props.value !== null && props.value !== undefined)

const displayText = computed(() => formatMetric(props.metric.id, props.value))

</script>

<template>
  <div class="scalar-metric">
    <div class="value neutral">{{ displayText }}</div>
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

.value.neutral {
  color: var(--el-text-color-primary);
}

.empty-tip {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
