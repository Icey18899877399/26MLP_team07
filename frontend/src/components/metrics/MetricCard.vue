<script setup>
import { computed } from 'vue'
import ScalarMetric from './ScalarMetric.vue'
import JsonFallback from './JsonFallback.vue'
import ConfusionHeatmap from '../charts/ConfusionHeatmap.vue'
import RocCurveChart from '../charts/RocCurveChart.vue'

/**
 * 指标卡片分发器 —— 可扩展性核心组件：
 * 根据注册表中指标的 type/chart 决定用哪个渲染器：
 *   scalar            → ScalarMetric 数值卡片
 *   matrix+heatmap    → ConfusionHeatmap 热力图
 *   curve+roc         → RocCurveChart ROC 曲线
 *   其它任何类型      → JsonFallback 通用表格（新增指标类型不崩溃）
 */
const props = defineProps({
  metric: { type: Object, required: true }, // 指标定义（来自注册表）
  value: { type: [Number, Object, Array, String], default: null },
  extra: { type: Object, default: null } // 同批结果里的其它键（如 class_names）
})

const kind = computed(() => {
  const m = props.metric
  if (m.type === 'scalar') return 'scalar'
  if (m.type === 'matrix' && m.chart === 'heatmap') return 'heatmap'
  if (m.type === 'curve' && m.chart === 'roc') return 'roc'
  return 'json'
})

const classNames = computed(() => {
  if (props.metric.id === 'confusion_matrix' && props.extra) {
    return props.extra.class_names || []
  }
  return []
})
</script>

<template>
  <div class="metric-card" :class="{ 'card-wide': kind === 'heatmap' || kind === 'roc' }">
    <div class="metric-title">
      <span>{{ metric.name }}</span>
      <el-tooltip v-if="metric.hint" :content="metric.hint" placement="top">
        <el-icon class="hint-icon"><QuestionFilled /></el-icon>
      </el-tooltip>
    </div>

    <!-- 数值指标 -->
    <ScalarMetric v-if="kind === 'scalar'" :metric="metric" :value="value" />

    <!-- 混淆矩阵 -->
    <ConfusionHeatmap v-else-if="kind === 'heatmap'" :matrix="value" :class-names="classNames" height="280px" />

    <!-- ROC 曲线 -->
    <RocCurveChart v-else-if="kind === 'roc'" :roc="value" height="280px" />

    <!-- 未知类型兜底 -->
    <JsonFallback v-else :value="value" />
  </div>
</template>

<style scoped>
.metric-card {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  padding: 12px;
  background: #fff;
  min-width: 0;
}

/* 矩阵/曲线类指标卡片占满结果网格的整行 */
.card-wide {
  grid-column: 1 / -1;
}

.metric-title {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  margin-bottom: 8px;
}

.hint-icon {
  font-size: 12px;
  cursor: help;
}
</style>
