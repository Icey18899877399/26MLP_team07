<script setup>
import { computed } from 'vue'
import LossCurveChart from '../charts/LossCurveChart.vue'
import { formatDuration } from '../../utils/format'

/**
 * 单次训练运行监控：状态、进度条、耗时、实时 loss 曲线、取消按钮。
 * 纯展示组件：所有数据来自 props.run（由 training store 维护），
 * 操作通过事件上抛，组件自身不写状态。
 */
const props = defineProps({
  run: { type: Object, required: true }
})

const emit = defineEmits(['cancel'])

const statusInfo = computed(() => {
  switch (props.run.status) {
    case 'queued':
      return { text: '排队中', type: 'info' }
    case 'running':
      return { text: '训练中', type: 'primary' }
    case 'lost':
      return { text: '已丢失', type: 'danger' }
    default:
      return { text: props.run.status, type: 'info' }
  }
})

const progressPercent = computed(() => {
  const { epoch, totalEpochs } = props.run.progress
  if (!totalEpochs) return 0
  return Math.min(100, Math.round((epoch / totalEpochs) * 100))
})

const elapsedText = computed(() => formatDuration(props.run.progress.elapsedMs))
</script>

<template>
  <div class="run-monitor">
    <div class="run-head">
      <div class="run-title">
        <el-tag :type="statusInfo.type" size="small">{{ statusInfo.text }}</el-tag>
        <span class="run-name">{{ run.algorithmName }} × {{ run.datasetName }}</span>
        <span class="run-id">{{ run.runId }}</span>
      </div>
      <div class="run-actions">
        <span v-if="run.status === 'running'" class="elapsed">
          正在训练与评价，完成后返回真实指标。请保留此页面。
        </span>
        <el-button
          v-if="false"
          size="small"
          type="danger"
          plain
          @click="emit('cancel')"
        >
          取消
        </el-button>
        <el-tag v-if="run.status === 'lost'" type="danger" size="small" effect="plain">
          后端已重启，该训练已丢失，可重新发起
        </el-tag>
      </div>
    </div>

    <el-progress
      v-if="run.progress.totalEpochs > 0"
      :percentage="progressPercent"
      :stroke-width="8"
      class="run-progress"
    />

    <!-- 实时曲线：仅当有序列数据时显示 -->
    <LossCurveChart
      v-if="Object.keys(run.progress.series).length > 0"
      :series="run.progress.series"
      :height="220"
    />
  </div>
</template>

<style scoped>
.run-monitor {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 12px;
}

.run-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  flex-wrap: wrap;
  gap: 8px;
}

.run-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.run-name {
  font-weight: 500;
}

.run-id {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.run-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.elapsed {
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

.run-progress {
  margin-bottom: 8px;
}
</style>
