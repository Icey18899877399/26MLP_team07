<script setup>
import { computed } from 'vue'
import { formatDuration } from '../../utils/format'

/**
 * 单次实验运行卡片。
 * HTTP 合同下训练是同步请求（无进度流/取消），因此只有两种状态：
 * - running：等待态（spinner + 走秒耗时）
 * - error：失败态（错误码 + 消息）
 * 完成态不在此渲染——完成的 run 进入"训练结果"面板。
 */
const props = defineProps({
  run: { type: Object, required: true }
})

const elapsedText = computed(() => formatDuration(props.run.elapsedMs))
</script>

<template>
  <div class="run-monitor">
    <div class="run-head">
      <div class="run-title">
        <el-tag v-if="run.status === 'running'" type="primary" size="small">训练中</el-tag>
        <el-tag v-else-if="run.status === 'error'" type="danger" size="small">失败</el-tag>
        <el-tag v-else type="info" size="small">{{ run.status }}</el-tag>
        <span class="run-name">{{ run.modelName }} × {{ run.datasetName }}</span>
        <span class="run-id">{{ run.id }}</span>
      </div>
      <div class="run-actions">
        <span v-if="run.status === 'running'" class="elapsed">
          <el-icon class="is-loading loading-icon"><Loading /></el-icon>
          已用 {{ elapsedText }}
        </span>
      </div>
    </div>

    <p v-if="run.status === 'running'" class="run-note">
      训练请求已提交，正在等待后端返回结果（同步训练）…
    </p>
    <p v-else-if="run.status === 'error'" class="run-note error-note">
      [{{ run.error?.code || 'unknown' }}] {{ run.error?.message }}。可修改参数后重新发起训练。
    </p>
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
  font-family: monospace;
}

.run-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.elapsed {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

.loading-icon {
  font-size: 13px;
}

.run-note {
  margin: 8px 0 0;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

.error-note {
  color: var(--el-color-danger);
}
</style>
