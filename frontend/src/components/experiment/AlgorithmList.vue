<script setup>
import { computed } from 'vue'

/**
 * 算法列表：完全由注册表（config.registry.algorithms）驱动渲染。
 * 后端新增算法 → 这里自动出现，零前端改动（可扩展性演示点）。
 */
const props = defineProps({
  algorithms: { type: Array, default: () => [] },
  activeId: { type: String, default: '' }
})

const emit = defineEmits(['select'])

const TASK_TYPE_NAMES = {
  classification: '分类',
  regression: '回归',
  clustering: '聚类',
  anomaly_detection: '异常检测'
}

/** 按任务类型分组：[{ taskType, name, items }] */
const groups = computed(() => {
  const map = new Map()
  for (const algo of props.algorithms) {
    const taskTypes = algo.taskTypes || []
    const label = taskTypes.map((t) => TASK_TYPE_NAMES[t] || t).join(' / ')
    for (const t of taskTypes) {
      const key = label || t
      if (!map.has(key)) map.set(key, [])
      if (!map.get(key).includes(algo)) map.get(key).push(algo)
    }
  }
  return [...map.entries()].map(([label, items]) => ({ label, items }))
})

function select(algo) {
  emit('select', algo)
}
</script>

<template>
  <div class="algorithm-list">
    <div v-for="group in groups" :key="group.label" class="algo-group">
      <div class="algo-group-title">{{ group.label }}</div>
      <button
        v-for="algo in group.items"
        :key="algo.id"
        class="algo-item"
        type="button"
        :data-model-id="algo.id"
        :aria-pressed="algo.id === activeId"
        :class="{ active: algo.id === activeId }"
        @click="select(algo)"
      >
        <div class="algo-name">
          <span>{{ algo.name }}</span>
          <el-tag v-if="algo.id === activeId" size="small" type="primary" effect="plain">已选</el-tag>
        </div>
        <div class="algo-desc">{{ algo.description }}</div>
      </button>
    </div>
  </div>
</template>

<style scoped>
.algo-group {
  margin-bottom: 8px;
}

.algo-group-title {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  padding: 4px 8px;
  border-bottom: 1px dashed var(--el-border-color-lighter);
  margin-bottom: 4px;
}

.algo-item {
  display: block;
  width: 100%;
  text-align: left;
  border: 0;
  background: transparent;
  font-family: inherit;
  color: inherit;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}

.algo-item:hover {
  background: var(--el-fill-color-light);
}

.algo-item.active {
  background: var(--el-color-primary-light-9);
  outline: 1px solid var(--el-color-primary-light-5);
}

.algo-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 500;
}

.algo-desc {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  margin-top: 2px;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
