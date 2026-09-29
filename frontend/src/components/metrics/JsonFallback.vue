<script setup>
import { computed } from 'vue'

/**
 * 未知指标类型的兜底渲染器：把任意值转成键值表格展示。
 * 保证后端新增指标类型时前端不崩溃（前向兼容），也让用户能看到原始数据。
 */
const props = defineProps({
  value: { default: null }
})

const rows = computed(() => {
  const v = props.value
  if (v === null || v === undefined) return []
  if (Array.isArray(v)) {
    return v.map((item, i) => ({ key: `[${i}]`, value: item }))
  }
  if (typeof v === 'object') {
    return Object.entries(v).map(([key, val]) => ({ key, value: val }))
  }
  return [{ key: '值', value: v }]
})

function displayValue(v) {
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}
</script>

<template>
  <el-table v-if="rows.length > 0" :data="rows" size="small" max-height="200">
    <el-table-column prop="key" label="字段" width="140" />
    <el-table-column label="值">
      <template #default="{ row }">
        <span class="mono">{{ displayValue(row.value) }}</span>
      </template>
    </el-table-column>
  </el-table>
  <div v-else class="no-data">无数据</div>
</template>

<style scoped>
.mono {
  font-family: Consolas, Menlo, monospace;
  font-size: 12px;
  word-break: break-all;
}

.no-data {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
