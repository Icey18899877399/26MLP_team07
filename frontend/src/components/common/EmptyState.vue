<script setup>
defineProps({
  title: { type: String, default: '暂无数据' },
  description: { type: String, default: '' }
})
</script>

<template>
  <div class="empty-state">
    <el-icon class="empty-icon"><Box /></el-icon>
    <div class="empty-title">{{ title }}</div>
    <div v-if="description" class="empty-desc">{{ description }}</div>
    <slot />
  </div>
</template>

<style scoped>
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px 16px;
  color: var(--el-text-color-secondary);
}

.empty-icon {
  font-size: 40px;
  margin-bottom: 8px;
  opacity: 0.5;
}

.empty-title {
  font-size: 14px;
  margin-bottom: 4px;
}

.empty-desc {
  font-size: 12px;
  opacity: 0.8;
}
</style>
