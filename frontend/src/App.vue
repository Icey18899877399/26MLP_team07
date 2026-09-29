<script setup>
import { onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useConnectionStore } from './stores/connection'
import ConnectionBadge from './components/common/ConnectionBadge.vue'

const route = useRoute()
const conn = useConnectionStore()

// 应用启动：接线 WebSocket 客户端并尝试连接后端（连不上会自动重试 + 兜底配置）
onMounted(() => {
  conn.init()
})
</script>

<template>
  <el-container class="app-shell">
    <el-aside width="220px" class="app-aside">
      <div class="app-logo">
        <img src="/favicon.svg" alt="logo" />
        <span>ML 可视化平台</span>
      </div>
      <el-menu :default-active="route.path" router class="app-menu">
        <el-menu-item index="/experiment">
          <el-icon><Cpu /></el-icon>
          <span>实验工作台</span>
        </el-menu-item>
        <el-menu-item index="/benchmark">
          <el-icon><DataAnalysis /></el-icon>
          <span>算法对比</span>
        </el-menu-item>
        <el-menu-item index="/datasets">
          <el-icon><Collection /></el-icon>
          <span>数据集</span>
        </el-menu-item>
        <el-menu-item index="/manual">
          <el-icon><Document /></el-icon>
          <span>使用手册</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="app-header">
        <div class="header-title">{{ route.meta.title }}</div>
        <div class="header-right">
          <ConnectionBadge />
        </div>
      </el-header>
      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-shell {
  height: 100vh;
}

.app-aside {
  background: #fff;
  border-right: 1px solid var(--el-border-color-light);
  display: flex;
  flex-direction: column;
}

.app-logo {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px;
  font-size: 16px;
  font-weight: 600;
  color: var(--el-color-primary);
}

.app-logo img {
  width: 28px;
  height: 28px;
}

.app-menu {
  border-right: none;
  flex: 1;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff;
  border-bottom: 1px solid var(--el-border-color-light);
}

.header-title {
  font-size: 16px;
  font-weight: 600;
}

.app-main {
  background: var(--el-bg-color-page);
  padding: 16px;
}
</style>
