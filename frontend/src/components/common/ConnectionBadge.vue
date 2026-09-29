<script setup>
import { computed, ref } from 'vue'
import { useConnectionStore } from '../../stores/connection'

const conn = useConnectionStore()
const showSettings = ref(false)
const urlInput = ref(conn.wsUrl)

const statusMap = computed(() => ({
  connecting: { type: 'warning', text: '连接中' },
  connected: { type: 'success', text: conn.isUsingFallback ? '已连接' : '已连接' },
  offline: { type: 'info', text: '离线（兜底配置）' }
}))

const current = computed(() => statusMap.value[conn.status] || statusMap.value.offline)

function applyUrl() {
  const url = (urlInput.value || '').trim()
  if (url !== '/api' && !/^https?:\/\//.test(url)) {
    conn.addLog('error', '请输入 /api 或以 http://、https:// 开头的 API 地址')
    return
  }
  conn.setWsUrl(url)
  showSettings.value = false
}

function reconnect() {
  conn.connect()
}
</script>

<template>
  <el-popover v-model:visible="showSettings" placement="bottom-end" :width="340" trigger="click">
    <template #reference>
      <div class="badge-wrapper">
        <el-tag :type="current.type" effect="light" round size="small" class="badge-tag">
          <span class="status-dot" :class="`dot-${conn.status}`" />
          {{ current.text }}
        </el-tag>
        <el-button text size="small" @click.stop="reconnect" title="手动重连">
          <el-icon><Refresh /></el-icon>
        </el-button>
      </div>
    </template>

    <div class="conn-settings">
      <div class="settings-title">后端连接设置</div>
      <div class="settings-item">
        <span class="settings-label">HTTP API 地址</span>
        <el-input v-model="urlInput" size="small" placeholder="/api" />
      </div>
      <div class="settings-hint">
        默认 /api 经开发服务器代理到本机 8000 端口。也可填写 http://127.0.0.1:8000/api。
      </div>
      <div class="settings-actions">
        <el-button size="small" @click="showSettings = false">取消</el-button>
        <el-button size="small" type="primary" @click="applyUrl">保存并重连</el-button>
      </div>
    </div>
  </el-popover>
</template>

<style scoped>
.badge-wrapper {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
}

.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 4px;
}

.dot-connected {
  background: var(--el-color-success);
}

.dot-connecting {
  background: var(--el-color-warning);
}

.dot-offline {
  background: var(--el-color-info);
}

.settings-title {
  font-weight: 600;
  margin-bottom: 8px;
}

.settings-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 8px;
}

.settings-label {
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

.settings-hint {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.6;
  margin-bottom: 8px;
}

.settings-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
</style>
