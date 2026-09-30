<script setup>
import { ref, watch, computed } from 'vue'
import { useConnectionStore } from '../../stores/connection'
import { CONFIG } from '../../config'

const conn = useConnectionStore()
const showSettings = ref(false)
const urlInput = ref(conn.baseUrl)

// 打开设置弹窗时同步当前地址
watch(showSettings, (visible) => {
  if (visible) urlInput.value = conn.baseUrl
})

const statusMap = computed(() => ({
  connecting: { type: 'warning', text: '连接中' },
  connected: { type: 'success', text: '已连接' },
  unavailable: { type: 'danger', text: '后端不可用' },
  offline: { type: 'info', text: '离线（兜底配置）' }
}))

const current = computed(() => statusMap.value[conn.status] || statusMap.value.offline)

const healthLine = computed(() => {
  const h = conn.health
  if (!h?.ml_backend) return ''
  const pkg = h.ml_backend.package ? `${h.ml_backend.package} · ` : ''
  return `${pkg}${h.ml_backend.detail || 'ML 包已连接'}`
})

function applyUrl() {
  const url = (urlInput.value || '').trim()
  if (url && !url.startsWith('http://') && !url.startsWith('https://')) {
    conn.addLog('error', '后端地址必须以 http:// 或 https:// 开头')
    return
  }
  conn.setBaseUrl(url)
  showSettings.value = false
}

function reconnect() {
  conn.refreshAll()
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
        <el-button text size="small" @click.stop="reconnect" title="手动刷新">
          <el-icon><Refresh /></el-icon>
        </el-button>
      </div>
    </template>

    <div class="conn-settings">
      <div class="settings-title">后端连接设置</div>
      <div class="settings-item">
        <span class="settings-label">HTTP 基础地址</span>
        <el-input v-model="urlInput" size="small" placeholder="留空使用本机代理 /api" />
      </div>
      <div class="settings-hint">
        留空使用同源 /api，开发代理连接本机 8000 端口真实后端。
        后端跑在队友电脑上时改为 http://&lt;IP&gt;:8000。
      </div>
      <div v-if="conn.status === 'connected'" class="settings-hint">
        健康信息：{{ healthLine }}
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

.dot-unavailable {
  background: var(--el-color-danger);
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
