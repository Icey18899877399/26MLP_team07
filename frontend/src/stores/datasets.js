/**
 * 数据集 store：数据集详情按需加载（dataset.load → dataset.info）。
 * 数据集列表本身来自连接 store 的注册表（config.registry.datasets），
 * 本 store 只缓存点击详情后后端返回的完整信息。
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { wsClient } from '../services/wsClient'
import { genMsgId } from '../utils/id'

export const useDatasetStore = defineStore('datasets', () => {
  // ---------- 状态 ----------
  /** datasetId → dataset.info 的 payload */
  const infos = ref(new Map())
  /** datasetId → 是否正在加载 */
  const loading = ref(new Set())

  // ---------- 消息归约 ----------
  function handleMessage(msg) {
    if (msg.type === 'dataset.info' && msg.payload?.id) {
      infos.value.set(msg.payload.id, msg.payload)
      loading.value.delete(msg.payload.id)
    } else if (msg.type === 'dataset.error' && msg.payload?.id) {
      loading.value.delete(msg.payload.id)
    }
  }

  // ---------- 动作 ----------
  /** 请求数据集详情；已缓存则直接返回 */
  function loadDataset(datasetId) {
    if (infos.value.has(datasetId)) return
    loading.value.add(datasetId)
    wsClient.send({ type: 'dataset.load', id: genMsgId(), payload: { datasetId } })
  }

  function getInfo(datasetId) {
    return infos.value.get(datasetId) || null
  }

  function isLoading(datasetId) {
    return loading.value.has(datasetId)
  }

  return { infos, loading, handleMessage, loadDataset, getInfo, isLoading }
})
