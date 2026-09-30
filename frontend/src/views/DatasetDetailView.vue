<script setup>
import { watch } from 'vue'
import { useRoute } from 'vue-router'
import { useConnectionStore } from '../stores/connection.js'
import { useDataLibraryStore } from '../stores/dataLibrary.js'
import DatasetTraining from '../components/datasets/DatasetTraining.vue'
const route = useRoute(), conn = useConnectionStore(), store = useDataLibraryStore()
watch([() => route.params.id, () => conn.baseUrl], ([id]) => { if (id) store.open(String(id)) }, { immediate: true })
</script>
<template>
  <router-link to="/datasets" class="dataset-back">← 返回数据集目录</router-link>
  <el-alert v-if="store.detailError" :title="store.detailError" type="error" :closable="false" class="page-alert"><el-button @click="store.open(String(route.params.id))">重试</el-button></el-alert>
  <section v-if="store.detailLoading" v-loading="true" class="page-card loading-card">正在读取数据结构…</section>
  <template v-else-if="store.detail">
    <section class="page-card dataset-overview"><div><div class="eyebrow">DATASET EXPLORER</div><h1>{{ store.detail.name }}</h1><el-tag effect="plain">{{ store.detail.source === 'upload' ? '本机上传' : '原始数据' }}</el-tag><code>{{ store.detail.id }}</code></div><div class="dataset-size"><strong>{{ store.detail.row_count.toLocaleString() }}<small>行</small></strong><strong>{{ store.detail.column_count }}<small>列</small></strong></div></section>
    <section class="page-card"><div class="card-title-row"><div><div class="eyebrow">SCHEMA</div><h2>数据结构</h2></div><span class="section-side-note">完整字段统计</span></div><div class="dataset-table-scroll"><table class="dataset-table schema-table"><thead><tr><th>字段</th><th>类型</th><th>缺失值</th><th>不同值数量</th><th>建议用途</th></tr></thead><tbody><tr v-for="column in store.detail.columns" :key="column.name"><td>{{ column.name }}</td><td>{{ column.dtype === 'number' ? '数值' : '文本 / 类别' }}</td><td>{{ column.missing_count }}</td><td>{{ column.unique_count }}</td><td>{{ column.name === store.detail.default_target ? '标签（可更改）' : column.dtype === 'number' ? '可选特征' : '需先编码后用于特征' }}</td></tr></tbody></table></div></section>
    <section class="page-card"><div class="card-title-row"><div><div class="eyebrow">PREVIEW</div><h2>样本预览</h2></div><span class="section-side-note">前 {{ store.detail.preview.length }} 行 · 非全部数据</span></div><div class="dataset-table-scroll"><table class="dataset-table preview-table"><thead><tr><th v-for="column in store.detail.columns" :key="column.name">{{ column.name }}</th></tr></thead><tbody><tr v-for="(row,index) in store.detail.preview" :key="index"><td v-for="column in store.detail.columns" :key="column.name">{{ row[column.name] ?? '—' }}</td></tr></tbody></table></div></section>
    <DatasetTraining :key="store.detail.id" />
  </template>
</template>
<style scoped>
.dataset-back{display:inline-block;margin-bottom:18px;font-size:12px;text-decoration:none}.dataset-overview{display:flex;align-items:center;justify-content:space-between;gap:20px}.dataset-overview h1{font-size:24px;margin:9px 0 14px;overflow-wrap:anywhere}.dataset-overview code{font-size:10px;color:#98a6a0;margin-left:12px;overflow-wrap:anywhere}.dataset-size{display:flex;gap:30px}.dataset-size strong{display:flex;flex-direction:column;gap:5px;font-size:26px;color:#2f5145}.dataset-size small{font-size:11px;font-weight:400;color:#8a9b93}.dataset-table-scroll{max-height:410px;overflow:auto;border:1px solid #e6edeb;border-radius:7px}.dataset-table{width:100%;border-collapse:collapse;font-size:12px;text-align:left}.dataset-table th{position:sticky;top:0;background:#f1f6f4;color:#6a8076;font-weight:600;white-space:nowrap}.dataset-table th,.dataset-table td{padding:11px 14px;border-bottom:1px solid #edf1ef}.dataset-table td{color:#53665c;white-space:nowrap}.preview-table td{max-width:360px;overflow:hidden;text-overflow:ellipsis}.dataset-table tbody tr:nth-child(even){background:#fafcfb}@media(max-width:700px){.dataset-overview{align-items:start;flex-direction:column}.dataset-overview h1{font-size:20px}.dataset-size strong{font-size:22px}}
</style>
