<script setup>
import { computed, onMounted, watch } from 'vue'
import { useConnectionStore } from '../stores/connection.js'
import { useOriginalExperimentsStore } from '../stores/originalExperiments.js'
import { useDataLibraryStore } from '../stores/dataLibrary.js'
import DatasetUpload from '../components/datasets/DatasetUpload.vue'
const conn = useConnectionStore(), originals = useOriginalExperimentsStore(), library = useDataLibraryStore()
const groups = computed(() => [
  { id: 'used', title: '原实验数据', hint: '点击数据集查看结构、样本与训练入口', items: library.items.filter(item => item.source === 'original' && !originals.datasets.find(d => d.id === item.id)?.archived_only) },
  { id: 'uploaded', title: '我的上传', hint: '独立保存在本机，刷新后仍可使用', items: library.items.filter(item => item.source === 'upload') },
  { id: 'archived', title: '原项目存档', hint: '可浏览原始结构；自主训练不会改变原实验输入', items: library.items.filter(item => item.source === 'original' && originals.datasets.find(d => d.id === item.id)?.archived_only) }
])
async function reload() { await Promise.all([library.load(), originals.loadDatasets()]) }
watch(() => conn.baseUrl, reload)
onMounted(reload)
</script>

<template>
  <section class="page-intro"><div><div class="eyebrow">DATA LIBRARY</div><h1>原始数据目录</h1><p>查看数据结构，或导入自己的数据开展新实验。</p></div><el-button :loading="library.loading" @click="reload"><el-icon><Refresh /></el-icon>刷新目录</el-button></section>
  <DatasetUpload />
  <el-alert v-if="library.error" :title="library.error" type="error" :closable="false" show-icon class="page-alert" />
  <section v-for="group in groups" :key="group.id" class="page-card dataset-section" :data-dataset-group="group.id" v-loading="library.loading">
    <div class="card-title-row"><div><h2>{{ group.title }}</h2><p class="section-hint">{{ group.hint }}</p></div><span class="section-side-note">{{ group.items.length }} 个数据集</span></div>
    <div v-if="group.items.length" class="original-dataset-grid">
      <router-link v-for="dataset in group.items" :key="dataset.id" :to="`/datasets/${encodeURIComponent(dataset.id)}`" class="original-dataset-card dataset-link" :aria-label="`查看数据集 ${dataset.name}`">
        <div class="dataset-card-title"><h3>{{ dataset.name }}</h3><el-icon><ArrowRight /></el-icon></div><code>{{ dataset.id }}</code>
        <div class="dataset-card-size"><strong>{{ dataset.row_count.toLocaleString() }} <small>行</small></strong><strong>{{ dataset.column_count }} <small>列</small></strong><span>{{ dataset.source === 'upload' ? '本机上传' : '原始文件' }}</span></div>
        <div class="dataset-card-action">查看字段、缺失值与样本 →</div>
      </router-link>
    </div>
    <el-empty v-else-if="!library.loading && !library.error" :description="group.id === 'uploaded' ? '尚未上传数据，可在上方选择 CSV / TSV' : '暂无数据集'" :image-size="55" />
  </section>
</template>

<style scoped>
.dataset-link{display:block;text-decoration:none;color:inherit;transition:border-color .15s,box-shadow .15s}.dataset-link:hover,.dataset-link:focus-visible{border-color:#62b3a2;box-shadow:0 5px 18px #267d6520}.dataset-link>code{font-size:10px;color:#95a69c;overflow-wrap:anywhere}.dataset-card-size{display:flex;align-items:baseline;flex-wrap:wrap;gap:22px;margin:24px 0 18px}.dataset-card-size strong{font-size:21px;color:#3f6152}.dataset-card-size small{font-size:11px;font-weight:400;color:#99aaa2}.dataset-card-size>span{font-size:11px;color:#8c9e95;margin-left:auto}.dataset-card-action{padding-top:14px;border-top:1px solid #eaf0ec;font-size:11px;color:#349080}.card-title-row h2{margin-bottom:8px}.card-title-row .section-hint{margin:0}
</style>
