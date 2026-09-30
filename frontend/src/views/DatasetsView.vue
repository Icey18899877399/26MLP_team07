<script setup>
import { computed, nextTick, onMounted, watch } from 'vue'
import { useConnectionStore } from '../stores/connection.js'
import { useOriginalExperimentsStore } from '../stores/originalExperiments.js'

const conn = useConnectionStore()
const store = useOriginalExperimentsStore()
const active = computed(() => store.datasets.filter(item => !item.archived_only))
const archived = computed(() => store.datasets.filter(item => item.archived_only))
const name = id => store.experiments.find(item => item.id === id)?.title || id
async function reload() { await Promise.all([store.loadDatasets(), store.loadCatalog()]) }
watch(() => conn.baseUrl, async () => { await nextTick(); reload() })
onMounted(reload)
</script>

<template>
  <section class="page-intro"><div><div class="eyebrow">ORIGINAL DATA DIRECTORY</div><h1>原始数据目录</h1><p>按原绘图脚本实际使用情况展示数据。目录中的存档文件不会被当作可替换输入。</p></div><el-button @click="reload"><el-icon><Refresh /></el-icon>刷新目录</el-button></section>
  <el-alert v-if="store.datasetsError" :title="store.datasetsError" type="error" :closable="false" class="page-alert" />
  <section class="page-card dataset-section" v-loading="store.datasetsLoading"><div class="card-title-row"><div><div class="eyebrow">USED BY ORIGINAL SCRIPTS</div><h2>实验实际使用</h2></div><span class="section-side-note">{{ active.length }} 个目录条目</span></div><div v-if="active.length" class="original-dataset-grid"><article v-for="dataset in active" :key="dataset.id" class="original-dataset-card"><div class="dataset-card-title"><h3>{{ dataset.name }}</h3><el-tag type="success" effect="plain" size="small">实际使用</el-tag></div><code>{{ dataset.id }}</code><div class="dataset-paths"><span>原始路径</span><code v-for="path in dataset.paths || []" :key="path">{{ path }}</code></div><div class="dataset-used"><span>相关原实验</span><div><router-link v-for="id in dataset.used_by || []" :key="id" :to="{ path: '/experiment', query: { experiment: id } }">{{ name(id) }}</router-link><span v-if="!dataset.used_by?.length">原脚本内部使用</span></div></div></article></div><el-empty v-else-if="!store.datasetsError" description="暂无原始数据条目" :image-size="80" /></section>
  <section class="page-card dataset-section archive-datasets"><div class="card-title-row"><div><div class="eyebrow">ARCHIVED IN REPOSITORY</div><h2>目录存档</h2></div><span class="section-side-note">{{ archived.length }} 个目录条目</span></div><p class="section-hint">Adult、California 等文件保存在原项目目录中。它们不是对应实验脚本当前使用的数据，完整复现不会用这些文件替换脚本输入。</p><div class="original-dataset-grid"><article v-for="dataset in archived" :key="dataset.id" class="original-dataset-card"><div class="dataset-card-title"><h3>{{ dataset.name }}</h3><el-tag type="info" effect="plain" size="small">仅存档</el-tag></div><code>{{ dataset.id }}</code><div class="dataset-paths"><span>存档路径</span><code v-for="path in dataset.paths || []" :key="path">{{ path }}</code></div></article></div></section>
  <section class="page-card"><div class="card-title-row"><div><div class="eyebrow">SCRIPT-GENERATED SAMPLES</div><h2>合成数据与机制演示</h2></div></div><p class="section-hint">部分原图通过脚本现场生成二维示意样本，解释邻域、密度、决策边界或异常分数。此类图像没有独立的仓库数据文件；复现时仍执行原脚本的生成步骤。</p></section>
</template>
