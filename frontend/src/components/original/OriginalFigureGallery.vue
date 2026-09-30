<script setup>
import { ref, watch } from 'vue'
import { originalApi } from '../../services/originalApi.js'
import { originalFigureTitle } from '../../config/originalTeaching.js'

const props = defineProps({ figures: { type: Array, default: () => [] }, sourceData: { type: Array, default: () => [] }, archived: Boolean, experimentId: { type: String, default: '' } })
const enlarged = ref(null)
const failedUrls = ref([])
const asset = path => originalApi.assetUrl(path)
const download = path => originalApi.downloadUrl(path)
const title = figure => originalFigureTitle(props.experimentId, figure)
watch(() => props.figures, () => { enlarged.value = null; failedUrls.value = [] })
function imageFailed(url) { if (!failedUrls.value.includes(url)) failedUrls.value.push(url) }
</script>

<template>
  <div v-if="figures.length" class="original-figure-grid">
    <article v-for="(figure, index) in figures" :key="figure.url" class="original-figure-card" :data-original-figure="figure.name">
      <div class="figure-card-head"><span class="figure-index">{{ String(index + 1).padStart(2, '0') }}</span><h3>{{ title(figure) }}</h3></div>
      <button class="figure-preview" type="button" :aria-label="`放大查看 ${title(figure)}`" :disabled="failedUrls.includes(figure.url)" @click="enlarged = figure"><span v-if="failedUrls.includes(figure.url)" class="image-failure">图像加载失败，请检查后端连接并刷新目录</span><img v-else :src="asset(figure.url)" :alt="title(figure)" loading="lazy" @error="imageFailed(figure.url)" /></button>
      <div class="figure-card-foot"><span>{{ archived ? '历史归档原图' : '本次运行生成' }}</span><div><el-button text size="small" @click="enlarged = figure">放大</el-button><a :href="download(figure.url)" :download="figure.name" target="_blank" rel="noopener">下载 PNG</a></div></div>
    </article>
  </div>
  <el-empty v-else description="当前没有可展示的图像" :image-size="76" />
  <div v-if="sourceData.length" class="source-data-row"><span>绘图数据</span><a v-for="file in sourceData" :key="file.url" :href="download(file.url)" :download="file.name" target="_blank" rel="noopener">{{ file.name }}</a></div>
  <el-dialog :model-value="Boolean(enlarged)" @update:model-value="value => { if (!value) enlarged = null }" :title="enlarged && title(enlarged)" class="figure-dialog" width="min(96vw, 1560px)" destroy-on-close append-to-body><div class="figure-dialog-content"><span v-if="enlarged && failedUrls.includes(enlarged.url)" class="image-failure">图像加载失败，请检查后端连接并刷新目录</span><img v-else-if="enlarged" :src="asset(enlarged.url)" :alt="title(enlarged)" @error="imageFailed(enlarged.url)" /></div><template #footer><a v-if="enlarged" :href="download(enlarged.url)" :download="enlarged.name" target="_blank" rel="noopener" class="dialog-download">下载原尺寸 PNG</a></template></el-dialog>
</template>
