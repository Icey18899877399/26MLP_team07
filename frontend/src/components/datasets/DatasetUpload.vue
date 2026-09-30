<script setup>
import { ref, onBeforeUnmount } from 'vue'
import { useRouter } from 'vue-router'
import { useDataLibraryStore } from '../../stores/dataLibrary.js'
const store = useDataLibraryStore(), router = useRouter()
const selectedFile = ref(null), localError = ref(''), reading = ref(false)
let active = true
onBeforeUnmount(() => { active = false })
function select(event) { selectedFile.value = event.target.files?.[0] || null; localError.value = ''; store.uploadError = '' }
async function submit() {
  const file = selectedFile.value
  if (!file || reading.value || store.uploading) return
  localError.value = ''
  if (!/\.(csv|tsv)$/i.test(file.name)) { localError.value = '请选择 CSV 或 TSV 文件'; return }
  if (file.size > 5 * 1024 * 1024) { localError.value = '文件不能超过 5 MiB'; return }
  reading.value = true
  try {
    const data = await store.uploadFile(file, () => active)
    if (data && active) router.push(`/datasets/${encodeURIComponent(data.id)}`)
  } catch { localError.value = '无法读取文件，请使用 UTF-8 编码的 CSV / TSV' }
  finally { reading.value = false }
}
</script>
<template>
  <section class="page-card dataset-upload">
    <div><div class="eyebrow">IMPORT YOUR DATA</div><h2>上传新数据集</h2><p>上传后可查看数据结构、选择字段与算法并调参训练。原数据保持不变。</p></div>
    <div class="upload-controls"><input type="file" accept=".csv,.tsv" aria-label="选择 CSV 或 TSV 数据集" :disabled="store.uploading || reading" @change="select"><el-button type="primary" :disabled="!selectedFile" :loading="store.uploading || reading" @click="submit">上传并查看</el-button></div>
    <p class="upload-limits">UTF-8 · 首行为字段名 · 最大 5 MiB / 20,000 行 / 128 列。仅保存于本机，不自动上传 GitHub。</p>
    <el-alert v-if="localError || store.uploadError" :title="localError || store.uploadError" type="error" :closable="false" show-icon />
  </section>
</template>
<style scoped>
.dataset-upload{border:1px solid #cbe4df;background:linear-gradient(115deg,#f4faf8,#fff)}h2{font-size:18px;margin:8px 0}p{color:#7b8b8b;font-size:12px;line-height:1.8}.upload-controls{display:flex;align-items:center;gap:15px;flex-wrap:wrap;margin-top:18px}input{max-width:100%;font-size:12px;color:#5b6f6c}input::file-selector-button{background:#fff;border:1px solid #cbded9;border-radius:6px;padding:9px 12px;color:#237f70;margin-right:12px;cursor:pointer}.upload-limits{font-size:11px}
</style>
