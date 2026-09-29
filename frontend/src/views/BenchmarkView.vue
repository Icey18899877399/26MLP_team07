<script setup>
import { computed, ref } from 'vue'
import { useTrainingStore } from '../stores/training'
const training = useTrainingStore()
const dataset = ref('')
const history = computed(() => training.history.filter(h => h.result?.status === 'done'))
const datasets = computed(() => [...new Set(history.value.map(h => h.datasetId))])
const rows = computed(() => history.value.filter(h => h.datasetId === dataset.value))
const metrics = computed(() => [...new Set(rows.value.flatMap(h =>
  Object.entries(h.result.metrics).filter(([,v]) => typeof v === 'number').map(([k]) => k)))])
</script>
<template>
  <div class="page-card">
    <h3>实验结果对比</h3>
    <p>选择同一数据集，比较已完成实验的真实指标；请同时核对参数、测试比例和随机种子。</p>
    <el-select v-model="dataset" placeholder="选择已运行的数据集">
      <el-option v-for="d in datasets" :key="d" :value="d" :label="d" />
    </el-select>
    <el-table :data="rows">
      <el-table-column prop="algorithmName" label="算法" />
      <el-table-column label="配置" min-width="250">
        <template #default="{row}">{{ JSON.stringify({params: row.hyperparams, split: row.split}) }}</template>
      </el-table-column>
      <el-table-column v-for="m in metrics" :key="m" :label="m">
        <template #default="{row}">{{ row.result.metrics[m] }}</template>
      </el-table-column>
    </el-table>
  </div>
</template>
