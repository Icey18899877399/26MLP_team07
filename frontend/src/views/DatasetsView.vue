<script setup>
import { useConnectionStore } from '../stores/connection'
const conn = useConnectionStore()
</script>
<template>
  <div class="page-card">
    <h3>可用数据集</h3>
    <el-alert title="列表来自当前模型服务；数据集随 ml_core 安装。" type="info" :closable="false" />
    <el-table :data="conn.registry.datasets || []">
      <el-table-column prop="name" label="名称" />
      <el-table-column prop="id" label="标识" />
      <el-table-column prop="taskType" label="任务" />
      <el-table-column prop="nSamples" label="样本数" />
      <el-table-column prop="nFeatures" label="特征数" />
    </el-table>
  </div>
</template>
