<script setup>
import { computed } from 'vue'
import { useConnectionStore } from '../stores/connection'
import { taskNames } from '../config/teaching'
const conn = useConnectionStore()
const datasets = computed(() => conn.registry.datasets || [])
const descriptions = {wdbc:'乳腺癌诊断数据，以细胞核的形态特征区分良性与恶性样本。',seeds:'小麦种子形态数据，用于观察无监督算法能否发现不同种子类型。',concrete:'混凝土配合比与龄期数据，预测连续的抗压强度。',california_housing:'加州房屋统计数据，学习地区特征与房价之间的关系。','6_cardio':'心电图特征异常检测数据，分析少数异常样本与正常样本的分布差异。','23_mammography':'乳腺影像特征异常检测数据，关注不平衡条件下的异常检出能力。'}
function compatible(ds) { return conn.registry.algorithms.filter(a => a.taskTypes.includes(ds.taskType) && (!a.compatibleDatasets?.length || a.compatibleDatasets.includes(ds.id))) }
</script>

<template>
  <section class="workspace-hero"><div><div class="eyebrow">DATA COLLECTION</div><h1>可用数据集</h1><p>覆盖分类、回归、聚类与异常检测，样本规模以当前后端注册表为准。</p></div><el-tag v-if="conn.isUsingFallback" type="warning">离线配置预览</el-tag></section>
  <div class="dataset-grid"><article v-for="ds in datasets" :key="ds.id" class="page-card"><div class="section-heading"><el-tag effect="plain">{{ taskNames[ds.taskType] }}</el-tag><span class="eyebrow">{{ ds.id }}</span></div><h2>{{ ds.name }}</h2><p>{{ descriptions[ds.id] || ds.description }}</p><div class="data-stats"><div><strong>{{ ds.nSamples.toLocaleString() }}</strong><span>样本数量</span></div><div><strong>{{ ds.nFeatures }}</strong><span>特征维度</span></div></div><div class="compatible"><span>支持的算法</span><div><el-tag v-for="algo in compatible(ds)" :key="algo.id" size="small" type="info" effect="plain">{{ algo.name }}</el-tag></div></div><p class="dataset-tip">{{ ds.taskType === 'anomaly_detection' ? '评估包含拟合样本，结果用于课程实验分析。' : ds.taskType === 'clustering' ? '全量聚类；已有类别标签仅用于外部评价。' : '预处理仅在训练集上拟合，测试集用于留出评估。' }}</p></article></div>
</template>

<style scoped>
.dataset-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.page-card{margin-bottom:0;display:flex;flex-direction:column}h2{font-size:19px;margin:0 0 10px}p{font-size:13px;line-height:1.9;color:#7b9183}.data-stats{display:flex;gap:38px;padding:16px 0}.data-stats>div{display:flex;flex-direction:column;gap:7px}.data-stats strong{font-size:27px;color:#347654;font-weight:550}.data-stats span,.compatible>span{font-size:11px;color:#90a395}.compatible{padding-top:17px;border-top:1px solid #e7eee9;margin-top:auto}.compatible>div{display:flex;flex-wrap:wrap;gap:7px;margin-top:9px}.dataset-tip{font-size:11px;margin-bottom:0}@media(max-width:1150px){.dataset-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:700px){.dataset-grid{grid-template-columns:1fr}}
</style>
