<script setup>
import { computed, ref, watch } from 'vue'
import { useConnectionStore } from '../stores/connection'
import { useTrainingStore } from '../stores/training'
import AlgorithmList from '../components/experiment/AlgorithmList.vue'
import HyperparamForm from '../components/experiment/HyperparamForm.vue'
import RunMonitor from '../components/experiment/RunMonitor.vue'
import MetricCard from '../components/metrics/MetricCard.vue'

/**
 * 实验工作台（核心页面）：
 * 左：算法列表（注册表驱动）｜右上：超参数表单 + 数据集/切分设置｜右下：训练监控与结果
 */
const conn = useConnectionStore()
const training = useTrainingStore()

const selectedAlgo = ref(null)
const selectedDatasetId = ref('')
const split = ref({ testRatio: 0.3, seed: 42, stratify: true })

const algorithms = computed(() => conn.registry.algorithms || [])
const datasets = computed(() => conn.registry.datasets || [])

/** 当前算法适用的数据集（按任务类型过滤） */
const availableDatasets = computed(() => {
  if (!selectedAlgo.value) return []
  return datasets.value.filter((d) => (selectedAlgo.value.taskTypes || []).includes(d.taskType)
    && (!selectedAlgo.value.compatibleDatasets?.length || selectedAlgo.value.compatibleDatasets.includes(d.id)))
})

const selectedDataset = computed(() =>
  datasets.value.find((d) => d.id === selectedDatasetId.value)
)

// 切换算法时自动选中第一个适用数据集，并重置切分参数为数据集默认值
watch(
  () => selectedAlgo.value?.id,
  () => {
    const list = availableDatasets.value
    if (list.length > 0) {
      selectedDatasetId.value = list[0].id
      const def = list[0].split || {}
      split.value = {
        testRatio: def.defaultTestRatio ?? 0.3,
        seed: def.defaultSeed ?? 42,
        stratify: def.stratifySupported ?? false
      }
    } else {
      selectedDatasetId.value = ''
    }
  }
)

function handleTrain(hyperparams) {
  if (!selectedAlgo.value || !selectedDatasetId.value) return
  training.startRun({
    algorithmId: selectedAlgo.value.id,
    algorithmName: selectedAlgo.value.name,
    datasetId: selectedDatasetId.value,
    datasetName: selectedDataset.value?.name || selectedDatasetId.value,
    hyperparams,
    split: { ...split.value }
  })
}

const selectedRunId = ref('')
const selectedRun = computed(() => training.getRun(selectedRunId.value))
watch(() => training.finishedRuns[0]?.runId, id => { if (id) selectedRunId.value = id })
watch(algorithms, list => {
  selectedAlgo.value = list.find(a => a.id === selectedAlgo.value?.id) || list[0] || null
}, { immediate: true })

/** 结果指标定义：按 run 所用数据集的任务类型从注册表取 */
const metricDefs = computed(() => {
  if (!selectedRun.value) return []
  const dataset = datasets.value.find((d) => d.id === selectedRun.value.datasetId)
  const taskType = dataset?.taskType || 'classification'
  return conn.registry.metrics?.[taskType] || []
})
</script>

<template>
  <div class="experiment-layout">
    <!-- 左：算法列表 -->
    <div class="panel-left page-card">
      <h3 class="page-card-title">算法选择</h3>
      <AlgorithmList :algorithms="algorithms" :active-id="selectedAlgo?.id" @select="selectedAlgo = $event" />
    </div>

    <!-- 右：配置 + 训练 + 结果 -->
    <div class="panel-right">
      <el-alert v-if="conn.lastError" :title="conn.lastError" type="error" :closable="false" />
      <el-alert v-for="run in training.runList.filter(r => r.error)" :key="run.runId"
        :title="run.algorithmName + ': ' + run.error" type="error" :closable="false" />
      <div class="page-card">
        <div v-if="selectedAlgo" class="algo-header">
          <div>
            <h3 class="page-card-title">{{ selectedAlgo.name }}</h3>
            <p class="algo-summary">{{ selectedAlgo.description }}</p>
          </div>
          <el-tag v-if="conn.isUsingFallback" type="info" effect="plain">未连接后端，仅本地演示配置</el-tag>
        </div>

        <el-form label-position="top" class="dataset-form">
          <div class="dataset-row">
            <el-form-item label="数据集">
              <el-select v-model="selectedDatasetId" style="width: 220px">
                <el-option
                  v-for="d in availableDatasets"
                  :key="d.id"
                  :label="`${d.name}（${d.nSamples} 样本 × ${d.nFeatures} 特征）`"
                  :value="d.id"
                />
              </el-select>
            </el-form-item>
            <el-form-item v-if="selectedDataset?.taskType === 'classification'" label="测试集比例">
              <el-slider v-model="split.testRatio" :min="0.1" :max="0.5" :step="0.05" style="width: 160px" show-input :show-input-controls="false" />
            </el-form-item>
            <el-form-item label="随机种子">
              <el-input-number v-model="split.seed" :min="0" :max="99999" controls-position="right" style="width: 130px" />
            </el-form-item>
            <el-form-item v-if="selectedDataset?.taskType === 'classification'" label="分层抽样（模型固定启用）">
              <el-switch :model-value="true" disabled />
            </el-form-item>
          </div>
        </el-form>

        <HyperparamForm :algorithm="selectedAlgo" :disabled="conn.status !== 'connected' || !selectedDatasetId || training.activeRuns.length > 0" @submit="handleTrain" />
      </div>

      <!-- 训练监控 -->
      <div v-if="training.activeRuns.length > 0" class="page-card">
        <h3 class="page-card-title">训练中</h3>
        <RunMonitor
          v-for="run in training.activeRuns"
          :key="run.runId"
          :run="run"
          @cancel="training.cancelRun(run.runId)"
          @select="selectedRunId = run.runId"
        />
      </div>

      <!-- 历史结果 -->
      <div v-if="training.finishedRuns.length > 0" class="page-card">
        <h3 class="page-card-title">
          训练结果
          <el-button text size="small" @click="training.clearRuns()">清空</el-button>
        </h3>
        <el-radio-group v-model="selectedRunId" size="small" class="run-selector">
          <el-radio-button v-for="run in training.finishedRuns" :key="run.runId" :value="run.runId">
            {{ run.algorithmName }} · {{ run.datasetName }}
          </el-radio-button>
        </el-radio-group>
        <div v-if="selectedRun" class="result-area">
          <div class="result-meta">
            <el-tag type="success" size="small">已完成</el-tag>
            <span class="meta-text">总请求耗时 {{ selectedRun.progress.elapsedMs }} ms</span>
          </div>
          <div class="metric-grid">
            <MetricCard
              v-for="metric in metricDefs"
              :key="metric.id"
              :metric="metric"
              :value="selectedRun.result?.metrics?.[metric.id]"
              :extra="selectedRun.result?.metrics"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.experiment-layout {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

.panel-left {
  width: 300px;
  flex-shrink: 0;
  position: sticky;
  top: 16px;
  max-height: calc(100vh - 100px);
  overflow-y: auto;
}

.panel-right {
  flex: 1;
  min-width: 0;
}

.algo-header {
  margin-bottom: 8px;
}

.algo-summary {
  margin: -4px 0 12px;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}

.dataset-row {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
  margin-bottom: 8px;
}

.run-selector {
  margin-bottom: 16px;
}

.result-area {
  margin-top: 8px;
}

.result-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.meta-text {
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}
</style>
