<script setup>
import { ref, watch } from 'vue'
import { buildFormModel, extractHyperparams, groupHyperparams } from '../../utils/hyperparams'

/**
 * 动态超参数表单：根据算法的 hyperparams schema 渲染对应控件。
 * schema 由 adapters.js 从合同 default_params 推断（无 min/max/options）：
 *   int → 整数输入框；float → 小数输入框；bool → 开关；
 *   text → 文本框；choice/range 等历史类型保留支持；未知类型 → 文本框兜底
 */
const props = defineProps({
  algorithm: { type: Object, default: null },
  /** 后端不可用（gating）时禁用训练按钮 */
  disabled: { type: Boolean, default: false },
  initialValues: { type: Object, default: null }
})

const emit = defineEmits(['submit'])

const model = ref({})
const grouped = ref([])
const formError = ref('')
const labels = {learning_rate: '学习率', max_iter: '最大迭代次数', threshold: '分类阈值', l2: 'L2 正则化', tol: '收敛容差', standardize: '标准化特征', n_neighbors: '近邻数量', p: '距离阶数', weights: '投票权重', var_smoothing: '方差平滑', max_depth: '最大树深', min_samples_split: '最小分裂样本', min_samples_leaf: '叶节点最少样本', n_estimators: '树的数量', max_features: '候选特征数', criterion: '分裂准则', hidden_layers: '隐藏层结构', hidden_layer_sizes: '隐藏层结构', activation: '激活函数', batch_size: '批大小', n_clusters: '聚类数量', eps: '邻域半径', min_samples: '核心点最少样本', contamination: '异常比例', nu: '异常边界参数', gamma: '核宽度', kernel: '核函数', random_state: '模型随机种子', fit_intercept: '拟合截距', bootstrap: '有放回采样', n_init: '初始化次数', max_samples: '子样本数量'}

// 算法切换时重建表单模型
watch(
  () => [props.algorithm, props.initialValues],
  () => {
    model.value = buildFormModel(props.algorithm?.hyperparams || [])
    grouped.value = groupHyperparams(props.algorithm?.hyperparams || [])
    if (props.initialValues) for (const p of props.algorithm?.hyperparams || []) {
      if (Object.hasOwn(props.initialValues, p.name)) model.value[p.name] = p.type === 'json' ? JSON.stringify(props.initialValues[p.name]) : props.initialValues[p.name]
    }
    formError.value = ''
  },
  { immediate: true }
)

function resetDefaults() {
  model.value = buildFormModel(props.algorithm?.hyperparams || [])
}

function submit() {
  try {
    formError.value = ''
    emit('submit', extractHyperparams(model.value, props.algorithm?.hyperparams || []))
  } catch (error) { formError.value = error.message }
}
</script>

<template>
  <el-form v-if="algorithm" label-position="top" size="default" class="hyperparam-form">
    <template v-for="g in grouped" :key="g.group">
      <el-divider v-if="grouped.length > 1" content-position="left" class="group-divider">
        {{ g.group }}
      </el-divider>
      <el-form-item v-for="p in g.items" :key="p.name">
        <template #label>
          <span class="param-label">
            {{ labels[p.name] || p.label || p.name }}
            <small v-if="labels[p.name]" class="param-key">{{ p.name }}</small>
            <el-tooltip v-if="p.hint" :content="p.hint" placement="top">
              <el-icon class="hint-icon"><QuestionFilled /></el-icon>
            </el-tooltip>
          </span>
        </template>

        <!-- 整数 -->
        <el-input-number
          v-if="p.type === 'int'"
          v-model="model[p.name]"
          :min="p.min"
          :max="p.max"
          :step="p.step || 1"
          controls-position="right"
          class="full-width"
        />
        <!-- 小数（合同推断的参数无边界，不设 precision 以免舍入小值如 1e-9） -->
        <el-input-number
          v-else-if="p.type === 'float'"
          v-model="model[p.name]"
          :min="p.min"
          :max="p.max"
          :step="p.step || 0.01"
          controls-position="right"
          class="full-width"
        />
        <!-- 下拉选择 -->
        <el-select v-else-if="p.type === 'choice'" v-model="model[p.name]" class="full-width">
          <el-option v-for="opt in p.options" :key="opt" :label="opt" :value="opt" />
        </el-select>
        <!-- 布尔开关 -->
        <el-switch v-else-if="p.type === 'bool'" v-model="model[p.name]" />
        <!-- 范围（双滑块） -->
        <el-slider
          v-else-if="p.type === 'range'"
          v-model="model[p.name]"
          range
          :min="p.min"
          :max="p.max"
          :step="p.step || 1"
          class="full-width"
        />
        <!-- 文本（合同推断：字符串默认值） -->
        <el-input v-else-if="p.type === 'text'" v-model="model[p.name]" class="full-width" />
        <el-input v-else-if="p.type === 'json'" v-model="model[p.name]" placeholder="null、数字或 [16, 8]" class="full-width" />
        <!-- 未知类型：文本输入兜底，绝不崩溃 -->
        <el-input v-else v-model="model[p.name]" :placeholder="`类型 ${p.type} 未识别，按文本处理`" />
      </el-form-item>
    </template>

    <el-alert v-if="formError" :title="formError" type="error" :closable="false" class="form-error" />
    <div class="form-actions">
      <el-button @click="resetDefaults">恢复默认</el-button>
      <el-button type="primary" :disabled="disabled" @click="submit">开始训练</el-button>
    </div>
  </el-form>

  <el-empty v-else description="请先在左侧选择一个算法" :image-size="60" />
</template>

<style scoped>
.hyperparam-form {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 0 18px;
}
.param-key { font-size: 10px; color: #879b92; font-weight: 400; }
.group-divider, .form-actions, .form-error { grid-column: 1 / -1; }

.group-divider {
  margin: 8px 0;
}

.param-label {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.hint-icon {
  color: var(--el-text-color-secondary);
  font-size: 13px;
  cursor: help;
}

.full-width {
  width: 100%;
}

.form-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}

.form-actions .el-button {
  min-width: 128px;
}
</style>
