<script setup>
import { ref, watch } from 'vue'
import { buildFormModel, extractHyperparams, groupHyperparams } from '../../utils/hyperparams'

/**
 * 动态超参数表单：根据算法的 hyperparams schema 渲染对应控件。
 * 控件类型映射（可扩展性演示点，协议见 docs/PROTOCOL.md）：
 *   int → 整数输入框；float → 小数输入框；choice → 下拉框；
 *   bool → 开关；range → 双滑块；未知类型 → 普通文本框兜底
 */
const props = defineProps({
  algorithm: { type: Object, default: null },
  disabled: { type: Boolean, default: false }
})

const emit = defineEmits(['submit'])

const model = ref({})
const grouped = ref([])

// 算法切换时重建表单模型
watch(
  () => props.algorithm?.id,
  () => {
    model.value = buildFormModel(props.algorithm?.hyperparams || [])
    grouped.value = groupHyperparams(props.algorithm?.hyperparams || [])
  },
  { immediate: true }
)

function resetDefaults() {
  model.value = buildFormModel(props.algorithm?.hyperparams || [])
}

function submit() {
  emit('submit', extractHyperparams(model.value, props.algorithm?.hyperparams || []))
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
            {{ p.label || p.name }}
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
        <!-- 小数 -->
        <el-input-number
          v-else-if="p.type === 'float'"
          v-model="model[p.name]"
          :min="p.min"
          :max="p.max"
          :step="p.step || 0.01"
          :precision="10"
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
        <!-- 未知类型：文本输入兜底，绝不崩溃 -->
        <el-input v-else v-model="model[p.name]" :placeholder="`类型 ${p.type} 未识别，按文本处理`" />
      </el-form-item>
    </template>

    <div class="form-actions">
      <el-button @click="resetDefaults">恢复默认</el-button>
      <el-button type="primary" :disabled="disabled" @click="submit">开始训练</el-button>
    </div>
  </el-form>

  <el-empty v-else description="请先在左侧选择一个算法" :image-size="60" />
</template>

<style scoped>
.hyperparam-form {
  padding: 0 4px;
}

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
  flex: 1;
}
</style>
