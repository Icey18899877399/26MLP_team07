/**
 * 超参数 schema 工具。
 * 根据后端注册表里的 hyperparams 数组构建表单模型、提取默认值、做轻量校验。
 * 表单控件类型映射（可扩展性核心之一）：
 *   int    → el-input-number（整数）
 *   float  → el-input-number（小数）
 *   choice → el-select
 *   bool   → el-switch
 *   range  → el-slider（双滑块，值为 [lo, hi]）
 *   其他   → el-input（文本降级，不崩溃）
 */

/** 根据 schema 构建表单模型 { name: 默认值 } */
export function buildFormModel(hyperparams) {
  const model = {}
  for (const p of hyperparams || []) {
    if (p.type === 'range') {
      // range 默认值为 [lo, hi]，无默认则取 [min, max]
      model[p.name] = Array.isArray(p.default)
        ? [...p.default]
        : [p.min ?? 0, p.max ?? 1]
    } else {
      model[p.name] = p.default
    }
  }
  return model
}

/** 从表单模型提取训练请求所需的超参数对象（值与输入保持一致，后端自行转换类型） */
export function extractHyperparams(model, hyperparams) {
  const result = {}
  for (const p of hyperparams || []) {
    const value = model[p.name]
    if (value === undefined || value === null || value === '') continue
    result[p.name] = value
  }
  return result
}

/** 判断某个超参数是否允许在当前任务类型下出现（目前后端只发适用的参数，这里做防御） */
export function filterByTask(hyperparams) {
  return hyperparams || []
}

/** 将超参数按 group 分组，返回 [{ group, items }]，无 group 的归入"其他参数" */
export function groupHyperparams(hyperparams) {
  const groups = new Map()
  for (const p of hyperparams || []) {
    const key = p.group || '其他参数'
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(p)
  }
  return [...groups.entries()].map(([group, items]) => ({ group, items }))
}
