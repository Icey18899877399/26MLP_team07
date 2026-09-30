import { inferHyperparam } from '../services/adapters.js'

export function interactiveAlgorithm(entry) {
  if (!entry) return null
  return {
    id: entry.model,
    hyperparams: Object.entries(entry.default_params).map(([key, value]) => {
      const field = inferHyperparam(key, value, entry.parameter_descriptions)
      if (key === 'class_weight') return { ...field, type: 'json', hint: `${field.hint || ''}；输入 null 或 "balanced"（带引号）` }
      if (key === 'gamma') return { ...field, type: 'text', default: String(value), hint: `${field.hint || ''}；可输入正数或 scale` }
      return field
    }).filter(Boolean)
  }
}
