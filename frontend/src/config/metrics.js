/**
 * 指标定义表（前端专属静态配置）
 *
 * HTTP 合同（HTTP_API_CONTRACT.md）没有指标注册端点，因此"某个任务类型
 * 展示哪些指标、用什么渲染器"是前端自己的知识，与传输无关。
 * 键集合与真实后端各任务适配器的输出一致（classification 8 键 /
 * regression 4 键 / clustering kmeans 与 dbscan 两套 / anomaly_detection 5 键）。
 *
 * 渲染规则（components/metrics/MetricCard.vue）：
 *   scalar → 数值卡片；matrix + chart:'heatmap' → 热力图；
 *   curve + chart:'roc' → ROC 曲线；其余 → JSON 表格
 *
 * 结果面板只渲染 metrics 中实际存在值的定义键；全部不命中时
 * 整包 metrics 走 JsonFallback。
 */
export const METRICS = {
  classification: [
    { id: 'accuracy', name: '准确率', type: 'scalar', higherIsBetter: true, hint: '预测正确的样本占比' },
    { id: 'precision', name: '精确率', type: 'scalar', higherIsBetter: true, hint: '预测为正类中真正的正类占比' },
    { id: 'recall', name: '召回率', type: 'scalar', higherIsBetter: true, hint: '真正的正类中被找出的占比' },
    { id: 'f1', name: 'F1分数', type: 'scalar', higherIsBetter: true, hint: '精确率与召回率的调和平均' },
    { id: 'tn', name: '真阴性', type: 'scalar' },
    { id: 'fp', name: '假阳性', type: 'scalar' },
    { id: 'fn', name: '假阴性', type: 'scalar' },
    { id: 'tp', name: '真阳性', type: 'scalar' }
  ],
  regression: [
    { id: 'mse', name: '均方误差', type: 'scalar', higherIsBetter: false },
    { id: 'rmse', name: '均方根误差', type: 'scalar', higherIsBetter: false },
    { id: 'mae', name: '平均绝对误差', type: 'scalar', higherIsBetter: false },
    { id: 'r2', name: 'R² 决定系数', type: 'scalar', higherIsBetter: true, hint: '越接近 1 拟合越好' }
  ],
  clustering: [
    { id: 'inertia', name: '簇内平方和', type: 'scalar', higherIsBetter: false },
    { id: 'adjusted_rand_index', name: '调整兰德指数', type: 'scalar', higherIsBetter: true, hint: '与真实标签比较，越接近 1 越好' },
    { id: 'n_clusters', name: '簇数', type: 'scalar' },
    { id: 'noise_points', name: '噪声点数', type: 'scalar', higherIsBetter: false },
    { id: 'silhouette', name: '轮廓系数', type: 'scalar', higherIsBetter: true, hint: '越接近 1 聚类越紧密' },
    { id: 'davies_bouldin', name: 'DB指数', type: 'scalar', higherIsBetter: false, hint: '越小聚类质量越好' }
  ],
  anomaly_detection: [
    { id: 'true_anomalies', name: '真实异常数', type: 'scalar' },
    { id: 'detected_anomalies', name: '检出异常数', type: 'scalar' },
    { id: 'anomaly_recall', name: '异常召回率', type: 'scalar', higherIsBetter: true, hint: '真实异常中被检出的比例' },
    { id: 'anomaly_precision', name: '异常精确率', type: 'scalar', higherIsBetter: true, hint: '判定异常中真实异常的比例' },
    { id: 'anomaly_f1', name: '异常F1', type: 'scalar', higherIsBetter: true, hint: '召回率与精确率的调和平均' }
  ]
}
