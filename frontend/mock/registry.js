/**
 * Mock 注册表 —— 与真实后端（ml_core registry.py + datasets.py）保持一致：
 * 同样的 12 个模型 id / 6 个数据集 id、中文 display_name 与参数说明。
 * HTTP mock 服务器（mock/server.js）直接返回这些条目；指标键集合与
 * 真实后端各任务适配器的输出一致。
 *
 * 说明：
 * - 前端离线兜底注册表（src/config/fallbackRegistry.js）是这些数据经
 *   src/services/adapters.js 适配后的静态镜像，两边应保持一致。
 * - svm 不在其中：真实后端未注册 svm（前端列表以真实后端为准）。
 * - class_weight: null 演示"合同 default_params 允许 null，前端表单跳过它"。
 */

export const MODELS = [
  {
    id: 'logistic_regression.optimized',
    display_name: '逻辑回归',
    task: 'classification',
    compatible_datasets: ['wdbc'],
    default_params: { learning_rate: 0.1, max_iter: 1000, threshold: 0.5, l2: 0.0, tol: 1e-8, standardize: true, class_weight: null },
    parameter_descriptions: {
      learning_rate: '梯度下降学习率（正数）',
      max_iter: '最大迭代次数（正整数）',
      threshold: '分类阈值（0~1）',
      l2: 'L2 正则化强度（非负）',
      tol: '早停容差（非负）',
      standardize: '是否对特征做标准化',
      class_weight: '类别权重（null 或 \'balanced\'）'
    }
  },
  {
    id: 'knn.optimized',
    display_name: 'K近邻',
    task: 'classification',
    compatible_datasets: ['wdbc'],
    default_params: { n_neighbors: 5, p: 2, weights: 'distance', standardize: true },
    parameter_descriptions: {
      n_neighbors: '参与投票的最近邻居个数',
      p: 'Minkowski 距离阶数（p=1 曼哈顿，p=2 欧氏）',
      weights: '投票权重（uniform/distance）',
      standardize: '是否对特征做标准化'
    }
  },
  {
    id: 'gaussian_naive_bayes.optimized',
    display_name: '朴素贝叶斯',
    task: 'classification',
    compatible_datasets: ['wdbc'],
    default_params: { var_smoothing: 1e-9 },
    parameter_descriptions: { var_smoothing: '方差平滑项（正数，防止零概率）' }
  },
  {
    id: 'cart_decision_tree.optimized',
    display_name: '决策树',
    task: 'classification',
    compatible_datasets: ['wdbc'],
    default_params: { max_depth: 8, min_samples_split: 2, min_samples_leaf: 1, min_impurity_decrease: 0.0, ccp_alpha: 0.0 },
    parameter_descriptions: {
      max_depth: '最大树深（正整数，null 表示不限制）',
      min_samples_split: '分裂所需最小样本数',
      min_samples_leaf: '叶节点最小样本数',
      min_impurity_decrease: '分裂最小不纯度下降',
      ccp_alpha: '代价复杂度剪枝系数（非负）'
    }
  },
  {
    id: 'random_forest.optimized',
    display_name: '随机森林',
    task: 'classification',
    compatible_datasets: ['wdbc'],
    default_params: { n_estimators: 50, max_depth: 8, min_samples_split: 2, min_samples_leaf: 1, voting: 'soft' },
    parameter_descriptions: {
      n_estimators: '树的数量（正整数）',
      max_depth: '单棵树最大深度（正整数）',
      min_samples_split: '分裂所需最小样本数',
      min_samples_leaf: '叶节点最小样本数',
      voting: '投票方式（soft 概率平均 / hard 多数投票）'
    }
  },
  {
    id: 'linear_regression.optimized',
    display_name: '线性回归',
    task: 'regression',
    compatible_datasets: ['concrete', 'california_housing'],
    default_params: { learning_rate: 0.01, max_iter: 1000, batch_size: 32, l2: 0.0, standardize: true, standardize_target: true },
    parameter_descriptions: {
      learning_rate: '学习率（正数）',
      max_iter: '最大迭代次数（正整数）',
      batch_size: '小批量大小（正整数）',
      l2: 'L2 正则化强度（非负）',
      standardize: '是否对特征做标准化',
      standardize_target: '是否对目标值做标准化（影响指标量纲）'
    }
  },
  {
    id: 'gbdt_regression.optimized',
    display_name: 'GBDT回归',
    task: 'regression',
    compatible_datasets: ['concrete'],
    default_params: { n_estimators: 100, learning_rate: 0.05, max_depth: 3, min_samples_split: 2, min_samples_leaf: 1, subsample: 1.0, validation_fraction: 0.1, l2_regularization: 0.0 },
    parameter_descriptions: {
      n_estimators: '树的数量（正整数）',
      learning_rate: '学习率（正数）',
      max_depth: '树最大深度（正整数）',
      min_samples_split: '分裂所需最小样本数',
      min_samples_leaf: '叶节点最小样本数',
      subsample: '每棵树样本采样比例（0~1）',
      validation_fraction: '早停验证集比例（0~1）',
      l2_regularization: '叶值 L2 正则化（非负）'
    }
  },
  {
    id: 'mlp_regression.optimized',
    display_name: 'MLP回归',
    task: 'regression',
    compatible_datasets: ['concrete', 'california_housing'],
    default_params: { activation: 'relu', learning_rate: 0.001, max_iter: 1000, batch_size: 32, l2: 0.0, validation_fraction: 0.2, n_iter_no_change: 20, standardize: true, standardize_target: true, gradient_clip: 5.0 },
    parameter_descriptions: {
      activation: '激活函数（relu/tanh）',
      learning_rate: '学习率（正数）',
      max_iter: '最大迭代次数（正整数）',
      batch_size: '小批量大小（正整数）',
      l2: 'L2 正则化强度（非负）',
      validation_fraction: '早停验证集比例（0~1）',
      n_iter_no_change: '早停耐心轮数（正整数）',
      standardize: '是否对特征做标准化',
      standardize_target: '是否对目标值做标准化',
      gradient_clip: '梯度裁剪阈值（正数）'
    }
  },
  {
    id: 'kmeans.optimized',
    display_name: 'K均值聚类',
    task: 'clustering',
    compatible_datasets: ['seeds'],
    default_params: { n_clusters: 3, init: 'k-means++', n_init: 10, max_iter: 300, tol: 0.0001, standardize: true },
    parameter_descriptions: {
      n_clusters: '簇数 k',
      init: "初始化方式：'random' 或 'k-means++'",
      n_init: '独立初始化次数，取惯性最低的一次',
      max_iter: '每次初始化的最大迭代次数',
      tol: '中心点移动的停止容差（非负）',
      standardize: '聚类前是否对特征做标准化'
    }
  },
  {
    id: 'dbscan.optimized',
    display_name: 'DBSCAN聚类',
    task: 'clustering',
    compatible_datasets: ['seeds'],
    default_params: { eps: 0.5, min_samples: 5, standardize: true },
    parameter_descriptions: {
      eps: '邻域半径（standardize=true 时作用于标准化空间）',
      min_samples: '核心点所需最小邻居数（正整数）',
      standardize: '是否对特征做标准化'
    }
  },
  {
    id: 'isolation_forest.optimized',
    display_name: '孤立森林',
    task: 'anomaly_detection',
    compatible_datasets: ['6_cardio', '23_mammography'],
    default_params: { n_estimators: 100, max_samples: 256, contamination: 0.1, max_features: 1.0 },
    parameter_descriptions: {
      n_estimators: '树的数量（正整数）',
      max_samples: '每棵树采样数（正整数或 0~1 比例）',
      contamination: '期望异常比例，决定判定阈值（心电图约 0.096，乳腺造影约 0.023）',
      max_features: '每次切分使用的特征比例（正整数或 0~1 比例）'
    }
  },
  {
    id: 'one_class_svm.optimized',
    display_name: '单类SVM',
    task: 'anomaly_detection',
    compatible_datasets: ['6_cardio', '23_mammography'],
    default_params: { nu: 0.1, gamma: 'scale', max_iter: 500, standardize: true },
    parameter_descriptions: {
      nu: '训练误差上限（0~1，越小边界越紧）',
      gamma: "RBF 核系数（'scale' 或正数）",
      max_iter: '最大迭代次数（正整数）',
      standardize: '是否对特征做标准化'
    }
  }
]

export const DATASETS = [
  { id: 'wdbc', display_name: '乳腺癌诊断', task: 'classification', sample_count: 569, feature_count: 30, has_target: true },
  { id: 'seeds', display_name: '小麦种子', task: 'clustering', sample_count: 210, feature_count: 7, has_target: true },
  { id: 'concrete', display_name: '混凝土强度', task: 'regression', sample_count: 1030, feature_count: 8, has_target: true },
  { id: 'california_housing', display_name: '加州房价', task: 'regression', sample_count: 20640, feature_count: 8, has_target: true },
  { id: '6_cardio', display_name: '心电图异常检测', task: 'anomaly_detection', sample_count: 1831, feature_count: 21, has_target: true },
  { id: '23_mammography', display_name: '乳腺造影异常检测', task: 'anomaly_detection', sample_count: 11183, feature_count: 6, has_target: true }
]

/**
 * 模拟性能表：每个真实兼容组合 (模型 × 数据集) 的基础指标。
 * 键集合与真实后端适配器输出一致（分类 8 键 / 回归 4 键 /
 * 聚类分 kmeans 与 dbscan 两套 / 异常 5 键）。
 * mock/server.js 在这些基础值上加小噪声生成最终结果。
 */
export const PERFORMANCE = {
  'logistic_regression.optimized@wdbc': { accuracy: 0.96, precision: 0.93, recall: 0.95, f1: 0.94, tn: 70, fp: 3, fn: 2, tp: 39, train_ms: 80, eval_ms: 30 },
  'knn.optimized@wdbc': { accuracy: 0.94, precision: 0.92, recall: 0.9, f1: 0.91, tn: 68, fp: 3, fn: 4, tp: 38, train_ms: 12, eval_ms: 90 },
  'gaussian_naive_bayes.optimized@wdbc': { accuracy: 0.92, precision: 0.9, recall: 0.88, f1: 0.89, tn: 66, fp: 4, fn: 5, tp: 37, train_ms: 6, eval_ms: 25 },
  'cart_decision_tree.optimized@wdbc': { accuracy: 0.93, precision: 0.91, recall: 0.9, f1: 0.9, tn: 67, fp: 3, fn: 4, tp: 38, train_ms: 60, eval_ms: 30 },
  'random_forest.optimized@wdbc': { accuracy: 0.95, precision: 0.93, recall: 0.93, f1: 0.93, tn: 69, fp: 2, fn: 3, tp: 39, train_ms: 900, eval_ms: 120 },
  'linear_regression.optimized@concrete': { mse: 115.0, rmse: 10.7, mae: 8.2, r2: 0.61, train_ms: 150, eval_ms: 20 },
  'linear_regression.optimized@california_housing': { mse: 0.52, rmse: 0.72, mae: 0.54, r2: 0.6, train_ms: 800, eval_ms: 90 },
  'gbdt_regression.optimized@concrete': { mse: 25.0, rmse: 5.0, mae: 3.7, r2: 0.9, train_ms: 8000, eval_ms: 60 },
  'mlp_regression.optimized@concrete': { mse: 35.0, rmse: 5.9, mae: 4.5, r2: 0.87, train_ms: 3000, eval_ms: 40 },
  'mlp_regression.optimized@california_housing': { mse: 0.3, rmse: 0.55, mae: 0.41, r2: 0.78, train_ms: 9000, eval_ms: 150 },
  'kmeans.optimized@seeds': { inertia: 180.0, adjusted_rand_index: 0.72, n_clusters: 3, train_ms: 50, eval_ms: 10 },
  'dbscan.optimized@seeds': { n_clusters: 3, noise_points: 12, adjusted_rand_index: 0.65, train_ms: 40, eval_ms: 10 },
  'isolation_forest.optimized@6_cardio': { true_anomalies: 176, detected_anomalies: 190, anomaly_recall: 0.81, anomaly_precision: 0.75, anomaly_f1: 0.78, train_ms: 2200, eval_ms: 500 },
  'isolation_forest.optimized@23_mammography': { true_anomalies: 260, detected_anomalies: 300, anomaly_recall: 0.77, anomaly_precision: 0.67, anomaly_f1: 0.72, train_ms: 9000, eval_ms: 2000 },
  'one_class_svm.optimized@6_cardio': { true_anomalies: 176, detected_anomalies: 150, anomaly_recall: 0.66, anomaly_precision: 0.77, anomaly_f1: 0.71, train_ms: 15000, eval_ms: 800 },
  'one_class_svm.optimized@23_mammography': { true_anomalies: 260, detected_anomalies: 210, anomaly_recall: 0.61, anomaly_precision: 0.76, anomaly_f1: 0.68, train_ms: 18000, eval_ms: 2500 }
}
