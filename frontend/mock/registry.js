/**
 * Mock 注册表 —— 模拟后端 config.registry 消息的内容。
 * 注意：与 src/config/fallbackRegistry.js 结构一致，但内容更多
 * （6 个算法 vs 5 个、5 个数据集 vs 4 个），用来演示：
 * "后端注册多少，前端就显示多少，前端零改动"。
 * 协议文档见 docs/PROTOCOL.md。
 */

export const ALGORITHMS = [
  {
    id: 'knn',
    name: 'K近邻',
    taskTypes: ['classification', 'regression'],
    description: '基于距离度量的惰性学习算法，投票/加权平均决定预测结果',
    hyperparams: [
      { name: 'n_neighbors', label: '邻居数 k', type: 'int', default: 5, min: 1, max: 50, step: 1, group: '基本参数', hint: '参与投票的最近邻居个数' },
      { name: 'weights', label: '权重方式', type: 'choice', default: 'uniform', options: ['uniform', 'distance'], group: '基本参数', hint: 'uniform 等权投票，distance 按距离倒数加权' },
      { name: 'metric', label: '距离度量', type: 'choice', default: 'minkowski', options: ['minkowski', 'euclidean', 'manhattan'], group: '基本参数' },
      { name: 'p', label: '闵氏距离 p 值', type: 'float', default: 2.0, min: 1.0, max: 5.0, step: 0.5, group: '高级参数', hint: 'p=1 为曼哈顿距离，p=2 为欧氏距离' }
    ]
  },
  {
    id: 'decision_tree',
    name: '决策树',
    taskTypes: ['classification', 'regression'],
    description: '基于信息增益/基尼系数的树形模型，可解释性强',
    hyperparams: [
      { name: 'criterion', label: '划分准则', type: 'choice', default: 'gini', options: ['gini', 'entropy'], group: '基本参数' },
      { name: 'max_depth', label: '最大深度', type: 'int', default: 10, min: 1, max: 30, step: 1, group: '基本参数', hint: '限制树深度防止过拟合' },
      { name: 'min_samples_split', label: '分裂最小样本数', type: 'int', default: 2, min: 2, max: 20, step: 1, group: '高级参数' },
      { name: 'min_samples_leaf', label: '叶节点最小样本数', type: 'int', default: 1, min: 1, max: 10, step: 1, group: '高级参数' }
    ]
  },
  {
    id: 'naive_bayes',
    name: '朴素贝叶斯',
    taskTypes: ['classification'],
    description: '基于贝叶斯定理与特征条件独立假设的概率模型',
    hyperparams: [
      { name: 'var_smoothing', label: '方差平滑系数', type: 'float', default: 1e-9, min: 1e-12, max: 1e-6, step: 1e-9, group: '基本参数', hint: '防止零概率，数值越小越接近无平滑' }
    ]
  },
  {
    id: 'logistic_regression',
    name: '逻辑回归',
    taskTypes: ['classification'],
    description: '线性分类模型，通过 sigmoid 输出类别概率',
    hyperparams: [
      { name: 'learning_rate', label: '学习率', type: 'float', default: 0.01, min: 0.0001, max: 1.0, step: 0.001, group: '基本参数' },
      { name: 'max_iter', label: '最大迭代次数', type: 'int', default: 100, min: 10, max: 2000, step: 10, group: '基本参数' },
      { name: 'regularization', label: '正则化', type: 'choice', default: 'l2', options: ['none', 'l1', 'l2'], group: '高级参数' },
      { name: 'fit_intercept', label: '拟合截距', type: 'bool', default: true, group: '高级参数' }
    ]
  },
  {
    id: 'svm',
    name: '支持向量机',
    taskTypes: ['classification'],
    description: '寻找最大间隔超平面，通过核函数处理非线性问题',
    hyperparams: [
      { name: 'kernel', label: '核函数', type: 'choice', default: 'rbf', options: ['linear', 'rbf', 'poly'], group: '基本参数' },
      { name: 'C', label: '惩罚系数 C', type: 'float', default: 1.0, min: 0.01, max: 100.0, step: 0.01, group: '基本参数', hint: 'C 越大越不允许分类误差（易过拟合）' },
      { name: 'gamma', label: '核系数 γ', type: 'choice', default: 'scale', options: ['scale', 'auto'], group: '高级参数' }
    ]
  },
  {
    id: 'kmeans',
    name: 'K均值聚类',
    taskTypes: ['clustering'],
    description: '基于质心的划分聚类算法，最小化簇内平方和',
    hyperparams: [
      { name: 'n_clusters', label: '簇数 k', type: 'int', default: 3, min: 2, max: 20, step: 1, group: '基本参数' },
      { name: 'n_init', label: '初始化次数', type: 'int', default: 10, min: 1, max: 100, step: 1, group: '基本参数', hint: '取多次运行中最优的一次，避免局部最优' },
      { name: 'tol', label: '收敛阈值', type: 'float', default: 0.0001, min: 0.00001, max: 0.1, step: 0.0001, group: '高级参数' }
    ]
  }
]

export const DATASETS = [
  {
    id: 'iris',
    name: '鸢尾花',
    taskType: 'classification',
    nSamples: 150,
    nFeatures: 4,
    nClasses: 3,
    description: '经典三分类数据集，根据花萼/花瓣尺寸区分三种鸢尾花',
    split: { defaultTestRatio: 0.3, defaultSeed: 42, stratifySupported: true },
    featureNames: ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'],
    classNames: ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾'],
    classDistribution: [50, 50, 50],
    sampleRows: [
      [5.1, 3.5, 1.4, 0.2, '山鸢尾'],
      [4.9, 3.0, 1.4, 0.2, '山鸢尾'],
      [7.0, 3.2, 4.7, 1.4, '变色鸢尾'],
      [6.4, 3.2, 4.5, 1.5, '变色鸢尾'],
      [6.3, 3.3, 6.0, 2.5, '维吉尼亚鸢尾'],
      [5.8, 2.7, 5.1, 1.9, '维吉尼亚鸢尾'],
      [5.4, 3.9, 1.7, 0.4, '山鸢尾'],
      [6.1, 2.8, 4.7, 1.2, '变色鸢尾'],
      [7.1, 3.0, 5.9, 2.1, '维吉尼亚鸢尾'],
      [5.0, 3.4, 1.5, 0.2, '山鸢尾']
    ]
  },
  {
    id: 'wine',
    name: '葡萄酒',
    taskType: 'classification',
    nSamples: 178,
    nFeatures: 13,
    nClasses: 3,
    description: '根据化学成分区分三种葡萄酒产地',
    split: { defaultTestRatio: 0.3, defaultSeed: 42, stratifySupported: true },
    featureNames: ['酒精', '苹果酸', '灰分', '灰分碱度', '镁', '总酚', '类黄酮', '非类黄酮酚', '原花青素', '颜色强度', '色调', '稀释酒OD280', '脯氨酸'],
    classNames: ['品种0', '品种1', '品种2'],
    classDistribution: [59, 71, 48],
    sampleRows: [
      [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065, '品种0'],
      [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050, '品种0'],
      [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520, '品种1'],
      [12.08, 1.83, 2.32, 18.5, 81, 1.6, 1.5, 0.52, 1.64, 2.4, 1.08, 2.27, 480, '品种1'],
      [12.86, 1.35, 2.32, 18, 122, 1.51, 1.25, 0.21, 0.94, 4.1, 0.76, 1.29, 630, '品种2'],
      [13.05, 1.77, 2.1, 17, 107, 3, 3, 0.28, 2.03, 5.04, 0.88, 3.35, 885, '品种2']
    ]
  },
  {
    id: 'breast_cancer',
    name: '乳腺癌诊断',
    taskType: 'classification',
    nSamples: 569,
    nFeatures: 30,
    nClasses: 2,
    description: '根据细胞核特征判断肿瘤良恶性（二分类）',
    split: { defaultTestRatio: 0.3, defaultSeed: 42, stratifySupported: true },
    featureNames: ['平均半径', '平均纹理', '平均周长', '平均面积', '平均平滑度', '平均紧密度', '平均凹度', '平均凹点', '平均对称性', '平均分形维数'],
    classNames: ['恶性', '良性'],
    classDistribution: [212, 357],
    sampleRows: [
      [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, '恶性'],
      [20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, '恶性'],
      [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, '良性'],
      [13.08, 15.71, 85.63, 520, 0.1075, 0.127, 0.04568, 0.0311, 0.1967, 0.06811, '良性'],
      [12.46, 24.04, 83.97, 475.9, 0.1186, 0.2396, 0.2273, 0.08543, 0.203, 0.08243, '良性'],
      [19.17, 24.8, 132.4, 1123, 0.0974, 0.2458, 0.2065, 0.1118, 0.2397, 0.078, '恶性']
    ]
  },
  {
    id: 'boston',
    name: '波士顿房价',
    taskType: 'regression',
    nSamples: 506,
    nFeatures: 13,
    nClasses: 0,
    description: '经典回归数据集，预测波士顿地区房价中位数（单位：千美元）',
    split: { defaultTestRatio: 0.3, defaultSeed: 42, stratifySupported: false },
    featureNames: ['犯罪率', '住宅用地比例', '非零售商业比例', '靠河', '氮氧化物浓度', '平均房间数', '房龄', '到就业中心距离', '高速路可达性', '税率', '师生比', '黑人比例', '低地位人口比例'],
    classNames: [],
    classDistribution: null,
    sampleRows: [
      [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98, 24.0],
      [0.02731, 0.0, 7.07, 0, 0.469, 6.421, 78.9, 4.967, 2, 242, 17.8, 396.9, 9.14, 21.6],
      [0.02729, 0.0, 7.07, 0, 0.469, 7.185, 61.1, 4.967, 2, 242, 17.8, 392.83, 4.03, 34.7],
      [0.03237, 0.0, 2.18, 0, 0.458, 6.998, 45.8, 6.062, 3, 222, 18.7, 394.63, 2.94, 33.4],
      [0.06905, 0.0, 2.18, 0, 0.458, 7.147, 54.2, 6.062, 3, 222, 18.7, 396.9, 5.33, 36.2]
    ]
  },
  {
    id: 'blobs',
    name: '高斯混合聚类数据',
    taskType: 'clustering',
    nSamples: 1500,
    nFeatures: 2,
    nClasses: 0,
    description: 'sklearn 生成的各向同性高斯簇数据（5 个簇），用于演示聚类算法',
    split: { defaultTestRatio: 0.2, defaultSeed: 42, stratifySupported: false },
    featureNames: ['x1', 'x2'],
    classNames: [],
    classDistribution: [300, 300, 300, 300, 300],
    sampleRows: [
      [2.0983, 1.1915, 0],
      [1.8122, 0.6023, 0],
      [-8.9721, 7.2309, 1],
      [-9.1327, 6.9301, 1],
      [3.5018, 6.7702, 2],
      [2.8821, 7.4244, 2]
    ]
  }
]

export const METRICS = {
  classification: [
    { id: 'accuracy', name: '准确率', type: 'scalar', higherIsBetter: true, hint: '预测正确的样本占比' },
    { id: 'precision', name: '精确率', type: 'scalar', higherIsBetter: true, hint: '预测为正类中真正的正类占比' },
    { id: 'recall', name: '召回率', type: 'scalar', higherIsBetter: true, hint: '真正的正类中被找出的占比' },
    { id: 'f1', name: 'F1分数', type: 'scalar', higherIsBetter: true, hint: '精确率与召回率的调和平均' },
    { id: 'confusion_matrix', name: '混淆矩阵', type: 'matrix', chart: 'heatmap', hint: '行列分别为真实/预测类别' },
    { id: 'roc_auc', name: 'ROC曲线', type: 'curve', chart: 'roc', higherIsBetter: true, hint: 'AUC 越大分类能力越强' }
  ],
  regression: [
    { id: 'mse', name: '均方误差', type: 'scalar', higherIsBetter: false },
    { id: 'rmse', name: '均方根误差', type: 'scalar', higherIsBetter: false },
    { id: 'mae', name: '平均绝对误差', type: 'scalar', higherIsBetter: false },
    { id: 'r2', name: 'R² 决定系数', type: 'scalar', higherIsBetter: true, hint: '越接近 1 拟合越好' }
  ],
  clustering: [
    { id: 'silhouette', name: '轮廓系数', type: 'scalar', higherIsBetter: true, hint: '越接近 1 聚类越紧密' },
    { id: 'davies_bouldin', name: 'DB指数', type: 'scalar', higherIsBetter: false, hint: '越小聚类质量越好' },
    { id: 'inertia', name: '簇内平方和', type: 'scalar', higherIsBetter: false }
  ]
}

/**
 * 模拟性能表：每对 (算法 × 数据集) 的基础指标。
 * mock/server.js 在这些基础值上加小噪声生成最终结果，
 * 让 benchmark 对比图看起来真实且有区分度。
 */
export const PERFORMANCE = {
  'knn@iris': { accuracy: 0.9667, precision: 0.968, recall: 0.9667, f1: 0.9664, train_ms: 12, eval_ms: 45, curve: 'fast' },
  'knn@wine': { accuracy: 0.7411, precision: 0.748, recall: 0.7411, f1: 0.7395, train_ms: 8, eval_ms: 40, curve: 'fast' },
  'knn@breast_cancer': { accuracy: 0.9591, precision: 0.963, recall: 0.9591, f1: 0.9590, train_ms: 15, eval_ms: 90, curve: 'fast' },
  'knn@boston': { mse: 26.71, rmse: 5.168, mae: 3.621, r2: 0.6632, train_ms: 10, eval_ms: 60, curve: 'fast' },
  'decision_tree@iris': { accuracy: 0.9333, precision: 0.938, recall: 0.9333, f1: 0.9329, train_ms: 25, eval_ms: 20, curve: 'step' },
  'decision_tree@wine': { accuracy: 0.8889, precision: 0.892, recall: 0.8889, f1: 0.8875, train_ms: 30, eval_ms: 22, curve: 'step' },
  'decision_tree@breast_cancer': { accuracy: 0.9298, precision: 0.934, recall: 0.9298, f1: 0.9297, train_ms: 60, eval_ms: 30, curve: 'step' },
  'decision_tree@boston': { mse: 19.24, rmse: 4.386, mae: 3.002, r2: 0.7575, train_ms: 55, eval_ms: 25, curve: 'step' },
  'naive_bayes@iris': { accuracy: 0.9556, precision: 0.958, recall: 0.9556, f1: 0.9554, train_ms: 5, eval_ms: 12, curve: 'fast' },
  'naive_bayes@wine': { accuracy: 0.9630, precision: 0.966, recall: 0.9630, f1: 0.9628, train_ms: 4, eval_ms: 14, curve: 'fast' },
  'naive_bayes@breast_cancer': { accuracy: 0.9298, precision: 0.936, recall: 0.9298, f1: 0.9285, train_ms: 6, eval_ms: 25, curve: 'fast' },
  'logistic_regression@iris': { accuracy: 0.9556, precision: 0.958, recall: 0.9556, f1: 0.9554, train_ms: 40, eval_ms: 15, curve: 'smooth' },
  'logistic_regression@wine': { accuracy: 0.9630, precision: 0.966, recall: 0.9630, f1: 0.9628, train_ms: 45, eval_ms: 16, curve: 'smooth' },
  'logistic_regression@breast_cancer': { accuracy: 0.9474, precision: 0.952, recall: 0.9474, f1: 0.9471, train_ms: 80, eval_ms: 30, curve: 'smooth' },
  'svm@iris': { accuracy: 0.9778, precision: 0.980, recall: 0.9778, f1: 0.9777, train_ms: 65, eval_ms: 18, curve: 'smooth' },
  'svm@wine': { accuracy: 0.9815, precision: 0.983, recall: 0.9815, f1: 0.9814, train_ms: 70, eval_ms: 20, curve: 'smooth' },
  'svm@breast_cancer': { accuracy: 0.9649, precision: 0.968, recall: 0.9649, f1: 0.9648, train_ms: 120, eval_ms: 40, curve: 'smooth' },
  'kmeans@blobs': { silhouette: 0.72, davies_bouldin: 0.55, inertia: 4350, train_ms: 50, eval_ms: 10, curve: 'flat' }
}

/** 各算法的模拟 loss 曲线形态 */
export const LOSS_CURVES = {
  fast: { baseLoss: 0.8, decay: 0.12 }, // 快速下降
  smooth: { baseLoss: 1.2, decay: 0.06 }, // 平滑下降
  step: { baseLoss: 1.0, decay: 0.09, stepwise: true }, // 阶梯式下降（决策树分裂）
  flat: { baseLoss: 900, decay: 0.05, floor: 420 } // 先快后平（kmeans 惯性）
}
