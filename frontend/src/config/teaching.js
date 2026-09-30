export const taskNames = {classification: '分类', regression: '回归', clustering: '聚类', anomaly_detection: '异常检测'}
const notes = {
  logistic_regression: ['逻辑回归', '将线性得分映射为类别概率，观察分类边界与误判。', '比较精确率与召回率，并结合混淆矩阵分析两类错误。'],
  linear_regression: ['线性回归', '学习特征与连续目标之间的线性关系。', '预测点越接近对角线越准确；残差中的规律提示模型尚未解释的结构。'],
  decision_tree: ['决策树', '递归选择特征与阈值，将样本划分到不同叶节点。', '增加深度提高表达能力，也可能过拟合；关注测试集表现。'],
  random_forest: ['随机森林', '结合多棵随机化决策树，降低单棵树预测的波动。', '比较树的数量与性能，结合特征重要性解释预测。'],
  adaboost: ['AdaBoost', '逐轮提高难分类样本的权重，组合多个弱学习器。', '弱学习器数量影响拟合能力；观察错误是否集中于少数样本。'],
  gradient_boosting: ['梯度提升', '逐步拟合当前模型的误差，使预测持续改进。', '学习率与迭代次数共同影响拟合；真实损失曲线反映优化过程。'],
  knn: ['K 近邻', '根据附近样本的标签进行投票或估计。', '小 K 更敏感，大 K 更平滑；距离受到特征尺度影响。'],
  naive_bayes: ['朴素贝叶斯', '在条件独立假设下组合各特征提供的类别证据。', '观察类别混淆，思考特征相关性对独立假设的影响。'],
  svm: ['支持向量机', '寻找能够区分类别并保持较大间隔的边界。', '正则化与核参数会改变边界复杂度，比较测试集错误分布。'],
  mlp: ['多层感知机', '通过多层非线性变换与反向传播学习预测函数。', '关注训练损失的实际变化，以及网络规模与泛化能力的关系。'],
  kmeans: ['K-Means', '交替分配样本与更新中心，使同簇样本更接近。', '簇数会影响簇内平方和；二维投影仅展示高维结构的一部分。'],
  dbscan: ['DBSCAN', '通过局部密度寻找簇，并识别稀疏区域的噪声点。', '邻域半径和最少样本数共同控制簇结构；噪声标签通常为 -1。'],
  isolation_forest: ['孤立森林', '用随机划分衡量样本被孤立的难易程度。', '异常分数与阈值决定检出结果；结合留出集标签检查误报和漏报。']
}
export function algorithmNotes(id) {
  const family = id.split('.')[0]
  const value = notes[family] || (family.includes('neural') ? notes.mlp : null)
  return { chineseName: value?.[0] || family, description: value?.[1] || '运行手写算法，分析真实样本上的预测表现。', insight: value?.[2] || '结合参数、评价指标与可视化结果解释模型行为。' }
}
const metricNames = {accuracy: '准确率', precision: '精确率', recall: '召回率', f1: 'F1 分数', roc_auc: 'ROC AUC', mse: '均方误差', rmse: '均方根误差', mae: '平均绝对误差', r2: 'R² 决定系数', inertia: '簇内平方和', adjusted_rand_index: '调整兰德指数', silhouette_score: '轮廓系数', n_clusters: '簇数', n_noise: '噪声点数', n_iter: '迭代次数', tn: '真负例', fp: '假正例', fn: '假负例', tp: '真正例'}
export function metricDefinition(id) {
  return {id, name: metricNames[id] || id, type: 'scalar', higherIsBetter: !['mse', 'rmse', 'mae', 'inertia', 'fp', 'fn'].includes(id)}
}
