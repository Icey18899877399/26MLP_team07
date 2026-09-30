export const originalTaskNames = {
  classification: '分类实验',
  regression: '回归实验',
  clustering: '聚类实验',
  anomaly_detection: '异常检测实验'
}

export const originalModelNotes = {
  logistic_regression: '以概率输出完成二分类，并观察优化过程、决策边界与评价曲线。',
  gaussian_naive_bayes: '用条件概率与独立性假设完成分类，观察分布估计和分类表现。',
  knn: '通过邻域投票预测类别，考察距离、邻居数与局部边界。',
  cart_decision_tree: '递归划分特征空间，展示分裂收益、树结构及剪枝影响。',
  random_forest: '汇集多棵自助采样树，观察袋外估计、树间差异和集成收益。',
  linear_regression: '用线性关系拟合连续目标，分析系数、误差与残差。',
  gbdt_regression: '逐轮拟合残差，观察提升过程、超参数和泛化表现。',
  mlp_regression: '通过多层神经网络拟合非线性关系，观察训练与预测行为。',
  kmeans: '交替分配样本与更新质心，观察轨迹、收敛和簇结构。',
  dbscan: '按局部密度识别簇与噪声，观察距离阈值和最少点数的影响。',
  isolation_forest: '通过随机切分路径检测离群点，分析异常分数与跨数据集表现。',
  one_class_svm: '学习正常样本边界，观察核函数、优化结构与异常检测表现。'
}

export const originalTaskOrder = ['classification', 'regression', 'clustering', 'anomaly_detection']

const figureTitles = {
  cart_decision_tree: ['分裂收益机制', '树结构对比', '决策边界对比', '剪枝验证曲线', '测试集诊断', '优化证据'],
  gaussian_naive_bayes: ['高斯特征拟合', '后验概率决策面', '独立性假设诊断', '基准性能比较', '概率预测诊断', '优化分析'],
  knn: ['局部邻域示意', '不同 K 值的决策边界', 'K 值验证曲线', 'ROC 与 PR 对比', '混淆矩阵', '优化与运行耗时'],
  logistic_regression: ['类别分布', '损失收敛曲线', '混淆矩阵', 'ROC 曲线', '精确率与召回率曲线', '优化方法消融'],
  random_forest: ['自助采样与袋外估计机制', '树数量收敛', '超参数响应', '树间差异与集成收益', '特征重要性稳定性', '最终基准对比'],
  gbdt_regression: ['逐轮残差拟合', '收敛与提前停止', '超参数表现', '特征重要性与依赖', '预测与残差诊断', '最终基准对比'],
  linear_regression: ['数据分布与相关性', '优化收敛', '正则化路径', '真实值与预测值', '残差诊断', '最终基准对比'],
  mlp_regression: ['网络结构与非线性拟合', '训练动态与提前停止', '超参数表现', '特征解释', '预测与残差诊断', '最终基准对比'],
  dbscan: ['聚类结果', '密度聚类机制', 'K 距离曲线', '参数敏感性', '标准化效果对比', '基准算法对比'],
  kmeans: ['簇分布', '质心移动轨迹', '收敛过程对比', '簇数量选择', '初始化稳定性', '基准算法对比'],
  isolation_forest: ['随机隔离机制', '异常分数分布图', '异常分数统计分布', 'ROC 与 PR 曲线', '参数敏感性', '基准算法对比'],
  one_class_svm: ['正常样本边界机制', '对偶与 KKT 结构', '优化收敛', '异常分数与 ROC / PR', '参数稳健性', '跨数据集基准对比']
}

export function originalFigureTitle(experimentId, figure) {
  const number = Number(/^([0-9]{2})_/.exec(figure.name || '')?.[1])
  return figureTitles[experimentId]?.[number - 1] || figure.title || figure.name
}
