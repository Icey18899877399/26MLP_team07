/**
 * 静态兜底注册表（fallback registry）
 *
 * 当后端不可达时，前端使用这份静态配置渲染界面，保证 UI 可以独立演示。
 * 内容 = mock/registry.js 的合同数据经 src/services/adapters.js 适配后的
 * 精确镜像（与真实后端 12 模型 / 6 数据集一致，中文名）。
 * 修改 mock/registry.js 后应重新生成。
 *
 * 可扩展性说明：
 * - 后端新增一个算法/数据集 → 前端界面自动出现，无需改任何前端代码
 * - 超参数 type 决定表单控件（从 default_params 值类型推断）：
 *   bool → 开关，int/float → 数字输入框，text → 文本框；null 值跳过
 * - 指标 type 决定渲染器（见 config/metrics.js）：scalar → 数值卡片，
 *   matrix → 热力图，curve → 曲线图；未知类型 → 通用 JSON 表格
 */
import { METRICS } from './metrics.js'

export const fallbackRegistry = {
  server: { name: '离线兜底配置' },
  taskTypes: ["classification","regression","clustering","anomaly_detection"],
  algorithms: [
  {
    "id": "logistic_regression.optimized",
    "name": "逻辑回归",
    "taskTypes": [
      "classification"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "learning_rate",
        "label": "learning_rate",
        "type": "float",
        "default": 0.1,
        "group": "其他参数",
        "hint": "梯度下降学习率（正数）"
      },
      {
        "name": "max_iter",
        "label": "max_iter",
        "type": "int",
        "default": 1000,
        "group": "其他参数",
        "hint": "最大迭代次数（正整数）"
      },
      {
        "name": "threshold",
        "label": "threshold",
        "type": "float",
        "default": 0.5,
        "group": "其他参数",
        "hint": "分类阈值（0~1）"
      },
      {
        "name": "l2",
        "label": "l2",
        "type": "int",
        "default": 0,
        "group": "其他参数",
        "hint": "L2 正则化强度（非负）"
      },
      {
        "name": "tol",
        "label": "tol",
        "type": "float",
        "default": 1e-8,
        "group": "其他参数",
        "hint": "早停容差（非负）"
      },
      {
        "name": "standardize",
        "label": "standardize",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "是否对特征做标准化"
      }
    ]
  },
  {
    "id": "knn.optimized",
    "name": "K近邻",
    "taskTypes": [
      "classification"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "n_neighbors",
        "label": "n_neighbors",
        "type": "int",
        "default": 5,
        "group": "其他参数",
        "hint": "参与投票的最近邻居个数"
      },
      {
        "name": "p",
        "label": "p",
        "type": "int",
        "default": 2,
        "group": "其他参数",
        "hint": "Minkowski 距离阶数（p=1 曼哈顿，p=2 欧氏）"
      },
      {
        "name": "weights",
        "label": "weights",
        "type": "text",
        "default": "distance",
        "group": "其他参数",
        "hint": "投票权重（uniform/distance）"
      },
      {
        "name": "standardize",
        "label": "standardize",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "是否对特征做标准化"
      }
    ]
  },
  {
    "id": "gaussian_naive_bayes.optimized",
    "name": "朴素贝叶斯",
    "taskTypes": [
      "classification"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "var_smoothing",
        "label": "var_smoothing",
        "type": "float",
        "default": 1e-9,
        "group": "其他参数",
        "hint": "方差平滑项（正数，防止零概率）"
      }
    ]
  },
  {
    "id": "cart_decision_tree.optimized",
    "name": "决策树",
    "taskTypes": [
      "classification"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "max_depth",
        "label": "max_depth",
        "type": "int",
        "default": 8,
        "group": "其他参数",
        "hint": "最大树深（正整数，null 表示不限制）"
      },
      {
        "name": "min_samples_split",
        "label": "min_samples_split",
        "type": "int",
        "default": 2,
        "group": "其他参数",
        "hint": "分裂所需最小样本数"
      },
      {
        "name": "min_samples_leaf",
        "label": "min_samples_leaf",
        "type": "int",
        "default": 1,
        "group": "其他参数",
        "hint": "叶节点最小样本数"
      },
      {
        "name": "min_impurity_decrease",
        "label": "min_impurity_decrease",
        "type": "int",
        "default": 0,
        "group": "其他参数",
        "hint": "分裂最小不纯度下降"
      },
      {
        "name": "ccp_alpha",
        "label": "ccp_alpha",
        "type": "int",
        "default": 0,
        "group": "其他参数",
        "hint": "代价复杂度剪枝系数（非负）"
      }
    ]
  },
  {
    "id": "random_forest.optimized",
    "name": "随机森林",
    "taskTypes": [
      "classification"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "n_estimators",
        "label": "n_estimators",
        "type": "int",
        "default": 50,
        "group": "其他参数",
        "hint": "树的数量（正整数）"
      },
      {
        "name": "max_depth",
        "label": "max_depth",
        "type": "int",
        "default": 8,
        "group": "其他参数",
        "hint": "单棵树最大深度（正整数）"
      },
      {
        "name": "min_samples_split",
        "label": "min_samples_split",
        "type": "int",
        "default": 2,
        "group": "其他参数",
        "hint": "分裂所需最小样本数"
      },
      {
        "name": "min_samples_leaf",
        "label": "min_samples_leaf",
        "type": "int",
        "default": 1,
        "group": "其他参数",
        "hint": "叶节点最小样本数"
      },
      {
        "name": "voting",
        "label": "voting",
        "type": "text",
        "default": "soft",
        "group": "其他参数",
        "hint": "投票方式（soft 概率平均 / hard 多数投票）"
      }
    ]
  },
  {
    "id": "linear_regression.optimized",
    "name": "线性回归",
    "taskTypes": [
      "regression"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "learning_rate",
        "label": "learning_rate",
        "type": "float",
        "default": 0.01,
        "group": "其他参数",
        "hint": "学习率（正数）"
      },
      {
        "name": "max_iter",
        "label": "max_iter",
        "type": "int",
        "default": 1000,
        "group": "其他参数",
        "hint": "最大迭代次数（正整数）"
      },
      {
        "name": "batch_size",
        "label": "batch_size",
        "type": "int",
        "default": 32,
        "group": "其他参数",
        "hint": "小批量大小（正整数）"
      },
      {
        "name": "l2",
        "label": "l2",
        "type": "int",
        "default": 0,
        "group": "其他参数",
        "hint": "L2 正则化强度（非负）"
      },
      {
        "name": "standardize",
        "label": "standardize",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "是否对特征做标准化"
      },
      {
        "name": "standardize_target",
        "label": "standardize_target",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "是否对目标值做标准化（影响指标量纲）"
      }
    ]
  },
  {
    "id": "gbdt_regression.optimized",
    "name": "GBDT回归",
    "taskTypes": [
      "regression"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "n_estimators",
        "label": "n_estimators",
        "type": "int",
        "default": 100,
        "group": "其他参数",
        "hint": "树的数量（正整数）"
      },
      {
        "name": "learning_rate",
        "label": "learning_rate",
        "type": "float",
        "default": 0.05,
        "group": "其他参数",
        "hint": "学习率（正数）"
      },
      {
        "name": "max_depth",
        "label": "max_depth",
        "type": "int",
        "default": 3,
        "group": "其他参数",
        "hint": "树最大深度（正整数）"
      },
      {
        "name": "min_samples_split",
        "label": "min_samples_split",
        "type": "int",
        "default": 2,
        "group": "其他参数",
        "hint": "分裂所需最小样本数"
      },
      {
        "name": "min_samples_leaf",
        "label": "min_samples_leaf",
        "type": "int",
        "default": 1,
        "group": "其他参数",
        "hint": "叶节点最小样本数"
      },
      {
        "name": "subsample",
        "label": "subsample",
        "type": "int",
        "default": 1,
        "group": "其他参数",
        "hint": "每棵树样本采样比例（0~1）"
      },
      {
        "name": "validation_fraction",
        "label": "validation_fraction",
        "type": "float",
        "default": 0.1,
        "group": "其他参数",
        "hint": "早停验证集比例（0~1）"
      },
      {
        "name": "l2_regularization",
        "label": "l2_regularization",
        "type": "int",
        "default": 0,
        "group": "其他参数",
        "hint": "叶值 L2 正则化（非负）"
      }
    ]
  },
  {
    "id": "mlp_regression.optimized",
    "name": "MLP回归",
    "taskTypes": [
      "regression"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "activation",
        "label": "activation",
        "type": "text",
        "default": "relu",
        "group": "其他参数",
        "hint": "激活函数（relu/tanh）"
      },
      {
        "name": "learning_rate",
        "label": "learning_rate",
        "type": "float",
        "default": 0.001,
        "group": "其他参数",
        "hint": "学习率（正数）"
      },
      {
        "name": "max_iter",
        "label": "max_iter",
        "type": "int",
        "default": 1000,
        "group": "其他参数",
        "hint": "最大迭代次数（正整数）"
      },
      {
        "name": "batch_size",
        "label": "batch_size",
        "type": "int",
        "default": 32,
        "group": "其他参数",
        "hint": "小批量大小（正整数）"
      },
      {
        "name": "l2",
        "label": "l2",
        "type": "int",
        "default": 0,
        "group": "其他参数",
        "hint": "L2 正则化强度（非负）"
      },
      {
        "name": "validation_fraction",
        "label": "validation_fraction",
        "type": "float",
        "default": 0.2,
        "group": "其他参数",
        "hint": "早停验证集比例（0~1）"
      },
      {
        "name": "n_iter_no_change",
        "label": "n_iter_no_change",
        "type": "int",
        "default": 20,
        "group": "其他参数",
        "hint": "早停耐心轮数（正整数）"
      },
      {
        "name": "standardize",
        "label": "standardize",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "是否对特征做标准化"
      },
      {
        "name": "standardize_target",
        "label": "standardize_target",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "是否对目标值做标准化"
      },
      {
        "name": "gradient_clip",
        "label": "gradient_clip",
        "type": "int",
        "default": 5,
        "group": "其他参数",
        "hint": "梯度裁剪阈值（正数）"
      }
    ]
  },
  {
    "id": "kmeans.optimized",
    "name": "K均值聚类",
    "taskTypes": [
      "clustering"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "n_clusters",
        "label": "n_clusters",
        "type": "int",
        "default": 3,
        "group": "其他参数",
        "hint": "簇数 k"
      },
      {
        "name": "init",
        "label": "init",
        "type": "text",
        "default": "k-means++",
        "group": "其他参数",
        "hint": "初始化方式：'random' 或 'k-means++'"
      },
      {
        "name": "n_init",
        "label": "n_init",
        "type": "int",
        "default": 10,
        "group": "其他参数",
        "hint": "独立初始化次数，取惯性最低的一次"
      },
      {
        "name": "max_iter",
        "label": "max_iter",
        "type": "int",
        "default": 300,
        "group": "其他参数",
        "hint": "每次初始化的最大迭代次数"
      },
      {
        "name": "tol",
        "label": "tol",
        "type": "float",
        "default": 0.0001,
        "group": "其他参数",
        "hint": "中心点移动的停止容差（非负）"
      },
      {
        "name": "standardize",
        "label": "standardize",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "聚类前是否对特征做标准化"
      }
    ]
  },
  {
    "id": "dbscan.optimized",
    "name": "DBSCAN聚类",
    "taskTypes": [
      "clustering"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "eps",
        "label": "eps",
        "type": "float",
        "default": 0.5,
        "group": "其他参数",
        "hint": "邻域半径（standardize=true 时作用于标准化空间）"
      },
      {
        "name": "min_samples",
        "label": "min_samples",
        "type": "int",
        "default": 5,
        "group": "其他参数",
        "hint": "核心点所需最小邻居数（正整数）"
      },
      {
        "name": "standardize",
        "label": "standardize",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "是否对特征做标准化"
      }
    ]
  },
  {
    "id": "isolation_forest.optimized",
    "name": "孤立森林",
    "taskTypes": [
      "anomaly_detection"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "n_estimators",
        "label": "n_estimators",
        "type": "int",
        "default": 100,
        "group": "其他参数",
        "hint": "树的数量（正整数）"
      },
      {
        "name": "max_samples",
        "label": "max_samples",
        "type": "int",
        "default": 256,
        "group": "其他参数",
        "hint": "每棵树采样数（正整数或 0~1 比例）"
      },
      {
        "name": "contamination",
        "label": "contamination",
        "type": "float",
        "default": 0.1,
        "group": "其他参数",
        "hint": "期望异常比例，决定判定阈值（心电图约 0.096，乳腺造影约 0.023）"
      },
      {
        "name": "max_features",
        "label": "max_features",
        "type": "int",
        "default": 1,
        "group": "其他参数",
        "hint": "每次切分使用的特征比例（正整数或 0~1 比例）"
      }
    ]
  },
  {
    "id": "one_class_svm.optimized",
    "name": "单类SVM",
    "taskTypes": [
      "anomaly_detection"
    ],
    "description": "",
    "hyperparams": [
      {
        "name": "nu",
        "label": "nu",
        "type": "float",
        "default": 0.1,
        "group": "其他参数",
        "hint": "训练误差上限（0~1，越小边界越紧）"
      },
      {
        "name": "gamma",
        "label": "gamma",
        "type": "text",
        "default": "scale",
        "group": "其他参数",
        "hint": "RBF 核系数（'scale' 或正数）"
      },
      {
        "name": "max_iter",
        "label": "max_iter",
        "type": "int",
        "default": 500,
        "group": "其他参数",
        "hint": "最大迭代次数（正整数）"
      },
      {
        "name": "standardize",
        "label": "standardize",
        "type": "bool",
        "default": true,
        "group": "其他参数",
        "hint": "是否对特征做标准化"
      }
    ]
  }
],
  datasets: [
  {
    "id": "wdbc",
    "name": "乳腺癌诊断",
    "taskType": "classification",
    "nSamples": 569,
    "nFeatures": 30,
    "nClasses": 0,
    "description": ""
  },
  {
    "id": "seeds",
    "name": "小麦种子",
    "taskType": "clustering",
    "nSamples": 210,
    "nFeatures": 7,
    "nClasses": 0,
    "description": ""
  },
  {
    "id": "concrete",
    "name": "混凝土强度",
    "taskType": "regression",
    "nSamples": 1030,
    "nFeatures": 8,
    "nClasses": 0,
    "description": ""
  },
  {
    "id": "california_housing",
    "name": "加州房价",
    "taskType": "regression",
    "nSamples": 20640,
    "nFeatures": 8,
    "nClasses": 0,
    "description": ""
  },
  {
    "id": "6_cardio",
    "name": "心电图异常检测",
    "taskType": "anomaly_detection",
    "nSamples": 1831,
    "nFeatures": 21,
    "nClasses": 0,
    "description": ""
  },
  {
    "id": "23_mammography",
    "name": "乳腺造影异常检测",
    "taskType": "anomaly_detection",
    "nSamples": 11183,
    "nFeatures": 6,
    "nClasses": 0,
    "description": ""
  }
],
  metrics: METRICS
}
