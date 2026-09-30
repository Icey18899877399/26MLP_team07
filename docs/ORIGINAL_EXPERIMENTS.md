# 原始实验来源索引

唯一协议来源是本仓库 `visualization/*_figures.py`；模型、数据、已有图分别来自 `Models/`、`data/`、`figures/`。2026-09-30 对照用户 D 盘原项目：72 PNG 和 18 CSV 哈希相同，代码与数据排除 CRLF 换行差异后内容相同。

界面不是将 12 个优化版模型统一 fit 一次，而是调用下列原流程；基础版、优化版、消融、基线及不同图形都由对应原脚本决定。

| 原实验 | 原数据 | 原入口默认配置摘要 | 六图内容举例 |
| --- | --- | --- | --- |
| Logistic Regression | WDBC | seed 42；max_iter 600 | 类别分布、损失、混淆、ROC、PR、累计消融 |
| KNN | WDBC | seed 42；5 折 | 邻域、K 决策边界、验证曲线、ROC/PR、混淆、优化/耗时 |
| Gaussian Naive Bayes | WDBC | seed 42；5 折 | 高斯拟合、后验面、假设诊断、基准、概率诊断、优化 |
| CART | WDBC | seed 42；5 折 | 分裂增益、树结构、边界、剪枝验证、测试诊断、优化证据 |
| Random Forest | WDBC | seed 42；5 折 | Bootstrap/OOB、树数收敛、参数响应、多样性、特征稳定性、基准 |
| Linear Regression | Concrete | seed 42；5 折 | 分布/相关、收敛、正则路径、实际/预测、残差、基准 |
| GBDT Regression | Concrete | seed 42；5 折；max_estimators 60 | 残差拟合、早停、参数响应、特征解释、预测/残差、基准 |
| MLP Regression | Concrete | seed 42；5 折；调参子集 240；调参迭代 80；最终迭代 300；5 次种子；3 次计时 | 网络/非线性、训练/早停、参数、特征解释、预测/残差、基准 |
| K-Means | Seeds | seed 42；3 簇；30 次稳定性；10 次基准 | 簇分布、质心轨迹、收敛、K 选择、初始化稳定性、基准 |
| DBSCAN | Seeds + 原脚本双月机制数据 | seed 42；min_samples 5 | 聚类结果、密度机制、K 距离、敏感性、标准化、基准 |
| Isolation Forest | Cardio + Mammography + 原脚本机制数据 | seed 13；quick=False | 隔离机制、得分面、分布、ROC/PR、敏感性、跨数据基准 |
| One-Class SVM | Cardio + Mammography + 原脚本机制数据 | quick=False；原内部 seeds 17/29/43；正常训练上限 72；评价分层上限 720 | 边界、对偶/KKT、收敛、得分/ROC/PR、鲁棒性、跨数据基准 |

此表只摘要入口参数，不替代脚本内完整协议。例如 MLP 的 240 是原调参子集，不是擅自截断最终训练数据；One-Class SVM 的 72/720 是原流程成本与评价策略的一部分，不是队友适配器的正常样本 300 上限。Isolation Forest 完整模式不使用 quick 中的 220/320 样本裁剪。

## 数据文件

- WDBC：`data/classification/wdbc/wdbc.data`
- Concrete：`data/regression/concrete/concrete_data.csv`
- Seeds：`data/clustering/seeds/seeds_dataset.txt`
- Cardio：`data/anomaly/6_cardio.npz`
- Mammography：`data/anomaly/23_mammography.npz`
- 原项目存档 Adult：`data/classification/adult/`，当前六图脚本未使用。
- 原项目存档 California Housing：`data/regression/california_housing/`，当前六图脚本未使用。

原压缩包、说明书与 Excel 文件一并保留。数据名称与队友测试目录相同并不意味着采用相同的处理流程；本次以原文件路径和原脚本为准，不走 `ml_core/resources/datasets` 的旧单次训练适配器。

## 原图与新图

已有图位于 `figures/<algorithm>/`，每算法六张。只有 logistic_regression、knn、kmeans 的原目录带有 `source_data/*.csv`，共 18 个；其余实验没有的 CSV 不凭空补造。

重新训练写入 `.original-runs/` 的独立目录，并标记 run_id 和状态。不得将旧图当作失败运行的替代产物，不覆盖已有成果，也不将随机演示图补入页面。
