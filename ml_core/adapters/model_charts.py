"""Algorithm diagnostics from the fitted estimator; never refit for a diagram.

The public helper also serves uploaded tables. Features must be in exactly the
space supplied to fit/predict; estimator-owned transformations are reused.
"""
from __future__ import annotations

import math

from .charts import DISPLAY_CAP, _axis, _chart, loss_charts


def dataset_feature_names(dataset):
    """Prefer selected upload columns; known original schemas match source order."""
    explicit = getattr(dataset, "feature_names", None)
    if explicit:
        return list(explicit)
    # Names and ordering from original cart/kmeans/linear figures schemas.
    known = {
        "wdbc": [f"{stat} {measure}" for stat in ("Mean", "SE", "Worst")
                 for measure in ("radius", "texture", "perimeter", "area", "smoothness",
                                 "compactness", "concavity", "concave points", "symmetry", "fractal dimension")],
        "seeds": ["Area", "Perimeter", "Compactness", "Kernel length", "Kernel width",
                  "Asymmetry coefficient", "Kernel groove length"],
        "concrete": ["Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
                     "Coarse aggregate", "Fine aggregate", "Age"],
    }
    names = known.get(dataset.info.id)
    return names if names and len(names) == dataset.info.feature_count else None


def _bar(key, title, note, names, values, ylabel="值"):
    return _chart(key, title, note, [{"name": ylabel, "type": "bar", "data": list(values)}],
                  {"type": "category", "data": list(names), "axisLabel": {"rotate": 30}}, _axis(ylabel))


def _line(key, title, note, values, ylabel, *, threshold=None):
    series = {"name": ylabel, "type": "line", "showSymbol": False,
              "data": [[i + 1, float(v)] for i, v in enumerate(values)]}
    if threshold is not None:
        series["markLine"] = {"silent": True, "data": [{"yAxis": float(threshold), "name": "判定阈值"}]}
    return _chart(key, title, note, [series], _axis("序号"), _axis(ylabel))


def _label(value, names):
    index = int(value)
    return str(names[index]) if names and 0 <= index < len(names) else str(value)


def model_charts(model_id, model, **kwargs):
    """Tag fitted diagnostics so clients can display algorithm-specific plots first."""
    return [{**chart, "category": "model_diagnostic"} for chart in _model_charts(model_id, model, **kwargs)]


def _model_charts(model_id, model, *, train_features, evaluation_features,
                 train_targets=None, evaluation_targets=None, predictions=None,
                 scores=None, feature_names=None, class_names=None):
    """Return finite JSON ECharts specs for a single already fitted model.

    ``scores`` for anomaly estimators means larger-is-more-anomalous (negative
    decision_function for OCSVM). Labels are optional. class_names is indexed
    by encoded class 0/1, independent of estimator first-seen class order.
    Display caps affect only diagnostics and are disclosed in each description.
    """
    name = model_id.removesuffix(".optimized")
    names = list(feature_names or [f"特征 {i + 1}" for i in range(len(train_features[0]))])
    rows = list(evaluation_features)
    if name in {"logistic_regression", "linear_regression"}:
        logistic = name == "logistic_regression"
        prefix = "logistic" if logistic else "linear"
        scale = "标准化输入" if model.standardize else "原始输入"
        charts = [_bar(prefix + "_coefficients", "逻辑回归系数" if logistic else "线性回归系数",
            f"实际拟合权重；{scale}空间，截距 {model.bias:.6g}。" +
            ("正权重增加类别 1 的对数优势。" if logistic else
             "目标使用标准化空间。" if model.standardize_target else "目标使用原始量纲。"),
            names, [float(v) for v in model.weights], "拟合权重")]
        if logistic:
            probabilities = [p[1] for p in model.predict_proba(rows)]
            charts.append(_line("logistic_probabilities", "正类概率与判定阈值",
                f"评估样本的 P({_label(1, class_names)}) 升序排列；显示前 {min(len(rows), DISPLAY_CAP)}/{len(rows)} 个排序位置。",
                sorted(probabilities)[:DISPLAY_CAP], "正类概率", threshold=model.threshold))
        return charts
    if name == "knn":
        neighbors = model._nearest_neighbors(model._transform(rows[:1])[0])
        shown = neighbors[:100]
        return [_bar("knn_neighborhood", "首个评估样本的近邻距离",
            f"复用模型标准化与 Minkowski p={model.p:g} 距离；横轴是训练行索引和类别。显示 {len(shown)}/{len(neighbors)} 个真实近邻；投票方式 {model.weights}。",
            [f"行 {i} · {_label(y, class_names)}" for _, i, y in shown],
            [float(d) for d, _, _ in shown], "距离"),
            _bar("knn_local_votes", "首个评估样本的类别投票",
                "实际近邻投票权重；距离加权时零距离近邻按模型原规则处理。",
                [_label(c, class_names) for c in model.classes],
                list(model._class_scores(model._transform(rows[:1])[0]).values()), "投票权重")]
    if name == "gaussian_naive_bayes":
        low = min(mu[0] - 4 * math.sqrt(var[0]) for mu, var in zip(model.means, model.variances))
        high = max(mu[0] + 4 * math.sqrt(var[0]) for mu, var in zip(model.means, model.variances))
        grid = [low + (high-low)*i/100 for i in range(101)]
        series = []
        for label, means, variances in zip(model.classes, model.means, model.variances):
            variance = variances[0]
            series.append({"name": _label(label, class_names), "type": "line", "showSymbol": False,
                "data": [[x, math.exp(-0.5*((x-means[0])/math.sqrt(variance))**2)/math.sqrt(2*math.pi*variance)] for x in grid]})
        return [_chart("gnb_gaussian_profiles", "朴素贝叶斯：类别高斯密度",
            f"展示第一个特征 {names[0]} 的真实类别均值及平滑方差对应的高斯密度；模型训练使用全部 {len(names)} 个特征。曲线是拟合密度，不是实测直方图。",
            series, _axis(names[0]), _axis("条件概率密度"))]
    if name == "cart_decision_tree":
        def node_data(node, depth=0, branch=""):
            label = f"预测 {_label(node.prediction, class_names)}" if node.is_leaf else f"{names[node.feature_index]} ≤ {node.threshold:.4g}"
            item = {"name": branch + label + f"\nn={node.sample_count}, Gini={node.impurity:.3g}", "value": node.sample_count}
            if not node.is_leaf and depth < 4:
                item["children"] = [node_data(node.left, depth + 1, "是："), node_data(node.right, depth + 1, "否：")]
            elif not node.is_leaf:
                item["name"] += "\n（后续分支省略）"
            return item
        return [{"id": "cart_tree", "title": "CART 实际决策树", "description":
            f"真实分裂阈值、样本数、Gini 与叶预测；显示根至深度 4，完整模型深度 {model.tree_depth_}，叶数 {model.n_leaves_}。可拖动缩放。",
            "option": {"animation": False, "tooltip": {"trigger": "item"}, "series": [{"type": "tree",
                "data": [node_data(model.root)], "top": "8%", "bottom": "8%", "left": "12%", "right": "25%",
                "roam": True, "initialTreeDepth": -1, "symbolSize": 8,
                "label": {"fontSize": 10, "position": "left"}, "leaves": {"label": {"position": "right"}}}]}}]
    if name == "random_forest":
        trees = model.estimators_[:64]
        displayed_rows = rows[:100]
        ensemble = model.predict(displayed_rows)
        disagreement = [sum(p != q for p, q in zip(tree.predict(displayed_rows), ensemble))/len(ensemble) for tree in trees]
        return [_bar("rf_tree_diversity", "随机森林：单树与集成分歧",
            f"各实际拟合树与集成预测的不一致率；前 {len(trees)}/{len(model.estimators_)} 棵树、前 {len(displayed_rows)}/{len(rows)} 个评估样本。分歧不等同于错误率。",
            [f"树 {i+1}" for i in range(len(trees))], disagreement, "分歧率"),
            _bar("rf_feature_importance", "随机森林特征重要性", "模型实际累计的归一化不纯度下降；不是因果效应。", names, model.feature_importances_, "重要性")]
    if name == "gbdt_regression":
        charts = loss_charts(model)
        for chart in charts:
            chart["id"] = "gbdt_stage_loss"
            chart["title"] = "GBDT 逐阶段损失"
        charts.append(_bar("gbdt_feature_importance", "GBDT 分裂增益重要性",
            f"来自实际保留的 {len(model.estimators_)} 棵提升树的归一化分裂增益。", names, model.feature_importances_, "重要性"))
        return charts
    if name == "mlp_regression":
        sizes = [len(model.weights_[0]), *[len(layer[0]) for layer in model.weights_]]
        nodes, links = [], []
        for layer, size in enumerate(sizes):
            for index in range(min(size, 12)):
                nodes.append({"id": f"{layer}:{index}", "name": names[index] if layer == 0 else f"L{layer}:{index+1}",
                    "x": layer * 180, "y": index * 30, "symbolSize": 12})
        for layer, matrix in enumerate(model.weights_):
            for source, weights in enumerate(matrix[:12]):
                for target, weight in enumerate(weights[:12]):
                    links.append({"source": f"{layer}:{source}", "target": f"{layer+1}:{target}", "value": float(weight),
                        "lineStyle": {"color": "#2563eb" if weight >= 0 else "#e05252", "width": min(4, 0.5 + abs(weight))}})
        return [{"id": "mlp_architecture", "title": "MLP 拟合网络与权重", "description":
            f"实际层宽 {' → '.join(map(str, sizes))}；每层最多展示前 12 个节点。蓝/红表示正/负权重，线宽随绝对值增加（上限 4）；悬停边显示权重。使用早停恢复后的实际参数。",
            "option": {"animation": False, "tooltip": {}, "series": [{"type": "graph", "layout": "none", "roam": True,
                "data": nodes, "links": links, "label": {"show": True, "fontSize": 9}, "lineStyle": {"opacity": 0.5}}]}},
            _bar("mlp_layer_weight_norms", "MLP 各层权重范数", "使用各层全部实际拟合权重计算 Frobenius 范数，无节点截断。",
                 [f"层 {i+1}" for i in range(len(model.weights_))],
                 [math.sqrt(sum(w*w for row in layer for w in row)) for layer in model.weights_], "权重范数")]
    if name == "kmeans":
        transformed = model._transform(rows[:DISPLAY_CAP])
        labels = model.predict(rows[:DISPLAY_CAP])
        distances = [math.sqrt(sum((a-b)**2 for a, b in zip(row, model._centers_transformed_[label]))) for row, label in zip(transformed, labels)]
        return [_line("kmeans_center_distances", "K-Means 样本到所属中心距离",
            f"模型实际特征空间中、使用全部特征计算的欧氏距离；原顺序展示前 {len(distances)}/{len(rows)} 个样本。", distances, "所属中心距离"),
            _chart("kmeans_centers", "K-Means 实际中心特征轮廓",
                "每条线对应真实拟合中心；使用模型训练空间（standardize=True 时为 z-score）。所有特征均展示。",
                [{"name": f"中心 {i}", "type": "line", "data": list(c)} for i, c in enumerate(model._centers_transformed_)],
                {"type": "category", "data": names, "axisLabel": {"rotate": 30}}, _axis("中心坐标"))]
    if name == "dbscan":
        import numpy as np
        core = set(model.core_sample_indices_)
        noise = sum(label == -1 for label in model.labels_)
        charts = [_bar("dbscan_density_roles", "DBSCAN 密度角色统计", "根据本次拟合的 core_sample_indices_ 与 labels_ 统计全部训练样本。",
            ["核心点", "边界点", "噪声点"], [len(core), len(model.labels_) - len(core) - noise, noise], "样本数")]
        k = model.min_samples
        if k <= len(train_features):
            matrix = np.asarray(model._transform(train_features), dtype=float)
            distances = [float(np.sqrt(np.partition(np.sum((matrix-row)**2, axis=1), k-1)[k-1])) for row in matrix[:64]]
            charts.append(_line("dbscan_k_distance", "DBSCAN k-distance 与 eps",
                f"k=min_samples={k}，与模型一致包含自身；查询前 {len(distances)}/{len(matrix)} 个训练样本，邻居搜索使用全部训练样本与全部特征。距离升序，仅辅助判断 eps。",
                sorted(distances), "第 k 邻居距离", threshold=model.eps))
        return charts
    if name == "isolation_forest":
        from Models.isolation_forest import average_path_length
        values = list(scores) if scores is not None else model.score_samples(rows)
        normalizer = average_path_length(model.max_samples_)
        ranked = sorted(map(float, values), reverse=True)
        return [_line("if_score_rank", "孤立森林异常分数排名",
            f"实际 score_samples 降序；越大越异常。显示前 {min(len(rows), DISPLAY_CAP)}/{len(rows)} 个排名，阈值来自训练集。",
            ranked[:DISPLAY_CAP], "异常分数", threshold=model.threshold_),
            _chart("if_path_lengths", "孤立森林：平均路径与异常分数",
                f"由实际分数 s=2^(-E[h]/c(n)) 还原平均调整路径长度；显示前 {min(len(rows), DISPLAY_CAP)}/{len(rows)} 个评估样本。c(n)=0 时路径长度为 0。",
                [{"name": "评估样本", "type": "scatter", "data": [[-math.log2(s)*normalizer, s] for s in values[:DISPLAY_CAP]]}],
                _axis("平均调整路径长度"), _axis("异常分数"))]
    if name == "one_class_svm":
        decisions = [-float(s) for s in scores] if scores is not None else model.decision_function(rows)
        return [_line("ocsvm_margins", "单类 SVM 决策间隔",
            f"实际 decision_function 升序；负值表示异常，模型容差阈值为 -1e-10。显示前 {min(len(rows), DISPLAY_CAP)}/{len(rows)} 个排序位置。",
            sorted(decisions)[:DISPLAY_CAP], "决策间隔", threshold=-1e-10),
            _bar("ocsvm_support_weights", "单类 SVM 支持向量权重",
                f"实际对偶系数 alpha；共 {len(model.support_indices_)} 个支持向量 / {len(train_features)} 个训练样本，显示前 100 个支持向量。",
                [f"训练行 {i}" for i in model.support_indices_[:100]], model.support_alphas_[:100], "alpha")]
    return []
