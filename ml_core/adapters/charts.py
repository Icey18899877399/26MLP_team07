"""JSON-only ECharts specifications derived from actual fitted model outputs.

Scatter displays are bounded; metrics and histograms always use all evaluated rows.
No Javascript formatters or invented training histories are emitted.
"""

from __future__ import annotations

import math
from collections import Counter

DISPLAY_CAP = 1000


def _chart(chart_id, title, description, series, x_axis, y_axis, **extra):
    return {"id": chart_id, "title": title, "description": description,
            "option": {"animation": False, "tooltip": {"trigger": "item"},
                       "grid": {"left": 65, "right": 35, "top": 55, "bottom": 65},
                       "legend": {"type": "scroll", "top": 5},
                       "xAxis": x_axis, "yAxis": y_axis, "series": series, **extra}}


def _axis(name):
    return {"type": "value", "name": name, "nameLocation": "middle", "nameGap": 38}


def roc_points(targets, scores):
    """Empirical ROC, grouping tied scores rather than ordering tied labels."""
    positives = sum(int(t) == 1 for t in targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        return []
    groups = {}
    for target, score in zip(targets, scores):
        counts = groups.setdefault(float(score), [0, 0])
        counts[int(target) == 1] += 1
    points = [[0.0, 0.0]]
    fp = tp = 0
    for score in sorted(groups, reverse=True):
        fp += groups[score][0]
        tp += groups[score][1]
        points.append([fp / negatives, tp / positives])
    return points


def classification_charts(targets, predictions, probabilities, *, class_names=None):
    names = list(class_names or ("0", "1"))
    categories = [f"{name} ({i})" for i, name in enumerate(names)]
    counts = [[x, y, sum(int(t) == y and int(p) == x
                         for t, p in zip(targets, predictions))]
              for y in (0, 1) for x in (0, 1)]
    charts = [_chart("confusion_matrix", "测试集混淆矩阵",
        f"分层留出测试集；横轴为预测类别，纵轴为真实类别。0 = {names[0]}，1 = {names[1]}。",
        [{"name": "样本数", "type": "heatmap", "data": counts,
          "label": {"show": True}}],
        {"type": "category", "name": "预测类别", "nameLocation": "middle", "nameGap": 30,
         "data": categories},
        {"type": "category", "name": "真实类别", "nameLocation": "middle", "nameGap": 48,
         "nameRotate": 90, "data": categories},
        visualMap={"min": 0, "max": max(row[2] for row in counts),
                   "calculable": False, "orient": "horizontal", "bottom": 0})]
    points = roc_points(targets, probabilities)
    if points:
        charts.append(_chart("roc_curve", "测试集 ROC 曲线",
            "根据模型对留出测试样本的真实正类概率逐阈值计算；相同概率合并处理。",
            [{"name": "模型 ROC", "type": "line", "showSymbol": False, "data": points},
             {"name": "随机参考", "type": "line", "showSymbol": False,
              "lineStyle": {"type": "dashed"}, "data": [[0, 0], [1, 1]]}],
            {**_axis("假阳性率"), "min": 0, "max": 1},
            {**_axis("真阳性率"), "min": 0, "max": 1}))
    return charts


def regression_charts(targets, predictions):
    pairs = [[float(a), float(b)] for a, b in zip(targets, predictions)][:DISPLAY_CAP]
    low = min(min(pair) for pair in pairs)
    high = max(max(pair) for pair in pairs)
    note = f"独立留出测试集；显示 {len(pairs)}/{len(targets)} 个样本，指标使用全部测试样本。目标值为原始量纲。"
    return [
        _chart("prediction_scatter", "测试集真实值与预测值", note,
            [{"name": "测试样本", "type": "scatter", "symbolSize": 6, "data": pairs},
             {"name": "理想预测", "type": "line", "showSymbol": False,
              "lineStyle": {"type": "dashed"}, "data": [[low, low], [high, high]]}],
            _axis("真实值"), _axis("预测值")),
        _chart("residual_scatter", "测试集残差", note + "残差 = 真实值 − 预测值。",
            [{"name": "残差", "type": "scatter", "symbolSize": 6,
              "data": [[p, a-p] for a, p in pairs],
              "markLine": {"silent": True, "data": [{"yAxis": 0}]}}],
            _axis("预测值"), _axis("残差")),
    ]


def loss_charts(model, *, target_standardized=False):
    train = getattr(model, "loss_history", None)
    if train is None:
        train = getattr(model, "train_loss_", [])
    if not train:
        return []
    series = [{"name": "训练损失", "type": "line", "showSymbol": False,
               "data": [[i+1, float(v)] for i, v in enumerate(train)]}]
    validation = getattr(model, "validation_loss_", [])
    if validation and getattr(model, "validation_fraction", 0) > 0:
        series.append({"name": "内部验证损失", "type": "line", "showSymbol": False,
                       "data": [[i+1, float(v)] for i, v in enumerate(validation)]})
    scale = "目标标准化空间" if target_standardized else "模型训练目标空间"
    return [_chart("training_loss", "实际训练损失轨迹",
        f"模型实际记录的损失（{scale}，可能包含正则项），不是测试集误差。内部验证集仅来自外层训练集；早停模型可能恢复最佳权重。",
        series, _axis("迭代 / 轮次"), _axis("训练目标损失"))]


def clustering_charts(features, labels, *, feature_names=None):
    count = len(features)
    names = list(feature_names or [f"特征 {i+1}" for i in range(len(features[0]))])
    projected = [[row[0], row[1] if len(row) > 1 else 0.0] for row in features]
    means = [sum(row[j] for row in projected)/count for j in (0, 1)]
    scales = [math.sqrt(sum((row[j]-means[j])**2 for row in projected)/count) or 1
              for j in (0, 1)]
    series = []
    for label in sorted(set(labels)):
        series.append({"name": "噪声 (-1)" if label == -1 else f"簇 {label}",
            "type": "scatter", "symbolSize": 7,
            "data": [[(row[0]-means[0])/scales[0], (row[1]-means[1])/scales[1]]
                     for row, cluster in list(zip(projected, labels))[:DISPLAY_CAP] if cluster == label]})
    sizes = Counter(labels)
    return [_chart("cluster_projection", "聚类结果：前两维标准化投影",
        f"使用全体样本拟合聚类；图中仅显示前两维特征的 z-score 投影（非 PCA、非重新训练），单特征时纵轴为 0。显示 {min(count, DISPLAY_CAP)}/{count} 个样本。聚类使用全部特征；有参考标签时才计算 ARI。",
        series, _axis(names[0] + " (z-score)"), _axis(names[1] + " (z-score)" if len(names) > 1 else "单特征占位 (0)")),
        _chart("cluster_sizes", "簇大小与噪声数量",
            "统计全部样本的预测簇标签；DBSCAN 的 -1 是噪声，不属于任何簇。高噪声比例可能意味着当前 eps 较小。",
            [{"name": "样本数", "type": "bar", "data": [sizes[label] for label in sorted(sizes)]}],
            {"type": "category", "data": ["噪声 (-1)" if label == -1 else f"簇 {label}"
                for label in sorted(sizes)]}, _axis("样本数"))]


def anomaly_charts(targets, scores, predictions, *, protocol, evaluation_label="样本内评估"):
    scores = [float(score) for score in scores]
    low, high = min(scores), max(scores)
    width = (high-low)/20 if high > low else 1.0
    groups = [[0]*20, [0]*20]
    for target, score in zip(targets, scores):
        index = min(19, int((score-low)/width))
        groups[int(target) == 1][index] += 1
    bins = [f"{low+i*width:.3g}–{low+(i+1)*width:.3g}" for i in range(20)]
    cells = [[x, y, sum(int(t) == y and int(p == -1) == x
                       for t, p in zip(targets, predictions))]
             for y in (0, 1) for x in (0, 1)]
    return [_chart("anomaly_score_distribution", f"异常分数分布（{evaluation_label}）",
        protocol + "分数越大越异常；按真实标签分组，统计全部样本。横轴为等宽分箱。",
        [{"name": name, "type": "bar", "data": group}
         for name, group in zip(("真实正常", "真实异常"), groups)],
        {"type": "category", "name": "异常分数", "data": bins,
         "axisLabel": {"rotate": 35, "interval": 3}}, _axis("样本数")),
        _chart("anomaly_confusion_matrix", f"异常检测混淆矩阵（{evaluation_label}）",
            protocol + "使用模型实际判定阈值；横轴预测类别，纵轴真实类别。",
            [{"name": "样本数", "type": "heatmap", "data": cells, "label": {"show": True}}],
            {"type": "category", "name": "预测类别", "nameLocation": "middle", "nameGap": 30,
             "data": ["正常", "异常"]},
            {"type": "category", "name": "真实类别", "nameLocation": "middle", "nameGap": 42,
             "nameRotate": 90, "data": ["正常", "异常"]},
            visualMap={"min": 0, "max": max(row[2] for row in cells), "calculable": False,
                       "orient": "horizontal", "bottom": 0})]
