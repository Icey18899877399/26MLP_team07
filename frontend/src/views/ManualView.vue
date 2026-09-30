<script setup>
const steps = [{title:'选择算法和数据',text:'在算法库选择分类、回归、聚类或异常检测模型。数据集下拉框仅展示当前模型支持的数据集。'}, {title:'设置与运行',text:'设置测试集比例、随机种子和超参数。提示图标说明参数含义。null、数组参数使用 JSON 输入，例如 null 或 [32, 16]。点击开始训练后显示请求状态与真实耗时。'}, {title:'解读结果',text:'先检查评估说明，再查看指标与图表。训练损失来自算法实际迭代；没有迭代记录的算法不显示损失曲线。所有图表可放大和导出。'}, {title:'复现与比较',text:'在实验历史点击载入参数，再次训练即可复现实验条件。算法对比页限定数据集、随机种子、切分和评估协议一致的结果。导出 JSON 可保留参数、指标和图表数据。'}]
</script>

<template>
  <section class="workspace-hero"><div><div class="eyebrow">QUICK START</div><h1>从第一次训练开始</h1><p>理解参数、评价协议与模型行为，完成一组可复现的机器学习实验。</p></div></section>
  <div class="manual-grid"><article v-for="(step,index) in steps" :key="step.title" class="page-card"><span class="section-number">0{{ index+1 }}</span><h2>{{ step.title }}</h2><p>{{ step.text }}</p></article></div>
  <section class="page-card"><h2>读懂评价结果</h2><el-descriptions :column="1" border><el-descriptions-item label="分类">准确率反映整体正确比例；精确率关注预测正例的可信度；召回率关注正例的检出能力；ROC 展示不同阈值的取舍。</el-descriptions-item><el-descriptions-item label="回归">RMSE、MAE 越小通常误差越低；R² 越接近 1 越好，也可能为负。关注预测散点是否贴近对角线、残差是否存在结构。</el-descriptions-item><el-descriptions-item label="聚类">二维标准化特征投影只显示部分高维结构。结合轮廓系数、簇内平方和及噪声点分析，不跨数据集直接比较数值。</el-descriptions-item><el-descriptions-item label="异常检测">异常比例很低时，准确率可能具有误导性。优先查看异常召回、精确率和 F1；当前评估包含拟合样本，不代表独立测试泛化表现。</el-descriptions-item></el-descriptions></section>
  <section class="page-card"><h2>运行与记录</h2><p>绿色连接状态表示后端可用。离线状态可浏览配置，但无法发起训练。训练使用同步 HTTP 请求，运行中不展示未经测量的进度百分比。请求超时后，服务端可能仍在执行。</p><p>历史记录保存在当前浏览器，最多保留 50 条会话记录；持久化最多保存最近 10 条，空间不足时减少缓存数量并在日志中提示。重要实验请主动导出 JSON。清空历史不会移除正在运行的请求。</p></section>
</template>

<style scoped>
.manual-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 20px}h2{font-size:18px;margin:15px 0}p{color:#768b7d;font-size:13px;line-height:2}@media(max-width:700px){.manual-grid{grid-template-columns:1fr}}
</style>
