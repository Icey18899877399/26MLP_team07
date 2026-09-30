<script setup>
const steps = [
  { number: '01', title: '选择原实验', detail: '仅使用左侧导航和实验目录，在分类、回归、聚类、异常检测中选择算法。全部使用项目自己的原始数据。' },
  { number: '02', title: '浏览历史原图', detail: '每套实验的六张现有 PNG 标注为历史归档。可放大查看完整图像、下载原尺寸 PNG；有绘图 CSV 的实验也提供原文件。' },
  { number: '03', title: '调整参数并训练', detail: '每个算法都有独立的调参区。修改参数后点击“开始训练”，等待真实计算结束，查看更新的指标和交互图表；不会覆盖六张原图。' },
  { number: '04', title: '需要时完整复现', detail: '展开页面底部“完整原实验协议与复现”，可启动原脚本的完整默认流程。运行日志和新产物会保存在服务端的独立目录中。' }
]
</script>

<template>
  <section class="page-intro"><div><div class="eyebrow">GUIDE / REPRODUCIBILITY</div><h1>使用说明</h1><p>区分既有成果与新运行产物，按原始协议完成复现。</p></div></section>
  <div class="manual-steps"><article v-for="step in steps" :key="step.number" class="page-card manual-step"><span class="manual-number">{{ step.number }}</span><h2>{{ step.title }}</h2><p>{{ step.detail }}</p></article></div>
  <section class="page-card manual-note"><div class="card-title-row"><div><div class="eyebrow">DATA & PROTOCOL</div><h2>关于数据与参数</h2></div></div><p>可视化分为两部分：第一部分是每算法六张历史原图，第二部分是调参后新计算的指标与动态图表。动态训练使用自己的原始数据及模型实现，页面显示参数来源和实际评估协议。单次调参不是完整交叉验证，也不声称复现原图中的最优结果。</p><p>原图中可能包含原脚本生成的合成机制演示及跨数据集比较。完整复现仍遵守原绘图脚本的只读默认设置；Adult、California 等仅存档文件不混入此处训练。</p></section>
  <section class="page-card manual-note"><div class="card-title-row"><div><div class="eyebrow">RUN HISTORY</div><h2>关于长时间运行</h2></div></div><p>完整复现可能执行交叉验证、重复实验与多组绘图，记录由服务端管理，可在“复现历史”刷新查看。调参训练则是同步请求：页面显示真实等待状态，不显示虚构百分比；结果仅保留在当前浏览器会话，刷新页面会清空，离开页面不会取消服务器计算。请求超时或断网时不会冒充训练成功。</p><p>连接失败时可通过左侧栏下方修改或刷新后端地址。原图与新图的下载链接会使用当前后端地址；切换地址后页面重新读取目录和历史，清除旧服务的调参结果。</p></section>
</template>
