<script setup>
const steps = [
  { number: '01', title: '选择原实验', detail: '在“原实验图谱”按分类、回归、聚类、异常检测浏览算法。数据来源、脚本参数和实验步骤均为只读。' },
  { number: '02', title: '浏览历史原图', detail: '每套实验的六张现有 PNG 标注为历史归档。可放大查看完整图像、下载原尺寸 PNG；有绘图 CSV 的实验也提供原文件。' },
  { number: '03', title: '启动完整复现', detail: '点击“启动原实验”只提交实验标识。后台串行运行原绘图脚本的完整默认流程，并写入独立输出目录。' },
  { number: '04', title: '查看真实运行', detail: '在本次产物或复现历史中查看排队、运行、完成、失败或中断状态。页面显示实际耗时和后端日志，不展示猜测的进度百分比。' }
]
</script>

<template>
  <section class="page-intro"><div><div class="eyebrow">GUIDE / REPRODUCIBILITY</div><h1>使用说明</h1><p>区分既有成果与新运行产物，按原始协议完成复现。</p></div></section>
  <div class="manual-steps"><article v-for="step in steps" :key="step.number" class="page-card manual-step"><span class="manual-number">{{ step.number }}</span><h2>{{ step.title }}</h2><p>{{ step.detail }}</p></article></div>
  <section class="page-card manual-note"><div class="card-title-row"><div><div class="eyebrow">DATA & PROTOCOL</div><h2>关于数据与参数</h2></div></div><p>原实验协议由项目的 12 套绘图脚本定义。界面不会提供临时修改切分比例、随机种子或超参数的表单；这些操作会偏离原实验。数据集页面标出实际使用的路径，Adult、California 等仅存档文件单独列示。</p><p>某些图片是使用脚本内生成的合成数据进行机制演示，另一些图片包含跨数据集比较。每套实验的“原实验协议”会显示该脚本对应的数据和步骤。</p></section>
  <section class="page-card manual-note"><div class="card-title-row"><div><div class="eyebrow">RUN HISTORY</div><h2>关于长时间运行</h2></div></div><p>原脚本可能执行交叉验证、重复实验与多组绘图，耗时取决于机器性能。浏览器可以离开当前页面；后台运行及记录由服务端管理，稍后可在“复现历史”刷新查看。服务重启中断的任务会明确标为中断。</p><p>连接失败时可通过右上角修改或刷新后端地址。原图与新图的下载链接会使用当前后端地址；切换地址后页面重新读取目录和历史。</p></section>
</template>
