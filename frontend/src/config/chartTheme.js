/**
 * 全局 ECharts 主题：所有图表共用一套配色、字体与提示样式，保证视觉一致性。
 * 在 BaseChart 中通过 echarts.registerTheme 注册为 'mlviz' 主题。
 */
export const chartTheme = {
  color: [
    '#409eff', // 主蓝
    '#67c23a', // 绿
    '#e6a23c', // 橙
    '#f56c6c', // 红
    '#909399', // 灰
    '#9b59b6', // 紫
    '#00bcd4', // 青
    '#ff9800' // 深橙
  ],
  textStyle: {
    // 中文字体栈：Windows 下回落到微软雅黑
    fontFamily: "system-ui, 'PingFang SC', 'Microsoft YaHei', 'Segoe UI', sans-serif"
  },
  tooltip: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#e4e7ed',
    borderWidth: 1,
    textStyle: { color: '#303133' },
    extraCssText: 'box-shadow: 0 2px 12px rgba(0,0,0,.12);'
  },
  legend: {
    textStyle: { fontSize: 12 }
  },
  categoryAxis: {
    axisLine: { lineStyle: { color: '#dcdfe6' } },
    axisTick: { alignWithLabel: true },
    axisLabel: { color: '#606266' }
  },
  valueAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: '#606266' },
    splitLine: { lineStyle: { color: '#f0f2f5' } }
  }
}
