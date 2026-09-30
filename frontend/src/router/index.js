import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/experiment' },
  { path: '/experiment', name: 'experiment', component: () => import('../views/ExperimentView.vue'), meta: { title: '原实验图谱' } },
  { path: '/benchmark', name: 'benchmark', component: () => import('../views/BenchmarkView.vue'), meta: { title: '复现历史' } },
  { path: '/datasets', name: 'datasets', component: () => import('../views/DatasetsView.vue'), meta: { title: '原始数据目录' } },
  { path: '/manual', name: 'manual', component: () => import('../views/ManualView.vue'), meta: { title: '使用说明' } }
]

const router = createRouter({ history: createWebHistory(), routes })
router.afterEach(to => { document.title = `${to.meta.title || '原实验图谱'} - ML 实验室` })
export default router
