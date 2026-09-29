import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/experiment' },
  {
    path: '/experiment',
    name: 'experiment',
    component: () => import('../views/ExperimentView.vue'),
    meta: { title: '实验工作台', icon: 'Cpu' }
  },
  {
    path: '/benchmark',
    name: 'benchmark',
    component: () => import('../views/BenchmarkView.vue'),
    meta: { title: '算法对比', icon: 'DataAnalysis' }
  },
  {
    path: '/datasets',
    name: 'datasets',
    component: () => import('../views/DatasetsView.vue'),
    meta: { title: '数据集', icon: 'Collection' }
  },
  {
    path: '/manual',
    name: 'manual',
    component: () => import('../views/ManualView.vue'),
    meta: { title: '使用手册', icon: 'Document' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.afterEach((to) => {
  document.title = to.meta.title
    ? `${to.meta.title} - 机器学习算法可视化平台`
    : '机器学习算法可视化平台'
})

export default router
