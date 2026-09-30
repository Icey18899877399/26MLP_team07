<script setup>
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useConnectionStore } from './stores/connection.js'
import ConnectionBadge from './components/common/ConnectionBadge.vue'

const route = useRoute()
const conn = useConnectionStore()
const collapsed = ref(false)
const mobileOpen = ref(false)
const nav = [
  { path: '/experiment', title: '原实验图谱', icon: 'DataAnalysis' },
  { path: '/benchmark', title: '复现历史', icon: 'Clock' },
  { path: '/datasets', title: '原始数据目录', icon: 'Collection' },
  { path: '/manual', title: '使用说明', icon: 'Document' }
]
const current = computed(() => nav.find(item => item.path === route.path) || nav[0])
conn.init({ originalOnly: true })
</script>

<template>
  <div class="app-shell" :class="{ 'nav-collapsed': collapsed, 'mobile-open': mobileOpen }">
    <aside class="app-sidebar">
      <router-link class="brand" to="/experiment" @click="mobileOpen = false"><span class="brand-mark">M</span><span class="brand-copy"><strong>ML 实验室</strong><small>TEAM 07 · ORIGINAL WORK</small></span></router-link>
      <div class="nav-section-title">工作空间</div>
      <nav aria-label="主导航"><router-link v-for="item in nav" :key="item.path" :to="item.path" class="nav-link" :aria-label="item.title" :class="{ active: route.path === item.path }" @click="mobileOpen = false"><el-icon><component :is="item.icon" /></el-icon><span>{{ item.title }}</span></router-link></nav>
      <div class="sidebar-spacer" />
      <div class="sidebar-note"><span class="note-dot" />原始实验 · 真实产物</div>
      <button class="collapse-button" type="button" :aria-label="collapsed ? '展开侧栏' : '折叠侧栏'" @click="collapsed = !collapsed"><el-icon><DArrowLeft v-if="!collapsed" /><DArrowRight v-else /></el-icon><span>折叠侧栏</span></button>
    </aside>
    <div v-if="mobileOpen" class="mobile-scrim" @click="mobileOpen = false" />
    <div class="app-body">
      <header class="app-topbar"><div class="topbar-left"><button class="mobile-menu-button" type="button" aria-label="打开导航" @click="mobileOpen = true"><el-icon><Menu /></el-icon></button><el-breadcrumb separator="/"><el-breadcrumb-item>机器学习实践</el-breadcrumb-item><el-breadcrumb-item>{{ current.title }}</el-breadcrumb-item></el-breadcrumb></div><div class="topbar-right"><span class="topbar-caption">原图与完整复现</span><ConnectionBadge /></div></header>
      <div class="page-tabs"><router-link v-for="item in nav" :key="item.path" :to="item.path" :class="{ active: route.path === item.path }">{{ item.title }}</router-link></div>
      <main class="app-main"><router-view /></main>
      <footer class="app-footer">TEAM 07 · 机器学习理论与实践 <span>原始实验协议由项目绘图脚本定义</span></footer>
    </div>
  </div>
</template>
