<script setup>
import { onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useConnectionStore } from './stores/connection'
import ConnectionBadge from './components/common/ConnectionBadge.vue'
const route = useRoute()
const conn = useConnectionStore()
onMounted(() => conn.init())
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <router-link to="/experiment" class="app-logo"><span class="logo-mark"><el-icon><DataAnalysis /></el-icon></span><div>ML 实验室<small>机器学习理论与实践 · TEAM 07</small></div></router-link>
      <el-menu :default-active="route.path" router mode="horizontal" :ellipsis="false" class="app-menu">
        <el-menu-item index="/experiment">实验工作台</el-menu-item><el-menu-item index="/benchmark">算法对比</el-menu-item><el-menu-item index="/datasets">数据集</el-menu-item><el-menu-item index="/manual">使用手册</el-menu-item>
      </el-menu><ConnectionBadge />
    </header>
    <main class="app-main"><router-view /></main>
    <footer class="app-footer"><span>TEAM 07 · 机器学习实验平台</span><span>从参数到结果，理解每一个模型</span></footer>
  </div>
</template>

<style scoped>
.app-shell{min-height:100vh;display:flex;flex-direction:column}.app-header{height:78px;padding:0 max(30px,calc((100vw - 1450px)/2));display:flex;align-items:center;justify-content:space-between;gap:25px;border-bottom:1px solid #e5eee8;background:#fff}
.app-logo{display:flex;align-items:center;gap:11px;text-decoration:none;color:#216a4e;font-weight:750;font-size:20px;white-space:nowrap}.app-logo small{display:block;font-size:9px;color:#8a9c90;font-weight:500;letter-spacing:1px;margin-top:3px}.logo-mark{width:38px;height:38px;border-radius:11px;background:#25855e;color:white;display:grid;place-items:center;font-size:23px}
.app-menu{border:0;height:77px;--el-menu-horizontal-height:77px;background:transparent}.app-menu .el-menu-item{font-size:14px;padding:0 22px}.app-main{width:100%;max-width:1450px;padding:30px;margin:0 auto;flex:1}.app-footer{padding:22px max(30px,calc((100vw - 1390px)/2));background:#ecf5ef;color:#81978a;display:flex;justify-content:space-between;font-size:11px;gap:15px}
@media(max-width:900px){.app-header{height:auto;flex-wrap:wrap;padding:15px 20px 0;gap:12px}.app-menu{order:3;width:100%;height:50px;--el-menu-horizontal-height:50px;justify-content:center}.app-logo{font-size:17px}.app-main{padding:18px}.app-menu .el-menu-item{padding:0 17px}.app-footer{padding:20px}}
</style>
