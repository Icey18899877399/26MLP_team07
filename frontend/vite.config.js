import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    // host: true 允许局域网访问（手机/队友电脑演示用）
    host: '127.0.0.1',
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } }
  },
  build: {
    // ECharts + Element Plus 会让单个 chunk 较大，这里只放宽警告阈值
    chunkSizeWarningLimit: 1500
  }
})
