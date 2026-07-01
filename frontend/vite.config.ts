import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Cấu hình Vite cho dự án NIDS VAE Dashboard
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Alias @ trỏ vào thư mục src để import ngắn gọn
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    // Proxy API sang backend FastAPI khi dev local
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
