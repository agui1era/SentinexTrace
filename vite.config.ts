import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  server: {
    watch: {
      ignored: ['**/.venv/**', '**/data/**'],
    },
    proxy: {
      '/api/rtsp': {
        target: 'http://localhost:8787',
        changeOrigin: true,
      },
      '/api/vision': {
        target: 'http://localhost:8890',
        changeOrigin: true,
      },
    },
  },
});
