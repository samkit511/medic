import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import fs from 'fs'
import path from 'path'

// Check if SSL certificates exist
const sslCertExists = fs.existsSync(path.resolve(__dirname, 'ssl/server.crt'))
const sslKeyExists = fs.existsSync(path.resolve(__dirname, 'ssl/server.key'))
const useHttps = sslCertExists && sslKeyExists

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true, // Don't try other ports if 5173 is in use
    https: useHttps ? {
      key: fs.readFileSync(path.resolve(__dirname, 'ssl/server.key')),
      cert: fs.readFileSync(path.resolve(__dirname, 'ssl/server.crt')),
    } : false,
    host: '0.0.0.0', // Allow external connections
  },
  define: {
    __BACKEND_URL__: useHttps ? '"https://localhost:8443"' : '"http://localhost:8000"',
  },
})
