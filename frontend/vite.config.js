import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * vite.config.ts 와 동기화 유지. (일부 환경에서 .js 가 우선 로드될 수 있음)
 */
const DEPLOY_TARGET =
  process.env.VITE_DEV_PROXY_TARGET ||
  "http://210.91.154.131:20443/deployment2/a05af76e431fe3ac";

const proxyCommon = {
  target: DEPLOY_TARGET,
  changeOrigin: true,
  secure: false,
  rewrite: (p) => p,
};

export default defineConfig({
  base: "/",
  resolve: {
    alias: { shared: path.resolve(__dirname, "../shared") },
  },
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": { ...proxyCommon },
      "/static": { ...proxyCommon },
      "/health": { ...proxyCommon },
      "/motions": { ...proxyCommon },
    },
  },
});
