import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * 로컬 Vite(3000) → 원격 배포 백엔드 프록시.
 * - 브라우저는 /api, /health 만 호출 (CORS 없음)
 * - target에 배포 base path 포함 시: 최종 요청은 {target}/api/... , {target}/health
 *
 * 로컬 백엔드만 쓸 때: `VITE_DEV_PROXY_TARGET=http://127.0.0.1:7000 npm run dev`
 */
const DEPLOY_TARGET =
  process.env.VITE_DEV_PROXY_TARGET ||
  "http://210.91.154.131:20443/deployment2/a05af76e431fe3ac";

const proxyCommon = {
  target: DEPLOY_TARGET,
  changeOrigin: true,
  secure: false,
  /** 경로는 그대로 전달 (백엔드가 /api/*, /health 를 base 아래에서 받음) */
  rewrite: (p: string) => p,
} as const;

export default defineConfig({
  base: "/",
  resolve: {
    // shared: 이 레포(zimage_webapp) 루트의 shared/ (다른 서버 git clone 시에도 동작)
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
