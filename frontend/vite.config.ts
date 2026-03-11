import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/",
  resolve: {
    // shared: 이 레포(zimage_webapp) 루트의 shared/ (다른 서버 git clone 시에도 동작)
    alias: { shared: path.resolve(__dirname, "../shared") },
  },
  plugins: [react()],
  server: {
    port: 3000,
    // 로컬: 백엔드를 7000에서 띄우고 프론트는 3000 → 같은 origin처럼 /api, /static, /health 를 7000으로 전달
    proxy: {
      "/api": { target: "http://localhost:7000", changeOrigin: true },
      "/static": { target: "http://localhost:7000", changeOrigin: true },
      "/health": { target: "http://localhost:7000", changeOrigin: true },
    },
  },
});
