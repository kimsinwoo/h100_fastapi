import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/",
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
