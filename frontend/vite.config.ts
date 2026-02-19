import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/",
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": { target: "http://210.91.154.131:20443/95ce287337c3ad9f", changeOrigin: true },
      "/static": { target: "http://210.91.154.131:20443/95ce287337c3ad9f", changeOrigin: true },
      "/health": { target: "http://210.91.154.131:20443/95ce287337c3ad9f", changeOrigin: true },
    },
  },
});
