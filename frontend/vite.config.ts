import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Vite dev server proxies /api/* to the FastAPI backend on :8421 so the
// frontend can talk to the agent + jobs API without CORS plumbing.
// Production builds emit to ../jobai/api/static/, where FastAPI mounts
// them as the SPA root (server.py serves index.html for unknown paths).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8421",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: path.resolve(__dirname, "../jobai/api/static"),
    emptyOutDir: true,
  },
});
