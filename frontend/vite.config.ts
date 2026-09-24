import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Local FastAPI (`python web_main.py` / `python service_main.py`) — the
// browser only ever talks to the Vite origin, so every API call stays
// same-origin and no backend CORS is involved (see README.md, "Web
// frontend (Stage 7B-1)").
const BACKEND_ORIGIN = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    // Literal loopback + a fixed port: the GitHub OAuth callback registered
    // for local development (http://127.0.0.1:5173/api/auth/github/callback)
    // must keep matching, so never silently fall back to another port.
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": BACKEND_ORIGIN,
      "/healthz": BACKEND_ORIGIN,
    },
  },
  build: {
    // Served by FastAPI (web/frontend.py): index.html at "/" and the
    // content-hashed files under dist/assets at "/assets/...".
    outDir: "dist",
    assetsDir: "assets",
  },
  test: {
    environment: "jsdom",
    // https so a Secure `__Host-` cookie can be exercised like production.
    environmentOptions: { jsdom: { url: "https://app.test/" } },
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    restoreMocks: true,
    unstubGlobals: true,
  },
});
