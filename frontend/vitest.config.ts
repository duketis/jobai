import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/**
 * Vitest config. Separate from vite.config.ts so the production build
 * stays free of test-only plugins / globals and the include patterns
 * don't accidentally pick up source files at build time.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      include: [
        "src/components/TailorButton.tsx",
        "src/components/TailorStatusPill.tsx",
        "src/lib/useTailorRuns.ts",
        "src/pages/TailorRunsPage.tsx",
      ],
      thresholds: {
        lines: 100,
        statements: 100,
        functions: 100,
        branches: 100,
      },
    },
  },
});
