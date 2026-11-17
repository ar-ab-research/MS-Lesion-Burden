import { defineConfig } from "vite";

// relative paths, so the built site works under any github pages address
export default defineConfig({
  base: "./",
  build: { outDir: "dist", sourcemap: false, target: "es2022" },
});
