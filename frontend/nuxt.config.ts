// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },

  modules: ["@nuxt/ui", "@nuxt/eslint"],

  css: ["~/assets/css/main.css"],

  future: {
    compatibilityVersion: 4,
  },
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || "http://localhost:8000",
    },
  },
  vite: {
    server: {
      proxy: {
        "/backend": {
          target: process.env.NUXT_PUBLIC_API_BASE || "http://127.0.0.1:8000/",
          changeOrigin: false,
          rewrite: (path: string) => path.replace(/^\/backend/, ""),
        },
      },
    },
  },

  compatibilityDate: "2024-11-27",
});
