// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },

  modules: ["@nuxt/ui", "@nuxt/eslint"],

  css: ["~/assets/css/main.css"],

  future: {
    compatibilityVersion: 4,
  },

  vite: {
    server: {
      proxy: {
        "/backend": {
          target: `http://127.0.0.1:8000/`,
          changeOrigin: true,
          rewrite: (path: string) => path.replace(/^\/backend/, ""),
        },
      },
    },
  },

  compatibilityDate: "2024-11-27",
});
