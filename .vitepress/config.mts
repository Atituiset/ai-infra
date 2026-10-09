import { defineConfig } from 'vitepress'

// GitHub Pages 项目站点 https://atituiset.github.io/ai-infra/
const base = '/ai-infra/'
const repo = 'https://github.com/Atituiset/ai-infra'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  lang: 'zh-CN',
  title: 'AI Infra 推理引擎知识全书',
  titleTemplate: '从核心概念到 vLLM/SGLang 源码精读',
  description:
    '面向训推平台/推理引擎研发工程师的 LLM 推理基础设施知识全书：从基础概念、vLLM/SGLang 源码精读，到生产部署、边缘 AI、Agentic/RL 负载与动手 Lab。',

  base,
  lastUpdated: true,
  cleanUrls: true,

  markdown: {
    // shiki 3.x 不再内置 cuda 语法, 用 cpp 高亮兜底
    languageAlias: { cuda: 'cpp' },
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#2e7d5b' }],
    ['meta', { property: 'og:title', content: 'AI Infra 推理引擎知识全书' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          '面向训推平台/推理引擎研发工程师的 LLM 推理基础设施知识全书：vLLM/SGLang 源码精读、生产部署、边缘 AI、动手 Lab。',
      },
    ],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
    // 学习笔记悬浮组件（localStorage），VitePress SPA 路由切换由 theme/index.ts 通知其重挂载
    ['script', { src: `${base}notes.js` }],
  ],

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/favicon.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '前言', link: '/preface' },
      { text: '全景图', link: '/ch00-overview' },
      { text: '实践手册', link: '/ch26-practice' },
    ],

    sidebar: [
      { text: '前言', link: '/preface' },
      {
        text: 'I 全景与入门',
        collapsed: false,
        items: [
          { text: '第0章 AI Infra 全景图', link: '/ch00-overview' },
          { text: '第1章 入门篇 — AI Infra 核心概念', link: '/ch01-fundamentals' },
          { text: '第7章 GPU 架构与 CUDA 编程基础', link: '/ch07-gpu-cuda-basics' },
        ],
      },
      {
        text: 'II 推理引擎进阶',
        collapsed: false,
        items: [
          { text: '第2章 进阶篇 — 高性能推理架构与优化', link: '/ch02-advanced' },
          { text: '第22章 模型压缩全景 — 量化、剪枝、蒸馏与稀疏化', link: '/ch22-compression' },
          { text: '第23章 训练 Infra 速览 — 训推平台工程师的最小知识集', link: '/ch23-training-infra' },
          { text: '第24章 AI 编译器与算子开发', link: '/ch24-compiler-operators' },
        ],
      },
      {
        text: 'III 源码精读',
        collapsed: false,
        items: [
          { text: '第3章 vLLM 源码深度解剖', link: '/ch03-vllm-source' },
          { text: '第4章 SGLang 源码深度解剖', link: '/ch04-sglang-source' },
        ],
      },
      {
        text: 'IV 生产与系统层',
        collapsed: false,
        items: [
          { text: '第8章 生产部署与运维', link: '/ch08-production' },
          { text: '第13章 成本模型与 ROI 分析', link: '/ch13-cost-roi' },
          { text: '第14章 安全、对齐与合规', link: '/ch14-security-compliance' },
          { text: '第18章 引擎之上 — 网关、调度与集群系统层', link: '/ch18-system-layer' },
        ],
      },
      {
        text: 'V 边缘与端侧',
        collapsed: false,
        items: [
          { text: '第12章 国产 GPU 与边缘芯片', link: '/ch12-domestic-gpu' },
          { text: '第15章 边缘 AI 硬件地图与约束', link: '/ch15-edge-hardware' },
          { text: '第16章 边缘推理框架与软件栈', link: '/ch16-edge-frameworks' },
          { text: '第17章 边缘场景优化与落地', link: '/ch17-edge-optimization' },
        ],
      },
      {
        text: 'VI 生态与前沿',
        collapsed: false,
        items: [
          { text: '第10章 推理引擎生态深度对比', link: '/ch10-engine-ecosystem' },
          { text: '第11章 具体模型部署案例', link: '/ch11-model-case-studies' },
          { text: '第19章 Agentic 与 RL 时代的推理负载', link: '/ch19-agentic-rl' },
          { text: '第20章 技术编年史 — AI Infra 演进脉络', link: '/ch20-history' },
          { text: '第21章 论文对照地图（Paper Map）', link: '/ch21-paper-map' },
          { text: '第25章 多模态生成与检索服务', link: '/ch25-multimodal-retrieval' },
        ],
      },
      {
        text: 'VII 成长：面试与实践',
        collapsed: false,
        items: [
          { text: '第5章 面试实战与自检清单', link: '/ch05-interview' },
          { text: '第6章 Deep Research 研究报告', link: '/ch06-research-report' },
          { text: '第26章 动手实践手册（Lab Manual）', link: '/ch26-practice' },
        ],
      },
      {
        text: '附录',
        collapsed: true,
        items: [{ text: '第9章 术语表与索引', link: '/ch09-glossary' }],
      },
    ],

    outline: {
      level: [2, 3],
      label: '本页大纲',
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    editLink: {
      pattern: `${repo}/edit/master/:path`,
      text: '在 GitHub 上编辑此页',
    },

    socialLinks: [{ icon: 'github', link: repo }],

    docFooter: { prev: '上一页', next: '下一页' },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: { dateStyle: 'short', timeStyle: 'short' },
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    skipToContentLabel: '跳转到内容',

    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2026 AI Infra Collect',
    },
  },
})
