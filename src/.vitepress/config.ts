import { defineConfig } from 'vitepress'

const base = '/ai-infra/'
const repo = 'https://github.com/Atituiset/ai-infra'

// 侧边栏条目与章节编号保持线性顺序（Part 0 → 26），
// 上下页导航 (prev/next) 也由该顺序驱动。
const chapters = [
  { text: 'Part 0：AI Infra 全景图', link: '/part-00-overview' },
  { text: 'Part 1：入门篇 — AI Infra 核心概念', link: '/part-01-fundamentals' },
  { text: 'Part 2：进阶篇 — 高性能推理架构与优化', link: '/part-02-advanced' },
  { text: 'Part 3：vLLM 源码深度解剖', link: '/part-03-vllm-source' },
  { text: 'Part 4：SGLang 源码深度解剖', link: '/part-04-sglang-source' },
  { text: 'Part 5：面试实战与自检清单', link: '/part-05-interview' },
  { text: 'Part 6：Deep Research 研究报告', link: '/part-06-research-report' },
  { text: 'Part 7：GPU 架构与 CUDA 编程基础', link: '/part-07-gpu-cuda-basics' },
  { text: 'Part 8：生产部署与运维', link: '/part-08-production' },
  { text: 'Part 9：术语表与索引', link: '/part-09-glossary' },
  { text: 'Part 10：推理引擎生态深度对比', link: '/part-10-engine-ecosystem' },
  { text: 'Part 11：具体模型部署案例', link: '/part-11-model-case-studies' },
  { text: 'Part 12：国产 GPU 与边缘芯片', link: '/part-12-domestic-gpu' },
  { text: 'Part 13：成本模型与 ROI 分析', link: '/part-13-cost-roi' },
  { text: 'Part 14：安全、对齐与合规', link: '/part-14-security-compliance' },
  { text: 'Part 15：边缘 AI 硬件地图与约束', link: '/part-15-edge-hardware' },
  { text: 'Part 16：边缘推理框架与软件栈', link: '/part-16-edge-frameworks' },
  { text: 'Part 17：边缘场景优化与落地', link: '/part-17-edge-optimization' },
  { text: 'Part 18：引擎之上 — 网关、调度与集群系统层', link: '/part-18-system-layer' },
  { text: 'Part 19：Agentic 与 RL 时代的推理负载', link: '/part-19-agentic-rl' },
  { text: 'Part 20：技术编年史 — AI Infra 演进脉络', link: '/part-20-history' },
  { text: 'Part 21：论文对照地图（Paper Map）', link: '/part-21-paper-map' },
  { text: 'Part 22：模型压缩全景 — 量化、剪枝、蒸馏与稀疏化', link: '/part-22-compression' },
  { text: 'Part 23：训练 Infra 速览 — 训推平台工程师的最小知识集', link: '/part-23-training-infra' },
  { text: 'Part 24：AI 编译器与算子开发', link: '/part-24-compiler-operators' },
  { text: 'Part 25：多模态生成与检索服务', link: '/part-25-multimodal-retrieval' },
  { text: 'Part 26：动手实践手册（Lab Manual）', link: '/part-26-practice' },
]

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI Infra 推理引擎知识全书',
  description:
    '面向训推平台/推理引擎研发工程师的 LLM 推理基础设施知识手册：从基础概念、vLLM/SGLang 源码，到生产部署与边缘 AI。',
  base,
  lastUpdated: true,

  markdown: {
    // shiki 3.x 不再内置 cuda 语法, 用 cpp 高亮兜底
    languageAlias: { cuda: 'cpp' },
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
    // 学习笔记悬浮组件（localStorage），VitePress SPA 路由切换由 theme/index.ts 通知其重挂载
    ['script', { src: `${base}notes.js` }],
  ],

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'GitHub', link: repo },
    ],

    sidebar: [
      { text: '关于本书', link: '/' },
      ...chapters,
    ],

    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    docFooter: { prev: '上一章', next: '下一章' },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: { dateStyle: 'short', timeStyle: 'short' },
    },

    editLink: {
      pattern: `${repo}/edit/master/src/:path`,
      text: '在 GitHub 上编辑此页',
    },

    socialLinks: [{ icon: 'github', link: repo }],

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '没有找到相关结果',
            resetButtonTitle: '清除查询条件',
            backButtonTitle: '返回',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭',
            },
          },
        },
      },
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © AI Infra Collect',
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
  },
})
