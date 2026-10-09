---
title: 前言
---

# 前言

> 这是一本面向**训推平台/推理引擎研发工程师**的 AI Infra 推理知识全书——从核心概念、vLLM/SGLang 源码精读，到生产部署、边缘 AI 与动手 Lab，27 章覆盖推理基础设施的完整知识地图。

## 内容基于

本书内容覆盖 AI Infra 推理领域从入门到源码级别的知识体系，基于：

- **vLLM 源码**：[vllm-project/vllm](https://github.com/vllm-project/vllm)，V1 引擎架构
- **SGLang 源码**：[sgl-project/sglang](https://github.com/sgl-project/sglang)，RadixAttention + Overlap 调度
- **互联网最新资料**：论文、博客、Release Notes、社区讨论
- **实战自测**：高频问题 + 源码定位 + 场景设计

## 如何使用

### 按角色阅读

| 角色 | 推荐路径 |
|------|----------|
| **新人/转行** | 第0章 → 第1章 → 第23章 (自测) → 第25章 L0-L1（动手） |
| **推理服务开发** | 第1章 → 第3章 → 第7章 (vLLM) 或 第8章 (SGLang) + 第25章 Track B/C |
| **架构师/TL** | 第0章 → 第3章 (精读) → 第7章 + 第8章 (速览) |
| **能力自检** | 第23章 (自检) → 按缺口回溯第1~8章 |
| **源码贡献** | 直接跳到 第7章 或 第8章 的"源码阅读路线" |
| **边缘 AI 工程师 / 平头哥生态** | 第14章 → 第15章 → 第16章（重点 §15.4 MNN） |
| **平台工程师 / SRE** | 第9章 → 第12章（网关、调度、容错） |
| **RL Infra / Agent 平台** | 第19章 → 第12章 §12.5（结构化输出） |
| **模型部署 / 压缩工程师** | 第4章 → 第3章 §3.7 → 第16章（端侧） |
| **内核 / 算子研发** | 第2章 → 第6章 → 第20章 §20.3 |
| **多模态 / RAG 平台** | 第22章 → 第12章 §12.3 |

### 按问题速查

| 我想了解... | 看这篇 |
|------------|--------|
| AI Infra 是什么？有哪些引擎？| 第0章 |
| KV Cache 是什么？为什么重要？| 第1章 §1.1.3-1.1.4 |
| TTFT / TPOT / Throughput 含义和优化 | 第1章 §1.2 |
| Continuous Batching 原理 | 第1章 §1.3.2, 第3章 §3.1 |
| TP / PP / EP 怎么切分？| 第3章 §3.2 |
| PD 分离 + Mooncake + HiCache | 第3章 §3.3 |
| PagedAttention / FlashAttention / MLA | 第3章 §3.4 |
| 稀疏注意力 NSA / DSA 是什么？ | 第3章 §3.4.4 |
| 投机解码 Eagle / MTP | 第3章 §3.6 |
| vLLM Scheduler / KV Manager 源码 | 第7章 §7.2-7.3 |
| vLLM Model Runner / Attention Backend | 第7章 §7.4 |
| vLLM KV Offload / 最新特性 | 第7章 §7.7 |
| SGLang RadixAttention 源码 | 第8章 §8.2 |
| SGLang Overlap 调度 / PD 分离 | 第8章 §8.3, §8.5 |
| SGLang 最新特性（D-LLM / EPLB / KV Canary）| 第8章 §8.10 |
| DeepSeek V3.2 / V4 怎么部署？ | 第18章 §18.1.1 |
| 高频问题 | 第23章 §23.1 |
| 场景设计题（千卡/MoE/多租户）| 第23章 §23.3 |
| 能力自检 Checklist | 第23章 §23.4 |
| GPU / CUDA 基础 | 第2章 |
| Benchmark / 部署 / 运维 | 第9章 |
| 术语速查 | 附录 |
| 边缘 AI 硬件怎么选？ | 第14章 §14.5 |
| TensorRT-LLM 在 Jetson 上怎么用？ | 第15章 §15.2 |
| llama.cpp 量化/GGUF 原理？ | 第15章 §15.3 |
| 边缘 KV Cache 怎么省？ | 第16章 §16.1 |
| VLM/VLA 怎么部署到边缘？ | 第16章 §16.5 |
| 平头哥玄铁/MNN 生态？ | 第15章 §15.4 + 第13章 |
| K8s 上 GPU 怎么调度/共享（DRA/MIG/MPS）？ | 第12章 §12.2 |
| 推理网关 / cache-aware 路由 / goodput？ | 第12章 §12.3 |
| 多租户 LoRA 服务（S-LoRA）？ | 第12章 §12.4 |
| 结构化输出 / constrained decoding？ | 第12章 §12.5 |
| Reasoning 模型对 infra 的影响？ | 第19章 §19.2 |
| Agent 会话 KV 生命周期 / sticky routing？ | 第19章 §19.3 |
| RL 训练的 rollout engine / weight sync？ | 第19章 §19.4 |
| MCP 是什么、影响哪些层？ | 第19章 §19.5 |
| Continuous batching 是怎么演化来的？ | 第20章 §20.2 |
| NVLink vs Ultra Ethernet vs UALink？ | 第20章 §20.5 |
| 某技术对应哪篇论文？阅读顺序？ | 第21章 |
| GPTQ/AWQ 原理与取舍？剪枝怎么选？ | 第4章 |
| 训练显存账 / ZeRO/FSDP / 万卡容错？ | 第5章 |
| AI 编译器栈对比？算子怎么开发融合？ | 第6章 |
| 扩散模型怎么 serving？向量检索 infra？ | 第22章 |
| 想动手做实验/攒作品集？ | 第25章（L0-L12） |

---

## 约定

| 标记 | 含义 |
|------|------|
| ★ | 高频重点，必须掌握 |
| ☆ | 加分项，了解即可 |
| `code` | 源码中实际存在的类/函数/变量名 |
| **粗体** | 核心概念，首次出现 |

## 维护

- 最后更新：2026-08-24
- 基于源码版本：[vLLM](https://github.com/vllm-project/vllm) V1 引擎开发主线、[SGLang](https://github.com/sgl-project/sglang)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[llama.cpp](https://github.com/ggml-org/llama.cpp) 最新主线
- 面向岗位：训推平台及引擎研发工程师

## 构建与部署（VitePress）

本书使用 [VitePress](https://vitepress.dev/zh/) 构建，结构如下：

```
docs/
├── package.json           # Node 依赖与脚本
├── index.md               # 首页（home 布局）
├── preface.md             # 前言（阅读路径与速查表）
├── ch*.md                 # 各章节（第0章 → 第26章）
├── public/                # 静态资源（notes.js、favicon）
├── .vitepress/
│   ├── config.mts         # 站点配置（分组侧边栏/搜索/编辑链接）
│   └── theme/             # 主题定制（书籍排版、章节元信息、SPA 路由钩子）
└── .github/workflows/     # GitHub Pages 自动部署
```

**本地构建预览**：

```bash
cd docs
npm install
npm run docs:dev        # 热更新预览 http://localhost:5173
npm run docs:build      # 构建静态站点到 .vitepress/dist
npm run docs:preview    # 预览构建产物
```

**部署到 GitHub Pages**：推送到 `master` 分支后，`.github/workflows/deploy-docs.yml` 会自动构建并发布。仓库 Settings → Pages 的 Source 需设为 **GitHub Actions**。
