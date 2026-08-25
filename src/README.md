# AI Infra 推理引擎知识全书

> **A comprehensive mbook on LLM inference infrastructure, covering fundamentals to vLLM/SGLang source-level analysis.**

---

## 关于本书

本书面向**训推平台/推理引擎研发工程师**岗位，覆盖 AI Infra 推理领域从入门到源码级别的知识体系。内容基于：

- **vLLM 本地源码**：`vllm/` 目录，V1 引擎架构
- **SGLang 本地源码**：`sglang/` 目录，RadixAttention + Overlap 调度
- **互联网最新资料**：论文、博客、Release Notes、社区讨论
- **面试实战**：高频问题 + 源码定位 + 场景设计

## 如何使用

### 按角色阅读

| 角色 | 推荐路径 |
|------|----------|
| **新人/转行** | Part 0 → Part 1 → Part 5 (面试题自检) → Part 26 L0-L1（动手） |
| **推理服务开发** | Part 1 → Part 2 → Part 3 (vLLM) 或 Part 4 (SGLang) + Part 26 Track B/C |
| **架构师/TL** | Part 0 → Part 2 (精读) → Part 3 + Part 4 (速览) |
| **面试备战** | Part 5 (自检) → 按缺口回溯 Part 1-4 |
| **源码贡献** | 直接跳到 Part 3 或 Part 4 的"源码阅读路线" |
| **边缘 AI 工程师 / 平头哥生态** | Part 15 → Part 16 → Part 17（重点 §16.4 MNN） |
| **平台工程师 / SRE** | Part 8 → Part 18（网关、调度、容错） |
| **RL Infra / Agent 平台** | Part 19 → Part 18 §18.5（结构化输出） |
| **模型部署 / 压缩工程师** | Part 22 → Part 2 §2.7 → Part 17（端侧） |
| **内核 / 算子研发** | Part 7 → Part 24 → Part 20 §20.3 |
| **多模态 / RAG 平台** | Part 25 → Part 18 §18.3 |

### 按问题速查

| 我想了解... | 看这篇 |
|------------|--------|
| AI Infra 是什么？有哪些引擎？| Part 0 |
| KV Cache 是什么？为什么重要？| Part 1 §1.1.3-1.1.4 |
| TTFT / TPOT / Throughput 含义和优化 | Part 1 §1.2 |
| Continuous Batching 原理 | Part 1 §1.3.2, Part 2 §2.1 |
| TP / PP / EP 怎么切分？| Part 2 §2.2 |
| PD 分离 + Mooncake + HiCache | Part 2 §2.3 |
| PagedAttention / FlashAttention / MLA | Part 2 §2.4 |
| 稀疏注意力 NSA / DSA 是什么？ | Part 2 §2.4.4 |
| 投机解码 Eagle / MTP | Part 2 §2.6 |
| vLLM Scheduler / KV Manager 源码 | Part 3 §3.2-3.3 |
| vLLM Model Runner / Attention Backend | Part 3 §3.4 |
| vLLM KV Offload / 最新特性 | Part 3 §3.7 |
| SGLang RadixAttention 源码 | Part 4 §4.2 |
| SGLang Overlap 调度 / PD 分离 | Part 4 §4.3, §4.5 |
| SGLang 最新特性（D-LLM / EPLB / KV Canary）| Part 4 §4.10 |
| DeepSeek V3.2 / V4 怎么部署？ | Part 11 §11.1.1 |
| 面试高频问题 | Part 5 §5.1 |
| 场景设计题（千卡/MoE/多租户）| Part 5 §5.3 |
| 能力自检 Checklist | Part 5 §5.4 |
| GPU / CUDA 基础 | Part 7 |
| Benchmark / 部署 / 运维 | Part 8 |
| 术语速查 | Part 9 |
| 边缘 AI 硬件怎么选？ | Part 15 §15.5 |
| TensorRT-LLM 在 Jetson 上怎么用？ | Part 16 §16.2 |
| llama.cpp 量化/GGUF 原理？ | Part 16 §16.3 |
| 边缘 KV Cache 怎么省？ | Part 17 §17.1 |
| VLM/VLA 怎么部署到边缘？ | Part 17 §17.5 |
| 平头哥玄铁/MNN 生态？ | Part 16 §16.4 + Part 12 |
| K8s 上 GPU 怎么调度/共享（DRA/MIG/MPS）？ | Part 18 §18.2 |
| 推理网关 / cache-aware 路由 / goodput？ | Part 18 §18.3 |
| 多租户 LoRA 服务（S-LoRA）？ | Part 18 §18.4 |
| 结构化输出 / constrained decoding？ | Part 18 §18.5 |
| Reasoning 模型对 infra 的影响？ | Part 19 §19.2 |
| Agent 会话 KV 生命周期 / sticky routing？ | Part 19 §19.3 |
| RL 训练的 rollout engine / weight sync？ | Part 19 §19.4 |
| MCP 是什么、影响哪些层？ | Part 19 §19.5 |
| Continuous batching 是怎么演化来的？ | Part 20 §20.2 |
| NVLink vs Ultra Ethernet vs UALink？ | Part 20 §20.5 |
| 某技术对应哪篇论文？阅读顺序？ | Part 21 |
| GPTQ/AWQ 原理与取舍？剪枝怎么选？ | Part 22 |
| 训练显存账 / ZeRO/FSDP / 万卡容错？ | Part 23 |
| AI 编译器栈对比？算子怎么开发融合？ | Part 24 |
| 扩散模型怎么 serving？向量检索 infra？ | Part 25 |
| 想动手做实验/攒作品集？ | Part 26（L0-L12） |

---

## 目录

- [Part 0：AI Infra 全景图](part-00-overview.md)
- [Part 1：入门篇 — AI Infra 核心概念](part-01-fundamentals.md)
- [Part 2：进阶篇 — 高性能推理架构](part-02-advanced.md)
- [Part 3：vLLM 源码深度解剖](part-03-vllm-source.md)
- [Part 4：SGLang 源码深度解剖](part-04-sglang-source.md)
- [Part 5：面试实战与自检清单](part-05-interview.md)
- [Part 6：Deep Research 研究报告](part-06-research-report.md)
- [Part 7：GPU 架构与 CUDA 编程基础](part-07-gpu-cuda-basics.md)
- [Part 8：生产部署与运维](part-08-production.md)
- [Part 9：术语表与索引](part-09-glossary.md)
- [Part 10：推理引擎生态深度对比](part-10-engine-ecosystem.md)
- [Part 11：具体模型部署案例](part-11-model-case-studies.md)
- [Part 12：国产 GPU 与边缘芯片](part-12-domestic-gpu.md)
- [Part 13：成本模型与 ROI 分析](part-13-cost-roi.md)
- [Part 14：安全、对齐与合规](part-14-security-compliance.md)
- [Part 15：边缘 AI 硬件地图与约束](part-15-edge-hardware.md)
- [Part 16：边缘推理框架与软件栈](part-16-edge-frameworks.md)
- [Part 17：边缘场景优化与落地](part-17-edge-optimization.md)
- [Part 18：引擎之上 — 网关、调度与集群系统层](part-18-system-layer.md)
- [Part 19：Agentic 与 RL 时代的推理负载](part-19-agentic-rl.md)
- [Part 20：技术编年史 — AI Infra 演进脉络](part-20-history.md)
- [Part 21：论文对照地图（Paper Map）](part-21-paper-map.md)
- [Part 22：模型压缩全景](part-22-compression.md)
- [Part 23：训练 Infra 速览](part-23-training-infra.md)
- [Part 24：AI 编译器与算子开发](part-24-compiler-operators.md)
- [Part 25：多模态生成与检索服务](part-25-multimodal-retrieval.md)
- [Part 26：动手实践手册（Lab Manual）](part-26-practice.md)

---

## 约定

| 标记 | 含义 |
|------|------|
| ★ | 面试高频，必须掌握 |
| ☆ | 加分项，了解即可 |
| `code` | 源码中实际存在的类/函数/变量名 |
| **粗体** | 核心概念，首次出现 |

## 维护

- 最后更新：2026-08-24
- 基于源码版本：vLLM V1 引擎开发主线、SGLang 最新主线、TensorRT-LLM 主线、llama.cpp 主线
- Profile 岗位参考：`profile.md` — 训推平台及引擎研发主任工程师

## 构建与部署（mdbook）

本书使用 [mdBook](https://github.com/rust-lang/mdBook) 管理，结构如下：

```
docs/
├── book.toml          # mdbook 配置
├── src/               # 全部 Markdown 源文件
│   ├── SUMMARY.md     # 目录（章节顺序）
│   ├── README.md      # 首页
│   └── part-*.md      # 各章节
└── .github/workflows/ # GitHub Pages 自动部署
```

**本地构建预览**：

```bash
cd docs
mdbook build          # 生成静态站点到 book/
mdbook serve          # 本地预览 http://localhost:3000
```

**部署到 GitHub Pages**：推送到 GitHub 仓库的 `master` 分支后，`.github/workflows/deploy-mdbook.yml` 会自动构建并发布。首次使用需在仓库 Settings → Pages 中把 Source 设为 **GitHub Actions**，并将 `book.toml` 中的 `git-repository-url` / `edit-url-template` 改成实际仓库地址。
