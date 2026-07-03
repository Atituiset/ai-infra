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
| **新人/转行** | Part 0 → Part 1 → Part 5 (面试题自检) |
| **推理服务开发** | Part 1 → Part 2 → Part 3 (vLLM) 或 Part 4 (SGLang) |
| **架构师/TL** | Part 0 → Part 2 (精读) → Part 3 + Part 4 (速览) |
| **面试备战** | Part 5 (自检) → 按缺口回溯 Part 1-4 |
| **源码贡献** | 直接跳到 Part 3 或 Part 4 的"源码阅读路线" |

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
| 投机解码 Eagle / MTP | Part 2 §2.6 |
| vLLM Scheduler / KV Manager 源码 | Part 3 §3.2-3.3 |
| vLLM Model Runner / Attention Backend | Part 3 §3.4 |
| SGLang RadixAttention 源码 | Part 4 §4.2 |
| SGLang Overlap 调度 / PD 分离 | Part 4 §4.3, §4.5 |
| 面试高频问题 | Part 5 §5.1 |
| 场景设计题（千卡/MoE/多租户）| Part 5 §5.3 |
| 能力自检 Checklist | Part 5 §5.4 |
| GPU / CUDA 基础 | Part 7 |
| Benchmark / 部署 / 运维 | Part 8 |
| 术语速查 | Part 9 |

---

## 目录

### [Part 0: AI Infra 全景图](part-00-overview.md)
- AI Infra 分层架构
- 训练 vs 推理的本质区别
- 推理引擎生态对比（vLLM / SGLang / TRT-LLM / TGI / LMDeploy）
- 部署形态演进：单卡 → TP → PP → PD 分离 → HiCache 集群
- 重要论文索引

### [Part 1: 入门篇 — 核心概念](part-01-fundamentals.md)
- Transformer 推理基础：自回归、Prefill vs Decode、KV Cache、GQA/MQA
- 性能指标：TTFT / TPOT / Throughput / Goodput
- 批处理：Static Batching → Continuous Batching → Chunked Prefill
- 内存与显存：HBM 层级、Memory Wall、KV Cache 显存计算
- 量化基础：AWQ / GPTQ / SmoothQuant / FP8 / KV Cache 量化
- 解码策略：Greedy / Sampling / Beam Search / 投机解码入门
- ★ 标注面试高频知识点

### [Part 2: 进阶篇 — 高性能推理架构](part-02-advanced.md)
- Continuous Batching 深度：vLLM Scheduler vs SGLang Scheduler
- Overlap 调度：SGLang 独有 CPU/GPU 并行
- 并行策略：TP / PP / EP / DP / SP / CP，MoE 专家并行
- PD Disaggregation：原理、SGLang 实现、Mooncake KV Transfer
- HiCache 三级缓存：L1 GPU / L2 CPU / L3 分布式
- Attention 优化：PagedAttention / FlashAttention / FlashInfer / MLA
- 量化进阶：W4A16 vs W8A8 vs FP8 实战选型
- 投机解码深度：Eagle / Medusa / MTP / dFlash

### [Part 3: vLLM 源码解剖](part-03-vllm-source.md)
- 架构总览：V1 Engine (EngineCore + Frontend)，数据流全景
- Scheduler：请求队列、KV 分配、Preemption、KV Connector
- PagedAttention / KVCacheManager：Block Pool、Prefix Caching、Hash Matching
- Model Runner：GPUModelRunner、CUDA Graph、Attention Backend 选择
- 投机解码：Eagle Proposer、验证流程、Scheduler 集成
- 量化支持：AWQ / GPTQ / FP8 配置与加载
- 源码阅读路线图

### [Part 4: SGLang 源码解剖](part-04-sglang-source.md)
- 架构总览：与 vLLM 的关键差异、数据流全景
- RadixAttention：Radix Tree 数据结构、前缀匹配、驱逐策略
- Overlap 调度：CPU/GPU 并行实现、RelayPayload、结果处理
- Model Runner：ScheduleBatch → ForwardBatch 转换
- PD 分离：Prefill/Decode Bootstrap + Transfer 完整生命周期
- Mooncake 传输后端：分层传输 + RDMA
- HiCache 集成：跨节点分布式 KV Cache
- 投机解码：Eagle v2 Worker、dFlash
- Mem Cache 体系：Radix / SWA / Mamba / Hi / Session / Unified Cache
- 源码阅读路线图

### [Part 5: 面试实战与自检](part-05-interview.md)
- 高频面试题（入门 5 题 + 进阶 5 题，含答题要点和参考章节）
- 源码定位题（4 题，含具体代码路径）
- 场景设计题（千卡 MoE 部署、低延迟优化、多租户调度）
- 能力自检 Checklist（对标 Profile 岗位要求）
- 必读论文/源码/关注人物

### [Part 6: 补充 — Deep Research 研究报告](part-06-research-report.md)
- 107-agent 多阶段对抗验证 → 7 条高置信度声明
- **Finding 1**: MLA + TP 下 KV Cache 8× 重复 → DP Attention 方案
- **Finding 2**: vLLM Wide-EP 架构（Attention DP + Expert EP）
- **Finding 3**: DP+EP vs TP+EP 实测数据（MI300X，DeepSeek-R1）
- **Finding 4**: MTP 投机解码生产环境衰减（+60.8% → +14.2%）
- **Finding 5**: Megatron-LM TP 通信量公式
- **Finding 6**: Mooncake RDMA 传输实测（87-190 GB/s）
- **Finding 7**: SGLang HiCache HiRadixTree 三级页表架构
- 16 条被否决声明 + 否决原因
- 25 篇完整来源列表

### [Part 7: GPU 架构与 CUDA 编程基础](part-07-gpu-cuda-basics.md) 🆕
- NVIDIA GPU 架构演进（Ampere / Hopper / Blackwell）
- GPU 核心组件：SM、Tensor Core、HBM、NVLink
- CUDA 基础：Kernel / Grid / Block / Thread / Warp
- 推理性能调优 Checklist

### [Part 8: 生产部署与运维](part-08-production.md) 🆕
- Benchmark 方法（vLLM / SGLang 工具、负载模型、结果解读）
- 关键部署参数
- 监控与可观测性
- 常见故障排查（OOM / 高 TTFT / 高 TPOT / 输出错误）
- 上线 checklist

### [Part 9: 术语表与索引](part-09-glossary.md) 🆕
- A-Z 术语表
- 按主题归类索引

### [Part 10: 推理引擎生态深度对比](part-10-engine-ecosystem.md) 🆕
- vLLM / SGLang / TensorRT-LLM / TGI / LMDeploy 逐引擎分析
- 横向对比矩阵（延迟/吞吐/易用性/生态/稳定性）
- 选型决策树 + 迁移成本

### [Part 11: 具体模型部署案例](part-11-model-case-studies.md) 🆕
- DeepSeek-V3/R1 (MoE + MLA)
- LLaMA-3-405B (Dense)
- Qwen3-235B-A22B (MoE)
- LLaMA-3.1-70B (中小规模)
- 模型部署决策模板

### [Part 12: 国产 GPU 与异构硬件适配](part-12-domestic-gpu.md) 🆕
- 华为昇腾、摩尔线程、海光 DCU、AMD ROCm
- 适配路径与关键参数
- 跨硬件迁移 checklist

### [Part 13: 成本模型与 ROI 分析](part-13-cost-roi.md) 🆕
- $/1M tokens 计算模型
- 硬件/电费/折旧/人力成本
- 不同引擎 TCO 对比
- ROI 决策框架与成本优化策略

### [Part 14: 安全、对齐与合规](part-14-security-compliance.md) 🆕
- 内容安全与过滤（输入/输出）
- 提示注入防护
- 隐私保护与水印
- 国内/国际合规要求

---

## 约定

| 标记 | 含义 |
|------|------|
| ★ | 面试高频，必须掌握 |
| ☆ | 加分项，了解即可 |
| `code` | 源码中实际存在的类/函数/变量名 |
| **粗体** | 核心概念，首次出现 |

## 维护

- 最后更新：2026-07-02
- 基于源码版本：vLLM V1 引擎开发主线、SGLang 最新主线
- Profile 岗位参考：`profile.md` — 训推平台及引擎研发主任工程师
