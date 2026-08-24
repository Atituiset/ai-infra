# Part 0：AI Infra 全景图

> **面向角色**：推理引擎研发工程师、AI Infra 架构师、GPU 集群运维  
> **前置知识**：基本了解 Transformer、LLM 推理概念  
> **目标**：建立 AI Infra 领域的全局认知，理解分层架构和各组件定位

---

## 0.1 AI Infra 是什么？

AI Infra（AI Infrastructure）是在裸 GPU 和终端用户之间搭建的**完整推理/训练服务栈**。它不是单一系统，而是从芯片指令到 API 网关的多层抽象。

```
   ┌─────────────────────────────────────┐
   │  边缘 / 端侧设备                      │  ← 边缘层
   │  (Jetson / 手机 / 车载 / IoT)         │
   └────┬────────────────────────────────┘
        │  请求 / 数据 / 推理结果 / 模型更新
        ▼
   ┌─────────────────────────────────────┐
   │  API 网关 / 负载均衡                  │  ← 平台层
   │  (Nginx, Envoy, K8s Ingress)         │
   └────┬─────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────┐
   │  推理引擎 (vLLM / SGLang / TRT-LLM)   │  ← 框架层 ★ 本书记重点
   │  · Scheduler · KV Cache · Executor   │
   └────┬─────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────┐
   │  运行时 & 编译器                      │  ← 系统软件层
   │  (CUDA / ROCm / Triton / torch.compile)│
   └────┬─────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────┐
   │  GPU 硬件 (H100 / B200 / A100 ...)    │  ← 芯片层
   │  · HBM · SM · Tensor Core · NVLink   │
   └──────────────────────────────────────┘
```

**核心矛盾**：LLM 推理是 **memory-bound**（受 HBM 带宽限制），而训练是 **compute-bound**（受计算吞吐限制）。因此推理引擎优化重心在**如何减少 HBM 读写**，而不是堆算力。图中的**边缘层**与数据中心不是互斥关系：很多场景先在端侧做预处理或轻量推理，再把结果回传到云端完成复杂生成，形成“云边协同”的完整链路。

---

## 0.2 训练 vs 推理：两个世界

| 维度 | 训练 (Training) | 推理 (Inference) |
|------|-----------------|------------------|
| **计算模式** | 一次处理整个序列（teacher forcing） | 逐 token 自回归生成 |
| **并行策略** | DP + TP + PP + ZeRO | TP + PP + EP（无梯度通信） |
| **内存瓶颈** | 优化器状态 + 梯度 + 激活值 | KV Cache（历史 Key/Value 缓存） |
| **延迟要求** | 吞吐优先（几小时/几天） | 延迟敏感（TTFT < 500ms, TPOT < 50ms） |
| **批处理** | 固定 batch size | Continuous Batching（动态进出） |
| **量化** | 训练中（QLoRA）或训练后（PTQ） | PTQ → AWQ/GPTQ/FP8 |

**关键洞察**：推理引擎不能复用训练框架（如 Megatron-LM / DeepSpeed）的架构，因为它们面向的是完全不同的硬件瓶颈和服务目标。推理需要**显式的 KV Cache 管理**和**动态调度**，这是 vLLM/SGLang 存在的根本原因。

---

## 0.3 推理引擎生态对比

### 主流引擎定位

| 引擎 | 特点 | 适用场景 | 语言/生态 |
|------|------|----------|-----------|
| **vLLM** | 社区最大、PagedAttention 先驱、V1 引擎重构中 | 通用推理服务 | Python/C++/CUDA |
| **SGLang** | RadixAttention、overlap 调度、编译优化 | 高吞吐 + 长 prompt 场景 | Python/C++/CUDA |
| **TensorRT-LLM** | NVIDIA 官方、闭源 C++ runtime、极致优化 | 英伟达 GPU 专用 | C++/Python |
| **TGI (HuggingFace)** | HF 生态集成好、watermarking 等特色功能 | HF 用户首选 | Rust/Python |
| **LMDeploy** | TurboMind C++ 引擎、低延迟 | 国产 GPU 兼容好 | C++/Python |

### 选型建议（2025-2026）

- **追求社区与通用性** → vLLM（模型支持最广、bug fix 最快）
- **追求吞吐与前沿优化** → SGLang（RadixAttention、overlap 调度）
- **追求极致延迟与英伟达锁定** → TensorRT-LLM（闭源、kernel 级优化）
- **追求 HuggingFace 兼容** → TGI（Lora adapter 热加载、水印）
- **国产 GPU / 定制硬件** → LMDeploy / SGLang（硬件 backend 抽象更好）

---

## 0.4 从单卡到集群：部署形态演进

### Level 1：单卡 (Single GPU)

```
GPU 0: [模型完整权重] + [KV Cache]
```

适用：7B 模型 FP16（~14GB）+ KV Cache（~4GB）= 18GB，一张 A100-80G 可承载。

### Level 2：TP（Tensor Parallelism）

```
GPU 0: [Attention Head 0-31, FFN 分片 0]
GPU 1: [Attention Head 32-63, FFN 分片 1]
```

适用：70B 模型 (≈140GB)，2→8 卡。每层权重按列/行切分，层内通信（AllReduce / AllGather）。

### Level 3：PP（Pipeline Parallelism）

```
GPU 0: [Layers 0-19]
GPU 1: [Layers 20-39]
GPU 2: [Layers 40-59]
GPU 3: [Layers 60-79]
```

适用：极深模型（176B+），减小通信粒度（仅层边界），但引入 pipeline bubble。

### Level 4：PD 分离 (Prefill-Decode Disaggregation)

```
  P 节点 (A100 x8)          D 节点 (H100 x2)
  [高计算→prefill]  ──KV Cache──▶  [高带宽→decode]
       ▲                              ▲
       │         Router / Gateway     │
       └────────────┬─────────────────┘
                    │
              用户请求
```

适用：生产级大规模服务。P 节点用高算力卡（A100），D 节点用高带宽卡（H100），通过 Mooncake / NCCL / RDMA 传输 KV Cache。

### Level 5：多模型集群 + HiCache

```
┌─────────────────────────────────────────────┐
│              Global KV Cache Store            │
│  (HiCache / Mooncake / 分布式共享内存)         │
└───┬──────────────┬──────────────┬────────────┘
    │              │              │
  P Pool        D Pool A      D Pool B
  (prefill)     (decode-A)    (decode-B)
```

适用：多模型共池、跨请求 Cache 复用、弹性伸缩。

---

## 0.5 推理引擎的内部架构（通用模板）

无论 vLLM 还是 SGLang，推理引擎的内部都可以抽象为以下流水线：

```
Request In
    │
    ▼
┌──────────────┐
│  Tokenizer    │  Tokenize → token_ids
│  Manager      │  (异步/多线程)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Scheduler    │  1. 检查 KV Cache 容量
│  (CPU)        │  2. 决定 batch 组成
│               │  3. 分配 KV Cache blocks
│               │  4. 处理 preemption
└──────┬───────┘
       │ SchedulerOutput (batch metadata)
       ▼
┌──────────────┐
│  Model Runner │  1. 准备 GPU Tensor（输入、位置、block table）
│  (GPU)        │  2. 执行模型 forward
│               │  3. 采样（greedy / top-p / beam）
└──────┬───────┘
       │ ModelRunnerOutput (next token ids)
       ▼
┌──────────────┐
│  Detokenizer  │  token_ids → 文本
└──────┬───────┘
       │
       ▼
Response Out (streaming / non-streaming)
```

### 关键设计决策

1. **Scheduler 和 Model Runner 分离**：Scheduler 在 CPU 上运行（逻辑复杂、需要全局信息）；Model Runner 在 GPU 上运行（计算密集、需要低延迟）。

2. **KV Cache 管理器独立**：KV Cache 是推理的核心资源，必须显式管理——不是 torch allocator 的事，而是需要**页式内存管理**和**前缀复用**的独立组件。

3. **Batch 元数据 GPU 化**：SchedulerOutput 中的 block_tables、position_ids 等元数据需要拷贝到 GPU，供 attention kernel 使用。

---

## 0.6 本书导航

| Part | 内容 | 适用读者 |
|------|------|----------|
| Part 1 | 入门篇：核心概念与基本工作原理 | 新人入职、基础面试 |
| Part 2 | 进阶篇：高性能推理架构与优化 | 高级面试、架构设计 |
| Part 3 | vLLM 源码深度解剖 | 源码面试、二次开发 |
| Part 4 | SGLang 源码深度解剖 | 源码面试、二次开发 |
| Part 5 | 面试实战与自检清单 | 面试备战、能力对标 |
| Part 6 | Deep Research 研究报告 | 面试备战、能力对标 |
| Part 7 | GPU 架构与 CUDA 编程基础 | 新人入职、基础面试 |
| Part 8 | 生产部署与运维 | 上线负责人、运维工程师 |
| Part 9 | 术语表与索引 | 所有人、快速查阅 |
| Part 10 | 推理引擎生态深度对比 | 技术选型、架构设计 |
| Part 11 | 具体模型部署案例 | 部署方案设计、面试实战 |
| Part 12 | 国产 GPU 与异构硬件适配 | 国产化项目、异构迁移 |
| Part 13 | 成本模型与 ROI 分析 | 技术选型、预算规划 |
| Part 14 | 安全、对齐与合规 | 上线负责人、安全工程师 |
| Part 15 | 边缘 AI 硬件地图与约束 | 边缘 AI 工程师、硬件选型 |
| Part 16 | 边缘推理框架与软件栈 | 端侧部署、框架集成 |
| Part 17 | 边缘场景优化与落地 | 边缘优化、生产落地 |
| Part 18 | 引擎之上：网关、调度与集群系统层 | 平台工程师、SRE、架构师 |
| Part 19 | Agentic 与 RL 时代的推理负载 | 架构师、RL Infra 工程师 |
| Part 20 | 技术编年史：AI Infra 演进脉络 | 所有人、面试叙事 |
| Part 21 | 论文对照地图（Paper Map） | 研究者、源码二开、面试深挖 |
| Part 22 | 模型压缩全景：量化/剪枝/蒸馏/稀疏化 | 部署工程师、端侧 AI 工程师 |
| Part 23 | 训练 Infra 速览 | 训推平台工程师 |
| Part 24 | AI 编译器与算子开发 | 引擎/内核研发工程师 |
| Part 25 | 多模态生成与检索服务 | 多模态/RAG 平台工程师 |

---

## 0.7 云侧 vs 边缘侧：同一问题，两种解法

把大模型从数据中心搬到边缘设备，不是简单地把 vLLM 编译成 ARM 版本，而是**硬件约束和服务目标的根本切换**。

| 维度 | 云侧 (Data Center) | 边缘侧 (Edge/Endpoint) |
|------|-------------------|------------------------|
| **算力** | 数百到数千 TFLOPS（H100 / B200） | 数到数十 TFLOPS（Jetson / NPU / 手机 SoC） |
| **内存** | 数十 GB HBM，可扩展多卡 | 数 GB 到十几 GB，固定不可扩 |
| **功耗** | 数百瓦每卡，有主动散热 | 数瓦到数十瓦，无风扇或电池供电 |
| **互联** | NVLink / RDMA / 高速网络 | PCIe / LPDDR / 片上总线，带宽有限 |
| **延迟要求** | 网络传输占主导，TTFT / TPOT 为百毫秒级 | 端侧交互，通常要求十毫秒级甚至实时 |

**云侧优化核心**：在算力和显存充足的前提下，最大化 **HBM 带宽利用率** 和 **多 GPU 通信效率**。因此云侧引擎重点做 TP / PP / EP、PD 分离、Continuous Batching、投机解码、FP8 / BF16 高吞吐计算。

**边缘侧优化核心**：在算力、内存、功耗三重受限下，让模型“塞得下、跑得动、耗得起”。因此边缘引擎重点做 **INT4 / INT8 / FP16 量化**、**算子兼容性适配**、**内存容量精打细算**、**异构调度（CPU / GPU / NPU / DSP）** 以及 **TinyLLM / 蒸馏 / 剪枝**。

本书后续 **Part 15** 会系统梳理边缘硬件地图与约束，**Part 16** 深入边缘推理框架（TensorRT-LLM、llama.cpp、MNN 等），**Part 17** 则聚焦边缘场景优化与落地实践。云侧和边缘侧不是非此即彼，而是同一 AI Infra 问题在两种资源条件下的不同解法。

---

## 附录：重要论文索引

> 完整的论文对照地图（15 条主题线 + 十篇最小必读集）见 **Part 21**。

| 论文 | 贡献 | 年份 |
|------|------|------|
| Attention Is All You Need | Transformer 架构 | 2017 |
| FlashAttention (Dao et al.) | IO-aware exact attention | 2022 |
| PagedAttention (vLLM) | 虚拟内存式 KV Cache 管理 | 2023 |
| Efficient Memory Management for LLM Serving (PagedAttention) | vLLM 核心论文 | 2023 |
| FlashAttention-2 / 3 | 更快、更省的注意力计算 | 2023/2024 |
| SGLang: Efficient Execution of Structured LM Programs | RadixAttention + SGLang DSL | 2024 |
| Splitwise (Mooncake) | PD 分离 + 高效 KV 传输 | 2024 |
| AWQ: Activation-aware Weight Quantization | 权重-only 量化 | 2023 |
| GPTQ: Accurate Post-Training Quantization | 权重-only 量化 | 2023 |
| SmoothQuant | W8A8 量化 | 2023 |
| Medusa / Eagle / MTP | 投机解码 | 2024 |
| DeepSeek-V2/V3 | MLA + MoE + Multi-Token Prediction | 2024 |
| DeepSeek NSA / V3.2 DSA | 原生稀疏注意力（压缩块 + 选择块 + Indexer） | 2025 |
| FastMTP / Dynamic Speculative Decoding | 生产环境投机解码优化 | 2025 |
