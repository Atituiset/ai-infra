# AI Infra 推理引擎知识全书 — 设计文档

## 背景与目标

基于 `profile.md` 中训推平台及引擎研发主任工程师岗位能力模型，结合 vLLM、SGLang 本地源码与互联网最新资料，整理一部面向面试备战与团队知识库双用途的 mbook。内容需覆盖 AI Infra 推理领域从入门到进阶的核心知识点，并对 vLLM/SGLang 源码进行深度解剖。

## 交付形式

一次性交付完整 mbook，存放于 `docs/ai-infra-handbook/`。

## 目录结构

```
docs/ai-infra-handbook/
├── README.md                 # 全书导航与使用说明
├── part-00-overview.md       # AI Infra 全景图
├── part-01-fundamentals.md   # 入门篇
├── part-02-advanced.md       # 进阶篇
├── part-03-vllm-source.md    # vLLM 源码解剖
├── part-04-sglang-source.md  # SGLang 源码解剖
└── part-05-interview.md      # 面试实战与自检
```

## 各 Part 内容要点

### Part 0：AI Infra 全景图
- AI Infra 分层：芯片层、系统软件层、框架层、平台层
- 训练 vs 推理：计算特征、优化目标、系统差异
- 推理引擎生态：vLLM、SGLang、TensorRT-LLM、TGI、LMDeploy 等定位与对比
- 从单机到集群：部署形态演进

### Part 1：入门篇
- Transformer 推理基础：自回归生成、KV Cache、Attention 计算量分析
- 推理性能指标：TTFT、TPOT、Throughput、Latency、Goodput
- 批处理：static batching vs continuous batching
- 内存与显存：HBM、页式内存、内存墙
- 量化基础：INT4/INT8/FP8、AWQ、GPTQ、SmoothQuant
- 解码策略：greedy、beam search、sample、speculative decoding

### Part 2：进阶篇
- Continuous Batching 与调度：vLLM Scheduler、SGLang Scheduler、chunked prefill
- 并行策略：TP、PP、DP、EP、SP，MoE 专家并行
- 分离式架构：PD Disaggregation、Mooncake KV Transfer、HiCache 三级缓存
- Attention 优化：PagedAttention、FlashAttention、FlashInfer、MLA
- 长文本与多模态：上下文窗口扩展、RoPE、YaRN、多模态输入处理
- 量化与压缩：FP8/INT4 推理、KV Cache 量化、MTP
- 投机解码：lookahead、Medusa、Eagle、MTP

### Part 3：vLLM 源码解剖
- 架构总览：LLMEngine、Scheduler、Worker、ModelRunner、Executor
- Scheduler 实现：continuous batching、preemption、swapping、recomputation
- PagedAttention：block manager、KV cache 分配、prefix caching
- Model Runner：模型加载、执行流程、attention backend（flash_attn/xformers/cuda）
- 分布式：Ray/PyTorch SPMD executor、TP/PP 实现
- 最新特性：V1 engine、prefix caching、multi-step scheduling、LoRA/SpecDecode

### Part 4：SGLang 源码解剖
- 架构总览：Runtime、Scheduler、Tokenizer Manager、Detokenizer、Model Runner
- RadixAttention：radix tree、prefix matching、cache reuse
- Schedule Policy：jump-forward、fast-forward、overlap scheduler
- Model Runner：SGLang 执行模型、attention backend、flashinfer 集成
- 编译与图优化：torch.compile、AST、structured generation
- 多模态与长文本：image token、long context、MMLU 等场景优化

### Part 5：面试实战与自检
- 高频面试题：按入门/进阶分类
- 源码定位题：vLLM/SGLang 关键类/函数
- 场景设计题：千卡集群部署、PD 分离、MoE 部署
- 自检 Checklist：岗位能力对照表

## 质量要求

- 每个知识点说明「为什么存在、解决什么问题、trade-off 是什么」
- vLLM/SGLang 部分需给出源码路径与关键类/函数名
- 进阶内容覆盖 profile 中提到的 PD Disaggregation、Mooncake、HiCache、TP/PP/EP、MoE、INT4/FP8、MTP/投机解码
- 区分「必须掌握」与「加分项」

## 研究方法

- 本地源码阅读：vLLM (`/home/atituiset/Projects/ai-infra-collect/vllm/`)、SGLang (`/home/atituiset/Projects/ai-infra-collect/sglang/`)
- 互联网深度检索：使用 `deep-research` 对最新架构、论文、博客、 release notes 进行多源验证
- 交叉验证：同一概念在不同来源间核对，标注不确定点
