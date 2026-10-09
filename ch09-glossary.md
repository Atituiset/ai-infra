# 第9章 术语表与索引

> **用途**：快速查阅 AI Infra 领域关键术语  
> **使用方式**：按字母顺序查找，或在各章中点击跳转

---

## A

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **AWQ**
| **Activation Recomputation** | - | 激活重计算：不存中间激活、反向重算以省显存 | 第23章 §23.2 | | Activation-aware Weight Quantization | 激活感知权重量化 | 第1章 §1.5.3, 第2章 §2.7 |
| **AllReduce** | - | 集合通信操作，各节点归约后广播 | 第2章 §2.2 |
| **AllToAll** | - | 集合通信操作，每个节点向所有节点发送数据 | 第2章 §2.2.3 |
| **Attention** | - | Transformer 核心机制 | 第1章 §1.1 |

## B

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Batching**
| **BF16** | Brain Floating Point 16 | bfloat16 格式，指数位同 FP32，训练默认 | 第23章 §23.4 | | - | 批处理 | 第1章 §1.3 |
| **Block Table** | - | PagedAttention 中虚拟 block 到物理 block 的映射 | 第2章 §2.4.1, 第3章 §3.3 |
| **Breakable CUDA Graph** | - | vLLM V1 中支持动态断点的 CUDA Graph | 第3章 §3.4.2 |

## C

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Continuous Batching** | - | 连续批处理 | 第1章 §1.3.2, 第2章 §2.1 |
| **Constrained Decoding** | - | 受限解码：用语法/Schema 在采样时屏蔽非法 token（XGrammar/Outlines） | 第18章 §18.5 |
| **Chunked Prefill**
| **Custom Op** | - | 引擎注册的自定义算子（vLLM custom_ops / sgl-kernel） | 第24章 §24.4 | | - | 将长 prompt 切分为多个 chunk 的 prefill | 第1章 §1.3.3, 第2章 §2.1 |
| **CUDA Graph** | - | 记录 CUDA kernel 序列并复用 | 第3章 §3.4.2, 第7章 §7.2 |
| **CP** | Context Parallelism | 上下文并行 | 第1章 §1.1.4, 第2章 §2.2.4 |

## D

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Decode**
| **Distillation** | Knowledge Distillation | 知识蒸馏：教师模型能力转移到小模型 | 第22章 §22.4 | | - | 自回归生成阶段（每次 1 token） | 第1章 §1.1.2 |
| **DLA** | Deep Learning Accelerator | 专用深度学习加速器，常与 GPU 共享内存 | 第15章 §15.2.1 |
| **DRA** | Dynamic Resource Allocation | K8s 声明式、拓扑感知的设备分配框架，逐步取代 device plugin | 第18章 §18.2.2 |
| **DP** | Data Parallelism | 数据并行 | 第2章 §2.2.4 |
| **DP Attention** | Data Parallel Attention | vLLM 针对 MLA 的按请求分区 attention | 第6章 Finding 1-2 |
| **dFlash** | draft-Flash | in-filling 式块草稿：整块并行出 draft（SGLang 首创，vLLM 已移植） | 第2章 §2.6.4, 第4章 §4.6.3 |
| **DSpark** | - | 半自回归草稿模型：dFlash 并行骨干 + 低秩 Markov 头注入块内依赖 | 第2章 §2.6.4, 第4章 §4.6.3 |
| **D-LLM** | Diffusion LLM | 面向 LLaDA / SDAR 等扩散式 LLM 的推理调度 | 第4章 §4.10 |
| **DSA** | DeepSeek Sparse Attention | DeepSeek V3.2 的分层稀疏注意力（Indexer + top-k 选择） | 第2章 §2.4.4, 第4章 §4.10 |

## E

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Eagle** | - | 投机解码算法，用轻量 draft model | 第2章 §2.6.2, 第3章 §3.5, 第4章 §4.6 |
| **Edge TPU** | - | Google 推出的低功耗 AI 加速器，主要用于 TFLite/LiteRT 生态 | 第16章 §16.5.3 |
| **EP** | Expert Parallelism | 专家并行（MoE 专用） | 第2章 §2.2.3 |
| **Elastic EP** | - | 运行期动态增删专家组的并行方案 | 第4章 §4.10 |
| **EPLB** | Expert Parallel Load Balancing | MoE 专家并行负载均衡（SGLang 按访问分布重排） | 第4章 §4.10 |

## F

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **FlashAttention** | - | IO-aware exact attention | 第1章 §1.5, 第2章 §2.4.2 |
| **FSDP** | Fully Sharded Data Parallel | PyTorch 原生全分片数据并行（ZeRO-3 思想） | 第23章 §23.7 |
| **FlashInfer** | - | 针对推理优化的 attention kernel 库 | 第2章 §2.4.2, 第4章 §4.4 |
| **FP8** | - | 8-bit floating point（E4M3/E5M2） | 第1章 §1.5.4, 第2章 §2.7 |

## G

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Gang Scheduling** | - | 成组调度：作业内全部 Pod 就位才启动（Volcano/Kueue） | 第18章 §18.2.3 |
| **GGUF** | Georgi Gerganov Universal Format | llama.cpp 使用的二进制模型格式，支持按张量选择量化类型 | 第16章 §16.3.2 |

| **Goodput** | - | 有效吞吐（成功请求的吞吐） | 第1章 §1.2 |
| **GPTQ** | - | 基于 Hessian 的权重量化 | 第1章 §1.5.2 |
| **GQA** | Grouped Query Attention | 分组查询注意力 | 第1章 §1.1.4 |
| **GPU** | Graphics Processing Unit | 图形处理器 | 第7章 |

## H

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **H2O** | Heavy-Hitter Oracle | 通过保留 heavy-hitter token 和最近窗口压缩 KV Cache | 第17章 §17.1.3 |
| **HBM** | High Bandwidth Memory | 高带宽显存 | 第1章 §1.4, 第7章 §7.1 |
| **HiCache** | - | SGLang 三级缓存架构 | 第2章 §2.3.4, 第4章 §4.5.3, 第6章 Finding 7 |
| **HiRadixTree** | - | HiCache 的跨三级页表结构 | 第6章 Finding 7 |
| **HTP** | Hexagon Tensor Processor | Qualcomm Hexagon NPU 上的张量处理单元，QNN 常用后端 | 第16章 §16.5.3 |

## I

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **In-flight Batching** | - | 在单条序列生成过程中动态拼接新请求的批处理策略 | 第16章 §16.2.4 |

## J

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Jetson** | - | NVIDIA 边缘 AI 计算平台系列（Orin Nano/NX/AGX/Thor） | 第15章 §15.2.1 |

## K

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Kernel Fusion** | - | 算子融合：合并算子减少 HBM 往返（垂直/水平/改数学三型） | 第24章 §24.5 |
| **KV Cache** | Key-Value Cache | Transformer 推理中存储历史 K/V 的缓存 | 第1章 §1.1.3 |
| **KV Cache Manager** | - | vLLM 中管理 block 分配的组件 | 第3章 §3.3 |
| **KV Cache Quantization** | - | 对 KV Cache 做 INT8/INT4 等低精度量化以节省内存 | 第17章 §17.1.4 |
| **KV Connector** | - | vLLM V1 中用于 PD 分离的 KV 传输连接器 | 第3章 §3.2.4 |
| **KV Offload / Tiering** | - | 显存不足时将 KV Cache 卸载到 CPU 或多级存储（FS/P2P） | 第2章 §2.5.2, 第3章 §3.7 |
| **KV Canary** | - | SGLang 的 KV Cache 完整性校验/扰动注入工具 | 第4章 §4.10 |

## L

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Low-Rank Decomposition** | - | 低秩分解压缩 | 第22章 §22.5 |
| **LLM** | Large Language Model | 大语言模型 | 全书 |
| **LPDDR** | Low-Power Double Data Rate | 低功耗双倍数据速率内存，边缘设备常用共享内存类型 | 第15章 §15.1.2 |
| **LRU** | Least Recently Used | 最近最少使用驱逐策略 | 第4章 §4.2.4 |

## M

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **MLA** | Multi-head Latent Attention | DeepSeek 的压缩 KV attention | 第2章 §2.4.3, 第6章 Finding 1 |
| **MNN** | - | 阿里巴巴开源的轻量级端侧推理框架 | 第16章 §16.4 |
| **MoE** | Mixture of Experts | 混合专家模型 | 第2章 §2.2.3 |
| **MTP** | Multi-Token Prediction | 多 token 预测（投机解码） | 第2章 §2.6.3, 第6章 Finding 4 |
| **Memory Wall**
| **Mixed Precision** | - | 混合精度训练（FP16/BF16/FP8 + FP32 主权重） | 第23章 §23.4 |
| **MLIR** | Multi-Level Intermediate Representation | 可组合编译器基础设施 | 第24章 §24.2 | | - | 内存带宽成为性能瓶颈 | 第1章 §1.4.2 |
| **MCP** | Model Context Protocol | Anthropic 开源的模型-工具连接标准（Tools/Resources/Prompts） | 第19章 §19.5 |
| **MIG** | Multi-Instance GPU | NVIDIA 硬件级单卡分区（最多 7 实例），显存/故障域隔离 | 第18章 §18.2.4 |
| **MPS** | Multi-Process Service | NVIDIA 多进程共享 GPU 的服务，SM 配额隔离 | 第18章 §18.2.4 |

## N

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **NPU** | Neural Processing Unit | 神经网络处理单元，边缘设备常见 AI 加速器 | 第15章 §15.2.2, 第17章 §17.4 |
| **NVLink** | - | NVIDIA 高速卡间互联 | 第1章 §1.4, 第7章 §7.1 |
| **NVL72 / Scale-up Domain** | - | NVLink5+NVSwitch 组成的 72 卡高带宽域（GB200） | 第20章 §20.5 |
| **NSA** | Native Sparse Attention | DeepSeek V3.2 的稀疏注意力：压缩块 + 选择块 + 滑动窗口 | 第2章 §2.4.4, 第4章 §4.10 |

## O

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Overlap Scheduling** | - | SGLang 的 CPU/GPU 并行调度 | 第2章 §2.1.2, 第4章 §4.3 |
| **OpenTelemetry GenAI** | - | LLM 可观测性的语义约定（gen_ai.* 属性与指标） | 第18章 §18.8 |

## P

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **PagedAttention** | - | vLLM 的页式 KV Cache 管理 | 第2章 §2.4.1, 第3章 §3.3 |
| **PD 分离** | Prefill-Decode Disaggregation | 将 prefill 和 decode 分离到不同节点 | 第1章 §0.4, 第2章 §2.3 |
| **Partial Rollout** | - | RL 中长轨迹在工具等待点挂起、跨 step 续跑的采样方式 | 第19章 §19.4.5 |
| **Pipeline Bubble** | - | PP 中 GPU 等待数据的时间 | 第2章 §2.2.2 |
| **PP** | Pipeline Parallelism | 流水线并行 | 第2章 §2.2.2 |
| **Prefill** | - | 处理 prompt 阶段 | 第1章 §1.1.2 |
| **Prefix Caching**
| **Pruning** | - | 剪枝：非结构化 / 2:4 半结构化 / 结构化三路线 | 第22章 §22.3 | | - | 缓存并复用公共前缀的 KV | 第2章 §2.4.1, 第3章 §3.3.3 |

## Q

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **QNN** | Qualcomm Neural Network | Qualcomm 神经网络 SDK，面向 Hexagon NPU / HTP | 第15章 §15.2.2, 第16章 §16.5.3 |
| **QPS** | Queries Per Second | 每秒查询数 | 第1章 §1.2 |
| **Quantization**
| **QAT** | Quantization-Aware Training | 量化感知训练，插入 fake-quant 节点训练 | 第22章 §22.2.1 | | - | 量化 | 第1章 §1.5, 第2章 §2.7 |

## R

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **RadixAttention** | - | SGLang 的前缀树式 KV Cache | 第2章 §2.1.2, 第4章 §4.2 |
| **RDMA** | Remote Direct Memory Access | 远程直接内存访问 | 第2章 §2.3.3 |
| **RLVR** | Reinforcement Learning with Verifiable Rewards | 用可验证奖励做 RL 后训练（R1 范式） | 第19章 §19.4.1 |
| **Rollout Engine** | - | RL 中负责策略采样的推理引擎（复用 vLLM/SGLang） | 第19章 §19.4.2-19.4.3 |
| **Ring Attention** | - | 序列维环形分块的 CP 实现，支撑超长上下文训练 | 第23章 §23.3 |
| **RKNN** | - | 瑞芯微为其 NPU 提供的模型转换与运行时工具链 | 第15章 §15.3.2, 第16章 §16.5.4 |
| **RoPE** | Rotary Position Embedding | 旋转位置编码 | 第2章 §2.5.1 |

## S

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Scheduler** | - | 推理引擎调度器 | 第2章 §2.1 |
| **S-LoRA** | - | 单 base model 服务数千 LoRA adapter 的系统（Unified Paging） | 第18章 §18.4 |
| **Semantic Cache** | - | 网关层按语义命中历史答案直接返回的缓存 | 第18章 §18.3.2 |
| **Session Affinity / Sticky Routing** | - | 同一会话固定路由到同一副本以最大化 prefix 命中 | 第19章 §19.3.2, 第18章 §18.3.3 |
| **SGLang** | - | 高性能结构化 LLM 编程/推理框架 | 全书 |
| **Shared Memory** | - / 共享内存 | CPU、GPU/NPU 共用同一颗 LPDDR 的内存架构 | 第15章 §15.1.2 |
| **SM** | Streaming Multiprocessor | NVIDIA GPU 流式多处理器 | 第7章 §7.1 |
| **SP** | Sequence Parallelism | 序列并行 | 第2章 §2.2.4 |
| **Speculative Decoding** | - | 投机解码 | 第1章 §1.6.2, 第2章 §2.6 |
| **Static Batching** | - | 静态批处理 | 第1章 §1.3.1 |
| **StreamingLLM** | - | 通过保留 sink token 和最近窗口实现超长序列推理 | 第17章 §17.1.3 |
| **Swapping** | - | 将 KV Cache 换出到 CPU | 第2章 §2.1.1 |

## T

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Tensor Core** | - | NVIDIA GPU 的矩阵乘单元 | 第7章 §7.1 |
| **Tensor Parallelism (TP)** | - | 张量并行 | 第2章 §2.2.1 |
| **Throughput** | - | 吞吐量 | 第1章 §1.2 |
| **Top-p / Top-k** | - | 采样策略 | 第1章 §1.6.1 |
| **TPOT**
| **torch.compile** | - | PyTorch 编译栈（Dynamo 图捕获 + Inductor 代码生成） | 第24章 §24.2-24.3 |
| **TVM** | Tensor Virtual Machine | 开源深度学习编译栈（OSDI'18） | 第24章 §24.2 |
| **Triton** | - | OpenAI 的 GPU kernel Python DSL | 第20章 §20.3, 第24章 | | Time Per Output Token | 每输出 token 时间 | 第1章 §1.2 |
| **TTFT** | Time To First Token | 首 token 时间 | 第1章 §1.2 |
| **Test-time Compute** | - | 推理时增加计算换取质量（长 CoT/best-of-N），o1/R1 范式 | 第19章 §19.2 |

## U

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **UALink** | Ultra Accelerator Link | AMD 阵营的加速器 scale-up 互联标准（spec 1.0 于 2025 发布） | 第20章 §20.5 |
| **UEC** | Ultra Ethernet Consortium | 改造以太网 RDMA 面向 AI 集群的标准组织（spec 1.0 于 2025-06） | 第20章 §20.5 |

## V

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **vLLM** | - | 开源 LLM 推理引擎 | 全书 |

## W

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **Warp** | - | 32 个 thread 的 SIMD 执行单元 | 第7章 §7.2.2 |
| **Wide-EP** | - | vLLM 的 DP+EP 混合并行架构 | 第6章 Finding 2 |
| **Weight Sync** | - | RL 训练后把新权重同步进 rollout engine（NCCL broadcast/CUDA IPC） | 第19章 §19.4.4 |
| **Wujian** | 无剑 | 平头哥 SoC 平台，配套玄铁 RISC-V 处理器 | 第15章 §15.3.1 |

## X

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **XuanTie** | 玄铁 | 平头哥 RISC-V 处理器系列 | 第15章 §15.3.1 |
| **XLA** | Accelerated Linear Algebra | Google 线性代数编译器，JAX/TPU 底座 | 第24章 §24.2 |

## Z

| 术语 | 英文全称 | 含义 | 所在章节 |
|------|---------|------|----------|
| **ZeRO** | Zero Redundancy Optimizer | 优化器状态/梯度/权重三级分片消除 DP 冗余 | 第23章 §23.2 |

---

## 索引表：按主题归类

### 核心概念
- AI Infra → 第0章
- Transformer 推理 → 第1章 §1.1
- KV Cache → 第1章 §1.1.3
- Prefill / Decode → 第1章 §1.1.2
- Continuous Batching → 第1章 §1.3.2

### 性能指标
- TTFT / TPOT / Throughput / Goodput → 第1章 §1.2
- Benchmark → 第8章 §8.1

### 并行策略
- TP / PP / DP / EP / SP / CP → 第2章 §2.2
- Wide-EP / DP Attention → 第6章 Finding 1-2

### 优化技术
- PagedAttention → 第2章 §2.4.1
- RadixAttention → 第2章 §2.1.2, 第4章 §4.2
- FlashAttention / FlashInfer → 第2章 §2.4.2
- MLA → 第2章 §2.4.3
- 量化 (FP8/AWQ/GPTQ) → 第1章 §1.5, 第2章 §2.7
- 投机解码 → 第1章 §1.6.2, 第2章 §2.6
- 压缩四件套 → 第22章
- 训练系统 → 第23章
- 编译器 / 算子 → 第24章
- 多模态生成 / 检索 → 第25章

### 高级架构
- PD 分离 → 第2章 §2.3
- Mooncake → 第2章 §2.3.3, 第6章 Finding 6
- HiCache → 第2章 §2.3.4, 第6章 Finding 7

### 源码
- vLLM → 第3章
- SGLang → 第4章

### 自测
- 高频题 → 第5章 §5.1
- 场景题 → 第5章 §5.3
- 自检 → 第5章 §5.4

### 硬件基础
- GPU / CUDA → 第7章

### 边缘 AI
- 边缘硬件 / Jetson / NPU / DLA / LPDDR / 共享内存 / 玄铁 / 无剑 → 第15章
- 边缘框架 / GGUF / QNN / RKNN / MNN / Edge TPU / In-flight Batching → 第16章
- 边缘优化 / StreamingLLM / H2O / KV Cache Quantization → 第17章

### 生产运维
- 部署 / 监控 / 故障排查 → 第8章
