# Part 9 补充：术语表与索引

> **用途**：快速查阅 AI Infra 领域关键术语  
> **使用方式**：按字母顺序查找，或在各 Part 中点击跳转

---

## A

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **AWQ** | Activation-aware Weight Quantization | 激活感知权重量化 | Part 1 §1.5.3, Part 2 §2.7 |
| **AllReduce** | - | 集合通信操作，各节点归约后广播 | Part 2 §2.2 |
| **AllToAll** | - | 集合通信操作，每个节点向所有节点发送数据 | Part 2 §2.2.3 |
| **Attention** | - | Transformer 核心机制 | Part 1 §1.1 |

## B

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Batching** | - | 批处理 | Part 1 §1.3 |
| **Block Table** | - | PagedAttention 中虚拟 block 到物理 block 的映射 | Part 2 §2.4.1, Part 3 §3.3 |
| **Breakable CUDA Graph** | - | vLLM V1 中支持动态断点的 CUDA Graph | Part 3 §3.4.2 |

## C

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Continuous Batching** | - | 连续批处理 | Part 1 §1.3.2, Part 2 §2.1 |
| **Chunked Prefill** | - | 将长 prompt 切分为多个 chunk 的 prefill | Part 1 §1.3.3, Part 2 §2.1 |
| **CUDA Graph** | - | 记录 CUDA kernel 序列并复用 | Part 3 §3.4.2, Part 7 §7.2 |
| **CP** | Context Parallelism | 上下文并行 | Part 1 §1.1.4, Part 2 §2.2.4 |

## D

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Decode** | - | 自回归生成阶段（每次 1 token） | Part 1 §1.1.2 |
| **DLA** | Deep Learning Accelerator | 专用深度学习加速器，常与 GPU 共享内存 | Part 15 §15.2.1 |
| **DP** | Data Parallelism | 数据并行 | Part 2 §2.2.4 |
| **DP Attention** | Data Parallel Attention | vLLM 针对 MLA 的按请求分区 attention | Part 6 Finding 1-2 |
| **dFlash** | draft-Flash | in-filling 式投机解码（SGLang 首创，vLLM 已移植） | Part 2 §2.6.3, Part 4 §4.6.3 |
| **D-LLM** | Diffusion LLM | 面向 LLaDA / SDAR 等扩散式 LLM 的推理调度 | Part 4 §4.10 |
| **DSA** | DeepSeek Sparse Attention | DeepSeek V3.2 的分层稀疏注意力（Indexer + top-k 选择） | Part 2 §2.4.4, Part 4 §4.10 |

## E

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Eagle** | - | 投机解码算法，用轻量 draft model | Part 2 §2.6.2, Part 3 §3.5, Part 4 §4.6 |
| **Edge TPU** | - | Google 推出的低功耗 AI 加速器，主要用于 TFLite/LiteRT 生态 | Part 16 §16.5.3 |
| **EP** | Expert Parallelism | 专家并行（MoE 专用） | Part 2 §2.2.3 |
| **Elastic EP** | - | 运行期动态增删专家组的并行方案 | Part 4 §4.10 |
| **EPLB** | Expert Parallel Load Balancing | MoE 专家并行负载均衡（SGLang 按访问分布重排） | Part 4 §4.10 |

## F

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **FlashAttention** | - | IO-aware exact attention | Part 1 §1.5, Part 2 §2.4.2 |
| **FlashInfer** | - | 针对推理优化的 attention kernel 库 | Part 2 §2.4.2, Part 4 §4.4 |
| **FP8** | - | 8-bit floating point（E4M3/E5M2） | Part 1 §1.5.4, Part 2 §2.7 |

## G

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **GGUF** | Georgi Gerganov Universal Format | llama.cpp 使用的二进制模型格式，支持按张量选择量化类型 | Part 16 §16.3.2 |
| **Goodput** | - | 有效吞吐（成功请求的吞吐） | Part 1 §1.2 |
| **GPTQ** | - | 基于 Hessian 的权重量化 | Part 1 §1.5.2 |
| **GQA** | Grouped Query Attention | 分组查询注意力 | Part 1 §1.1.4 |
| **GPU** | Graphics Processing Unit | 图形处理器 | Part 7 |

## H

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **H2O** | Heavy-Hitter Oracle | 通过保留 heavy-hitter token 和最近窗口压缩 KV Cache | Part 17 §17.1.3 |
| **HBM** | High Bandwidth Memory | 高带宽显存 | Part 1 §1.4, Part 7 §7.1 |
| **HiCache** | - | SGLang 三级缓存架构 | Part 2 §2.3.4, Part 4 §4.5.3, Part 6 Finding 7 |
| **HiRadixTree** | - | HiCache 的跨三级页表结构 | Part 6 Finding 7 |
| **HTP** | Hexagon Tensor Processor | Qualcomm Hexagon NPU 上的张量处理单元，QNN 常用后端 | Part 16 §16.5.3 |

## I

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **In-flight Batching** | - | 在单条序列生成过程中动态拼接新请求的批处理策略 | Part 16 §16.2.4 |

## J

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Jetson** | - | NVIDIA 边缘 AI 计算平台系列（Orin Nano/NX/AGX/Thor） | Part 15 §15.2.1 |

## K

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **KV Cache** | Key-Value Cache | Transformer 推理中存储历史 K/V 的缓存 | Part 1 §1.1.3 |
| **KV Cache Manager** | - | vLLM 中管理 block 分配的组件 | Part 3 §3.3 |
| **KV Cache Quantization** | - | 对 KV Cache 做 INT8/INT4 等低精度量化以节省内存 | Part 17 §17.1.4 |
| **KV Connector** | - | vLLM V1 中用于 PD 分离的 KV 传输连接器 | Part 3 §3.2.4 |
| **KV Offload / Tiering** | - | 显存不足时将 KV Cache 卸载到 CPU 或多级存储（FS/P2P） | Part 2 §2.5.2, Part 3 §3.7 |
| **KV Canary** | - | SGLang 的 KV Cache 完整性校验/扰动注入工具 | Part 4 §4.10 |

## L

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **LLM** | Large Language Model | 大语言模型 | 全书 |
| **LPDDR** | Low-Power Double Data Rate | 低功耗双倍数据速率内存，边缘设备常用共享内存类型 | Part 15 §15.1.2 |
| **LRU** | Least Recently Used | 最近最少使用驱逐策略 | Part 4 §4.2.4 |

## M

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **MLA** | Multi-head Latent Attention | DeepSeek 的压缩 KV attention | Part 2 §2.4.3, Part 6 Finding 1 |
| **MNN** | - | 阿里巴巴开源的轻量级端侧推理框架 | Part 16 §16.4 |
| **MoE** | Mixture of Experts | 混合专家模型 | Part 2 §2.2.3 |
| **MTP** | Multi-Token Prediction | 多 token 预测（投机解码） | Part 2 §2.6.3, Part 6 Finding 4 |
| **Memory Wall** | - | 内存带宽成为性能瓶颈 | Part 1 §1.4.2 |

## N

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **NPU** | Neural Processing Unit | 神经网络处理单元，边缘设备常见 AI 加速器 | Part 15 §15.2.2, Part 17 §17.4 |
| **NVLink** | - | NVIDIA 高速卡间互联 | Part 1 §1.4, Part 7 §7.1 |
| **NSA** | Native Sparse Attention | DeepSeek V3.2 的稀疏注意力：压缩块 + 选择块 + 滑动窗口 | Part 2 §2.4.4, Part 4 §4.10 |

## O

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Overlap Scheduling** | - | SGLang 的 CPU/GPU 并行调度 | Part 2 §2.1.2, Part 4 §4.3 |

## P

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **PagedAttention** | - | vLLM 的页式 KV Cache 管理 | Part 2 §2.4.1, Part 3 §3.3 |
| **PD 分离** | Prefill-Decode Disaggregation | 将 prefill 和 decode 分离到不同节点 | Part 1 §0.4, Part 2 §2.3 |
| **Pipeline Bubble** | - | PP 中 GPU 等待数据的时间 | Part 2 §2.2.2 |
| **PP** | Pipeline Parallelism | 流水线并行 | Part 2 §2.2.2 |
| **Prefill** | - | 处理 prompt 阶段 | Part 1 §1.1.2 |
| **Prefix Caching** | - | 缓存并复用公共前缀的 KV | Part 2 §2.4.1, Part 3 §3.3.3 |

## Q

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **QNN** | Qualcomm Neural Network | Qualcomm 神经网络 SDK，面向 Hexagon NPU / HTP | Part 15 §15.2.2, Part 16 §16.5.3 |
| **QPS** | Queries Per Second | 每秒查询数 | Part 1 §1.2 |
| **Quantization** | - | 量化 | Part 1 §1.5, Part 2 §2.7 |

## R

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **RadixAttention** | - | SGLang 的前缀树式 KV Cache | Part 2 §2.1.2, Part 4 §4.2 |
| **RDMA** | Remote Direct Memory Access | 远程直接内存访问 | Part 2 §2.3.3 |
| **RKNN** | - | 瑞芯微为其 NPU 提供的模型转换与运行时工具链 | Part 15 §15.3.2, Part 16 §16.5.4 |
| **RoPE** | Rotary Position Embedding | 旋转位置编码 | Part 2 §2.5.1 |

## S

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Scheduler** | - | 推理引擎调度器 | Part 2 §2.1 |
| **SGLang** | - | 高性能结构化 LLM 编程/推理框架 | 全书 |
| **Shared Memory** | - / 共享内存 | CPU、GPU/NPU 共用同一颗 LPDDR 的内存架构 | Part 15 §15.1.2 |
| **SM** | Streaming Multiprocessor | NVIDIA GPU 流式多处理器 | Part 7 §7.1 |
| **SP** | Sequence Parallelism | 序列并行 | Part 2 §2.2.4 |
| **Speculative Decoding** | - | 投机解码 | Part 1 §1.6.2, Part 2 §2.6 |
| **Static Batching** | - | 静态批处理 | Part 1 §1.3.1 |
| **StreamingLLM** | - | 通过保留 sink token 和最近窗口实现超长序列推理 | Part 17 §17.1.3 |
| **Swapping** | - | 将 KV Cache 换出到 CPU | Part 2 §2.1.1 |

## T

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Tensor Core** | - | NVIDIA GPU 的矩阵乘单元 | Part 7 §7.1 |
| **Tensor Parallelism (TP)** | - | 张量并行 | Part 2 §2.2.1 |
| **Throughput** | - | 吞吐量 | Part 1 §1.2 |
| **Top-p / Top-k** | - | 采样策略 | Part 1 §1.6.1 |
| **TPOT** | Time Per Output Token | 每输出 token 时间 | Part 1 §1.2 |
| **TTFT** | Time To First Token | 首 token 时间 | Part 1 §1.2 |

## V

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **vLLM** | - | 开源 LLM 推理引擎 | 全书 |

## W

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **Warp** | - | 32 个 thread 的 SIMD 执行单元 | Part 7 §7.2.2 |
| **Wide-EP** | - | vLLM 的 DP+EP 混合并行架构 | Part 6 Finding 2 |
| **Wujian** | 无剑 | 平头哥 SoC 平台，配套玄铁 RISC-V 处理器 | Part 15 §15.3.1 |

## X

| 术语 | 英文全称 | 含义 | 所在 Part |
|------|---------|------|----------|
| **XuanTie** | 玄铁 | 平头哥 RISC-V 处理器系列 | Part 15 §15.3.1 |

---

## 索引表：按主题归类

### 核心概念
- AI Infra → Part 0
- Transformer 推理 → Part 1 §1.1
- KV Cache → Part 1 §1.1.3
- Prefill / Decode → Part 1 §1.1.2
- Continuous Batching → Part 1 §1.3.2

### 性能指标
- TTFT / TPOT / Throughput / Goodput → Part 1 §1.2
- Benchmark → Part 8 §8.1

### 并行策略
- TP / PP / DP / EP / SP / CP → Part 2 §2.2
- Wide-EP / DP Attention → Part 6 Finding 1-2

### 优化技术
- PagedAttention → Part 2 §2.4.1
- RadixAttention → Part 2 §2.1.2, Part 4 §4.2
- FlashAttention / FlashInfer → Part 2 §2.4.2
- MLA → Part 2 §2.4.3
- 量化 (FP8/AWQ/GPTQ) → Part 1 §1.5, Part 2 §2.7
- 投机解码 → Part 1 §1.6.2, Part 2 §2.6

### 高级架构
- PD 分离 → Part 2 §2.3
- Mooncake → Part 2 §2.3.3, Part 6 Finding 6
- HiCache → Part 2 §2.3.4, Part 6 Finding 7

### 源码
- vLLM → Part 3
- SGLang → Part 4

### 面试
- 高频题 → Part 5 §5.1
- 场景题 → Part 5 §5.3
- 自检 → Part 5 §5.4

### 硬件基础
- GPU / CUDA → Part 7

### 边缘 AI
- 边缘硬件 / Jetson / NPU / DLA / LPDDR / 共享内存 / 玄铁 / 无剑 → Part 15
- 边缘框架 / GGUF / QNN / RKNN / MNN / Edge TPU / In-flight Batching → Part 16
- 边缘优化 / StreamingLLM / H2O / KV Cache Quantization → Part 17

### 生产运维
- 部署 / 监控 / 故障排查 → Part 8
