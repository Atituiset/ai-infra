# Part 17：边缘场景优化与落地

> **面向角色**：边缘 AI 推理优化工程师、量产交付工程师
> **前置知识**：Part 15 硬件约束、Part 16 框架、Part 1/2 KV Cache 与量化基础
> **目标**：掌握边缘场景下的性能优化、精度保障和问题排查方法

---

## 17.1 内存受限下的 KV Cache 管理 ★

边缘设备的内存是共享的：CPU、GPU/NPU 共用同一颗 LPDDR5/DDR5，没有独立的 HBM。对于 LLM 来说，**权重**只占内存的一部分，**KV Cache** 在长序列下会迅速膨胀，成为真正的瓶颈。本节讨论边缘场景下压缩和管理 KV Cache 的核心技术。

### 17.1.1 KV Cache 为什么会成为边缘瓶颈

对于一个标准的 Decoder-only Transformer，KV Cache 的峰值显存估算为：

```
KV Cache (bytes) ≈ 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

以 **Llama 3.2 3B Instruct** 为例：

- `num_layers = 28`
- `num_kv_heads = 8`
- `head_dim = 128`
- `batch_size = 1`
- `seq_len = 4096`
- `dtype = FP16`（2 bytes）

则 KV Cache ≈ `2 × 28 × 8 × 128 × 4096 × 1 × 2 ≈ 3.7 GB`。

如果序列拉到 8192，仅 KV Cache 就要约 7.4 GB。这还没算权重、激活值、系统开销。在 Jetson Orin Nano 8 GB 或手机 8 GB 设备上，长上下文很容易把内存打满。因此边缘优化的第一项工作，通常就是**压缩 KV Cache**。

### 17.1.2 GQA / MQA / MLA 在边缘的收益

**Multi-Head Attention（MHA）** 每个头都有独立的 K/V，KV Cache 最大。**Grouped-Query Attention（GQA）** 让多个 query head 共享一组 K/V head，显著减少 KV Cache。**Multi-Query Attention（MQA）** 更进一步，所有 query head 共享同一组 K/V。

| 注意力变体 | KV head 数量 | KV Cache 相对 MHA | 典型代表 | 边缘收益 |
|-----------|-------------|------------------|---------|---------|
| **MHA** | = query heads | 100% | 早期 Llama 2 7B | 基准 |
| **GQA** | 中间值 | 25%-50% | Llama 3 系列、Qwen2.5 | 主流折中 |
| **MQA** | 1 | 1/num_heads | PaLM、ChatGLM | 最大压缩 |
| **MLA** | 低秩压缩向量 | 10%-25% | DeepSeek-V2/V3 | 长上下文神器 |

**MLA（Multi-head Latent Attention）** 是更激进的压缩方案。它把 K/V 投影到一个低维的 latent 向量，解码时只缓存这个低秩向量，而不是完整的 K/V head。在 DeepSeek-V2/V3 中，MLA 把 KV Cache 压缩到传统 MHA 的 10%-20%，对长上下文边缘部署极具吸引力。

**选型建议**：

- 如果模型原生已经使用 GQA，优先用它，不要强行改成 MQA（可能掉精度）。
- 如果目标设备内存极度受限，可以考虑 MQA 或 MLA 架构的模型。
- 边缘端不建议自己从零训练注意力变体，优先选择已经验证好的开源模型。

### 17.1.3 Sliding-window / StreamingLLM / H2O

当序列长度超过设备能承载的 KV Cache 上限时，需要**主动遗忘**部分历史 token。常见策略有三种。

**Sliding-window Attention**

只允许每个 token 看到最近的 `W` 个 token。超出窗口的旧 token 被直接丢弃。实现简单，但问题在于**丢失远距离依赖**。如果 system prompt 或关键上下文在窗口外，模型会完全遗忘。

```python
# 概念示意：只保留最近 window_size 个 token 的 KV
if len(kv_cache) > window_size:
    kv_cache = kv_cache[-window_size:]
```

**StreamingLLM**

StreamingLLM 的核心观察是：模型对**初始的几个 sink token**（通常是 bos/padding 或 system prompt 开头）有持续的高注意力。因此策略变为：

- 保留最开始的 **sink tokens**（通常 4 个）。
- 保留最近的 **local window**（例如 1024 个 token）。
- 中间部分全部驱逐。

这样可以在不重新训练的情况下，让模型支持**理论上无限长**的流式输入，同时保留对初始上下文的记忆。

```
序列： [sink0, sink1, sink2, sink3] + [被驱逐的中间 token] + [最近 W 个 token]
```

**H2O（Heavy-Hitter Oracle）**

H2O 认为，真正重要的 token 是那些**被很多后续 token 高频访问**的 heavy hitters。它维护两类 token：

- **Heavy hitters**：累计被注意力访问次数最高的 top-k token。
- **Recent window**：最近 W 个 token。

 eviction 时只保留这两部分。与 StreamingLLM 相比，H2O 不依赖固定的 sink token，而是动态地发现关键 token，对开放域对话更友好。

| 方法 | 保留策略 | 是否需要训练 | 长上下文效果 | 边缘适用性 |
|-----|---------|------------|------------|----------|
| **Sliding-window** | 最近 W 个 | 否 | 差（丢远距离） | 简单场景 |
| **StreamingLLM** | sink + 最近 W 个 | 否 | 好 | 流式输入、边缘首选 |
| **H2O** | heavy hitters + 最近 W 个 | 否 | 较好 | 开放域对话 |

### 17.1.4 KV Cache INT8/INT4 量化

除了减少 KV Cache 的 token 数量，还可以通过**降低每个值的精度**来压缩。

| 精度 | 相对 FP16 显存 | 典型精度损失 | 边缘支持度 | 推荐场景 |
|-----|--------------|------------|----------|---------|
| **FP16** | 100% | 基准 |  universal | 短序列、高精度 |
| **INT8** | 50% | 低（<1% ppl 增加） | 广泛 | 长序列、精度敏感 |
| **INT4** | 25% | 中-高（2-5% ppl 增加） | 部分框架 | 极限内存、短序列 |
| **FP8** | 50% | 低 | H100/Blackwell/Jetson 部分 | 新硬件优先 |

**实现要点**：

1. **per-channel / per-token 量化**：对 K 和 V 分别计算 scale/zero point。K 通常按 head 维度 per-channel 量化，V 按 token 维度 per-token 量化。
2. **保留部分 FP16**：对于第一层和最后一层，或对精度敏感的 head，可以保留 FP16，其余用 INT8/INT4。
3. **动态 vs 静态量化**：动态量化根据运行时统计调整 scale，精度更好但开销大；静态量化用校准得到的固定 scale，边缘更常用。
4. **反量化位置**：在 attention score 计算前反量化回 FP16，还是在 INT8 上完成 matmul，取决于硬件支持和框架能力。

```python
# 概念：INT8 KV Cache 的 per-channel 量化
scale_k = max(abs(K)) / 127
K_int8 = round(K / scale_k)
K_fp16 = K_int8 * scale_k  # 计算时反量化
```

### 17.1.5 与云侧 PagedAttention 的差异

云侧 vLLM 的 **PagedAttention** 把 KV Cache 分成固定大小的 block，像操作系统的虚拟内存一样动态分配，支持 preemptive scheduling 和请求间共享前缀。

边缘侧通常不会直接复用 PagedAttention，原因如下：

| 维度 | 云侧 PagedAttention | 边缘 KV Cache 管理 |
|-----|-------------------|------------------|
| **内存模型** | 独立 HBM，容量大 | 共享 LPDDR，容量小 |
| **调度目标** | 高吞吐、高并发 | 低延迟、确定性 |
| **block 管理** | 动态分配、preemption | 静态预分配、避免 overhead |
| **batch 大小** | 大（数十到数百） | 小（1-8） |
| **优化重点** | 隐藏延迟、提高利用率 | 压缩、减少访存、保精度 |

边缘侧更常见的做法是：

- **静态 KV Cache 池**：根据 `max_batch_size × max_seq_len` 一次性分配，避免运行时 malloc。
- **紧凑布局**：GQA/MQA 直接减少 head 数；INT8/INT4 降低每个单元大小。
- **按需 eviction**：Sliding-window 或 StreamingLLM 在生成长序列时释放旧 block。

---

## 17.2 量化与校准实战 ★

量化是边缘部署的必修课。同样的模型，FP16 可能在 8 GB 设备上跑不起来，INT4 后就能在 4 GB 设备上流畅运行。但量化不是简单地把权重改成 INT8/INT4，**校准流程和精度评估**决定了最终能否交付。

### 17.2.1 PTQ / QAT / GPTQ / AWQ / SmoothQuant 在边缘的适用性

| 方法 | 是否需要训练数据 | 是否需要反向传播 | 典型精度 | 边缘落地成本 | 推荐场景 |
|-----|--------------|--------------|---------|------------|---------|
| **PTQ** | 少量校准数据 | 否 | 中-高 | 最低 | 边缘最常用 |
| **QAT** | 完整训练数据 | 是 | 最高 | 高 | 自研模型、量产 |
| **GPTQ** | 少量校准数据 | 否（基于 Hessian） | 高（INT4） | 低 | 大模型 INT4 压缩 |
| **AWQ** | 少量校准数据 | 否 | 高（INT4） | 低 | 对精度敏感的 INT4 |
| **SmoothQuant** | 少量校准数据 | 否 | 高（INT8） | 低 | 激活值分布大的模型 |

**PTQ（Post-Training Quantization）**

最轻量的量化方式。用校准数据跑一遍模型，统计每层权重和激活的 min/max，确定 scale 和 zero point。PTQ 的优点是快、不需要训练；缺点是对于 INT4 这种低比特，精度容易崩。

**QAT（Quantization-Aware Training）**

在训练过程中模拟量化误差，让模型学会适应低精度。效果最好，但需要完整的训练流程和算力，通常只在自研模型或大规模量产时使用。

**GPTQ（Generative Pre-trained Transformer Quantization）**

一种 layer-wise 的 one-shot 量化方法。它利用 Hessian 矩阵信息，按层逐步量化权重并补偿误差。GPTQ 在 INT4 下通常比朴素 PTQ 好很多，是边缘部署 7B-13B 模型的常用选择。

**AWQ（Activation-aware Weight Quantization）**

AWQ 的核心发现是：模型中大约 1% 的权重对激活特别敏感。AWQ 通过保护这 1% 的 salient weight，在 INT4 下获得接近 FP16 的精度。实际工程中，AWQ 经常比 GPTQ 更稳定，尤其适合小 batch、低延迟场景。

**SmoothQuant**

Transformer 的激活值通常比权重更难量化（outlier 多）。SmoothQuant 通过数学变换把激活的难度迁移到权重上：

```
X' = X · diag(s)^(-1)
W' = diag(s) · W
```

这样激活变得好量化，权重稍微难量化一点，但整体 INT8 精度可以接近 FP16。SmoothQuant 在边缘 INT8 部署中非常有效。

### 17.2.2 Calibration 流程与数据集选择

校准数据集的质量直接决定量化后的精度。常见实践：

1. **数据量**：通常 **256-1024 条样本** 足够。太少会导致统计不稳定，太多则收益递减。
2. **数据分布**：校准数据应该尽量贴近真实业务分布。例如对话模型用真实对话语料，代码模型用代码片段。
3. **长度分布**：覆盖目标 max_seq_len 的 50%-100%，避免校准时全是短句。
4. **避免极端异常值**：如果校准数据包含大量特殊 token 或超长样本，scale 会被拉偏。

```python
# 伪代码：PTQ 校准流程
calibration_data = load_dataset("target_domain_samples", n=512)
for sample in calibration_data:
    model.forward(sample)  # 收集每层激活统计

for layer in model.layers:
    scale, zero_point = compute_quantization_params(layer.activation_stats)
    layer.set_activation_quantizer(scale, zero_point)
```

**数据集选择建议**：

| 场景 | 推荐校准数据来源 | 样本数 | 备注 |
|-----|----------------|-------|------|
| 通用对话 | SlimOrca、ShareGPT 子集 | 256-512 | 覆盖多轮对话 |
| 代码助手 | The Stack 子集、LeetCode | 512 | 包含代码结构 |
| 多语言 | 目标语种 Wikipedia/对话 | 512 | 不能用纯英文校准多语言模型 |
| 视觉语言 | 目标图像-文本对 | 256-512 | 覆盖目标分辨率 |

### 17.2.3 精度评估（perplexity + task-level）

量化后的模型不能只测 perplexity，必须测**下游任务精度**。

**Perplexity（PPL）**

PPL 反映模型对下一个 token 的预测能力，是快速筛查量化损失的指标。一般要求：

- INT8：PPL 增加 **< 1%** 相对 FP16。
- INT4：PPL 增加 **< 3%** 相对 FP16（AWQ/GPTQ 通常能做到）。

**Task-level Accuracy**

PPL 好不代表任务表现好。必须测真实任务：

- 对话模型：MT-Bench、AlpacaEval。
- 代码模型：HumanEval、MBPP。
- 数学模型：GSM8K、MATH。
- 多语言模型：目标语种的阅读理解、翻译。

| 评估层级 | 指标 | 用途 | 边缘验收标准 |
|---------|------|------|------------|
| **PPL** | Wikitext-2 / C4 | 快速筛查 | 相对 FP16 < 3% |
| **任务精度** | MMLU、GSM8K、HumanEval | 能力验证 | 相对 FP16 < 2% |
| **端到端** | 客户业务指标 | 真实交付 | 满足 SLA |

### 17.2.4 Per-layer / Per-channel / Group-wise 量化影响

量化粒度越细，精度越好，但计算和存储 overhead 越大。

| 粒度 | 定义 | 精度 | 存储/计算开销 | 边缘适用性 |
|-----|------|------|------------|----------|
| **Per-tensor** | 整个张量一个 scale | 最低 | 最小 | 简单硬件 |
| **Per-channel** | 输出通道一个 scale | 高 | 小 | 推荐 |
| **Per-token** | 每个 token 一个 scale | 高 | 中 | 激活常用 |
| **Group-wise** | 每 G 个元素一个 scale | 最高 | 中 | INT4 推荐 |

**权重量化**：通常用 per-channel 或 group-wise。INT4 几乎必须用 group-wise（例如 group size = 128），否则精度损失太大。

**激活量化**：通常用 per-token 或 dynamic per-token，因为激活值的分布随输入变化大。

**KV Cache 量化**：K 常用 per-channel（按 head 维度），V 常用 per-token，整体精度损失可控。

---

## 17.3 模型轻量化 ★

量化之外，模型本身的大小和结构也需要为边缘优化。蒸馏、剪枝、NAS 和专门的 edge-friendly 模型家族，是三种主要思路。

### 17.3.1 蒸馏、剪枝、NAS for edge

**知识蒸馏（Knowledge Distillation）**

用小模型（student）学习大模型（teacher）的输出分布。蒸馏可以传递教师模型的"暗知识"，让小模型在同等参数量下表现更好。

常见蒸馏目标：

- **Logit 蒸馏**：让学生匹配教师的 softmax 输出。
- **Hidden-state 蒸馏**：让学生匹配教师的中间层表示。
- **Reasoning 蒸馏**：例如 DeepSeek-R1 的蒸馏，把推理能力传递给小模型。

边缘部署中，**自蒸馏 + 量化联合训练**往往能取得最佳精度-效率平衡。

**剪枝（Pruning）**

- **非结构化剪枝**：把单个权重置零，压缩率可达 50%-70%，但需要硬件支持稀疏加速（如 NVIDIA 2:4 structured sparsity）。
- **结构化剪枝**：剪掉整个 head、channel 或 layer，可以直接减少计算量，但精度损失通常更大。
- **运动剪枝（Movement Pruning）**：训练过程中学习哪些权重重要，适合与微调结合。

边缘设备通常更偏好**结构化剪枝**，因为非结构化稀疏如果没有硬件加速，实际加速比很低。

**NAS（Neural Architecture Search）**

对于 CV 小模型，NAS 可以自动搜索适合目标 NPU 的算子组合和通道数。例如 **Once-for-All**、**BigNAS**、**Hardware-aware NAS**。在 LLM 领域，NAS 更多用于搜索 attention head 数、FFN 维度、层数等。

### 17.3.2 Edge-friendly 模型家族

近年来，多个模型家族专门针对边缘场景优化：

| 模型家族 | 典型尺寸 | 架构特点 | 边缘优势 | 适用平台 |
|---------|---------|---------|---------|---------|
| **Phi-3 / Phi-4** | 3.8B-14B | 高质量训练数据 | 小尺寸高性能 | 手机、PC、Jetson |
| **Gemma 2 / 3** | 2B-27B | Google 官方优化 | 开放权重、多尺寸 | Android、Core ML |
| **Qwen2.5** | 0.5B-72B | 全尺寸覆盖 | 中文/多语言强 | 手机、边缘盒子 |
| **Llama 3.2** | 1B-3B | Meta 官方小模型 | 生态广、工具链成熟 | 通用边缘 |
| **DeepSeek-V2/V3** | 16B-671B MoE | MLA 低秩 KV | 长上下文边缘 | 高端边缘/服务器 |
| **StableLM / TinyLlama** | 1B-3B | 小尺寸实验模型 | 极低资源 | MCU 级设备 |

**选型建议**：

- 如果业务以中文为主，优先考虑 Qwen2.5 系列。
- 如果需要与 Android / Google 生态深度集成，考虑 Gemma。
- 如果需要跨平台部署和丰富工具链，Llama 3.2 是稳妥选择。
- 如果需要长上下文且内存受限，DeepSeek-V2/V3 的 MLA 值得尝试。

### 17.3.3 1B-4B vs 量化 7B-13B 选型

边缘部署常面临一个选择：用原生小模型（1B-4B），还是把 7B-13B 量化到 INT4？

| 维度 | 原生 1B-4B | 量化 7B-13B |
|-----|-----------|------------|
| **内存占用** | 低（1-3 GB FP16） | 中（4-8 GB INT4） |
| **推理速度** | 快 | 较慢 |
| **精度上限** | 较低 | 较高（即使量化） |
| **长上下文能力** | 通常较弱 | 通常较强 |
| **工具链支持** | 好 | 取决于量化格式 |
| **功耗** | 低 | 中-高 |

**决策逻辑**：

- 如果任务是简单分类、短回答、指令跟随，**1B-4B** 足够。
- 如果需要复杂推理、代码生成、多轮长对话，**量化 7B** 通常比原生 3B 更强。
- 如果设备只有 4 GB 内存，优先尝试 **INT4 量化的 3B** 或 **原生 1.5B-2B**。
- 如果设备有 8 GB 内存，**INT4 量化的 7B** 或 **INT8 量化的 3B-4B** 是甜点区。

---

## 17.4 异构调度与多模型部署 ★

边缘设备通常有多个计算单元：CPU、GPU、NPU、DSP。如何把它们组合起来，同时运行多个模型，是量产中的核心工程问题。

### 17.4.1 CPU + GPU/NPU 混合执行

一个典型的异构执行策略：

- **CPU**：负责前处理、后处理、控制逻辑、不支持的算子 fallback。
- **GPU/NPU**：负责密集计算，如 attention、conv、matmul。
- **DSP/ISP**：负责图像信号处理、语音前端。

```
输入数据
  │
  ▼
CPU 预处理（resize、normalize、tokenize）
  │
  ▼
NPU 执行 encoder / backbone
  │
  ▼
CPU 解量化 / reshape / 后处理
  │
  ▼
GPU 执行 LLM 解码
  │
  ▼
CPU 组装最终输出
```

**关键原则**：

1. **减少跨单元拷贝**：每次 CPU↔NPU 或 CPU↔GPU 的数据搬移都会消耗时间和带宽。尽量让数据在目标单元上生成和消费。
2. **流水线重叠**：当 NPU 处理第 N 帧时，CPU 可以预处理第 N+1 帧。
3. **避免 NPU 空闲等 CPU**：如果后处理太重，NPU 会空转，需要把部分后处理移到 NPU 或优化 CPU 代码。

### 17.4.2 Graph Partitioning 与 Fallback

边缘 NPU 的算子支持通常不完整。遇到不支持的算子时，框架需要把这部分图切下来，fallback 到 CPU 或 GPU 执行。

| 策略 | 优点 | 缺点 |
|-----|------|------|
| **整图回退 CPU** | 简单 | 性能差 |
| **算子级 fallback** | 保留大部分 NPU 加速 | 切分点多，数据拷贝多 |
| **子图级 fallback** | 平衡 | 需要手动标记边界 |
| **重写为支持算子** | 性能最好 | 工作量大 |

**常见 fallback 算子**：

- 自定义 activation（如 GELU、SwiGLU 的某些实现）。
- 复杂索引 / gather / scatter。
- 动态 shape 算子（如可变长度 padding）。
- 某些 normalization 的特定维度。

**优化建议**：

- 在模型设计阶段就参考目标 NPU 的算子白名单。
- 用 `onnx-simplifier` 或框架自带工具把图简化成 NPU 友好形式。
- 对于必须 fallback 的算子，尽量合并成一个大的 CPU 子图，减少切换次数。

### 17.4.3 内存预算分配、模型热切换、Pipeline 组合

当设备需要同时驻留多个模型时，必须做严格的内存预算。

**内存预算公式**：

```
总内存预算 = 系统预留 + 权重总量 + KV Cache 峰值 + 激活值峰值 + 多模型切换缓冲

其中：
  权重总量 = Σ(模型_i_权重大小)
  KV Cache 峰值 = Σ(模型_i_max_batch × max_seq_len × kv_head_dim × layers × dtype_bytes × 2)
  切换缓冲 = max(单个模型权重)  # 用于热切换时临时加载
```

例如一个 8 GB 设备要同时跑一个 3B LLM 和一个 ViT encoder：

- 系统预留：1 GB
- 3B LLM INT4 权重：约 1.8 GB
- ViT INT8 权重：约 0.3 GB
- LLM KV Cache（batch=1, seq=2048）：约 1.0 GB
- 激活值 + 缓冲：约 1.5 GB
- 总计：约 4.6 GB，剩余 3.4 GB 给热切换和其他任务。

**模型热切换**：

- 预先把多个模型权重存在存储（eMMC/SSD）中，运行时按需求加载到内存。
- 使用 mmap 或框架提供的 weight streaming 减少加载延迟。
- 对于不常用模型，可以卸载到存储，释放内存。

**Pipeline 组合**：

```
camera → YOLO（NPU）→ 目标 crop → CLIP（NPU）→ LLM（GPU）→ TTS（CPU/DSP）
```

这种 pipeline 需要在不同模型之间共享内存池，避免每个模型独立分配导致碎片化。

### 17.4.4 RTOS / Android / Linux 约束

不同操作系统对边缘 AI 的运行有不同约束：

| 系统 | 典型场景 | 关键约束 | 优化方向 |
|-----|---------|---------|---------|
| **RTOS** | 工业控制、车载 MCU | 实时性、确定性、内存固定 | 静态分配、WCET 分析、避免 GC |
| **Android** | 手机、平板、XR | 功耗、后台限制、热设计 | 使用 NNAPI / QNN / Core ML、低功耗模式 |
| **Linux** | 边缘盒子、机器人 | 资源竞争、驱动版本 | cgroup 限流、实时内核、BSP 锁定 |
| **Bare Metal** | MCU、极低功耗 | 无 OS、KB 级内存 | 微模型、INT8/INT4、手写 kernel |

**Android 特别注意事项**：

- 应用进入后台后，NPU 可能被系统回收。
- 高负载推理会触发 thermal throttling，需要监听 `PowerManager` 温度事件。
- 使用 `NNAPI` 或厂商 SDK 的省电模式，在电量低时降级。

---

## 17.5 VLM / VLA 边缘部署 ★

多模态模型（VLM、VLA）正在快速进入边缘场景：智能摄像头、机器人、AR 眼镜、车载感知。但多模态的部署复杂度远高于纯文本 LLM。

### 17.5.1 视觉编码器量化与图优化

VLM 通常包含两部分：

1. **Vision Encoder**（如 CLIP ViT、SigLIP、DINOv2）：把图像变成 visual tokens。
2. **LLM Decoder**：把 visual tokens 和 text tokens 一起解码。

在边缘，**视觉编码器往往是延迟瓶颈**。例如一张 336×336 的图经过 ViT-L/14，可能产生 576 个 visual tokens，每个 token 维度 1024，计算量不可忽略。

**优化策略**：

| 策略 | 收益 | 注意 |
|-----|------|------|
| **INT8 量化视觉编码器** | 2x 加速，50% 内存 | 注意 attention softmax 精度 |
| **降低输入分辨率** | 显著减少 tokens 数 | 可能丢失小目标细节 |
| **Patch 合并 / 池化** | 减少 visual tokens | 需要模型支持 |
| **图优化 / 算子融合** | 减少 kernel launch | 依赖框架能力 |
| **FP16 保留关键层** | 保精度 | 第一层、最后一层常用 |

**经验法则**：视觉编码器对量化比 LLM 更敏感。如果 INT8 后 VQA 精度下降明显，可以尝试：

- 只对 CNN/linear 量化，attention 保留 FP16。
- 使用 per-channel 量化而非 per-tensor。
- 对视觉编码器单独做校准，而不是和 LLM 共用一套 scale。

### 17.5.2 多模态 Pipeline：camera → preprocess → encoder → LLM

一个典型的边缘 VLM pipeline：

```
camera 捕获帧（YUV / RGB）
    │
    ▼
ISP 处理（去噪、白平衡、缩放）
    │
    ▼
CPU/GPU 预处理（resize 到 336×336，归一化）
    │
    ▼
Vision Encoder（NPU/GPU）→ visual tokens
    │
    ▼
Projector（MLP / linear，NPU）→ LLM embedding 空间
    │
    ▼
Tokenizer + text embedding（CPU/NPU）
    │
    ▼
LLM Decoder（GPU/NPU）生成文本
```

**延迟优化要点**：

1. **零拷贝**：让 camera buffer 直接进 NPU，避免 CPU 中转。
2. **分辨率权衡**：336×336 和 448×448 的精度差异 vs 延迟差异需要实测。
3. **Batch visual tokens**：如果是视频流，相邻帧的视觉特征可以缓存或复用。
4. **Pipeline 并行**：预处理下一帧时，LLM 正在解码当前帧的文本。

### 17.5.3 LLaVA / Qwen-VL / Phi-vision / OpenVLA 部署路径

| 模型 | 视觉编码器 | LLM | 边缘部署特点 |
|-----|-----------|-----|------------|
| **LLaVA 1.5/NeXT** | CLIP ViT-L/14 | Vicuna / Llama | 成熟、工具链多 |
| **Qwen-VL / Qwen2-VL** | ViT | Qwen | 中文/多语言强 |
| **Phi-3/4 Vision** | CLIP ViT | Phi-3/4 | 微软优化，端侧友好 |
| **OpenVLA** | DINOv2 + SigLIP | Llama + action head | 机器人 VLA 专用 |

**LLaVA 部署路径**：

1. 导出视觉编码器为 ONNX / TensorRT。
2. 导出 projector 为 ONNX。
3. 用 llama.cpp / TensorRT-LLM / MLC-LLM 部署 LLM。
4. 在应用层把 visual tokens 注入到 text embedding 前。

**Qwen-VL 部署路径**：

- Qwen 官方通常提供 `qwen-vl-utils` 和转换脚本。
- 注意 Qwen2-VL 的 **M-RoPE**（Multimodal Rotary Position Embedding），需要运行时支持。
- 视觉编码器可以用 `qwen2-vl-2b` 等轻量版本。

**Phi-vision 部署路径**：

- Phi-3 Vision 的视觉编码器较小，适合手机 NPU。
- Microsoft 提供了 ONNX 导出示例，可直接用 ONNX Runtime / Core ML / QNN。

**OpenVLA 部署路径**：

OpenVLA 是面向机器人操作的 **Vision-Language-Action** 模型，架构为：

```
图像 → DINOv2（视觉特征）
    ↓
图像 → SigLIP（视觉语言对齐特征）
    ↓
Concat / Fusion → Projector → Llama backbone
    ↓
Action head → 机器人动作（end-effector pose / joint angles）
```

部署时通常把 DINOv2 + SigLIP 两个编码器量化到 INT8，Llama 量化到 INT4/INT8，action head 保留 FP16。

### 17.5.4 视频流帧率/延迟权衡

视频流场景下，VLM 面临一个基本矛盾：

- **高帧率**：能捕捉快速变化，但每帧都跑完整 pipeline 延迟高、功耗大。
- **低帧率**：延迟低，但可能丢失关键动作或事件。

**常用策略**：

| 策略 | 描述 | 适用 |
|-----|------|------|
| **跳帧** | 每 N 帧处理 1 帧 | 静态监控 |
| **关键帧** | 只在运动检测触发时处理 | 事件驱动 |
| **时序池化** | 多帧特征平均 / LSTM 聚合 | 动作识别 |
| **稀疏 attention** | 视频 tokens 只保留关键帧 | 长视频理解 |
| **双分支** | 轻量 CNN 实时检测 + VLM 低频分析 | 安防、机器人 |

**帧率选择经验**：

- 工业质检：1-5 FPS 足够。
- 机器人操作：10-30 FPS。
- AR 眼镜：≥30 FPS，但通常只处理关键 ROI。

---

## 17.6 Inflight Batching 在边缘

Inflight Batching（也叫 Continuous Batching）是云侧提高吞吐的核心技术。但在边缘，它不一定总是最优选择。

### 17.6.1 小 batch、低延迟场景的差异

云侧 inflight batching 的目标是：

- 高并发时最大化 GPU 利用率。
- 通过动态 batching 把多个请求的 prefill/decode 阶段混合调度。
- 允许一定程度的 TPOT 抖动换取更高吞吐。

边缘场景则完全不同：

- 并发请求通常只有 1-4 个。
- 用户要求**确定性的低延迟**。
- 内存本来就紧张，batch 大了 KV Cache 会爆。

### 17.6.2 什么时候有用，什么时候有害

**Inflight Batching 在边缘有用的场景**：

1. **多租户边缘服务器**：边缘盒子同时服务多个摄像头或用户，请求到达时间不齐。
2. **异步 pipeline**：LLM 在等待 I/O 时，可以插入其他请求的 decode。
3. **高吞吐优先**：例如批量文档处理，延迟不敏感。

**Inflight Batching 在边缘有害的场景**：

1. **单用户实时对话**：batch 带来的调度 overhead 反而增加首 token 延迟。
2. **内存受限设备**：batch size 增大会线性增加 KV Cache。
3. **延迟 SLA 严格**： inflight batching 的 TPOT 抖动可能违反 SLA。

| 场景 | 是否推荐 inflight batching | 替代方案 |
|-----|-------------------------|---------|
| 单用户语音助手 | 否 | static batch = 1 |
| 多摄像头视频分析 | 是 | continuous batching |
| 边缘 API 网关 | 是 | dynamic batching |
| 机器人实时控制 | 否 | static batch + 优先级调度 |

**工程建议**：

- 边缘设备默认使用 **static batching**，只有当实测并发 > 2 且有吞吐压力时，再启用 inflight batching。
- 如果启用，设置 `max_num_seqs` 很小（例如 4-8），避免 KV Cache 过度占用。
- 对延迟敏感请求设置优先级队列，避免被低优先级请求阻塞。

---

## 17.7 问题排查与客户支持 ★

边缘项目落地过程中，问题往往不在算法本身，而在**硬件、驱动、SDK、客户环境**的耦合。建立标准化的排查流程和协作模板，是量产交付的关键。

### 17.7.1 功能 / 性能 / 精度 triage 流程

收到客户问题后，第一步是按类型分类：

```
客户反馈
  │
  ├─ 功能问题：崩溃 / 无输出 / 输出乱码 / 工具调用失败
  │     ├─ 复现步骤？
  │     ├─ 是否在所有输入都出现？
  │     └─ 日志中是否有 segfault / OOM / op not supported？
  │
  ├─ 性能问题：慢 / 卡顿 / 吞吐不够
  │     ├─ 测 TTFT / TPOT / end-to-end latency
  │     ├─ 是否触发 thermal throttling？
  │     └─ batch size / 并发数 / 序列长度是否匹配设备能力？
  │
  └─ 精度问题：输出不对 / 幻觉严重 / 任务指标下降
        ├─ FP16 基线是否正常？
        ├─ 量化模型是否正常？
        ├─ 输入预处理是否与训练一致？
        └─ 后处理（如 temperature、top-p）是否正确？
```

**功能问题常见根因**：

- 算子 fallback 到 CPU 导致超时或异常。
- 输入 shape 不在 engine 支持范围内。
- SDK / 驱动版本不兼容。
- 模型格式转换时权重 shape 错误。

**性能问题常见根因**：

- KV Cache 太大导致频繁换页或 OOM。
- 视觉编码器分辨率过高。
- 未启用 NPU/GPU 加速，纯 CPU 运行。
- 散热不足导致降频。

**精度问题常见根因**：

- 量化 calibration 数据分布与真实输入不符。
- 视觉预处理归一化参数错误。
- tokenizer 与训练时不一致（especially chat template）。
- KV Cache 驱逐策略丢失关键上下文。

### 17.7.2 SDK / BSP / 驱动版本冲突

边缘部署最痛苦的问题之一就是**版本矩阵**。一个 Jetson 项目可能涉及：

- JetPack 版本（L4T）
- CUDA / cuDNN / TensorRT 版本
- Python / PyTorch 版本
- TensorRT-LLM / llama.cpp 版本
- 自定义 kernel 或 plugin 版本

**版本冲突表现**：

| 现象 | 可能原因 |
|-----|---------|
| engine 加载失败 | TensorRT 版本与生成 engine 时不一致 |
| 算子报错 `op not supported` | SDK 版本太旧，不支持新算子 |
| 推理结果随机错误 | cuDNN / driver bug |
| 性能比官方 demo 差很多 | 未使用优化后的 BSP |
| 量化工具链崩溃 | Python 包版本不兼容 |

**建议做法**：

1. **锁定版本矩阵**：在项目文档中明确记录所有依赖版本。
2. **Docker 化**：尽可能把环境打包成容器，避免客户本地环境差异。
3. **CI 覆盖目标版本**：在多个 JetPack / BSP 版本上跑回归测试。
4. **与客户对齐升级窗口**：边缘设备升级 BSP 成本高，不能随意升级。

### 17.7.3 温度墙/降频检测

边缘设备散热有限，长时间高负载会触发 thermal throttling，性能断崖式下降。

**Jetson 平台检测**：

```bash
# 实时查看温度、频率、功耗
sudo tegrastats --interval 1000

# 典型输出
RAM 3562/7770MB ... CPU [34%@1190,28%@1190,...] ... 
TOT 8238 mW ... PMIC@50C ... thermal@49C ... 
AO@55C ... PLL@42C ... POM_5V_IN 4375/4375 POM_5V_GPU 1312/1312
```

**关键指标**：

- `CPU [X%@freq]`：CPU 利用率与当前频率。如果频率从标称值下降，说明在降频。
- `thermal@XXC`：当前温度。超过 85°C 通常会触发保护。
- `POM_5V_IN / GPU`：功耗。

**Android 平台检测**：

```bash
# 查看温度节点
adb shell cat /sys/class/thermal/thermal_zone*/type
adb shell cat /sys/class/thermal/thermal_zone*/temp

# 查看 CPU/GPU 频率
adb shell cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
adb shell cat /sys/class/devfreq/*gpu*/cur_freq
```

**应对策略**：

| 策略 | 做法 |
|-----|------|
| **硬件散热** | 加风扇、散热片、改善风道 |
| **功耗墙调整** | 限制 TDP（如 Jetson `nvpmodel`） |
| **动态降负载** | 温度高时降低 batch size 或分辨率 |
| **间歇运行** | 推理与空闲交替，给设备散热时间 |
| **选择低功耗模型** | 用更小模型或更低精度 |

### 17.7.4 跨团队协作模板（算法 / 硬件 / 客户）

边缘问题通常需要算法、硬件、框架、客户多方协作。一个标准的 issue 模板可以大幅减少沟通成本。

```markdown
## 问题报告模板

### 基本信息
- 设备型号：NVIDIA Jetson AGX Orin 64GB
- BSP/SDK 版本：JetPack 6.0 / TensorRT-LLM 0.10.0
- 模型：Qwen2.5-7B-Instruct INT4 AWQ
- 框架：TensorRT-LLM + Triton
- 复现频率：100% / 偶发

### 问题类型
- [ ] 功能（崩溃/报错/无输出）
- [ ] 性能（慢/吞吐低）
- [ ] 精度（输出不对/指标下降）

### 复现步骤
1. 启动服务...
2. 发送 prompt...
3. 观察结果...

### 已收集日志
- 错误日志：...
- tegrastats 输出：...
- benchmark 结果：...

### 已尝试方案
- FP16 基线是否正常：...
- 降低 batch size 是否有改善：...
- 降低序列长度是否有改善：...

### 期望结果 vs 实际结果
- 期望：TTFT < 500ms，TPOT < 50ms
- 实际：TTFT 2s，TPOT 200ms

### 附件
- 最小复现脚本
- engine 配置文件
- 性能 profiling 报告
```

**团队协作原则**：

1. **先复现，再定位**：没有复现步骤的问题很难修。
2. **分层隔离**：先确认 FP16 基线是否正常，再确认量化模型，最后确认部署环境。
3. **数据说话**：用 benchmark 数字替代主观描述。
4. **客户侧信息要全**：散热条件、电源功率、并发模式、输入样本都可能是根因。

---

## 17.8 边缘上线 Checklist

在把边缘 AI 模型交付给客户之前，建议逐项检查以下清单。

### 模型与精度

- [ ] FP16 基线在目标输入上精度正确。
- [ ] 量化模型 PPL 增加 < 3%（INT4）或 < 1%（INT8）。
- [ ] 下游任务指标下降在客户可接受范围内。
- [ ] Calibration 数据覆盖真实业务分布。
- [ ] Chat template / tokenizer 与训练时一致。
- [ ] 视觉预处理（resize、normalize、interpolation）与训练一致。

### 性能与内存

- [ ] TTFT、TPOT、end-to-end latency 满足 SLA。
- [ ] 内存峰值 < 设备容量的 80%（留热切换/突发缓冲）。
- [ ] KV Cache 在长序列下不 OOM。
- [ ] 视觉编码器分辨率与延迟目标匹配。
- [ ] 已测试最大 batch size 和并发数。
- [ ] 已做 warmup，首次推理延迟在可接受范围。

### 稳定性与散热

- [ ] 连续运行 30 分钟以上无崩溃。
- [ ] 长时间运行后未触发 thermal throttling。
- [ ] tegrastats / 温度节点输出在正常范围。
- [ ] 电源功率稳定，无欠压导致降频。
- [ ] OOM 时有优雅降级策略（如缩短上下文、降低分辨率）。

### 部署与运维

- [ ] SDK / BSP / 驱动版本已锁定并文档化。
- [ ] 模型文件、配置文件、脚本已版本控制。
- [ ] 提供最小复现脚本和 benchmark 工具。
- [ ] 日志包含足够信息（错误码、shape、内存、温度）。
- [ ] OTA 升级流程已验证。
- [ ] 客户环境已做现场验收测试（FAT/SAT）。

### 安全与合规

- [ ] 模型文件有完整性校验（hash / 签名）。
- [ ] 不泄露训练数据或敏感 prompt。
- [ ] 满足目标行业的合规要求（车规、医疗、安防）。
- [ ] 权限最小化，避免 root 运行推理服务。

---

## 17.9 小结

边缘 AI 优化是一个端到端的系统工程：

1. **内存是首要约束**：通过 GQA/MQA/MLA、KV Cache 量化、StreamingLLM/H2O 等方法压缩上下文占用。
2. **量化要精要稳**：PTQ/GPTQ/AWQ/SmoothQuant 各有利弊，关键是校准数据和 task-level 精度验证。
3. **模型选型决定天花板**：原生小模型 vs 量化大模型要根据设备、任务、延迟综合判断。
4. **异构调度是落地关键**：CPU/GPU/NPU 协同、graph partitioning、内存预算、热切换都需要精细设计。
5. **多模态更复杂**：VLM/VLA 的视觉编码器是瓶颈，预处理、分辨率、帧率都需要权衡。
6. **不是所有云侧技术都适合边缘**：inflight batching 在边缘要谨慎使用。
7. **问题排查需要体系化**：功能/性能/精度 triage、版本矩阵、thermal 检测、跨团队协作缺一不可。

最终，边缘 AI 的竞争力不只取决于模型精度，更取决于**在真实设备、真实功耗、真实散热条件下能否稳定、可复现地满足客户 SLA**。这份 checklist 和优化方法论，希望能帮助团队把边缘项目从 demo 推进到量产。
