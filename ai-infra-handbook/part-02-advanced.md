# Part 2：进阶篇 — 高性能推理架构与优化

> **面向角色**：有 1-2 年经验的推理工程师、高级面试候选人  
> **目标**：深入理解推理引擎的核心架构决策与优化技术  
> **标星**：★ = 必须掌握（面试高频）；☆ = 加分项

---

## 2.1 Continuous Batching 与调度深度剖析 ★

### 2.1.1 vLLM Scheduler 设计

vLLM V1 的 Scheduler（`vllm/v1/core/sched/scheduler.py`）是一个**基于优先级的连续批处理器**。

**核心数据结构**：

```python
class Scheduler:
    waiting: RequestQueue       # 待调度的请求（按 FCFS + Priority 排序）
    running: dict[str, Request] # 正在执行的请求
    kv_cache_manager: KVCacheManager  # KV Cache 分配器

    max_num_running_reqs: int          # 最多同时运行的请求数
    max_num_scheduled_tokens: int      # 每步最多调度的 token 数
```

**调度流程**（每个 step 执行一次）：

```
schedule() → SchedulerOutput:
  1. 从 waiting 队列中选择请求（按 FCFS + Priority）
  2. 为每个请求检查 KV Cache 是否有可用 block
  3. 如果可以分配 → 加入 scheduled_running
  4. 如果不能分配 → 尝试 preemption（抢占低优先级请求）
  5. 返回 SchedulerOutput（包含 block_tables、token_ids 等元数据）
```

**调度策略**：
- 默认 **FCFS**（先来先服务）+ **Priority**（优先级队列）
- `max_num_scheduled_tokens` 限制每步 prefill 的 token 总量（用于 chunked prefill）
- 支持 **preemption**（当 KV Cache 不足时，抢占/换出低优先级请求的 block）

### 2.1.2 SGLang Scheduler 设计

SGLang 的 Scheduler（`sglang/srt/managers/scheduler.py`，~4300 行！）采用了更激进的优化策略。

**核心区别：RadixAttention 驱动的调度**

```python
class Scheduler:
    tree_cache: BasePrefixCache    # Radix Tree（不是简单的 block 池！）
    waiting_queue: Deque[Req]      # 等待队列
    running_batch: ScheduleBatch   # 当前运行 batch

    # 关键：每次调度前，用 Radix Tree 匹配前缀
    def get_new_batch_prefill(self):
        for req in self.waiting_queue:
            match_result = self.tree_cache.match_prefix(req.token_ids)
            # 如果前缀匹配 > 某个阈值（如 256 tokens）
            # → 可以直接复用历史 KV Cache，跳过匹配的 tokens！
```

**Overlap 调度**（SGLang 独有特色）：

```
传统调度:  CPU Scheduler  →  GPU Forward  →  CPU Scheduler  →  GPU Forward  → ...
                     ↑__________|  (GPU 空闲时 CPU 在调度)

SGLang:     [CPU Scheduler]                        [CPU Scheduler]
                     ↘                                   ↗
                        GPU Forward (step N)   GPU Forward (step N+1)
                     ↗                                   ↘
            [Result Processing]                 [Result Processing]
            
效果: CPU 和 GPU 完全并行，消除 CPU 调度开销（~2-10ms per step → 几乎免费）
```

这就是 `scheduler.py` 中 `overlap_utils.py`、`batch_overlap/` 模块的作用——通过 CUDA Stream 管理实现计算和调度的 overlap。

### 2.1.3 Jump-Forward / Fast-Forward (SGLang 独有) ☆

当 SGLang 检测到请求有**长前缀匹配**（如多个请求共享同一个 system prompt），它会：

1. **Jump-Forward**：直接跳过已被 Radix Cache 命中的 tokens，在 forward 时通过 attention mask 忽略它们
2. **Fast-Forward**：对于 decode 阶段被 pause 的请求，快速恢复到之前的 KV Cache 状态

```python
# schedule_policy.py: match_prefix_for_req()
match_result = tree_cache.match_prefix(MatchPrefixParams(
    key=RadixKey(token_ids=req.origin_input_ids + req.output_ids),
    ...
))
# req.prefix_indices → 告诉 Model Runner 哪些位置的 KV 已缓存
```

---

## 2.2 并行策略：TP / PP / EP / DP / SP ★

### 2.2.1 Tensor Parallelism (TP) ★

**切分方式**：每层的权重矩阵按列/行切分到多个 GPU。

```
MHA Attention Layer (8 heads, TP=2):
  GPU 0: W_Q[:, :4], W_K[:, :4], W_V[:, :4], W_O[:4, :]
  GPU 1: W_Q[:, 4:], W_K[:, 4:], W_V[:, 4:], W_O[4:, :]

FFN Layer (intermediate_size=28672, TP=2):
  GPU 0: W_gate[:, :14336], W_up[:, :14336], W_down[:14336, :]
  GPU 1: W_gate[:, 14336:], W_up[:, 14336:], W_down[14336:, :]
```

**通信模式**：
- FFN: f → AllReduce(g) → g → AllReduce(f)
- Attention: f → AllReduce(g) → g → AllReduce(f)

**关键特性**：
- 每张卡存储**切分后的权重**（内存降低 N 倍）
- 每张卡存储**完整的 KV Cache**（内存不降低！）
- 通信开销随 TP 大小增长（AllReduce 的通信量与数据量成正比）

**适用场景**：
- 单模型 > 单卡显存（如 70B 模型 → TP=2~8）
- 需要低延迟（层内通信 vs 层间通信，TP 比 PP 延迟更低）

### 2.2.2 Pipeline Parallelism (PP) ★

**切分方式**：模型按层切分到多个 GPU。

```
Llama-80 layers, PP=4:
  GPU 0: Layers 0-19
  GPU 1: Layers 20-39
  GPU 2: Layers 40-59
  GPU 3: Layers 60-79
```

**关键问题：Pipeline Bubble**

```
Time →   [GPU0: L0-19] [GPU0: idle        ] [GPU0: L0-19]
         [GPU1: idle  ] [GPU1: L20-39     ] [GPU1: idle     ]
         [GPU2: idle  ] [GPU2: idle       ] [GPU2: L40-59   ]
                        ↑ pipeline bubble
```

**推理中的 PP**：
- 推理没有 optimizer state，PP 的主要好处是**减少每个 GPU 的权重内存**
- 但 PP 在推理中不如 TP 常用，因为 TOPT 对延迟敏感，pipeline bubble 难以接受
- 实际中更常见的做法是 **TP + PP 混合**：如 Llama-405B → TP=8 + PP=2

### 2.2.3 Expert Parallelism (EP) — MoE 专用 ★

**MoE 模型特点**：每个 token 只激活部分专家（如 8/256），计算量只占总参数的 ~10%。

```
DeepSeek-V3: 671B 总参数 → 每 token 激活约 37B

MoE Layer:
  Router: W_gate → Softmax → Top-k selection
  Experts: E1, E2, ..., E256  (每个专家都是一个小 FFN)
  对每个 token: 只计算 top-k 个专家 → 加起来
```

**EP 的切分方式**：

```
EP=4, 256 experts:
  GPU 0: Experts 0-63
  GPU 1: Experts 64-127
  GPU 2: Experts 128-191
  GPU 3: Experts 192-255

每个 token:
  1. Router 在本地计算 top-k expert ids
  2. 通过 All-to-All 将 token 分发到对应 GPU
  3. 各 GPU 计算自己的 experts
  4. 通过 All-to-All 将结果汇总回来
```

**通信成本**：All-to-All 的开销随 EP 增大而增长（O(N²) 连接数）。需要仔细 trade-off：EP 越大，每卡内存越少，但通信开销越大。

**实战考量**：DeepSeek-V3/R1 这类巨型 MoE 模型通常 TP=1 + EP=较大，因为 MoE FFN 是最重的部分。

### 2.2.4 DP / SP 及其他并行

| 策略 | 切分维度 | 通信 | 推理中适用性 |
|------|---------|------|------------|
| **DP** | Batch → 多个副本 | 无通信（推理）/ AllReduce (训练) | 高吞吐部署 |
| **SP (Sequence Parallelism)** | 序列长度维度 | AllReduce/ReduceScatter | 长序列（>32K）|
| **CP (Context Parallelism)** | 类似 SP，针对 Attention | Ring Attention / AllGather | 极长序列（>128K）|

**DP 在推理中的特殊性**：由于没有梯度同步，推理 DP 每个 replica 完全独立运行，前端做简单的 round-robin 路由即可。但需要在模型更新时同步权重。

---

## 2.3 PD 分离 (Prefill-Decode Disaggregation) ★

### 2.3.1 为什么需要 PD 分离？

回顾 1.1.2 节的结论：prefill 是 compute-bound，decode 是 memory-bound。同一个 GPU 无法同时高效处理两者。

```
同构部署 (colocate):
  H100 × 8:
    Prefill 占 60% 算力, 30% HBM → P/D 竞争同一个 GPU 资源
    Decode  占 5% 算力,  80% HBM → 互相影响对方延迟

PD 分离 (disaggregate):
  P Pool: A100 × 8 (高 FP16 算力, HBM 不重要)
  D Pool: H100 × 4 (高 HBM 带宽, 算力不重要)
  P → D: KV Cache 通过网络传输 (NCCL/RDMA/Mooncake)
```

**收益**：
1. **硬件异构降本**：P 用便宜的 A100，D 用贵但带宽高的 H100
2. **独立弹性伸缩**：P 和 D 独立扩容（读多写少场景可以 D 多 P 少）
3. **批量优化**：P 可以做更大的 batch（token-level batching），D 可以做并发 decode

### 2.3.2 SGLang 的 PD 分离实现 ★

SGLang 在 `sglang/srt/disaggregation/` 中实现了完整的 PD 分离架构：

```
Prefill Server (prefill.py):
  1. BootstrapQueue: 初始化 KV Sender + 握手
  2. WaitingQueue: PrefillAdder 从队列取请求
  3. 执行 prefill forward
  4. InflightQueue: 通过 KV Sender 传输 KV Cache 到 Decode Server

Decode Server (decode.py):
  1. PreallocQueue: 初始化 KV Receiver + 预分配 KV 内存
  2. TransferQueue: 轮询 Receiver 等待传输完成
  3. WaitingQueue: 构造 PrebuiltExtendBatch（跳过 prefill forward）
  4. RunningBatch: 合并到 running batch 执行 decode
```

**KV 传输后端** (`disaggregation/` 目录下)：
- `mooncake/`: Mooncake SDK 集成（腾讯元宝的 KV 传输框架）
- `nixl/`: NVIDIA NIXL（NVLink 跨节点传输）
- `mori/`: Mori 传输后端
- `ascend/`: 华为昇腾专用 backend

### 2.3.3 Mooncake KV Transfer ☆

Mooncake（来自腾讯/Moonshot 的 Splitwise 工程）核心思想：

```
传统 NCCL 传输: 粗粒度 → P 完成全部 prefill → 传输完整 KV Cache → D 开始 decode
                 (延迟 = prefill 时间 + 传输时间 + decode 时间)

Mooncake:       细粒度 → P 每计算完一层，即刻传输该层的 KV 到 D
                 (延迟 ≈ max(prefill 时间, 传输时间) + decode 时间)
                 
                 使用 RDMA + chunked transfer
                 支持分层传输（layer-wise pipelining）
```

SGLang 的 Mooncake 集成在 `disaggregation/mooncake/` 中，通过 RDMA 实现低延迟 KV 传输。

### 2.3.4 HiCache 三级缓存 ☆

```
L1: GPU HBM (热缓存)
  ├── 当前在 GPU 上的 KV Cache (最快, ~3.35 TB/s)
  │
L2: CPU DRAM (温缓存)
  ├── Swapped out KV Cache blocks
  ├── 带宽: NVLink ~900 GB/s or PCIe ~32 GB/s
  │
L3: 分布式存储 / 共享内存 (冷缓存)
  ├── HiCache: 跨节点共享 KV Cache (RDMA)
  ├── 可以复用历史请求的前缀
  └── 跨实例共享（如多个模型实例共享 system prompt 的 KV）
```

SGLang 的 HiCache 实现在 `mem_cache/hicache_storage.py` 和 `disaggregation/decode_hicache_mixin.py`。

**HiCache 的价值**：
- 在 LLM 场景中，system prompt 往往相同 → 可以全局复用
- 多轮对话中，历史轮次 KV 可以恢复（免除重复 prefill）
- 跨模型场景：同一 prefix 的 KV 被多个下游模型复用

---

## 2.4 Attention 优化 ★

### 2.4.1 PagedAttention (vLLM 核心创新) ★

**问题**：KV Cache 的显存碎片化。如果为每个请求预留连续的 max_seq_len 空间，浪费严重（大部分请求用不完）。

**PagedAttention 思想**：

```
传统: 每个请求分配连续 KV Cache
  Req1: [████████░░░░░░░░]  (预留 4096，实际用 500 → 87% 浪费)

PagedAttention: 页式管理，block 粒度分配
  Block Size = 16 tokens
  Req1: [B0][B3][B7][B2]...  (按需分配 16-token blocks)
  Req2: [B1][B5][B8]...
```

**Block Table**：

```
Request → Block Table (映射虚拟block→物理block)
  Req1: [block_0 → B0, block_1 → B3, block_2 → B7, ...]
  
Attention 计算:
  Q = compute for new token
  K = gather from B0, B3, B7... (通过 block table 索引)
  V = gather from same blocks
  attn = softmax(Q @ K^T / sqrt(d)) @ V
```

**源码定位**：
- vLLM: `vllm/v1/core/kv_cache_manager.py` — 管理 block 池、分配/回收
- vLLM: `vllm/v1/core/kv_cache_utils.py` — `KVCacheBlock` 数据结构

### 2.4.2 FlashAttention / FlashInfer ★

| 实现 | 特点 | 适用场景 |
|------|------|---------|
| **FlashAttention-2** | IO-aware, 分块计算，避免完整 attention matrix | 通用 |
| **FlashAttention-3** | H100 专用，FP8 支持，更细粒度并行 | H100+ |
| **FlashInfer** | 针对推理场景优化的 attention kernel 集合 | 推理（SGLang 默认）|
| **xFormers** | Meta 出品，Memory-Efficient Attention | 训练/推理 |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/3 专用，KV 压缩 | DeepSeek 系列 |

**FlashInfer 对推理的特殊优化**：
- `decode_attention`: 针对 token-by-token decode 优化的 kernel（batch > 1 时显著优于 FlashAttention）
- `prefill_attention`: 针对长序列 prefill 的优化
- `append_attention`: 增量 KV cache 追加操作

SGLang 深度集成了 FlashInfer（`sglang/srt/layers/attention/flashinfer_backend.py`），vLLM V1 也支持 FlashInfer。

### 2.4.3 MLA (Multi-head Latent Attention) - DeepSeek 贡献 ★

MLA 的核心思想：**压缩 KV Cache**。

```
标准 Attention:
  Q = W_Q @ x, K = W_K @ x, V = W_V @ x
  KV Cache: 完整的 K, V 矩阵

MLA:
  C_KV = W_DKV @ x           # 低维潜在表示（如 512 dims）
  K = W_UK @ C_KV             # 上投影到完整 K
  V = W_UV @ C_KV             # 上投影到完整 V
  
  KV Cache 只存 C_KV！        # 512 dims vs 128*heads dims
  
  以 DeepSeek-V2 为例：KV Cache 从 128 * 128 = 16K → 512 → ~10x 压缩！
```

**MLA 的 trade-off**：
- 收益：KV Cache 压缩 5-10x，同等显存支持更多并发
- 代价：decode 时每次需要上投影（额外的 matmul），但因为 decode 是 memory-bound，所以影响不大

---

## 2.5 长文本与多模态 ☆

### 2.5.1 RoPE 与位置编码扩展

```
RoPE (Rotary Position Embedding):
  - 通过旋转矩阵编码相对位置信息
  - 优势：外推到训练长度外时的衰减可控

YaRN (Yet another RoPE extensioN):
  - 通过 NTK-aware 插值扩展上下文窗口
  - Llama-2-4K → 32K 的扩展方式
```

### 2.5.2 长上下文优化的关键技术

| 技术 | 原理 | 加速比 |
|------|------|--------|
| **Ring Attention** | 序列分片到多 GPU，环形传递 K/V | 线性扩展 |
| **DistFlashAttn** | 分布式 FlashAttention（序列维 TP）| ~N×（N GPU）|
| **KV Cache Offload** | 不活跃 block 换出到 CPU | ∞（不限制 GPU 内存）|
| **StreamingLLM** | 保留 attention sink + 最近 window | O(1) 显存 |

### 2.5.3 多模态推理

vLLM 和 SGLang 都支持以下多模态模型：
- **LLaVA / LLaVA-NeXT**: Image → Vision Encoder → Projection → LLM
- **Qwen-VL / Qwen2-VL**: 原生视觉语言模型
- **Pixtral / InternVL**: 动态分辨率多模态

**多模态推理的特殊挑战**：
1. **Image token 的 KV Cache** 比 text token 更长（一张图 → 数百到数千 tokens）
2. **Preprocessing**（图像缩放/裁剪）在 CPU 上耗时，需要异步处理
3. **Batch 混合**（图文请求混批）需要处理不同模态的 token 布局

SGLang 在 `sglang/srt/managers/multimodal_processor.py` 和 `sglang/srt/multimodal/` 中实现了这些逻辑。

---

## 2.6 投机解码深度剖析 ★

### 2.6.1 Eagle / Medusa / MTP

| 方法 | Draft 模型 | 特点 |
|------|-----------|------|
| **Medusa** | 多个 prediction heads（从大模型中间层 fork）| 无需额外模型 |
| **Eagle** | 小 decoder（1-2 层 Transformer） | 更准确、更快 |
| **MTP (Multi-Token Prediction)** | 大模型自带的多 token 预测头（DeepSeek-V3）| 无需额外模型 |
| **ngram** | n-gram 模式匹配 | 零额外计算 |
| **dFlash (SGLang)** | 轻量 draft 模型 + Eagle | 延迟最优 |

### 2.6.2 Eagle 工作原理 ★

```
大模型 (Target):  80 layers Transformer
Eagle Draft:      1-2 layers Transformer（共享大模型的 embedding + LM head）

Step 1: 大模型 forward → hidden states (last layer)
Step 2: hidden states → Eagle Draft → draft 3 tokens
Step 3: 大模型验证（一次 forward，处理 1 real + 3 draft tokens）
Step 4: 接受/拒绝采样（使用 speculative sampling 的 accept probability 公式）
```

**接受概率**：
```
给定 draft token x_d 和 target distribution p:
  accept_prob = min(1, p(x_d) / q(x_d))
  
直观理解：
  - 如果 target 认为 draft token 的概率高 → 大概率接受
  - 如果 target 认为 draft token 的概率低 → 可能拒绝
```

**SGLang 的 Eagle 实现**：
- `sglang/srt/speculative/eagle_worker_v2.py`: Eagle v2 worker
- `sglang/srt/speculative/eagle_info.py`: Eagle 元数据管理
- `sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`: CUDA Graph 加速 draft

**vLLM 的 SpecDecode 实现**：
- `vllm/v1/spec_decode/eagle.py`: Eagle proposer
- `vllm/v1/spec_decode/medusa.py`: Medusa proposer
- `vllm/v1/spec_decode/draft_model.py`: 通用 draft model 接口

### 2.6.3 MTP (Multi-Token Prediction) — DeepSeek 贡献 ☆

DeepSeek-V3 原生支持 MTP：模型训练时就加入了额外的 prediction heads，可以直接预测未来 1-4 个 token。

```
标准 Transformer:
  h → LM Head → token(t)

MTP Transformer:
  h → LM Head → token(t)
  h → MTP Head 1 → token(t+1)
  h → MTP Head 2 → token(t+2)
  h → MTP Head 3 → token(t+3)
  
训练时: 联合优化所有 head 的 loss
推理时: 主 head 生成 token，MTP heads 作为"免费"的 draft
```

**MTP vs Eagle**：
- MTP 不需要额外模型 → 零额外显存和参数 → **更好的成本效率**
- MTP 的 draft 质量通常不如专门的 Eagle 模型 → **accept rate 可能更低**
- DeepSeek-V3 的 MTP 在实践中达到 ~1.5-2x 的 decode 加速

---

## 2.7 量化进阶：INT4/FP8 实战 ★

### 2.7.1 W4A16 vs W8A8 vs FP8 的选择

| 方案 | 权重 | 激活 | HBM 节省 | 精度损失 | 硬件要求 |
|------|------|------|----------|---------|---------|
| W4A16 (AWQ/GPTQ) | INT4 | FP16 | 4x 权重 | 轻微（<1% perplexity）| 所有 CUDA GPU |
| W8A8 (SmoothQuant) | INT8 | INT8 | 2x 全部 | 极小 | 所有 CUDA GPU |
| FP8 (E4M3) | FP8 | FP8 | 2x 全部 | 几乎无 | H100+ |
| W4A8 | INT4 | INT8 | 4x 权重+2x 激活 | 中等 | 需要特殊 kernel |

### 2.7.2 vLLM 的量化支持结构

```
vllm/model_executor/layers/quantization/
├── awq/               # AWQ 量化
├── gptq/              # GPTQ 量化
├── fp8.py             # FP8 (E4M3/E5M2)
├── compressed_tensors/ # 稀疏+量化混合
├── kv_cache.py         # KV Cache 量化
└── ...
```

### 2.7.3 SGLang 的量化支持

```
sglang/srt/layers/quantization/
├── awq/               # AWQ
├── gptq/              # GPTQ
├── fp8.py / fp8_utils.py / fp8_kernel.py  # FP8 完整支持
├── int8_kernel.py     # INT8 kernel
├── w8a8_int8.py       # W8A8 量化
├── modelopt_quant.py  # NVIDIA ModelOpt 集成
└── ...
```

### 2.7.4 KV Cache 量化的实现细节 ☆

```
FP16 KV → FP8 KV:
  1. 计算完当前 token 的 K, V (FP16)
  2. K_fp8 = K_fp16.to(torch.float8_e4m3fn)  # 原生转换
  3. 写入 KV Cache (FP8 block)
  4. 后续 attention 计算时从 FP8 cache 读取

逐 token vs 逐 block 量化:
  - 逐 token: 每个 token 独立 scale → 精度高 → 开销大
  - 逐 block: block 内共享 scale → 精度略低 → 开销小
  - vLLM/SGLang 默认使用逐 block，因为 block 内 token 的值分布相近
```

---

## 2.8 本章小结

| 领域 | 核心知识点 | 重要程度 |
|------|-----------|---------|
| 调度 | Continuous Batching + Chunked Prefill + Overlap | ★★ |
| 并行 | TP（每层切分）+ EP（MoE 专家分布）+ PP（层间流水）| ★★ |
| PD 分离 | Prefill/Decode 分离 + Mooncake KV 传输 + HiCache 三级缓存 | ★ |
| Attention | PagedAttention + FlashInfer + MLA | ★★ |
| 长文本 | Ring Attention + KV Offload | ☆ |
| 投机解码 | Eagle 架构 + MTP + Accept Probability | ★ |
| 量化 | AWQ/GPTQ (W4A16) + FP8 (W8A8) + KV Cache 量化 | ★★ |
