# Part 1：入门篇 — AI Infra 核心概念

> **面向角色**：刚接触 LLM 推理的工程师、面试准备中的候选人  
> **目标**：建立对推理引擎核心概念的系统性理解  
> **标星**：★ = 面试高频，必掌握；☆ = 了解即可

---

## 1.1 Transformer 推理基础 ★

### 1.1.1 自回归生成

LLM 的推理遵循**自回归**范式：

```
输入: "今天天气"
  → model forward → logits → sample → token: "真"  (step 1)
输入: "今天天气真"
  → model forward → logits → sample → token: "好"  (step 2)
输入: "今天天气真好"
  → model forward → logits → sample → token: "！"  (step 3)
...直到生成 <eos> 或达到 max_tokens
```

**核心问题**：每个 step 都需要完整执行一遍模型 forward（走过所有 80 层 Transformer + attention + FFN），但实际新增的只有 1 个 token。这就是推理的**计算效率困境**。

### 1.1.2 Prefill vs Decode 两个阶段

推理引擎将一次请求拆分为两个截然不同的阶段：

| 阶段 | 输入 | 产出 | 并行度 | 瓶颈 |
|------|------|------|--------|------|
| **Prefill** | 全部 prompt tokens（如 4096 tokens） | 第 1 个生成 token | 可完全并行 | **compute-bound**（需处理所有 prompt 位置） |
| **Decode** | 1 个新 token | 下一个 token | 逐 token 串行 | **memory-bound**（受 HBM 带宽限制，每个 token 都要读全部权重） |

```
Prefill 阶段:
  token:  [t0, t1, t2, ..., t4095]  →  一次处理所有位置 → first output token

Decode 阶段 (逐 token):
  token:  t4096  →  依赖于 t4095  →  t4097
  token:  t4097  →  依赖于 t4096  →  t4098
  ...（每次只输入 1 个新 token）
```

**为什么这个区分重要？** prefll 和 decode 对硬件的要求完全不同：
- Prefill 需要**高算力**（大量 matmul 并行），A100 这类卡已经足够
- Decode 需要**高内存带宽**（每次都要从 HBM 读全部权重），H100 的 HBM3e 优势明显

这就是 **PD 分离（Prefill-Decode Disaggregation）** 的理论基础（详见 Part 2）。

### 1.1.3 KV Cache 机制 ★

**问题**：Transformer Attention 每次都要计算所有历史 token 的 Key 和 Value。如果没有缓存，第 N 步会重复计算前 N-1 步的 K/V，计算量呈 O(N²) 增长。

**方案**：将每层的 Key 和 Value 存起来，后续步骤直接复用。

```python
# 无 KV Cache（概念性代码，实际不可行）
for step in range(max_new_tokens):
    all_tokens = prefix + generated  # 越来越长
    Q, K, V = linear_projections(all_tokens)  # 重复计算！
    output = attention(Q, K, V)

# 有 KV Cache（实际推理做法）
K_cache, V_cache = [], []
for step in range(max_new_tokens):
    q = linear_q(new_token)
    k, v = linear_kv(new_token)
    K_cache.append(k); V_cache.append(v)
    output = attention(q, K_cache, V_cache)  # 只计算新 token 的 QKV
```

**KV Cache 的内存占用**：

```
KV Cache 大小 = 2 × n_layers × n_kv_heads × head_dim × max_seq_len × dtype_bytes
```

以 Llama-2-70B 为例：
```
n_layers = 80, n_kv_heads = 8 (GQA), head_dim = 128, max_seq_len = 4096, FP16
= 2 × 80 × 8 × 128 × 4096 × 2 = 1.34 GB per request
```

这意味着**一张 80GB GPU 的 KV Cache 只能支持约 50 个并发 4096-token 的请求**。KV Cache 管理是整个推理引擎最核心的设计挑战。

### 1.1.4 GQA/MQA：减少 KV Cache 的注意力变体

| 注意力变体 | Q heads | KV heads | KV Cache 节省 | 精度损失 |
|-----------|---------|----------|---------------|---------|
| MHA（标准）| H | H | 基准 | 0 |
| GQA（Llama-2-70B）| H | H/8 | **8x** | 极小 |
| MQA（PaLM）| H | 1 | **Hx** | 轻微 |

GQA（Grouped Query Attention）将多个 Q head 共享同一对 K/V head，显著降低 KV Cache 占用。**几乎所有现代大模型都使用 GQA 或 MQA**。

---

## 1.2 推理性能指标 ★

### 1.2.1 核心指标定义

| 指标 | 全称 | 含义 | 优化方向 |
|------|------|------|----------|
| **TTFT** | Time To First Token | 从收到请求到产出第一个 token 的时间 | 优化 prefill 速度、调度优先级 |
| **TPOT** | Time Per Output Token | 每生成一个 token 的平均时间（不含首 token） | 优化 decode 阶段、减少 memory 瓶颈 |
| **Throughput** | 吞吐量 | 每秒产出 token 总数（跨所有请求） | 增大 batch、提高 GPU 利用率 |
| **QPS** | Queries Per Second | 每秒完成的请求数 | 减少端到端延迟、提高并发 |
| **Latency (P50/P95/P99)** | 延迟分位数 | 请求端到端的完成时间 | 调度公平性、长尾优化 |
| **Goodput** | 有效吞吐 | 仅计算成功（非 abort）请求的吞吐 | 与前几个指标正交，体现服务质量 |

### 1.2.2 TTFT vs TPOT 的 trade-off

```
请求: "写一篇关于 AI 的论文" (prompt ~100 tokens)

TTFT = 500ms   ← 用户感知的"响应速度"
TPOT = 30ms    ← 用户感知的"生成速度"

总延迟 = TTFT + n_tokens × TPOT
       = 500ms + 1000 × 30ms = 30.5s
```

**面试常考题**：如何降低 TTFT？
- 答：增加 prefill batch 的 token 预算、chunked prefill（将长 prompt 拆分为多个 chunk，混合 prefill/decode）、PD 分离（专门的高算力 P 节点负责 prefill）

**面试常考题**：TPOT 主要由什么决定？
- 答：HBM 带宽。Decode 阶段每次 forward 需要读取全部权重（如 70B × 2 bytes = 140GB），H100 HBM3e 带宽约 3.35 TB/s，理论最快 TOPT ≈ 140GB / 3.35TB/s ≈ 42ms

### 1.2.3 Batching 如何影响指标

| Batch Size | TTFT | TPOT | Throughput | GPU 利用率 |
|-----------|------|------|------------|-----------|
| 1 | 最低（立刻处理）| 最低（无竞争）| 最低 | <5% |
| 8 | 中等 | 轻微增加 | 高 | ~30% |
| 64 | 较高（排队）| 增加明显 | 最高 | ~60-80% |
| 256 | 很高 | 显著增加 | 可能下降（mem 压力）| ~90%+ |

**洞察**：batch size 不是越大越好——当 KV Cache 耗尽时，会触发 preemption（抢占），反而降低整体吞吐。

---

## 1.3 批处理：从静态到连续 ★

### 1.3.1 Static Batching（静态批处理）

```
Batch = [req1, req2, req3]  ←  三个请求必须同时开始
  →  如果 req1 生成 100 tokens, req2 生成 10 tokens, req3 生成 500 tokens
  →  req2 完成后 GPU 资源浪费（padding）
  →  整个 batch 必须等 req3 完成
```

**问题**：GPU 利用率极低（~20-30%）。最早期的推理服务（如 HF TGI v0.x）使用这种方式。

### 1.3.2 Continuous Batching（连续批处理）★

```
Step 1: Batch = [req1(prefill), req2(prefill), req3(decode)]
Step 2: Batch = [req1(decode), req2(decode), req3(decode), req4(prefill)]
Step 3: Batch = [req1(decode), req3(decode), req4(decode)]  ← req2 完成，移出
Step 4: Batch = [req1(decode), req3(decode), req4(decode), req5(prefill)]
```

**核心思想**：每个 step 结束后重新组成 batch——新请求可以随时加入（prefill），完成的请求随时移出。这就是 vLLM 和 SGLang 都使用的**核心调度机制**。

**实现要点**：
1. Scheduler 维护 Waiting Queue（等待的请求）和 Running Batch（正在执行的请求）
2. 每个 step 从 Waiting Queue 中选择请求加入 Running Batch
3. KV Cache 管理必须支持**动态分配和回收**（PagedAttention / RadixAttention 的核心价值）
4. 需要**chunked prefill**：长 prompt 不要一次 prefill 完（会导致 TPOT 的 spike），而是切成多个 chunk，均匀分配到多个 step

### 1.3.3 Chunked Prefill ★

```
传统 prefill (4096 token prompt):
  Step N: [reqA(prefill 4096 tokens)]  ← TPOT 大幅尖刺（~200ms）

Chunked prefill:
  Step N:   [reqA(prefill 1024 tokens)] + [decode batch]
  Step N+1: [reqA(prefill next 1024)]   + [decode batch]
  Step N+2: [reqA(prefill next 1024)]   + [decode batch]
  Step N+3: [reqA(prefill last 1024)]   + [decode batch]
  Step N+4: [reqA(decode)]              + [decode batch]  ← 平滑过渡
```

**权衡**：chunked prefill 让 TTFT 轻微增加（被切分了），但大幅改善 TPOT 的稳定性（消除了尖刺）。

---

## 1.4 内存与显存 ★

### 1.4.1 GPU 内存分层

```
┌──────────────────────────────────────┐
│  HBM (High Bandwidth Memory)         │  ← 80GB (A100) / 141GB (H200)
│  ├── 模型权重 (Weights)              │  ← ~14GB (7B FP16), ~140GB (70B FP16)
│  ├── KV Cache                        │  ← 动态分配，受 max_num_seqs 限制
│  ├── 激活值 (Activations)            │  ← 临时，每层 forward 后释放
│  └── CUDA Context / 其他              │  ← ~1-2GB
│  带宽: ~2 TB/s (A100) / ~3.35 TB/s (H100) / ~4.8 TB/s (B200)
└──────────────────────────────────────┘
        ↕ (PCIe 4.0: ~32 GB/s, NVLink: ~900 GB/s)
┌──────────────────────────────────────┐
│  CPU DRAM                            │  ← 512GB-2TB
│  ├── 模型权重副本（不使用时 swap）     │
│  └── Swapped KV Cache blocks         │
└──────────────────────────────────────┘
```

### 1.4.2 内存墙 (Memory Wall) ★

**推理的核心矛盾**：

```
Decode 阶段：
  每次 forward 需要读取: 模型权重 (70B × 2 bytes = 140 GB)
  每次 forward 计算量: ~70B × 2 FLOPs = 140 GFLOPs
  算术强度 = 140 GFLOPs / 140 GB = 1 FLOP/byte

H100 HBM3e 带宽 (3.35 TB/s) vs H100 FP16 算力 (989 TFLOPs)
  最大可用算力 = 3.35 TB/s × 1 FLOP/byte = 3.35 TFLOPs  (仅 0.3% 利用率！)
```

这意味着在 decode 阶段，**GPU 99%+ 的计算单元在闲着等数据**。这就是为什么推理优化的核心是**减少内存读写**：
- 量化 → 权重体积缩小（FP16→INT4，4x 减少）
- KV Cache 量化 → KV 体积缩小
- FlashAttention → 避免将完整 attention matrix 写入 HBM
- Batching → 一次读取权重，服务多个请求（摊销带宽成本）

### 1.4.3 KV Cache 显存计算（实战）

**面试题**：一个 70B 模型（GQA, 80 layers, 8 KV heads, 128 head_dim, FP16，max_seq_len=4096），H100-80G 最多支持多少个并发请求？

```
单请求 KV Cache = 2 × 80 × 8 × 128 × 4096 × 2 / (1024³) ≈ 1.25 GB

H100-80G 可用：
  模型权重: ~140 GB → 需要 TP=2（两张卡各 70GB）
  留 5GB 给激活值/系统

每张卡可用 KV Cache ≈ 10 GB（80 - 70 - 5，省略了TP细节）
 但 H100 通常用 TP=4（每卡 35GB），则 KV Cache ≈ 45 GB / 卡

并发请求数 ≈ 45 / 1.25 ≈ 36 requests per card

4 卡 TP 集群: 36 × 4 / 4 = 36 concurrent requests  ← 注意 TP 不增加 KV Cache 总量
```

**关键洞察**：TP 会增加 KV Cache 开销（每张卡有完整的 KV），所以是拿**内存换算力**。这在大型 MoE 模型上尤其显著。

---

## 1.5 量化基础 ☆→★（面试高频）

### 1.5.1 为什么需要量化？

| 精度 | 每参数体积 | 70B 模型大小 | HBM 带宽需求（TPOT=40ms 基准）|
|------|-----------|-------------|------------------------------|
| FP32 | 4 bytes | 280 GB | 7 TB/s |
| FP16/BF16 | 2 bytes | 140 GB | 3.5 TB/s |
| INT8 | 1 byte | 70 GB | 1.75 TB/s |
| FP8 | 1 byte | 70 GB | 1.75 TB/s |
| INT4 | 0.5 bytes | 35 GB | 0.875 TB/s |

**结论**：量化是解决 Memory Wall 的最直接手段。INT4 能让 70B 模型跑在单张 H100-80G 上。

### 1.5.2 量化方法分类

| 方法 | 类型 | 精度恢复手段 | 适用 GPU |
|------|------|-------------|---------|
| **GPTQ** | PTQ (W4A16) | 逐层最优量化 + Hessian | 所有 NVIDIA |
| **AWQ** | PTQ (W4A16) | 激活值感知权重缩放 | 所有 NVIDIA |
| **SmoothQuant** | PTQ (W8A8) | 平滑 pre-LayerNorm activation 异常值 | 所有 NVIDIA |
| **FP8** | PTQ (W8A8) | 原生精度（E4M3/E5M2）| H100+ (FP8 原生支持) |
| **bitsandbytes** | 在线量化 (W4A16) | 逐块量化 + 反量化 | 所有 |
| **GGUF** | PTQ (各种精度) | llama.cpp 社区标准 | CPU/GPU |

### 1.5.3 AWQ 核心思想（简化版）★

AWQ 的关键洞察：**权重中只有 ~1% 的 channel（salient channels）对输出质量关键**。

```
GPTQ: 对所有 weight channel 一视同仁，找到全局最优量化参数
AWQ:  先找到 salient channels（激活值最大的维度），给它们更高的精度权重

实践：
  - 找到 activation 中绝对值最大的 1% channels
  - 对这些 channels 的 weight 乘以 scale > 1（相当于放大）
  - 量化后再除以 scale（恢复）
  - 效果：在没有额外计算开销的情况下，保护了重要 channels
```

### 1.5.4 FP8 为什么是未来

```
FP16: 1 位符号 + 5 位指数 + 10 位尾数 → 范围大但精度冗余

FP8 (E4M3): 1 位符号 + 4 位指数 + 3 位尾数 → 范围适中、推理够用
FP8 (E5M2): 1 位符号 + 5 位指数 + 2 位尾数 → 范围大、梯度用

优势：
  1. H100+ 有原生 FP8 Tensor Core → 2x 吞吐 vs FP16
  2. 精度足够推理（transformer 不需要 FP16 的精度）
  3. W8A8 同时减少权重和激活的带宽需求
  4. DeepSeek-V3 证明了 FP8 训练+推理全流程可行
```

### 1.5.5 KV Cache 量化

```
原始 KV Cache: FP16 → 每 token 每层 2 × n_kv_heads × head_dim × 2 bytes

FP8 KV Cache: 压缩到 1/2 → 如 8 KB → 4 KB per token
INT4 KV Cache: 压缩到 1/4 → 2 KB per token

实现方式：
  - 逐 token 量化（online）：每计算完一个 token，量化 KV 后再写入 cache
  - 逐 block 量化（offline）：在 block 级别管理量化参数（scale）
  
挑战：
  - KV 的数值分布不稳定（不同层、不同位置差异大）
  - 需要动态 scale（类似 SmoothQuant 思路）
  - SGLang 和 vLLM 都支持 FP8 KV Cache
```

---

## 1.6 解码策略 ☆

### 1.6.1 核心解码方式

| 策略 | 原理 | 多样性 | 使用场景 |
|------|------|--------|---------|
| **Greedy** | 每次都选概率最高的 token | 无 | 评估/benchmark |
| **Temperature Sampling** | 调整 logits → softmax 的分布尖锐度 | 可控 | 对话/创作 |
| **Top-k** | 只从概率最高的 k 个 token 中采样 | 中等 | 质量与多样平衡 |
| **Top-p (Nucleus)** | 从累积概率 ≥ p 的 token 中采样 | 中等 | 主流（ChatGPT 默认） |
| **Beam Search** | 维护 k 条最优路径 | 低 | 翻译/代码生成 |

### 1.6.2 投机解码 (Speculative Decoding) 入门

**问题**：自回归生成是**串行**的（每步 1 个 token），无法利用 GPU 并行能力。

**思路**：用一个轻量的"草稿模型"快速预测未来 N 个 token，然后让大模型**并行验证**。

```
Step 1: Draft Model (小模型，如 1B) 快速生成 [t1, t2, t3]
Step 2: Target Model (70B) 一次 forward 验证这 3 个 token
Step 3: 如果全部匹配 → 接受 → 等价于 1 步生成 3 个 token ← 3x 加速！
         如果部分匹配 → 接受匹配部分 → 拒绝不匹配的 → 回退继续
```

**为什么能加速？** 大模型一次 forward 处理 3 个 token 的时间 ≈ 处理 1 个 token 的时间（decode 阶段），所以如果草稿模型足够好，能获得 2-3x 的 decode 加速。

- vLLM 实现：`vllm/v1/spec_decode/` — 支持 Eagle、Medusa、ngram、MTP（Multi-Token Prediction）
- SGLang 实现：`sglang/srt/speculative/` — 支持 Eagle、dFlash、ngram

（投机解码的深入分析见 Part 2）

---

## 1.7 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| KV Cache | 推理的核心数据结构，占用显存的主角 |
| Prefill vs Decode | 两个阶段，两种硬件瓶颈 |
| Continuous Batching | vLLM/SGLang 的调度基础 |
| Memory Wall | decode 阶段 HBM 带宽是瓶颈 |
| PagedAttention | KV Cache 的页式管理 |
| 量化 | 对抗 Memory Wall 的最有效武器 |
| 投机解码 | 用"草稿+验证"突破串行瓶颈 |

### 源码定位速查

| 想了解... | vLLM 源码路径 | SGLang 源码路径 |
|-----------|-------------|-----------------|
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `sglang/srt/managers/scheduler.py` |
| KV Cache 管理 | `vllm/v1/core/kv_cache_manager.py` | `sglang/srt/mem_cache/radix_cache.py` |
| Model Runner | `vllm/v1/worker/gpu_model_runner.py` | `sglang/srt/model_executor/model_runner.py` |
| 量化 | `vllm/model_executor/layers/quantization/` | `sglang/srt/layers/quantization/` |
