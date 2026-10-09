# 第18章 具体模型部署案例

> **面向角色**：需要为特定大模型设计部署方案的工程师  
> **目标**：通过 4 个典型模型，展示从模型特性到部署参数的完整推导

---

## 18.1 DeepSeek-V3 / R1（671B MoE）★

### 模型特性

```
总参数: 671B
每 token 激活参数: ~37B
架构: MoE + MLA + Multi-Token Prediction
注意力: MLA (num_kv_heads = 1)
专家数: 256, 每 token top-8
上下文: 64K
```

### 关键约束

1. **MLA 单 KV head → TP 下 KV Cache 8× 重复**（第24章 Finding 1）
2. **MoE 层巨大 → 需要 EP 分布专家**
3. **激活稀疏 → 不能简单用 TP 切分所有层**

### 推荐部署方案

#### 方案 A：vLLM Wide-EP（推荐用于生产）

```bash
# 8 节点 × 8 GPU = 64 GPU
# DP=8, EP=8, TP=1
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --num-scheduler-steps 10 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

**配置逻辑**：
- TP=1：避免 MLA KV 重复
- DP=8：8 个独立 replica，前端路由
- EP=8：每个 replica 内 8 GPU 分布 256 experts（每 GPU 32 experts）
- FP8：H100 原生支持，2x 吞吐

#### 方案 B：SGLang DP + EP

```bash
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --tp 1 \
  --dp 8 \
  --ep 8 \
  --max-running-requests 256 \
  --enable-torch-compile \
  --radix-cache \
  --quantization fp8
```

### 性能预期

| 指标 | 8×H100 (FP8) | 64×H100 (FP8, DP=8) |
|------|--------------|---------------------|
| 单请求 TTFT | ~2-5s | ~0.5-1s |
| TPOT | ~25-35ms | ~25-35ms |
| Throughput | ~500-800 tok/s | ~5,000-8,000 tok/s |

### 注意事项
- DeepSeek-V3 对 EP 的 AllToAll 带宽要求极高，建议使用 NVLink/IB
- MLA 压缩 KV 后，prefix caching 收益更大
- MTP 可开启，但大规模生产收益衰减（第24章 Finding 4）

### 18.1.1 DeepSeek-V3.2 / V4 演进（2025-2026）

DeepSeek 家族的注意力机制迭代非常快，部署前必须先确认引擎版本支持：

| 版本 | 关键变化 | 部署影响 |
|------|---------|---------|
| **V3 / V3.1** | MLA + MoE + MTP | 本手册 第18章 主体案例 |
| **V3.2-Exp** | 引入 DSA（DeepSeek Sparse Attention，Indexer + top-k 选择），128K 上下文 | 需要支持 DSA 的 attention backend（SGLang `nsa_backend.py` / vLLM IndexCache） |
| **V3.2** | 上下文扩到 160K | 同上 |
| **V4-Flash** | 284B 总参 / 13B 激活，DSA2 = CSA（Compressed Sparse Attention）+ HCA（Heavily Compressed Attention）混合，原生 1M token 上下文 | 稀疏模式与 V3 完全不同，必须用新版引擎 + 专用内存池（SGLang `deepseek_v4_memory_pool.py`） |
| **V4 Pro** | 1.6T MoE，混合注意力架构，MIT license | 大集群部署，EP + 稀疏注意力协同 |

**工程要点**：
1. **DSA/CSA 的选择索引需要特殊处理**：top-k 索引如果不缓存，每个 layer 都要重复计算（vLLM IndexCache、SGLang 均在运行时复用）。
2. **KV Cache 复用粒度变化**：稀疏注意力下"跳过"的块不参与计算，但仍占据内存；Radix/Block Cache 的前缀命中逻辑需要适配稀疏 mask。
3. **引擎版本锁定**：DeepSeek 新架构往往需要配套的新 kernel（MLA → DSA → CSA/HCA 是三条不同的 kernel 路径），不能混用。
4. **成本结构变化**：V3.2-Exp 官方 API 价格下调 50%+，稀疏注意力是核心原因——这也解释了为什么 `$/1M tokens`（第10章）需要按模型代数重估。

---

## 18.2 LLaMA-3-405B（Dense 模型）★

### 模型特性

```
总参数: 405B (Dense, 无 MoE)
注意力: GQA (8 KV heads)
层数: 126
上下文: 128K
```

### 关键约束

1. **单卡放不下**：405B × FP16 = 810GB
2. **Dense 模型**：不需要 EP，重点在 TP + PP
3. **GQA**：KV Cache 相对可控

### 推荐部署方案

#### 方案：TP=8 + PP=2（16 GPU）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 128 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.92
```

**配置逻辑**：
- TP=8：单节点 8 GPU 层内并行
- PP=2：跨节点层间并行，突破单节点显存
- FP8：减少权重和 KV 体积

### 性能预期

| 指标 | 16×H100 (FP8) |
|------|---------------|
| 单请求 TTFT | ~1-3s |
| TPOT | ~30-45ms |
| Throughput | ~800-1,500 tok/s |

### 注意事项
- PP 引入 pipeline bubble，低并发下吞吐受影响
- 128K 上下文需要 KV Cache 量化或 offload
- TP=16 时通信开销可能反而降低性能

---

## 18.3 Qwen3-235B-A22B（MoE）

### 模型特性

```
总参数: 235B (MoE)
每 token 激活参数: 22B
架构: MoE + Qwen3 改进
上下文: 128K
```

### 关键约束

1. **MoE 但激活参数少**：22B 激活，TP 压力小
2. **总参数 235B**：仍需多卡
3. **128K 上下文**：KV Cache 管理重要

### 推荐部署方案

```bash
# 4 节点 × 8 GPU = 32 GPU
# TP=4, EP=8, DP=1
python -m sglang.launch_server \
  --model Qwen/Qwen3-235B-A22B \
  --tp 4 \
  --ep 8 \
  --max-running-requests 256 \
  --max-model-len 32768 \
  --radix-cache \
  --enable-torch-compile \
  --quantization fp8
```

**配置逻辑**：
- TP=4：22B 激活参数在 4 卡上可接受
- EP=8：32 GPU 分 235B 总参数中的 experts
- RadixCache：128K 上下文场景前缀复用收益大

---

## 18.4 LLaMA-3.1-70B（中小规模 Dense）

### 模型特性

```
总参数: 70B
注意力: GQA (8 KV heads)
上下文: 128K
FP16 权重: ~140GB
```

### 推荐部署方案

#### 单节点 8×H100（最常用）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --quantization fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9
```

#### 成本优化：INT4 (AWQ) 单节点 4×A100

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-3.1-70B-AWQ \
  --tensor-parallel-size 4 \
  --quantization awq \
  --max-model-len 8192 \
  --max-num-seqs 64
```

### 性能预期

| 配置 | TTFT | TPOT | Throughput |
|------|------|------|------------|
| 8×H100 FP8 | ~200ms | ~15-20ms | ~3,000-5,000 tok/s |
| 4×A100 AWQ | ~500ms | ~30-40ms | ~800-1,200 tok/s |

### 注意事项
- 70B 是"甜点"模型，单节点即可服务
- FP8 vs AWQ 的 trade-off：FP8 延迟低，AWQ 成本低
- 128K 上下文需要谨慎设置 max_num_seqs，避免 OOM

---

## 18.5 模型部署决策模板

```
Step 1: 确定模型特性
  - Dense vs MoE
  - 总参数 / 激活参数
  - Attention 类型 (MHA/GQA/MQA/MLA)
  - 上下文长度
  - 量化格式

Step 2: 计算显存需求
  - 权重: params × dtype_bytes
  - KV Cache: 2 × layers × kv_heads × head_dim × max_len × seqs × dtype
  - 激活: ~2-5GB per GPU

Step 3: 选择并行策略
  - Dense: TP (+ PP if needed)
  - MoE: EP (+ DP if high throughput)
  - MLA: avoid TP, use DP+EP

Step 4: 选择量化
  - H100+: FP8 W8A8
  - 其他: AWQ/GPTQ W4A16
  - KV Cache: FP8 if H100

Step 5: 调优关键参数
  - max_num_seqs / max-running-requests
  - chunked prefill size
  - prefix caching on/off
  - CUDA Graph on/off

Step 6: Benchmark
  - 真实负载分布
  - P99 TTFT/TPOT
  - Throughput vs Latency trade-off
```

---

## 18.6 本章小结

| 模型 | 架构 | 推荐并行 | 推荐量化 | 推荐引擎 |
|------|------|---------|---------|---------|
| DeepSeek-V3/R1 | MoE + MLA | DP+EP, TP=1 | FP8 | vLLM Wide-EP / SGLang |
| LLaMA-3-405B | Dense + GQA | TP=8 + PP=2 | FP8 | vLLM / TRT-LLM |
| Qwen3-235B-A22B | MoE + GQA | TP=4 + EP=8 | FP8 | SGLang / vLLM |
| LLaMA-3.1-70B | Dense + GQA | TP=8 | FP8 / AWQ | vLLM / SGLang |
