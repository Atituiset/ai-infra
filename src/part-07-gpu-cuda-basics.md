# Part 7 补充：GPU 架构与 CUDA 编程基础

> **面向角色**：需要理解推理性能瓶颈的工程师  
> **目标**：补齐 AI Infra 工程师必须掌握的 GPU 硬件与 CUDA 基础  
> **标星**：★ = 面试常问，必须掌握

---

## 7.1 GPU 硬件基础 ★

### 7.1.1 NVIDIA GPU 架构演进

| 架构 | 代表型号 | 关键特性 | 年代 |
|------|---------|---------|------|
| Ampere | A100 | TF32, Sparse Tensor Core | 2020 |
| Hopper | H100/H200 | FP8 Tensor Core, HBM3e, NVLink4 | 2022 |
| Blackwell | B100/B200 | FP4, 2nd Gen Transformer Engine | 2024 |
| Rubin | R100 | 下一代，2026 后 | 2025+ |

### 7.1.2 GPU 核心组件

```
┌──────────────────────────────────────┐
│  GPU (H100)                           │
│  ┌─────────────────────────────────┐ │
│  │  GPC (Graphics Processing Clusters)│ │
│  │  ┌───────────────────────────┐  │ │
│  │  │  SM (Streaming Multiprocessor) │  │ │  ← 144 个 SM (H100)
│  │  │  · CUDA Cores (INT/FP)     │  │ │
│  │  │  · Tensor Cores (FP16/FP8) │  │ │
│  │  │  · Register File (~256 KB) │  │ │
│  │  │  · Shared Memory (L1, ~228 KB)│ │ │
│  │  │  · L2 Cache                 │  │ │
│  │  └───────────────────────────┘  │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │  HBM (High Bandwidth Memory)    │ │  ← 80 GB (A100) / 141 GB (H200)
│  │  带宽: ~3.35 TB/s (H100)        │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
```

### 7.1.3 关键性能指标 ★

| 指标 | H100 SXM | A100 SXM | 意义 |
|------|----------|----------|------|
| FP16 Tensor Core | 989 TFLOPS | 312 TFLOPS | 半精度算力 |
| FP8 Tensor Core | 1,979 TFLOPS | 不支持 | 8 位浮点算力 |
| HBM 带宽 | 3.35 TB/s | 2.0 TB/s | 内存瓶颈 |
| NVLink 带宽 | 900 GB/s | 600 GB/s | 卡间高速互联 |
| PCIe 带宽 | ~64 GB/s | ~64 GB/s | 卡与 CPU 互联 |
| SM 数量 | 132/144 | 108 | 并行度 |

**核心洞察**：LLM 推理 decode 阶段 FP16 算力利用率通常 <5%，因为瓶颈在 HBM 带宽，不在算力。

### 7.1.4 内存层次与延迟

```
寄存器 (Register):   ~1 cycle      ← 由编译器/线程自动管理
Shared Memory:       ~20-30 cycles ← CUDA block 内共享，显式管理
L1 Cache:            ~30-60 cycles
L2 Cache:            ~200 cycles   ← H100 约 50 MB
HBM:                 ~400-600 cycles ← 80GB+
CPU DRAM:            ~10,000 cycles ← 通过 PCIe/NVLink
```

**FlashAttention 的核心价值**：通过分块（tiling）将 attention 计算保持在 SRAM（Shared Memory）内，避免频繁访问 HBM。

---

## 7.2 CUDA 编程基础 ★

### 7.2.1 Kernel / Grid / Block / Thread

```
Kernel: 在 GPU 上执行的函数
  Grid  → 多个 Block
  Block → 多个 Thread (最多 1024)

CUDA Thread Hierarchy:
  Grid(2, 2):
    Block(0,0) → Thread(0..1023)
    Block(0,1) → Thread(0..1023)
    Block(1,0) → Thread(0..1023)
    Block(1,1) → Thread(0..1023)
```

### 7.2.2 重要 CUDA 概念

| 概念 | 含义 | 面试关联 |
|------|------|---------|
| **Warp** | 32 个 thread 一组，SIMT 执行 | warp divergence 会降低效率 |
| **Coalesced Memory Access** | 相邻 thread 访问相邻内存地址 | 合并访问可最大化 HBM 带宽 |
| **Shared Memory Bank Conflict** | 多个 thread 同时访问同一个 bank | 导致串行化，降低性能 |
| **Occupancy** | 每个 SM 上活跃 warp 数 / 最大 warp 数 | 高 occupancy 隐藏延迟 |
| **Latency Hiding** | 用足够多的 warp 掩盖内存延迟 | LLM kernel 优化核心 |

### 7.2.3 一个极简 CUDA Kernel（概念）

```cuda
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// 启动: add<<<n/256, 256>>>(a, b, c, n);
```

### 7.2.4 CUDA Kernel 调优方向

```
1. 内存访问模式
   - 合并访问 (coalesced)
   - 使用 Shared Memory 减少 HBM 访问
   
2. 计算强度
   - 增加每个 thread 的计算量
   - 减少 kernel launch 次数
   
3. Occupancy
   - 平衡 register 使用与 block 大小
   - 避免过度使用 shared memory
   
4. Tensor Core 利用
   - 使用 WMMA / CUTLASS / cuBLAS
   - 确保矩阵维度对齐 (8/16/32 倍数)
```

---

## 7.3 推理性能调优 Checklist ★

### 7.3.1 延迟敏感场景

```
□ 使用 FP8 / INT4 量化减少 HBM 读取
□ 启用 CUDA Graph 减少 kernel launch overhead
□ 选择正确的 attention backend (decode 优先 FlashInfer)
□ 避免 batch size 过大导致 TPOT 上升
□ 使用 chunked prefill 平滑 TTFT 尖刺
□ PD 分离：prefill 不阻塞 decode
□ 投机解码：Eagle / MTP / dFlash
```

### 7.3.2 吞吐敏感场景

```
□ 最大化 batch size（直到 KV Cache 或 compute 饱和）
□ 使用 continuous batching 提高 GPU 利用率
□ 开启 prefix caching 减少重复 prefill
□ 长 prompt 场景用 SGLang RadixAttention
□ 多卡部署：DP + EP 优于纯 TP（MoE 模型）
□ 使用 FlashAttention-3 / FlashInfer 加速 attention
```

### 7.3.3 成本敏感场景

```
□ 量化到 INT4（AWQ/GPTQ）降低单卡需求
□ 使用 A100 替代 H100（如果 latency 可接受）
□ PD 分离：P 节点用便宜卡
□ 动态 batching + 自动扩缩容
□ KV Cache offload 到 CPU 支持超长上下文
```

---

## 7.4 面试常见 GPU/CUDA 题

**Q: 为什么 LLM decode 阶段 GPU 算力利用率很低？**
- 答：decode 是 memory-bound。每次 forward 读取全部权重，HBM 带宽成为瓶颈，算力等待数据。

**Q: H100 相比 A100，对推理的最大提升是什么？**
- 答：HBM 带宽 3.35 TB/s vs 2.0 TB/s（+67%），以及原生 FP8 Tensor Core（2x 吞吐）。

**Q: 什么是 Tensor Core？为什么 FP8 比 FP16 快？**
- 答：Tensor Core 是专用的矩阵乘单元。FP8 每个操作处理的数据量减半，且 H100 原生支持，所以 2x 吞吐。

**Q: CUDA 中什么是 warp divergence？如何避免？**
- 答：同一个 warp 内的 thread 执行不同分支，导致串行化。避免方法：按 warp 对齐分支条件，或在 kernel 设计时考虑数据布局。

**Q: 为什么 FlashAttention 能减少 HBM 访问？**
- 答：将 attention 分块计算，每次只加载一小块 Q/K/V 到 SRAM，计算完立即写回，避免存储完整 attention matrix。

---

## 7.5 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| HBM 带宽 | decode 阶段推理瓶颈 |
| Tensor Core | 专用矩阵乘单元，FP8 带来 2x 加速 |
| Warp / Occupancy | 并行执行与延迟隐藏的关键 |
| Coalesced Access | 最大化内存带宽利用 |
| FlashAttention | 通过 tiling 减少 HBM 访问 |
