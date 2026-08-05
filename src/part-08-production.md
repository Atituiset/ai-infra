# Part 8 补充：生产部署与运维

> **面向角色**：需要上线 LLM 推理服务的工程师  
> **目标**：补齐 benchmark、部署、监控、故障排查的实战经验

---

## 8.1 Benchmark 方法 ★

### 8.1.1 关键指标测量

```bash
# vLLM benchmark 工具
python vllm/benchmarks/benchmark_throughput.py \
  --model meta-llama/Llama-2-70b-hf \
  --input-len 2048 \
  --output-len 512 \
  --num-prompts 1000 \
  --tensor-parallel-size 4

# SGLang benchmark 工具
python -m sglang.bench_serving \
  --model meta-llama/Llama-2-70b-hf \
  --dataset-name random \
  --random-input 2048 \
  --random-output 512 \
  --num-prompts 1000 \
  --tp 4
```

### 8.1.2 负载模型

| 负载 | 特点 | 适用场景 |
|------|------|---------|
| **Fixed** | 固定 input/output len | 对比不同引擎 |
| **ShareGPT** | 真实对话长度分布 | 模拟线上聊天 |
| **Random** | 可配置长度 | 压测边界 |
| **Burst** | 瞬时大量请求 | 测试调度与排队 |

### 8.1.3 如何解读结果

```
关键输出:
  Throughput:  X tokens/s
  TTFT:        mean=X, P50=X, P90=X, P99=X
  TPOT:        mean=X, P50=X, P90=X, P99=X
  QPS:         X req/s
  
注意:
  - 不要只看平均，要看 P99
  - 高吞吐可能伴随高 TPOT
  - 负载分布影响极大（ShareGPT vs Fixed 可能差 2-3x）
```

### 8.1.4 常见 Benchmark 陷阱

1. **warmup 不足**：首次运行包含 CUDA cache 编译，结果偏低
2. **单客户端瓶颈**：客户端本身成为瓶颈，测不出引擎极限
3. **网络延迟**：客户端和服务器不在同一机房
4. **KV Cache 未预热**：prefix caching 效果未体现
5. **太短时间**：只跑几十秒，未稳定状态

---

## 8.2 生产部署关键参数 ★

### 8.2.1 vLLM 关键参数

| 参数 | 含义 | 调优建议 |
|------|------|---------|
| `--tensor-parallel-size` | TP 大小 | 70B→4-8，7B→1 |
| `--pipeline-parallel-size` | PP 大小 | 405B+ 考虑 |
| `--max-model-len` | 最大上下文 | 按业务需求，不要设过大（浪费 KV） |
| `--max-num-seqs` | 最大并发请求数 | 受 KV Cache 限制 |
| `--max-num-batched-tokens` | 每步最大 token 数 | 影响 chunked prefill |
| `--gpu-memory-utilization` | GPU 内存使用率上限 | 默认 0.9，留空间给 activation |
| `--enable-prefix-caching` | 前缀缓存 | 长 system prompt 必开 |
| `--kv-cache-dtype fp8` | KV Cache 量化 | H100 上可开 |
| `--quantization fp8` | 权重 FP8 量化 | H100+ |
| `--speculative-model` | 投机解码 | 延迟敏感场景 |

### 8.2.2 SGLang 关键参数

| 参数 | 含义 | 调优建议 |
|------|------|---------|
| `--tp` | TP 大小 | 同 vLLM |
| `--dp` | DP 大小 | 高吞吐 MoE 推荐 |
| `--max-running-requests` | 最大运行请求数 | 控制 KV Cache 压力 |
| `--chunked-prefill-size` | Chunked prefill 大小 | 默认即可，可调平滑 TTFT |
| `--enable-torch-compile` | torch.compile 优化 | 通常开启 |
| `--mem-fraction-static` | 静态内存比例 | 同 gpu-memory-utilization |
| `--radix-cache` | 前缀缓存 | 默认开启，长 prompt 收益大 |
| `--enable-hicache` | HiCache 三级缓存 | 多实例共享场景 |
| `--speculative-algorithm EAGLE` | 投机解码 | 延迟敏感 |
| `--disaggregation-mode prefill` / `decode` | PD 分离 | 大规模生产 |

---

### 8.2.3 KV Cache Offload / Tiering 部署实操

当单卡显存不足、但又不希望牺牲 `max_model_len` 或并发数时，除了 KV Cache 量化，还可以把不活跃的 KV 块换出到 CPU / 远端存储（Part 3 §3.7）。适用场景与配置如下。

**什么时候值得开**：

| 场景 | 判断依据 | 建议 |
|------|---------|------|
| 长空闲序列多 | 大量请求有很长的 thinking/等待段，期间不再产生新 token | 开启后收益最大 |
| 并发高但单请求短 | KV 总量超过 GPU 容量 20% 以上 | 可开，观察 TPOT 抖动 |
| 延迟 SLA 极严 | TPOT 要求 < 20ms | 谨慎，offload 回读会增加 P99 延迟 |
| 短 prompt 高频问答 | 每个请求 KV 都很小 | 通常不需要 |

**vLLM 配置示例**：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --kv-offloading-size 40 \
  --kv-offloading-backend native
```

参数说明（基于当前主线，`vllm/config/cache.py` + `engine/arg_utils.py`）：

| 参数 | 含义 | 备注 |
|------|------|------|
| `--kv-offloading-size` | 每卡允许换出到 CPU 的 KV 空间（GiB） | 0 = 关闭（默认） |
| `--kv-offloading-backend` | 卸载后端 | `native`（默认）或 `lmcache` |
| `--cpu-offload-gb` / `--offload-mode cpu` | 权重 offload（与 KV offload 是两件事） | 权重常驻 CPU，按层换入，适合极小显存 |

**SGLang 配置示例**：

```bash
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --cpu-offload-gb 100 \
  --offload-mode cpu
```

PD 分离场景下，decode 侧可独立开启异步 KV 卸载：
`--disaggregation-decode-enable-offload-kvcache`（配合 Mooncake/HiCache 使用）。

**部署注意点**：
1. **offload 带宽是硬约束**：PCIe 带宽（~64 GB/s）远低于 HBM（3 TB/s+），回读大量 KV 会直接推高 TPOT。实测应对比"开/关"的 P50/P99。
2. **不要与投机解码叠加**：投机解码需要频繁访问 draft KV，offload 的往返开销容易吃掉加速收益。
3. **监控指标**：关注 offload 命中率、换入/换出次数、PCIe 吞吐（`nvidia-smi dmon` / 引擎 metrics），出现频繁抖动说明 offload 边界参数不合理。
4. **与量化优先级**：先做 KV Cache INT8/FP8（几乎无延迟代价），仍不够再考虑 offload；offload 是"用带宽换容量"的最后手段。

---

## 8.3 监控与可观测性

### 8.3.1 必看指标

```
系统层:
  - GPU 利用率 (not just overall, but SM/Tensor/Memory)
  - HBM 使用率
  - NVLink / PCIe 带宽
  - 温度 / 功耗

引擎层:
  - TTFT (P50/P90/P99)
  - TPOT (P50/P90/P99)
  - Throughput (tokens/sec)
  - Batch size per step
  - KV Cache 使用率
  - Prefix cache hit rate
  - Queue length / wait time
  - Speculative decoding accept rate

业务层:
  - QPS
  - Error rate
  - Timeout rate
  - Cost per 1M tokens
```

### 8.3.2 常用工具

| 工具 | 用途 |
|------|------|
| **nvidia-smi** / **dcgm** | GPU 基础监控 |
| **Nsight Systems** | CUDA kernel 级 profiling |
| **PyTorch Profiler** | Python 层 profiling |
| **vLLM metrics** | `/metrics` Prometheus endpoint |
| **SGLang metrics** | `/metrics` Prometheus endpoint |
| **Grafana + Prometheus** | 可视化 + 告警 |

---

## 8.4 常见故障排查

### 8.4.1 OOM（显存溢出）

**诊断步骤**：
```
1. 检查模型权重大小：model_size × dtype
2. 检查 KV Cache 占用：
   KV = 2 × layers × kv_heads × head_dim × max_seq_len × dtype × num_seqs
3. 检查激活值：与 batch size 和 sequence length 成正比
4. 检查是否有内存碎片：vLLM 的 block pool 机制通常可避免

解决方案:
  - 降低 max_num_seqs
  - 降低 max_model_len
  - 开启 KV Cache 量化 (fp8)
  - 启用 KV Cache offload
  - 增加 TP 大小
```

### 8.4.2 高 TTFT

**可能原因**：
```
1. 长 prompt 一次性 prefill → chunked prefill 切分
2. 队列堆积 → 增加实例数 / 优化调度
3. prefix caching 未命中 → 检查 system prompt 是否一致
4. prefill 被 decode 阻塞 → PD 分离
```

### 8.4.3 高 TPOT

**可能原因**：
```
1. batch size 过大 → 降低 max_num_seqs
2. 未启用 CUDA Graph → 开启
3. attention backend 不合适 → decode 用 FlashInfer
4. 量化未启用 → FP8 / INT4
5. TP 过大导致通信开销 → 尝试减小 TP
```

### 8.4.4 输出不正确 / 乱码

**可能原因**：
```
1. 量化精度损失过大 → 换 AWQ/GPTQ 配置或 FP8
2. KV Cache 损坏 → 检查 dtype / flashinfer 版本
3. attention mask 错误 → 多见于自定义模型支持
4. 采样参数异常 → temperature=0 是否走 greedy
```

---

## 8.5 线上 checklist

```
上线前:
  □ 完成 warmup（至少 100 个请求）
  □ 在真实负载分布下 benchmark（ShareGPT 或业务数据）
  □ 测试 P99 延迟是否满足 SLA
  □ 验证 prefix caching 命中率
  □ 确认 max_num_seqs 不会导致 OOM
  □ 配置监控告警（TTFT/TPOT/OOM/Error rate）
  □ 准备回滚方案

上线后:
  □ 持续观察 24-48 小时
  □ 监控 GPU 利用率和 HBM 使用率
  □ 关注长文本请求的长尾延迟
  □ 定期检查量化精度是否漂移
```

---

## 8.6 本章小结

| 主题 | 关键点 |
|------|--------|
| Benchmark | 用真实负载、看 P99、充分 warmup |
| 参数调优 | max_num_seqs / batch tokens / prefix cache 是核心 |
| 监控 | 系统层 + 引擎层 + 业务层 |
| 故障排查 | OOM / 高 TTFT / 高 TPOT / 输出错误 |
| 上线 | warmup → benchmark → 监控 → 回滚 |
