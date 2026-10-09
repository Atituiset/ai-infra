# 第5章 面试实战与自检清单

> **面向角色**：准备训推平台/推理引擎岗位面试的候选人  
> **目标**：面试高频问题 + 源码定位 + 场景设计 + 能力自检  
> **使用方式**：先自检后查漏补缺

---

## 5.1 面试高频问题

### 5.1.1 入门级（1-3 年经验）

**Q1: 解释 Continuous Batching 如何提高 GPU 利用率。**
- 静态 batch 必须等所有请求完成才能开始新 batch
- Continuous batching 每个 step 后重新组 batch，完成的出，新的进
- GPU 利用率从 20-30% → 60-80%
- 核心挑战：KV Cache 必须支持动态分配/回收（PagedAttention / RadixAttention）
- 参考：第1章 §1.3，第2章 §2.1

**Q2: Prefill 和 Decode 阶段的区别是什么？为什么需要分别对待？**
- Prefill: compute-bound，并行处理所有 prompt tokens
- Decode: memory-bound，每次 1 token，受 HBM 带宽限制
- 分开对待 → PD Disaggregation：P 用便宜高算力卡，D 用贵高带宽卡
- 参考：第1章 §1.1.2，第2章 §2.3

**Q3: KV Cache 的内存占用如何计算？给定模型参数，估算最大并发数。**
- 公式: `2 × n_layers × n_kv_heads × head_dim × max_seq_len × dtype_bytes`
- 示例计算见 第1章 §1.4.3
- 关键点：GQA/MQA 显著减少 KV Cache，TP 不减少总量

**Q4: PagedAttention 和传统 KV Cache 管理有什么区别？**
- 传统：每个请求预分配连续 max_seq_len 空间 → 严重浪费
- PagedAttention：block 粒度（16 tokens），按需分配，页表映射
- 额外好处：block-level prefix caching，hash matching
- 参考：第2章 §2.4.1

**Q5: 什么是 FlashAttention？为什么比标准 Attention 快？**
- IO-aware 算法：将 attention 计算分块，避免将完整 N×N matrix 写入 HBM
- 标准 Attention: HBM 读写 O(N²)
- FlashAttention: HBM 读写 O(N²/d)（d 为 SRAM 大小），实际约 7-8x 加速
- 参考：第2章 §2.4.2

### 5.1.2 进阶级（3-5 年经验）

**Q6: TP 和 PP 在推理中分别适合什么场景？它们的通信开销如何？**
- TP: 层内切分，每层 forward 后 AllReduce，适合中小模型（70B 级别），延迟低
- PP: 层间切分，层边界通信，适合极大模型（405B+），但有 pipeline bubble
- 实际混合：405B 模型 → TP=8 + PP=2
- 推理中 TP 更重要（低延迟优先），PP 用于突破单卡显存极限
- 参考：第2章 §2.2.1-2.2.2

**Q7: MoE 模型的 EP（Expert Parallelism）如何工作？All-to-All 通信的开销如何分析？**
- Router → Top-K expert ids → All-to-All dispatch → Expert compute → All-to-All combine
- All-to-All 通信量 = batch_tokens × hidden_dim × 2 / EP_size（双向）
- EP 增大 → 通信量/卡不变，但 latency 增加（更多连接）
- DeepSeek-V3 策略：TP=1 + EP=较大 → 最大化每卡 expert 数 → 减少通信
- 参考：第2章 §2.2.3

**Q8: PD 分离架构的核心收益是什么？SGLang 是如何实现的？**
- 硬件异构降本（P 用 A100，D 用 H100）
- 独立弹性伸缩（P 多 D 少 vs P 少 D 多）
- SGLang: BootstrapQueue → WaitingQueue → Prefill → InflightQueue → KV Transfer → Decode
- 关键：分层传输（Mooncake）+ RDMA + HiCache 三级缓存
- 参考：第2章 §2.3.2-2.3.4

**Q9: 投机解码为什么能加速？Eagle 和 MTP 的区别是什么？**
- 原理：用轻量 draft model 预测 N tokens → 大模型并行验证 → 接受/拒绝
- 加速比取决于 draft accept rate（通常 60-80%）→ 实际 1.5-2.5x
- Eagle: 额外 1-2 层 Transformer，需要额外显存
- MTP: 模型自带 multi-token heads，零额外显存
- 参考：第2章 §2.6

**Q10: FP8 和 INT4 量化在推理中的 trade-off 是什么？你会如何为 70B 模型选择量化方案？**
- FP8: 2x 压缩（weight + activation），精度几乎无损，需 H100+
- INT4 (AWQ): 4x 压缩（仅 weight），轻微精度损失（<1%），所有 GPU
- 选择：有 H100 → FP8 W8A8；其他 GPU → AWQ W4A16
- 对延迟敏感场景：INT4 权重更小 → decode 更快（减少 HBM 读取）
- 参考：第2章 §2.7

**Q11: DeepSeek-V3.2 的稀疏注意力（NSA/DSA）如何工作？为什么说训练-推理一致性是前提？**
- NSA：压缩块 + 选择块（可学习门控）+ 滑动窗口，KV 访问从 O(N) 降到 O(√N) 量级
- DSA：每层前加 Lightning Indexer 输出 top-k 索引，相邻层复用/缓存选择结果
- 一致性：如果训练时用全注意力、推理时用稀疏，分布偏移会直接导致精度崩坏，所以必须在训练阶段就采用相同稀疏模式
- 工程影响：top-k 索引可缓存（vLLM IndexCache），Radix/Block Cache 需要感知稀疏 mask
- 参考：第2章 §2.4.4, 第4章 §4.10

**Q12: 显存不足时，KV Cache Offload 和传统 Swapping 有什么区别？**
- 传统 Swapping（vLLM Preemption）：KV 换出到 CPU，请求被抢占，换回时重新调度
- KV Offload / Tiering：按访问频率/热度分层放置（GPU → CPU → FS/P2P/远端），对请求透明，类似操作系统的 page cache 分级
- 关键指标：offload 带宽与 HBM 带宽的差距决定收益；长空闲序列收益最大
- 参考：第3章 §3.7

> 注：Q11/Q12 属于"当前主线加分题"，考察候选人是否跟踪 2025-2026 年的新架构，答不上不影响基础评分。

---

## 5.2 源码定位题

> 以下问题考察候选人是否真正阅读过源码。

**Q: 在 vLLM 中，Scheduler 如何处理 KV Cache 分配失败？定位相关代码。**

```
答案：
1. scheduler.py → schedule()
2. 调用 kv_cache_manager.allocate_slots(req, num_tokens)
3. 如果返回 None → 缓存不足
4. 触发 preemption：选择优先级最低的 running request
5. preemption mode: RECOMPUTE（丢弃 KV）或 SWAP（换出到 CPU）
6. 释放被抢占请求的 blocks → 重新尝试分配

源码: vllm/v1/core/sched/scheduler.py ~line 250-350
```

**Q: SGLang 的 RadixAttention 如何找到请求的前缀匹配？**

```
答案：
1. schedule_policy.py → match_prefix_for_req()
2. 调用 tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=...)))
3. radix_cache.py → RadixCache.match_prefix()
4. 从 root_node 出发，沿 children 逐 token 匹配
5. 返回 MatchResult(device_indices=GPU cached indices, ...)
6. 结果赋值给 req.prefix_indices，传给 ModelRunner

源码: 
  sglang/srt/managers/schedule_policy.py → match_prefix_for_req()
  sglang/srt/mem_cache/radix_cache.py → RadixCache.match_prefix()
```

**Q: vLLM V1 的 EngineCore 和 Frontend 如何通信？**

```
答案：
使用 msgspec 消息协议，支持跨进程通信：
- EngineCoreRequest: Frontend → EngineCore (ADD)
- EngineCoreOutput: EngineCore → Frontend (结果回传)
- EngineCoreRequestType: 定义消息类型 (ADD/ABORT/UTILITY/...)

通信方式:
- 同进程: 直接调用
- 多进程: multiprocessing.Queue
- 多节点: 通过网络（TCP/gRPC）+ msgspec 序列化

源码: vllm/v1/engine/__init__.py (消息定义)
```

**Q: SGLang 的 Overlap 调度如何实现 CPU/GPU 并行？**

```
答案：
1. 主 CUDA Stream: GPU forward
2. 辅助 Stream: CPU schedule (准备下一个 batch)
3. 使用 CUDA Event 同步关键依赖点

关键流程:
- forward 异步启动 (non-blocking)
- CPU 立即开始处理结果 + 准备下一 batch
- CUDA Event 确保依赖正确:
  - 需要 GPU 结果时 → wait event
  - 写入 KV Cache 时 → 确保上一轮 forward 完成

源码: sglang/srt/managers/overlap_utils.py
```

---

## 5.3 场景设计题

### 场景 1：千卡集群部署 DeepSeek-V3

**题目**：你需要在一个 1000 卡 H100 集群上部署 DeepSeek-V3（671B，MoE，每 token 激活 37B）服务。设计部署方案。

**考察点**：
1. **并行策略**：这个模型太大不能 TP=8。EP 是关键。
   - EP = 64（每个 GPU 4 experts × 256 = 1024 太多，应该是 EP = 32 或 64）
   - 实际方案参考：EP=64, TP=1, DP=8（8 个 replica）
   - 每 replica：8 node × 8 GPU = 64 GPU (EP=64)
   - 总：8 × 64 = 512 GPU 用于推理，其余用于其他服务
   
2. **MLA 处理**：KV Cache 被 MLA 压缩约 10x，支持更高并发
   
3. **MTP**：利用 MTP 做免费投机解码，1.5-2x decode 加速
   
4. **PD 分离**：是否需要？取决于 P/D 比例。如果读多写少（system prompt 长），值得做分离
   
5. **网络**：All-to-All 需要高带宽，IB/RoCE 必需

### 场景 2：低延迟服务的优化路径

**题目**：当前推理服务 TTFT=2s, TPOT=60ms。目标是 TTFT<500ms, TPOT<30ms。诊断与优化方案？

**考察点**：
1. **TTFT 优化**：
   - 检查 chunked prefill 配置：减小 `max_num_batched_tokens` 降低单 step prefill 延迟
   - PD 分离：专门 P 节点处理 prefill
   - 前缀缓存命中率：如果 system prompt 相同，开启 prefix caching
   
2. **TPOT 优化**：
   - 量化：FP8 (W8A8) → 2x HBM 读取减少
   - 投机解码：Eagle/MTP 突破串行瓶颈
   - Check attention kernel: FlashInfer decode kernel vs FlashAttention-2
   - 是否 TP 过大？TP=8 通信开销可能占比 >20%，尝试 TP=4
   
3. **系统层面**：
   - 检查 CUDA Graph 是否启用
   - 检查是否有 CPU-GPU 同步等待
   - NVLink 带宽是否打满

### 场景 3：多租户场景的 Scheduler 设计

**题目**：多个租户共享推理集群，需要 QoS 保障 + 公平性。设计调度策略。

**考察点**：
1. **优先级队列**：vLLM SchedulingPolicy + priority 参数
2. **资源隔离**：为每个租户预留 KV Cache quota（类似 CPU cgroup）
3. **抢占策略**：当高优先级请求到达而 KV Cache 不足时，抢占低优先级租户
4. **指标**：per-tenant TTFT P50/P99, TPOT P50/P99
5. **实现思路**：在 RequestQueue 中支持 priority class + per-class token budget

---

## 5.4 自检 Checklist

对标 `profile.md` 中的岗位要求：

### 必需项（≥ 8/10 才算合格）

- [ ] 能画出 vLLM Scheduler → KV Cache Manager → Model Runner 的完整数据流
- [ ] 能解释 PagedAttention 的 block 分配和回收机制
- [ ] 能计算给定模型的 KV Cache 大小和最大并发数
- [ ] 理解 TP/PP/EP 的切分方式、通信模式和适用场景
- [ ] 能用代码示例解释 Continuous Batching vs Static Batching
- [ ] 了解 FP8/INT4 量化在推理中的原理和效果
- [ ] 能解释投机解码的 accept probability 公式
- [ ] 知道 SGLang RadixAttention 相比 PagedAttention 的优势
- [ ] 了解 PD 分离的架构和 Mooncake 的分层传输思想
- [ ] 能独立完成一个推理引擎的 benchmark（TTFT/TPOT/Throughput）

### 加分项（有 2 项以上显著加分）

- [ ] 阅读过 vLLM scheduler.py 或 SGLang radix_cache.py 的完整实现
- [ ] 了解 CUDA Graph / Breakable CUDA Graph 的原理
- [ ] 了解 FlashInfer 针对 decode 优化的 kernel 细节
- [ ] 有千卡集群部署 MoE 模型的实际经验
- [ ] 了解 MLA (Multi-head Latent Attention) 的实现
- [ ] 能设计多级 KV Cache (L1 HBM / L2 CPU / L3 Remote)
- [ ] 了解 RDMA/NVLink 在 KV 传输中的使用
- [ ] 了解多种 attention backend 的选择策略 (FlashAttention/FlashInfer/xFormers/MLA)

---

## 5.5 拓展阅读

### 必读论文

| 论文 | 要点 | 链接 |
|------|------|------|
| vLLM (PagedAttention) | KV Cache 页式管理 | SOSP 2023 |
| SGLang | RadixAttention + DSL | 2024 |
| FlashAttention | IO-aware attention | NeurIPS 2022 |
| FlashAttention-2 | 更快的并行策略 | 2023 |
| Splitwise / Mooncake | PD 分离 | 2024 |
| AWQ | 激活感知量化 | MLSys 2024 |
| DeepSeek-V2 (MLA) | Multi-head Latent Attention | 2024 |
| DeepSeek-V3 | FP8 训练 + MTP | 2024 |
| Eagle | 投机解码 | 2024 |

### 必读源码

- `vllm/v1/core/sched/scheduler.py` — 调度核心
- `vllm/v1/core/kv_cache_manager.py` — KV Cache 管理
- `sglang/srt/mem_cache/radix_cache.py` — RadixAttention
- `sglang/srt/managers/scheduler.py` — Overlap 调度
- `sglang/srt/disaggregation/prefill.py` + `decode.py` — PD 分离

### 关注的人与组织

- **LMSYS (SGLang 团队)**：Ying Sheng, Lianmin Zheng 等
- **vLLM 团队 (UC Berkeley)**：Woosuk Kwon, Zhuohan Li 等
- **NVIDIA TRT-LLM 团队**：官方推理引擎
- **Mooncake (Moonshot/Kimi)**：PD 分离与 KV 传输
- **DeepSeek**：MLA/MTP/MoE 架构创新
