# 第4章 SGLang 源码深度解剖

> **面向角色**：需要阅读或修改 SGLang 源码的工程师、高级面试候选人  
> **目标**：建立从请求到 GPU forward 的 SGLang 完整数据流认知  
> **源码版本**：基于 AI Infra Collect 中 clone 的 SGLang 最新代码

---

## 4.1 SGLang 架构总览

### 4.1.1 与 vLLM 的架构差异

SGLang 和 vLLM 解决问题的思路有本质不同：

| 维度 | vLLM | SGLang |
|------|------|--------|
| **KV Cache 管理** | PagedAttention（block 页表）| **RadixAttention**（前缀树）|
| **调度策略** | FCFS + Priority | **Overlap 调度** + Jump-Forward |
| **编译优化** | 可选 torch.compile | **深度 torch.compile** + 自定义 kernel |
| **前缀复用** | Block-level hash matching | **Radix Tree**（字节级匹配）|
| **PD 分离** | KVConnector 架构 | **Disaggregation Mixin** + Mooncake |
| **代码组织** | V1/V0 双引擎 | 单一 Runtime（但模块化更细）|
| **内存池** | Fixed block pool | **Token-level + SWA + Mamba** 多池 |

### 4.1.2 核心模块与数据流

```
HTTP /gremlin → TokenizerManager → Scheduler → ModelRunner → GPU
     ↑               ↓                  ↓              ↓
  FastAPI      Async Tokenize    CPU Scheduling   CUDA Forward
  (entrypoints) (tokenizer_      (scheduler.py)   (model_runner.py)
                manager.py)

Mem Cache (Radix Tree):
  ┌─────────────────────────────────────────────┐
  │  mem_cache/                                  │
  │  ├── radix_cache.py       ← Radix Tree      │
  │  ├── swa_radix_cache.py   ← Sliding Window  │
  │  ├── hiradix_cache.py     ← HiCache 集成    │
  │  ├── memory_pool.py       ← Token Pool      │
  │  ├── hicache_storage.py   ← 远程分布式缓存   │
  │  └── allocator/            ← 各后端分配器    │
  └─────────────────────────────────────────────┘

Disaggregation:
  ┌─────────────────────────────────────────────┐
  │  disaggregation/                             │
  │  ├── prefill.py           ← P 侧逻辑        │
  │  ├── decode.py            ← D 侧逻辑        │
  │  ├── mooncake/            ← Mooncake 后端   │
  │  ├── nixl/                ← NIXL 后端       │
  │  └── mori/                ← Mori 后端       │
  └─────────────────────────────────────────────┘
```

**源码路径图**：

| 层级 | 主要文件 | 核心类/函数 |
|------|---------|-------------|
| API 入口 | `sglang/srt/entrypoints/http_server.py` | `app` (FastAPI) |
| Tokenizer | `sglang/srt/managers/tokenizer_manager.py` | `TokenizerManager` |
| **调度核心** | `sglang/srt/managers/scheduler.py` | `class Scheduler` (~4300 行) |
| 调度策略 | `sglang/srt/managers/schedule_policy.py` | `PrefillAdder`, `match_prefix_for_req()` |
| 调度批 | `sglang/srt/managers/schedule_batch.py` | `ScheduleBatch`, `Req` |
| Overlap | `sglang/srt/managers/overlap_utils.py` | `resolve_forward_inputs()` |
| **Radix Cache** | `sglang/srt/mem_cache/radix_cache.py` | `class RadixCache` |
| 内存池 | `sglang/srt/mem_cache/memory_pool.py` | `ReqToTokenPool`, `KVCache` |
| **Model Runner** | `sglang/srt/model_executor/model_runner.py` | `class ModelRunner` |
| Attention | `sglang/srt/layers/attention/` | FlashInfer backend 等 |
| 分布式 | `sglang/srt/distributed/` | TP/PP/EP 通信组 |
| PD 分离 | `sglang/srt/disaggregation/` | Prefill/Decode Bootstrap + Transfer |

---

## 4.2 RadixAttention：前缀树式 KV Cache ★

### 4.2.1 为什么需要 Radix Tree？

PagedAttention 的 prefix caching 是基于 block hash 的——粒度是 block（16 tokens），只能匹配完全相同的 block sequence。RadixAttention 使用**前缀树（Radix Tree）**——对 token 序列逐 token 匹配，可以找到任意长度的公共前缀。

```
场景: 三个请求共享 system prompt + 不同的 user prompt

  请求 A: "You are a helpful assistant. What is Python?"
  请求 B: "You are a helpful assistant. What is Rust?"
  请求 C: "You are a helpful assistant. How to cook?"

PagedAttention (block hash):
  只有 block 完全匹配才共享 → 16 tokens 对齐 → 可能在 user prompt 第一个 block 开始分歧

RadixAttention:
  "You are a helpful assistant. " → 30 tokens → 全部共享！
  然后分别分叉到 "What is Python?", "What is Rust?", "How to cook?"
```

### 4.2.2 Radix Cache 核心数据结构

**文件**：`sglang/srt/mem_cache/radix_cache.py`

```python
class RadixKey:
    """Radix Tree 的 key：token 序列 + 可选 extra_key"""
    token_ids: array[int]          # int array (更省内存)
    extra_key: Optional[str]       # 额外 key (lora_id, cache_salt)
    is_bigram: bool                # bigram 模式（用于 ngram 投机解码）
    limit: Optional[int]           # 长度截断（不复制数据）


class TreeNode:
    """Radix Tree 的节点"""
    children: defaultdict[TreeNode]  # 子节点映射
    parent: TreeNode
    key: RadixKey                    # 该节点存储的 token 子序列
    value: Optional[torch.Tensor]    # 对应的 GPU KV Cache indices
    host_value: Optional[torch.Tensor]   # CPU 侧的 KV Cache indices
    lock_ref: int                    # 锁定引用计数
    last_access_time: float          # LRU 驱逐的时间戳
    hit_count: int                   # 命中次数
    hash_value: Optional[List[str]] # 每个 page 的 hash


class RadixCache(BasePrefixCache):
    """
    Radix Tree KV Cache 管理器。
    
    核心操作:
    - match_prefix(): 匹配请求的 token 序列，返回已缓存的前缀长度
    - insert(): 将新计算的 KV Cache 插入树中
    - evict(): 驱逐最少使用的节点
    """
```

### 4.2.3 前缀匹配流程 ★

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    """
    在 Radix Tree 中查找最长公共前缀。
    
    返回:
      MatchResult(
        device_indices: tensor  # GPU 上已缓存的 KV indices
        last_device_node: TreeNode  # 匹配到的最后一个 GPU 节点
        last_host_node: TreeNode    # 匹配到的最后一个 CPU 节点
        host_hit_length: int        # CPU 侧命中长度
      )
    """

# 在 schedule_policy.py 中调用:
match_result = tree_cache.match_prefix(MatchPrefixParams(
    key=RadixKey(token_ids=req.origin_input_ids + req.output_ids),
    ...
))
req.prefix_indices = match_result.device_indices  # 直接传给 ModelRunner
```

### 4.2.4 驱逐策略

```python
class RadixCache:
    evictable_leaves: set[TreeNode]   # 可被驱逐的叶子节点
    
    def evict(self, num_tokens: int) -> EvictResult:
        """
        驱逐 num_tokens 大小的 KV Cache。
        
        策略:
        1. 从 evictable_leaves 中选择 LRU 叶子
        2. 沿着父节点向上递归驱逐（直到遇到 lock_ref > 0 的节点）
        3. 更新 evictable_size
        """
```

**关键：lock_ref 机制**
- `lock_ref > 0`：节点正在被某个请求使用，不可驱逐
- `lock_ref == 0`：节点空闲，可以被驱逐
- 请求的生命周期：match → lock_ref++ → 使用 → lock_ref--

**会话级 Radix Cache**：
`mem_cache/session_radix_cache.py` — `SessionRadixCacheMixin` — 支持跨请求的多轮会话前缀复用。

---

## 4.3 Overlap 调度：CPU/GPU 并行 ★

### 4.3.1 传统调度 vs Overlap 调度

```
传统 (vLLM 模式):
  [CPU: Schedule] → [GPU: Forward] → [CPU: Process Results + Schedule] → [GPU: Forward] → ...
        ↑__________|                    ↑__________________________|
        (GPU 空闲，等待 CPU 调度)              (GPU 空闲，等待 CPU 处理结果)
        
  Wall clock: T_schedule + T_forward + T_process + T_schedule + T_forward + ...

SGLang Overlap 模式:
  主 Stream:  [GPU: Forward N-1]  [GPU: Forward N]    [GPU: Forward N+1]
  调度 Stream: [CPU: Schedule N]  [CPU: Schedule N+1]  [CPU: Schedule N+2]
  结果 Stream: [CPU: Process N-1] [CPU: Process N]     [CPU: Process N+1]
  
  Wall clock: T_forward + T_forward + T_forward + ...  (CPU 开销被完全隐藏！)
```

### 4.3.2 Overlap 实现

**文件**：`sglang/srt/managers/overlap_utils.py`

```python
class RelayPayload:
    """在 GPU forward 期间传递到 CPU 的中继数据"""
    # 包含 batch metadata、采样结果等

def resolve_forward_inputs(batch: ScheduleBatch):
    """
    决定当前 step 的 forward 模式:
    - DECODE: 正常 decode
    - EXTEND: prefill 新请求（可能有 chunked prefill）
    - IDLE: GPU 空闲，等待新请求
    """
```

**Overlap 调度的关键设计**：
1. GPU forward 是异步的（通过 CUDA Stream），CPU 可以在 forward 执行期间准备下一个 batch
2. 使用 `relay_payload` 在 forward 完成时触发结果处理
3. 结果处理也是异步的（detokenize + 发送响应 + 更新 Radix cache）

### 4.3.3 Scheduler 的 Overlap 方法

在 `scheduler.py` 的 `Scheduler` 类中：

```python
class Scheduler(
    SchedulerDisaggregationDecodeMixin,  # PD 分离 - Decode 侧
    SchedulerDisaggregationPrefillMixin, # PD 分离 - Prefill 侧
    SchedulerPPMixin,                    # Pipeline Parallelism
    SchedulerDllmMixin,                  # dLLM 支持
    SchedulerMultiplexMixin,             # Multiplex 支持
):
    """
    Scheduler 通过 Mixin 模式组合功能，而不是继承单一基类。
    这是 SGLang 架构的一个特点：每个功能维度独立的 Mixin。
    """
```

---

## 4.4 Model Runner 与 Forward Batch ★

### 4.4.1 ModelRunner

**文件**：`sglang/srt/model_executor/model_runner.py`  
**类**：`class ModelRunner`

```python
class ModelRunner:
    model: nn.Module                       # 加载的模型
    server_args: ServerArgs                # 服务器参数
    
    def forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """
        Decode 阶段的 forward:
        1. 从 ScheduleBatch 构造 ForwardBatch
        2. 执行 model.forward()
        3. 采样（greedy / top-p / beam）
        4. 返回 GenerationBatchResult
        """
    
    def forward_extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """
        Extend (prefill) 阶段的 forward:
        1. 处理 chunked prefill（按 max_tokens_per_batch 切分）
        2. 执行 model.forward() — 可并行处理所有 prompt tokens
        3. 返回最后一个 token 的 logits 用于采样
        """
```

### 4.4.2 ForwardBatch 与 ForwardMode

**文件**：`sglang/srt/model_executor/forward_batch_info.py`

```python
class ForwardMode(Enum):
    DECODE = auto()           # 正常 decode
    EXTEND = auto()           # Prefill（可能 chunked）
    IDLE = auto()             # GPU 空闲
    TARGET_VERIFY = auto()    # 投机解码验证阶段
    DRAFT_EXTEND = auto()     # Draft 模型 prefill
    
class ForwardBatch:
    """
    GPU 侧的 "batch" 表示。由 ScheduleBatch 转换而来。
    
    核心字段:
    - input_ids: [total_tokens] tensor
    - positions: [total_tokens] tensor
    - out_cache_loc: [total_tokens] tensor    # token → KV cache slot
    - req_pool_indices: [num_reqs] tensor     # 每个请求的 pool index
    - seq_lens: [num_reqs] tensor             # 每个请求的序列长度
    - extend_seq_lens: [num_extend] tensor    # prefill 请求的扩展长度
    - out_cache_loc: 指向 KV Cache 的写入位置
    """
```

### 4.4.3 ScheduleBatch → ForwardBatch 转换

```
ScheduleBatch (CPU, high-level):
  ├── reqs: List[Req]                    # 请求列表
  ├── req_to_token_pool: ReqToTokenPool  # token → KV cache slot 的池
  ├── token_to_kv_pool: KVCache          # KV Cache 物理存储
  ├── prefix_indices: Tensor             # 从 Radix Tree 匹配的前缀
  └── ...

        ↓ ForwardBatch.init_new()

ForwardBatch (GPU, low-level):
  ├── input_ids: Tensor[batch_size, ...]
  ├── positions: Tensor[batch_size, ...]
  ├── req_pool_indices: Tensor[batch_size]
  ├── seq_lens: Tensor[batch_size]
  ├── out_cache_loc: Tensor[total_tokens]
  └── ...
```

---

## 4.5 PD 分离实现 ★

### 4.5.1 整体生命周期

SGLang 的 PD 分离是目前所有推理引擎中实现最完整的。

**Prefill 侧生命周期**（`disaggregation/prefill.py`）：

```
1. BootstrapQueue:
   - 初始化 KV Sender（握手 + 预分配）
   - 请求在此排队直到 bootstrap 完成
   
2. WaitingQueue:
   - PrefillAdder 从此队列取请求
   - 执行 prefill forward (extend mode)
   
3. InflightQueue:
   - 请求 prefill 完成后进入此队列
   - 轮询 (poll) KV Sender，检查传输状态
   - 传输完成后，请求从 P 侧移除
```

**Decode 侧生命周期**（`disaggregation/decode.py`）：

```
1. PreallocQueue:
   - 初始化 KV Receiver
   - 握手后预分配 KV Cache 内存
   - 有可用内存时移到 TransferQueue
   
2. TransferQueue:
   - 轮询 Receiver，检查传输进度
   - 传输完成后移到 WaitingQueue
   
3. WaitingQueue → RunningBatch:
   - PrebuiltExtendBatch (跳过 prefill forward)
   - 仅填充 KV Cache 元数据
   - 合并到 RunningBatch 执行 decode
```

### 4.5.2 Mooncake 传输后端

**文件**：`sglang/srt/disaggregation/mooncake/`

Mooncake 传输的核心优势：
1. **分层传输**：每算完一层 Transformer，立即传输该层的 KV Cache（不等待整个 prefill 完成）
2. **RDMA 零拷贝**：绕过 CPU，GPU → 网络 → GPU 直接传输
3. **带宽利用**：流水线式的传输 + 计算 overlap

### 4.5.3 HiCache 集成

HiCache（`mem_cache/hicache_storage.py`）提供跨节点的 KV Cache 存储抽象：

```python
@dataclass
class HiCacheStorageConfig:
    tp_rank: int / tp_size: int
    pp_rank: int / pp_size: int
    is_mla_model: bool           # MLA 模型特殊处理
    model_name: Optional[str]
    extra_config: Optional[dict]

# 支持多种存储后端:
# - Local SSD/NVMe (本地冷缓存)
# - RDMA 共享内存 (跨节点热缓存)
# - S3/对象存储 (长期持久化缓存)
```

**HiCache 在 Decode 侧的集成**（`decode_hicache_mixin.py`）：

```python
class DecodeHiCachePreallocMixin:
    """HiCache 预分配逻辑"""
    def prealloc_from_hicache(self, req):
        # 1. 检查 HiCache 中是否有该请求的 KV
        # 2. 如果有 → 预分配 GPU 内存 + 发起异步加载
        # 3. 加载期间请求在 PreallocQueue 等待

class DecodeHiCacheTransferMixin:
    """HiCache 传输完成后处理"""
    def on_hicache_restore_complete(self, req):
        # 1. 验证数据完整性
        # 2. 恢复 Radix Tree 节点
        # 3. 将请求移入 WaitingQueue
```

---

## 4.6 投机解码实现 ☆

### 4.6.1 SGLang 支持的投机方法

**文件**：`sglang/srt/speculative/`

```
speculative/
├── eagle_worker_v2.py                     # Eagle v2 worker
├── eagle_info.py                          # Eagle metadata
├── eagle_draft_cuda_graph_runner.py       # CUDA Graph 加速
├── eagle_draft_extend_cuda_graph_runner.py
├── dflash_worker_v2.py                    # dFlash worker
├── dflash_info.py / dflash_utils.py
├── ngram_worker.py / ngram_info.py        # ngram worker
├── standalone_worker_v2.py               # 独立 draft model worker
├── multi_layer_eagle_worker_v2.py        # 多层 Eagle
├── reject_sampling.py                     # 接受/拒绝采样
├── spec_info.py                           # SpeculativeAlgorithm 枚举
└── ...
```

### 4.6.2 Eagle v2 Worker 关键流程

```python
class EagleWorkerV2:
    """
    Eagle v2 实现了 draft model 的独立 GPU 管理。
    
    核心改进 vs v1:
    - Draft model 和 target model 可以运行在不同 GPU 上
    - CUDA Graph 加速 draft forward
    - 支持 extend (prefill) + decode 两种 draft 模式
    """
    
    def draft(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,  # 来自 target forward
    ) -> DraftResult:
        """
        1. hidden_states → draft_model.forward() → draft_logits
        2. 采样 draft_tokens (按 num_speculative_tokens 配置)
        3. 返回 draft_tokens
        """
    
    def verify(
        self,
        batch: ScheduleBatch,
        draft_tokens: list[list[int]],
    ) -> VerifyResult:
        """
        1. 构造 TARGET_VERIFY forward batch
        2. 执行 target model.forward() (一次处理 real + draft tokens)
        3. reject_sampling: 对每个 draft token 做 accept/reject
        4. 返回 accepted_tokens
        """
```

### 4.6.3 dFlash（draft-Flash）☆

dFlash（draft-Flash）是 SGLang 提出的投机解码方法，现已被 vLLM 移植支持（`vllm/vllm/v1/spec_decode/dflash.py`，面向 Qwen3.5 等支持 in-filling 的模型）：

```
传统 Eagle:   Target → hidden_states → Draft → draft_tokens → Target verify
dFlash:      Target → 在 attention 中直接"填充" draft tokens
             (in-filling style decoding，效率更高)

优势:
- 减少了 draft model 的独立 forward → 更低延迟
- 不需要额外的 draft model 参数
- 对短 prompt 场景尤其有效
```

> 演进：SGLang 最早实现（第4章 源码），vLLM 后续在 V1 引擎中提供 `DFlashProposer`。两者在 mask token 处理和上下文 K/V 复用上思路一致，实现细节略有差异。
>
> **DSpark（半自回归增强版）**：Qwen3-DSpark 草稿模型在 dFlash 的并行骨干上叠加低秩 Markov 头，采样时逐位注入块内依赖（机制详解见 第2章 §2.6.4）。vLLM 侧对应 `model_executor/models/qwen3_dspark.py`。
>
> **组合与零成本家族**：ngram（`cpp_ngram/` + `ngram_worker.py`）、投机解码 × PD 分离的编排（`speculative/eagle_disaggregation.py`）等机制对比与选型见 第2章 §2.6.5。

---

## 4.7 Mem Cache 体系

### 4.7.1 多缓存类型

SGLang 的 `mem_cache/` 目录包含了业界最完整的 KV Cache 类型支持：

| 缓存类型 | 文件 | 用途 |
|---------|------|------|
| **RadixCache** | `radix_cache.py` | 标准 Transformer 的 KV Cache |
| **SWARadixCache** | `swa_radix_cache.py` | Sliding Window Attention |
| **MambaRadixCache** | `mamba_radix_cache.py` | Mamba SSM 的 State Cache |
| **HiRadixCache** | `hiradix_cache.py` | 带 HiCache 存储层的 Radix Cache |
| **UnifiedRadixCache** | `unified_radix_cache.py` | 统一多种缓存类型 |
| **ChunkCache** | `chunk_cache.py` | Chunked prefill 的临时缓存 |
| **MultimodalCache** | `multimodal_cache.py` | 多模态 (图像/视频) 的 Embedding 缓存 |
| **SessionRadixCache** | `session_radix_cache.py` | 多轮对话的会话级缓存 |
| **DeepSeekV4** | `deepseek_v4_memory_pool.py` | DeepSeek V4 专用内存池 |

### 4.7.2 内存池层级

```
ReqToTokenPool:
  每个请求分配一个 "slot"，存储该请求所有 token → KV Cache 位置的映射

TokenToKVPool (KVCache):
  物理 KV Cache 存储（GPU HBM）
  
  SGLang 使用 token 级别的分配（不是 vLLM 的 block 级别）
  → 理论上更精细，但管理开销稍大
  → 通过 triton_ops/ 中的自定义 kernel 加速索引操作
```

---

## 4.8 关键设计决策总结

| 决策 | SGLang 的选择 | 理由 |
|------|-------------|------|
| KV Cache 粒度 | Token 级别（Radix Tree） vs vLLM 的 Block 级别 | 更细粒度的前缀匹配，更好复用 |
| 调度模型 | Overlap 调度（CPU/GPU 并行）| 消除 CPU 调度开销 |
| 缓存结构 | Radix Tree vs 哈希表 | 支持部分前缀匹配，不仅完整 block |
| 分布式 | Mixin 模式组合功能 | 灵活扩展，PD/PP/EP 独立 |
| 量化 | 自定义 kernel（sgl-kernel/）| 极致性能，不依赖第三方 |
| 编译 | 深度 torch.compile | 减少 Python overhead |
| 代码风格 | `msgspec.Struct` 而非 `@dataclass` | 性能 + 跨语言兼容 |

---

## 4.9 源码阅读路线

```
第一遍（理解 SGLang 的独特设计）:
  1. sglang/srt/managers/scheduler.py      ← Scheduler 类定义 (class Scheduler)
  2. sglang/srt/mem_cache/radix_cache.py   ← RadixCache 类的 match_prefix + insert
  3. sglang/srt/managers/schedule_policy.py ← PrefillAdder + 前缀匹配逻辑
  4. sglang/srt/managers/schedule_batch.py ← Req + ScheduleBatch 数据结构

第二遍（理解执行流程）:
  5. sglang/srt/model_executor/model_runner.py ← forward_decode + forward_extend
  6. sglang/srt/model_executor/forward_batch_info.py ← ForwardBatch + ForwardMode
  7. sglang/srt/managers/overlap_utils.py     ← Overlap 调度核心

第三遍（理解高级特性）:
  8. sglang/srt/disaggregation/prefill.py     ← PD 分离 P 侧
  9. sglang/srt/disaggregation/decode.py      ← PD 分离 D 侧
  10. sglang/srt/speculative/eagle_worker_v2.py  ← Eagle v2
  11. sglang/srt/mem_cache/hicache_storage.py    ← HiCache

深入定制:
  12. sglang/srt/layers/attention/            ← Attention kernel
  13. sglang/srt/layers/quantization/         ← 量化 kernel

---

## 4.10 最新特性速览

> 基于 SGLang 当前主线（2026 年中），以下特性在 第2章/4 主干章节之外值得单独跟踪。

| 特性 | 文件位置 | 说明 |
|------|---------|------|
| **DeepSeek V3.2 稀疏注意力（NSA / DSA）** | `sglang/srt/layers/attention/nsa_backend.py`、`dsa/` | Native Sparse Attention 与 DeepSeek Sparse Attention：压缩 KV 块 + 硬件对齐的稀疏掩码，长上下文下显著降内存/算力 |
| **DeepSeek V4 专用 Attention** | `layers/attention/deepseek_v4_backend.py`、`deepseek_v4_memory_pool.py` | V4 架构专用 backend 与内存池（含 HIP/Radix 变体） |
| **Hybrid Attention** | `layers/attention/hybrid_attn_backend.py`、`hybrid_linear_attn_backend.py` | 注意力 + 线性注意力（Mamba 类）混合架构的联合调度 |
| **D-LLM（Diffusion LLM）调度** | `sglang/srt/dllm/` | 面向 LLaDA、SDAR 等扩散式 LLM 的调度：`joint_threshold` / `low_confidence` 算法、mask token 管理 |
| **Elastic EP** | `sglang/srt/elastic_ep/` | 运行期动态增删专家组（Expert Backup 机制），应对负载波动 |
| **EPLB（专家并行负载均衡）** | `sglang/srt/eplb/` | 按实际专家访问分布做负载均衡（LPLB solver + 分布记录器），缓解 MoE 热点专家倾斜 |
| **KV Canary** | `sglang/srt/kv_canary/` | KV Cache 完整性校验/扰动注入工具，用于定位静默损坏与精度问题 |
| **Grammar / Function Calling** | `sglang/srt/constrained/`、`sglang/srt/function_call/` | xgrammar / outlines / llguidance 多种语法后端；DeepSeek-V3/V4、Qwen、Gemma 等数十个模型的 function-call 格式检测器 |
| **远端模型权重 Connector** | `sglang/srt/connector/` | 从 S3 / Redis / Azure 等远端加载权重（`weight_iterator` 抽象），便于集群化部署 |
| **可观测性** | `sglang/srt/observability/` | OpenTelemetry 等可观测性接入 |

**跟踪建议**：这些特性大多随模型架构演进（V3.2/V4、扩散 LLM）而来，面试或方案设计时可将其作为"了解当前主线"的加分项；核心的调度 / RadixAttention / Overlap / PD 分离仍以正文各节为准。
  14. sgl-kernel/                             ← C++/CUDA 自定义 kernel
```
