# 第3章 vLLM 源码深度解剖

> **面向角色**：需要阅读或修改 vLLM 源码的工程师  
> **目标**：建立从 API 入口到 GPU forward 的完整代码路径认知  
> **源码版本**：基于 [vLLM](https://github.com/vllm-project/vllm) 最新主线代码（V1 引擎架构）

---

## 3.1 vLLM 架构总览

### 3.1.1 核心模块与数据流

```
┌──────────────────────────────────────────────────────────────────┐
│                        vLLM V1 Engine                             │
│                                                                    │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐ │
│   │ Entry    │───▶│ Engine   │───▶│ Scheduler│───▶│ Executor   │ │
│   │ Points   │    │ Frontend │    │ (V1)     │    │ (GPU)      │ │
│   └──────────┘    └──────────┘    └──────────┘    └────────────┘ │
│        │               │               │               │          │
│  OpenAI API       AsyncLLM       Core Engine     Worker /        │
│  /v1/chat         Engine         Scheduler +     ModelRunner /   │
│  /v1/completions  (async_llm     KV Manager      Attention       │
│                    _engine.py)                    Backend         │
└──────────────────────────────────────────────────────────────────┘
```

**源码路径图**：

| 层级 | 主要文件 | 核心类 |
|------|---------|--------|
| API 入口 | `vllm/entrypoints/openai/api_server.py` | `api_server` (FastAPI) |
| 异步引擎 | `vllm/engine/async_llm_engine.py` | `AsyncLLMEngine` |
| 引擎前端 | `vllm/v1/engine/__init__.py` | `EngineCoreRequest`, `EngineCoreOutput` |
| 核心调度 | `vllm/v1/core/sched/scheduler.py` | `Scheduler` |
| KV Cache | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` |
| 执行器 | `vllm/v1/executor/uniproc_executor.py` | `UniProcExecutor` |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | `GPUWorker` |
| Model Runner | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner` |
| Attention | `vllm/v1/attention/backends/` | FlashAttention / FlashInfer / MLA |
| 量化 | `vllm/model_executor/layers/quantization/` | `FP8Config`, `AWQConfig` |

### 3.1.2 V1 vs Legacy 引擎

vLLM 目前处于 V0 → V1 的迁移期。V1 引擎的核心改进：

| 方面 | V0 (Legacy) | V1 (当前主线) |
|------|------------|---------------|
| 架构 | 单一 LLMEngine 类 | EngineCore + Frontend 分离 |
| 通信 | Python 对象传递 | msgspec 序列化（支持跨进程） |
| 调度 | 简单的 FCFS | Priority + Policy + KV Connector |
| Attention | 运行时选择 backend | 编译时静态选择 |
| CUDA Graph | 手动管理 | BreakableCUDAGraph + 多配置 |
| 扩展性 | 单进程 | DP + EEP 多 Engine Core |

**V1 的核心设计**：`EngineCore` 和 `Frontend` 通过 msgspec 消息通信，可以运行在不同进程甚至不同节点上。这使得 DP 和 PD 分离成为可能。

---

## 3.2 Scheduler：连续批处理的心脏 ★

### 3.2.1 Scheduler 类

**文件**：`vllm/v1/core/sched/scheduler.py`  
**类**：`class Scheduler(SchedulerInterface)`

**核心状态**：

```python
class Scheduler:
    # 请求管理
    requests: dict[str, Request]        # req_id → Request
    waiting: RequestQueue               # 等待队列（优先级排序）
    running: list[Request]              # 正在执行的请求

    # 调度约束
    max_num_running_reqs: int           # 最大并发请求数
    max_num_scheduled_tokens: int       # 每步最大调度 token 数
    max_model_len: int                  # 模型最大上下文长度

    # KV 管理
    kv_cache_manager: KVCacheManager    # KV Cache 分配器
    block_size: int                     # 每个 block 的 token 数（默认 16）

    # KV Connector (PD 分离)
    connector: KVConnectorBase_V1       # KV 传输连接器

    # 投机解码
    num_spec_tokens: int                # 投机 token 数
    use_eagle: bool                     # 是否使用 Eagle
```

### 3.2.2 关键方法

**`schedule()` — 核心调度逻辑**：

```
schedule() → SchedulerOutput:
  │
  ├── 1. 处理已完成请求
  │     running = [r for r in running if r.request_id not in finished]
  │
  ├── 2. 调度新请求（从 waiting queue）
  │     for req in waiting:
  │       if len(running) >= max_num_running_reqs: break
  │       if num_scheduled_tokens >= max_num_scheduled_tokens: break
  │       │
  │       ├── 为新请求分配 KV Cache blocks
  │       │   kv_blocks = kv_cache_manager.allocate_slots(req)
  │       │   if 分配失败 → break (no more memory)
  │       │
  │       └── 将请求加入 running
  │           scheduled_running.append(req)
  │
  ├── 3. 为 running batch 准备每个请求的 block table
  │     for req in running:
  │       block_table.append(req.kv_block_ids)
  │
  ├── 4. 处理前缀缓存命中 (prefix caching)
  │     对相同 hash 的 block → 复用而非重新计算
  │
  └── 5. 构造 SchedulerOutput
        (scheduled_new_reqs, scheduled_running_reqs,
         finished_req_ids, block_tables, ...)
```

### 3.2.3 Preemption 机制

当 KV Cache 不足时，Scheduler 可以**抢占**正在运行的请求：

```
Preemption 等级 (由高到低):
  1. RECOMPUTE: 丢弃请求的 KV Cache，后续重新 prefill（最便宜的重启方式）
  2. SWAP: 将 KV Cache block 换出到 CPU 内存（保留进度但开销大）
  
触发条件:
  - 新请求需要 KV Cache 但 free blocks 不足
  - 选择优先级最低的请求进行 preemption
```

### 3.2.4 KV Connector：PD 分离的 Scheduler 层 ★

```python
# scheduler.py __init__
self.connector = KVConnectorFactory.create_connector(
    config=self.vllm_config,
    role=KVConnectorRole.SCHEDULER,
    kv_cache_config=self.kv_cache_config,
)

# 在 schedule() 中:
# 1. 检查是否有从 P 节点传输过来的 KV Cache
# 2. 为 D 节点预分配 block，等待 KV 传输完成
# 3. 传输完成后，将请求加入 running batch
```

**KV Connector 的角色**：
- `SCHEDULER`: 在 Scheduler 中运行，负责协调 KV 传输
- `WORKER`: 在 Worker 中运行，负责实际的 KV 数据搬运

---

## 3.3 PagedAttention / KV Cache Manager ★

### 3.3.1 KVCacheManager

**文件**：`vllm/v1/core/kv_cache_manager.py`  
**类**：`class KVCacheManager`

```python
class KVCacheManager:
    """
    基于页式内存管理的 KV Cache 分配器。
    
    核心概念:
    - Block Size: 每个 block 存储的 token 数（通常 16 或 32）
    - Block Pool: 预分配的所有物理 blocks
    - Free Queue: 空闲 blocks 列表
    - Block Table: 每个请求的 虚拟block → 物理block 映射
    """
    
    block_pool: KVCacheBlock[]   # 所有物理 block
    free_blocks: deque            # 空闲 block 队列
    
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_lookahead_tokens: int = 0,  # 投机解码需要额外 slots
    ) -> KVCacheBlocks | None:
        """为请求分配 num_new_tokens 个 token 的 KV Cache slots。
        
        Returns:
            KVCacheBlocks 如果分配成功
            None 如果没有足够的 free blocks
        """
    
    def free(self, request: Request) -> None:
        """释放请求占用的所有 blocks 回 free queue"""
    
    def get_block_table(self, request: Request) -> list[int]:
        """获取请求的 block table（物理 block ID 列表）"""
```

### 3.3.2 KVCacheBlock

**文件**：`vllm/v1/core/kv_cache_utils.py`

```python
class KVCacheBlock:
    block_id: int              # 物理 block ID
    ref_cnt: int               # 引用计数（多个请求共享时 > 1）
    block_hash: int | None     # block 内容哈希（prefix caching 用）
    is_null: bool              # 是否为 padding block
```

### 3.3.3 Prefix Caching 实现

```
前缀缓存的流程:
  
  1. 请求到达时，计算其 prompt tokens 的 block-level hash
  2. 在 block pool 中查找 hash 匹配的 block
  3. 如果命中 → 共享该 block (ref_cnt += 1)，跳过这部分 prefill
  4. 如果未命中 → 分配新 block，prefill 后计算 hash

  hash 计算:
    对 block 内的 token 序列做 xxhash (又快又准)
    
  Key Insight:
    system prompt 在所有请求中完全相同
    → 所有请求共享 system prompt 的 KV blocks
    → 只需 prefill system prompt 一次！
```

### 3.3.4 混合 KV Cache Coordinator

**文件**：`vllm/v1/core/kv_cache_coordinator.py`  
**类**：`class HybridKVCacheCoordinator`

支持**多层 KV Cache**（GPU HBM + CPU DRAM + 分布式存储）的统一协调：

```python
class HybridKVCacheCoordinator:
    """
    管理多层 KV Cache 之间的 block 迁移。
    
    gpu → cpu: swap out (内存压力时)
    cpu → gpu: swap in (请求恢复时)
    remote → gpu: KV load (PD 分离 / HiCache)
    """
```

---

## 3.4 Model Runner：GPU 执行引擎 ★

### 3.4.1 GPUModelRunner

**文件**：`vllm/v1/worker/gpu_model_runner.py`  
**类**：`class GPUModelRunner`  
**规模**：约 3000+ 行（vLLM 中最大的单文件之一）

**核心方法**：

```python
class GPUModelRunner:
    model: nn.Module                      # 加载的 HuggingFace 模型
    kv_cache_config: KVCacheConfig        # KV Cache 配置
    input_batch: GPUInputBatch            # 批输入数据
    
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """
        核心执行流程：
        
        1. prepare_inputs():
           - 将 SchedulerOutput 转换为 GPU tensors
           - 构造 block_tables, input_ids, positions
           - 处理多模态输入
        
        2. model.forward():
           - 逐层执行 Transformer
           - 每层: Attention → FFN
           - CUDA Graph 加速（跳过 kernel launch overhead）
        
        3. sample():
           - 从最后一层 hidden states 计算 logits
           - 应用采样参数（temperature, top_p, top_k）
           - 返回采样的 token ids
        """
    
    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelInputForGPU:
        """
        准备 GPU 输入：
        - input_ids: [num_tokens] tensor
        - positions: [num_tokens] tensor (在序列中的位置)
        - block_tables: [num_requests, max_blocks_per_seq]
        - slot_mapping: [num_tokens] (token → KV cache slot)
        """
    
    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """从 logits 采样下一个 token。"""
```

### 3.4.2 CUDA Graph 优化 ★

```
问题: 每个 decode step 都需要重新 launch 数百个 CUDA kernel
      kernel launch overhead (~5-10us per kernel) 加起来可观

CUDA Graph 方案:
  1. 记录一次完整的 forward pass 的 kernel launch 序列
  2. 后续 step 直接 replay 整个 graph（只需一次 launch）
  3. overhead 从 ~100us → ~5us

限制:
  - Graph 中的 input shapes 必须固定 → 需要 padding
  - 动态 batch size 需要 pre-warm 多个 graph 配置
  
vLLM 对此的解决方案 (BreakableCUDAGraph):
  - 支持在 graph 中"打孔"（breakable），在固定点注入动态逻辑
  - 编译模式: CompilationMode.VLLM_COMPILE
```

源码位置：`vllm/compilation/breakable_cudagraph.py` — `BreakableCUDAGraphWrapper`

### 3.4.3 Attention Backend 选择 ★

**文件**：`vllm/v1/attention/selector.py`

```python
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    ...
) -> type[AttentionBackend]:
    """
    根据模型配置自动选择最优 attention backend:
    
    1. use_mla=True → MLAAttentionBackend (DeepSeek-V2/3)
    2. H100 + FP8 KV Cache → FlashAttention-3
    3. decode 阶段 + batch > 1 → FlashInfer (更快)
    4. 默认 → FlashAttention-2
    5. CPU → Torch SDPA
    """
```

### 3.4.4 分布式支持

**文件**：`vllm/v1/worker/gpu_model_runner.py`（后半部分）

```python
class GPUModelRunner:
    # TP 通信组
    tp_group: GroupCoordinator        # 层内通信 (AllReduce)
    pp_group: GroupCoordinator        # 层间通信 (send/recv activations)
    dcp_group: GroupCoordinator       # Context Parallelism
    
    def execute_model(...):
        # TP: 不需要模型层面的修改，AllReduce 在每层 forward 后自动执行
        # PP: 通过 send/recv 在层边界传递 activations
        # CP: 在 attention 计算中分序列长度
```

**并行化的 Model Shard**：

```
TP=4 时:
  GPUModelRunner 在每个 rank 上独立运行
  但模型权重是切分后的
  每层 forward 后自动做 AllReduce (由 parallel_state 管理)
  
PP=2 时:
  rank 0: GPUModelRunner 加载 layers[0:N/2]
  rank 1: GPUModelRunner 加载 layers[N/2:N]
  通过 PPGroup.send/recv 传递 hidden states
```

---

## 3.5 投机解码实现 ★

### 3.5.1 架构概览

**文件**：`vllm/v1/spec_decode/`

```
spec_decode/
├── eagle.py                     # Eagle proposer
├── medusa.py                    # Medusa proposer
├── ngram_proposer.py            # ngram proposer
├── draft_model.py               # Draft model 接口
├── eagle_worker_v2.py           # Eagle v2 worker
├── eagle_info.py                # Eagle metadata
├── eagle_draft_cuda_graph_runner.py  # CUDA Graph 加速
├── reject_sampling.py           # 接受/拒绝采样逻辑
└── metrics.py                   # 投机解码指标
```

### 3.5.2 Eagle Proposer 关键流程

```python
class EagleProposer(SpecDecodeProposer):
    """
    Eagle 草稿模型：
    - 从 target model 的 hidden states 出发
    - 通过 1-2 层 decoder-only Transformer 预测 future tokens
    - 共享 target model 的 embedding + LM head
    """
    
    def propose(
        self,
        target_hidden_states: torch.Tensor,  # 来自 target model
        sampling_metadata: SamplingMetadata,
    ) -> list[list[int]]:
        """
        1. target_hidden_states → Eagle draft model → draft logits
        2. draft logits → sample → draft_tokens (每个请求 k 个)
        3. 返回 draft_tokens
        """
    
    def verify(
        self,
        draft_tokens: list[list[int]],
        target_model: nn.Module,
    ) -> tuple[list[int], list[bool]]:
        """
        1. target model 一次 forward (处理 real + draft tokens)
        2. 对每个 draft token 做 accept/reject 采样
        3. 返回 (accepted_tokens, accept_mask)
        """
```

### 3.5.3 投机解码的 Scheduler 集成

```python
# scheduler.py 中的关键参数
self.num_spec_tokens = vllm_config.num_speculative_tokens  # 投机 token 数
self.use_eagle = True                                       # 使用 Eagle
self.num_lookahead_tokens = self.num_spec_tokens            # 需要预留的 KV slots

# 调度时:
# 1. 为每个请求分配 extra lookahead slots (num_spec_tokens 个)
# 2. Model Runner 在执行时:
#    - 先做 normal forward (1 token)
#    - Eagle proposer.predict() → draft k tokens
#    - Target forward verify() → accept/reject
```

---

## 3.6 量化支持

### 3.6.1 量化配置与加载

**文件**：`vllm/model_executor/layers/quantization/`

vLLM 的量化采用 **方法注册** 模式：

```python
# quantization/__init__.py
QUANTIZATION_METHODS = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "fp8": FP8Config,
    "bitsandbytes": BitsAndBytesConfig,
    "compressed-tensors": CompressedTensorsConfig,
    ...
}

# 模型加载时:
quant_config = QUANTIZATION_METHODS[quant_method].from_config(model_config)
# quant_config 提供:
#   - get_quant_method(layer) → 该层的量化方法
#   - 量化后的 layer wrapper
```

### 3.6.2 FP8 推理流程

```python
# quantization/fp8.py
class FP8Config:
    """
    FP8 推理配置：
    
    支持的 FP8 格式：
    - E4M3 (4 exponent, 3 mantissa): 推理默认
    - E5M2 (5 exponent, 2 mantissa): 训练/梯度
    
    量化流程：
    1. 加载 FP8 checkpoint (如果提供了)
    2. 否则，在加载时做 online quantization:
       weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
    3. 每层 forward:
       - Weight: FP8 (包在量化 wrapper 中)
       - Activation: FP8 (通过 dynamic scaling 量化)
       - Matmul: FP8 Tensor Core
    4. KV Cache: 可选 FP8 存储
    """
```

---

## 3.7 最新特性速览

| 特性 | 文件位置 | 状态 |
|------|---------|------|
| **V1 Engine** | `vllm/v1/` | 主线开发中 |
| **Multi-Engine DP** | `vllm/v1/engine/` — `EngineCoreRequest.data_parallel_rank` | 已支持 |
| **EEP (Elastic EP)** | `vllm/v1/engine/` — `ReconfigureDistributedRequest`（运行期动态调整 DP rank/EP 分组） | 已支持 |
| **KV Events** | `vllm/distributed/kv_events/` | 已支持 |
| **EC Transfer** | `vllm/distributed/ec_transfer/` | 已支持 |
| **Structured Output** | `vllm/v1/structured_output/` | 已支持 |
| **Breakable CUDA Graph** | `vllm/compilation/breakable_cudagraph.py` | 已支持 |
| **Prefix Caching** | `vllm/v1/core/kv_cache_manager.py` → `enable_caching` | 已支持 |
| **LoRA** | `vllm/lora/` + `vllm/v1/worker/lora_model_runner_mixin.py` | 已支持 |
| **Multimodal** | `vllm/multimodal/` + model implementations | 已支持 |
| **KV Cache Offload / Tiering** | `vllm/v1/kv_offload/`（CPU 卸载 + 多级 tiering，支持 FS/P2P/async-lookup 二级存储） | 已支持 |
| **SimpleKVOffload** | `vllm/v1/simple_kv_offload/`（面向单卡的轻量 CPU offload） | 已支持 |
| **DFlash 投机解码** | `vllm/v1/spec_decode/dflash.py`（in-filling 式投机，支持 Qwen3.5 多模态） | 已支持 |
| **动态投机解码 (Dynamic Spec Decode)** | `vllm/v1/spec_decode/dynamic/`（运行时调整 draft 长度/策略） | 已支持 |
| **Suffix Decoding** | `vllm/v1/spec_decode/suffix_decoding.py`（基于后缀树的采样级缓存，需 arctic_inference） | 实验特性 |
| **Late Interaction Pooling** | `vllm/v1/pool/late_interaction.py`（多模态 token 延迟融合池化） | 已支持 |
| **Qwen3-DSpark / DeepSeek V4 等新架构** | `vllm/model_executor/models/qwen3_dspark.py` 等 | 已支持 |

---

## 3.8 源码阅读路线建议

如果你想深入 vLLM 源码，推荐以下阅读顺序：

```
新人第一遍（理解架构）:
  1. vllm/entrypoints/openai/api_server.py   ← 看请求怎么进来的
  2. vllm/v1/engine/__init__.py              ← 看 EngineCoreRequest/Output 协议
  3. vllm/v1/core/sched/scheduler.py         ← 看调度逻辑
  4. vllm/v1/worker/gpu_model_runner.py      ← 看前 200 行的 execute_model

进阶（理解细节）:
  5. vllm/v1/core/kv_cache_manager.py         ← 看 block 分配
  6. vllm/v1/attention/backends/              ← 看 attention kernel
  7. vllm/model_executor/layers/quantization/ ← 看量化实现
  8. vllm/v1/spec_decode/eagle.py             ← 看投机解码

专家（修改/贡献）:
  9. vllm/v1/core/sched/scheduler.py          ← 调度策略修改
  10. vllm/v1/worker/gpu_model_runner.py      ← 性能优化
  11. vllm/distributed/                        ← 分布式修改
```
