# Part 16：边缘推理框架与软件栈

> **面向角色**：边缘 AI 推理优化工程师、框架适配工程师、端侧部署工程师  
> **前置知识**：Part 15 边缘硬件约束、基本量化概念、LLM 推理流程  
> **目标**：掌握主流边缘推理框架的定位、用法和关键源码结构，能够在 Jetson、手机 NPU、RISC-V 等平台上完成模型转换与运行时调优

---

## 16.1 边缘框架全景

### 16.1.1 边缘推理框架的分类

与云侧相对统一的 CUDA / PyTorch / vLLM 生态不同，边缘侧的推理框架呈现明显的"垂直化"特征：芯片厂商为了发挥自家 NPU 的峰值算力，往往会提供一套从模型转换到运行时调度的私有工具链。按照开放程度和硬件绑定关系，可以把主流边缘框架分为三类：

| 类别 | 代表框架 | 硬件绑定 | 开放程度 | 典型场景 |
|------|---------|---------|---------|---------|
| **NVIDIA 锁定生态** | TensorRT、TensorRT-LLM | NVIDIA GPU / Jetson / DLA | 半开放（runtime 闭源） | 机器人、工业视觉、高端边缘盒子 |
| **跨平台通用框架** | llama.cpp、ONNX Runtime、OpenVINO、MNN、MLC-LLM、ExecuTorch | CPU / GPU / 部分 NPU | 开源 | 手机、PC、AIoT、快速原型 |
| **芯片厂商专用 SDK** | Qualcomm QNN、Rockchip RKNN、Apple Core ML、华为 CANN / MindSpore Lite | 各自 NPU | 私有工具链为主 | 手机、车载、低成本摄像头、国产替代 |

**关键洞察**：边缘框架选型的核心不是"哪个框架最快"，而是**目标平台是否支持、模型是否能跑通、工具链是否成熟**。一个闭源但能在目标 NPU 上跑通的 SDK，往往比一个开源但算子支持缺失的框架更有工程价值。

### 16.1.2 与云侧引擎的关系

云侧推理引擎（vLLM、SGLang、TensorRT-LLM）和边缘推理框架并非完全割裂，它们在技术栈上有明显的继承关系：

- **模型格式向下转换**：云侧训练/导出的 PyTorch / ONNX / HuggingFace 模型，通常要先经过量化、图优化，再转换为边缘框架的私有格式（如 TensorRT engine、RKNN、QNN DLC、GGUF）。
- **核心模块复用**：TensorRT-LLM 的 C++ runtime、llama.cpp 的 GGML backend 都直接继承或移植了云侧 kernel 优化思路。
- **调度策略简化**：云侧强调 Continuous Batching、PD 分离、HiCache；边缘侧更关注单 batch 低延迟、静态形状、内存受限下的 KV Cache 管理。

```
云侧训练/导出
     │
     ▼
PyTorch / ONNX / HuggingFace
     │
     ├── 量化（INT8/INT4/FP8）───┐
     ├── 图优化 / 算子融合        │
     └── 目标格式转换            ▼
                          TensorRT engine / RKNN / QNN DLC / GGUF / MNN / OV IR
                                      │
                                      ▼
                              边缘运行时（Jetson / 手机 NPU / RISC-V / ARM）
```

### 16.1.3 框架选型矩阵 ★

| 框架 | 主要硬件 | 量化支持 | 易用性 | 性能天花板 | 生态/社区 | 最佳场景 |
|------|---------|---------|--------|-----------|----------|---------|
| **TensorRT-LLM** | NVIDIA GPU / Jetson / DLA | FP8/FP16/INT8/INT4/W4A16 | 中 | 极高（NVIDIA 锁定） | 官方主导 | 高端边缘、延迟敏感 |
| **llama.cpp** | CPU / CUDA / Metal / Vulkan / OpenCL / 国产后端 | GGML 量化（Q4/Q5/Q8/IQ） | 高 | 中-高 | 极活跃 | 端侧 LLM、快速部署 |
| **ONNX Runtime** | CPU / GPU / 部分 NPU（EP） | INT8/FP16/动态量化 | 高 | 中 | 活跃 | 通用 CV/NLP、跨平台 |
| **OpenVINO** | Intel CPU / GPU / NPU | INT8/FP16/NNCF | 高 | 中 | 官方+社区 | Intel AIPC、工业质检 |
| **MNN** | ARM / x86 / RISC-V / NPU / GPU | INT8/FP16/混合精度 | 中 | 中-高 | 阿里生态 | AIoT、平头哥玄铁 |
| **QNN** | Qualcomm Hexagon NPU | INT8/FP16 | 中 | 高（Q 系锁定） | 官方 | 手机、车载、XR |
| **RKNN** | Rockchip NPU | INT8/FP16 | 中 | 中 | 官方+社区 | 低成本边缘盒子、摄像头 |
| **MLC-LLM** | 多种 GPU / NPU（TVM） | INT4/INT8/FP16 | 中 | 高 | 社区 | 手机/边缘 LLM 编译优化 |
| **ExecuTorch** | ARM / Apple / Qualcomm | 量化（PyTorch 生态） | 中 | 中 | Meta+社区 | 移动端 PyTorch 模型 |
| **TensorFlow Lite / LiteRT** | ARM / Edge TPU / 多种 NPU | INT8/FP16 | 高 | 中 | 极广 | 移动端、嵌入式 CV |

**选型原则**：

1. **先定硬件，再选框架**：没有跨平台框架能通吃所有 NPU，硬件白名单决定候选集。
2. **先跑通模型，再谈性能**：算子 fallback 到 CPU 往往比理论峰值更致命。
3. **量化是边缘必修课**：FP16 7B 模型在 8 GB 共享内存设备上几乎无法运行，必须降到 INT4/INT8。
4. **工具链版本锁定**：边缘 SDK 与 BSP、驱动版本强耦合，升级成本远高于云侧。

---

## 16.2 TensorRT / TensorRT-LLM 源码级分析 ★

### 16.2.1 仓库布局

TensorRT-LLM 是 NVIDIA 面向 LLM 推理的优化框架，核心定位是**把 HuggingFace 模型编译成 TensorRT engine，并在 NVIDIA GPU / Jetson 上以低延迟运行**。仓库主要分为三层：

| 目录 | 作用 | 关键文件 |
|------|------|---------|
| `tensorrt_llm/` | Python 层：模型定义、builder、量化、命令行 | `builder.py`、`commands/build.py`、`network.py`、`models/`、`quantization/` |
| `cpp/` | C++ runtime、调度器、KV Cache、kernel、plugin | `executor/`、`batch_manager/`、`plugins/`、`runtime/`、`kernels/` |
| `examples/` / `docs/` | 模型示例与部署文档 | `examples/llama/`、`docs/source/deployment-guide/` |

> 注：根据 NVIDIA 主线最新规划，`backend="tensorrt"`（即传统 engine build 路径）被标记为 legacy，新特性主要投向 PyTorch / AutoDeploy 后端；但在 Jetson 和边缘低延迟场景中，TensorRT engine 路径仍是当前主流。

### 16.2.2 Builder 路径：从 HuggingFace 到 engine ★

**`TensorRT-LLM/tensorrt_llm/builder.py`** 是 engine 编译的 Python 入口。核心类 `Builder` 封装了 `nvinfer1.Builder`，负责创建 `Network`、`BuilderConfig` 并最终调用 `build_engine`。

```python
class Builder():
    def create_network(self) -> Network:
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        return Network()._init(
            self.trt_builder.create_network(
                explicit_batch_flag
                | (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            )
        )

    def create_builder_config(self, precision, timing_cache, tensor_parallel, ...):
        config = self.trt_builder.create_builder_config()
        # FP16 / BF16 / INT8 / FP8 / weight streaming / refit 等开关
        ...
```

**`TensorRT-LLM/tensorrt_llm/commands/build.py`** 则提供 `trtllm-build` 命令行入口。它解析 `--checkpoint_dir`、`--output_dir`、`--max_batch_size`、`--max_input_len`、`--max_seq_len`、`--max_num_tokens`、`--tp_size`、`--pp_size` 等参数，并调用 `Builder.build()`。

关键参数含义：

| 参数 | 含义 | 边缘调优注意 |
|------|------|-------------|
| `--max_batch_size` | engine 支持的最大 batch size | 边缘通常设小（1-8），避免 KV Cache 过度预留 |
| `--max_input_len` / `--max_seq_len` | 最大输入/总长度 | 直接决定 KV Cache 显存占用 |
| `--max_num_tokens` | 每步最大 token 数（padding removed） | 影响 prefill 阶段算子优化粒度 |
| `--opt_num_tokens` | 优化目标 token 数 | 越接近真实负载，kernel 选择越优 |
| `--tp_size` / `--pp_size` | 张量/流水线并行 | Jetson 单卡场景通常都为 1 |
| `--quantization` / `--kv_cache_dtype` | 权重量化与 KV Cache 类型 | 边缘常用 `fp8` / `int8` / `int4_awq` |

**`TensorRT-LLM/tensorrt_llm/network.py`** 封装了 `nvinfer1.INetworkDefinition`，提供 `net_guard` 上下文管理器、插件注册、mark output 等能力。Builder 会把 `PretrainedModel.forward` 中定义的 TensorRT 算子逐层添加到这个 network 里。

**完整 build 流程**：

```
HuggingFace checkpoint
        │
        ▼
PretrainedConfig + PretrainedModel  (tensorrt_llm/models/)
        │
        ▼
Network()._init(trt_builder.create_network(...))   (network.py)
        │
        ▼
Builder.create_builder_config(precision, quant_mode, ...)   (builder.py)
        │
        ▼
trllm-build CLI 调用 build()  →  serialized TensorRT engine
        │
        ▼
Executor 加载 engine，运行推理
```

### 16.2.3 Runtime / Executor ★

**`TensorRT-LLM/cpp/include/tensorrt_llm/executor/executor.h`** 定义了 C++ 侧的对外 API。核心类 `Executor` 是用户请求的入口，提供 request 入队、response 等待、统计信息导出等能力。

```cpp
namespace tensorrt_llm::executor {
class Executor {
public:
    Executor(std::filesystem::path const& modelPath, ModelType modelType,
             ExecutorConfig const& executorConfig);

    IdType enqueueRequest(Request const& llmRequest);
    std::vector<Response> awaitResponses(
        std::optional<std::chrono::milliseconds> const& timeout);
    void cancelRequest(IdType requestId);
    void shutdown();
    std::optional<std::shared_ptr<KVCacheEventManager>> getKVCacheEventManager() const;
    ...
};
}
```

同文件还定义了 `SamplingConfig`，覆盖 `beamWidth`、`topK`、`topP`、`temperature`、`repetitionPenalty`、`frequencyPenalty`、`minP`、`numReturnSequences` 等所有常见采样参数。

**`TensorRT-LLM/cpp/tensorrt_llm/executor/executor.cpp`** 中的 `Executor` 实现非常薄，只是把调用转发给 `Executor::Impl`：

```cpp
IdType Executor::enqueueRequest(Request const& llmRequest) {
    return mImpl->enqueueRequest(llmRequest);
}

std::vector<Response> Executor::awaitResponses(
    std::optional<std::chrono::milliseconds> const& timeout) {
    return mImpl->awaitResponses(timeout);
}
```

真正的调度逻辑在 `ExecutorImpl`、以及 `BatchManager` / `Scheduler` / `KVCacheManager` 中协同完成。Python 层的 `GenerationExecutor`（`tensorrt_llm/executor/executor.py`）通过 nanobind 绑定到这套 C++ core。

### 16.2.4 In-flight Batching / KV Cache Manager ★

TensorRT-LLM 的调度系统由 `BatchManager` 统一负责，其中 `capacityScheduler` 决定**哪些请求进入当前 batch**，`microBatchScheduler` 负责**chunked prefill 的 token 切分**，`kvCacheManager` 负责**物理 KV Cache 分配与复用**。

**`TensorRT-LLM/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h`** 定义了页式 KV Cache 的核心结构：

```cpp
namespace tensorrt_llm::batch_manager::kv_cache_manager {

class KVCacheBlock;
class BlockManager;
class KVCacheManager;

struct PoolConfiguration {
    SizeType32 windowSize;      // SWA 窗口大小
    SizeType32 sizePerHead;     // 每个 head 的字节数
    nvinfer1::DataType dtype;   // KV Cache 数据类型
};

} // namespace
```

`KVCacheBlock` 对应一个物理 block，包含 block id、memory pool index、引用计数、调度引用计数、radix tree lookup node、hash 等。`BlockManager` 维护 free block 列表、分配与回收；`KVCacheManager` 则在 block 之上管理每个序列的 block 链，并支持前缀复用（prefix caching）。

**`TensorRT-LLM/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`** 实现了 block 分配、序列 block 链遍历、前缀复用 token 计数等逻辑，关键函数包括 `getAllSequenceBlocks()`、`getUsableUniqueTokenCountForReuse()` 等。

**`TensorRT-LLM/cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp`** 实现了多种容量调度策略：

- `MaxRequestsScheduler`：尽量塞满并发数。
- `MaxUtilizationScheduler`：追求最大利用率，支持 two-steps look-ahead 和 prefix-aware scheduling。
- `GuaranteedNoEvictScheduler` / `StaticBatchScheduler`：保证已调度请求不会被驱逐，适合边缘确定性场景。

其中 `beneficialToSkip()` 体现了前缀感知的调度思想：如果某个请求的首个新 block 已经被同 batch 中另一个 context 请求生成，则延迟该请求到下一轮，以利用下一轮的前缀复用。

**`TensorRT-LLM/cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp`** 处理 chunked prefill：

- `ContextChunkingPolicy::kEQUAL_PROGRESS`：所有请求同步推进，公平分配 chunk。
- `ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED`：按请求顺序贪婪分配剩余算力预算。
- `reuse_adjusted_compute()`：对可复用的 KV Cache 前缀，计算实际需 forward 的 token 数，避免重复计算。

### 16.2.5 Plugin 机制 ★

TensorRT-LLM 的大量性能来自**自定义 TensorRT plugin**，尤其是 attention、quantization、MoE、speculative decoding 等算子。

**`TensorRT-LLM/cpp/include/tensorrt_llm/plugins/api/tllmPlugin.h`** 负责插件注册与日志：

```cpp
namespace tensorrt_llm::plugins::api {
    auto constexpr kDefaultNamespace = "tensorrt_llm";
    class LoggerManager { ... };
}

extern "C" {
    bool initTrtLlmPlugins(void* logger = ..., char const* libNamespace = ...);
    nvinfer1::v_1_0::IPluginCreator* const* getPluginCreators(std::int32_t& nbCreators);
}
```

**`TensorRT-LLM/cpp/tensorrt_llm/plugins/common/plugin.h`** 提供插件基类：

```cpp
namespace tensorrt_llm::plugins {

class BasePlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    void setPluginNamespace(char const* libNamespace) noexcept override;
    [[nodiscard]] char const* getPluginNamespace() const noexcept override;
protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BasePluginV3 : public nvinfer1::IPluginV3,
                     public nvinfer1::IPluginV3OneCore,
                     public nvinfer1::IPluginV3OneBuild,
                     public nvinfer1::IPluginV3OneRuntime {
    ...
};

} // namespace
```

**`TensorRT-LLM/cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h`** 是最重要的 plugin 之一，封装了 GPT 风格的 MHA/GQA/MLA，并支持：

- paged KV cache（`block_offsets`）与非 paged（`past_key_value_pool`）两种模式。
- remove input padding（`num_tokens` 而非 `batch_size * seq_len`）。
- FP8 / INT8 KV cache 量化（`kv_cache_quantization_scale`）。
- speculative decoding（`spec_decoding_*` 输入）。
- MLA（`q_a_proj_tensor`、`kv_a_proj_with_mqa_tensor` 等）。

```cpp
class GPTAttentionPlugin : public GPTAttentionPluginCommon {
public:
    GPTAttentionPlugin(int layer_idx, int num_heads, int num_kv_heads,
        int head_size, float q_scaling, ...,
        tensorrt_llm::kernels::ContextFMHAType context_fmha_type,
        int kv_cache_quant_mode, bool remove_input_padding,
        bool paged_kv_cache, int tokens_per_block,
        nvinfer1::DataType type, int32_t max_context_length, ...);

    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                nvinfer1::PluginTensorDesc const* outputDesc,
                void const* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override;
    ...
};
```

### 16.2.6 Jetson 部署路径与命令

在 Jetson 上部署 TensorRT-LLM 的典型路径如下：

```
HuggingFace 模型
        │
        ▼
convert_checkpoint.py 或 llmapi 转换
        │
        ▼
trllm-build 编译 TensorRT engine
        │
        ▼
jetson-containers / JetPack 环境运行
        │
        ▼
trllm-serve / 自定义 C++ Executor 推理服务
```

常用命令示例：

```bash
# 1. 转换 HuggingFace checkpoint 为 TensorRT-LLM checkpoint
python convert_checkpoint.py \
  --model_dir ./Llama-2-7b-hf \
  --output_dir ./tllm_checkpoint \
  --dtype float16

# 2. 编译 TensorRT engine（Jetson 上常设 max_batch_size=1，开启 INT4/AWQ 或 FP8）
trtllm-build \
  --checkpoint_dir ./tllm_checkpoint \
  --output_dir ./llama2_7b_engine \
  --gemm_plugin float16 \
  --max_batch_size 1 \
  --max_input_len 2048 \
  --max_seq_len 4096 \
  --max_num_tokens 2048

# 3. 启动 OpenAI 兼容服务
trllm-serve ./llama2_7b_engine --port 8000

# 4. 或者用 Python API 直接调用
python -c "from tensorrt_llm import LLM; llm = LLM(model='./llama2_7b_engine'); print(llm.generate('Hello'))"
```

**Jetson 调优注意点**：

- 共享内存容量有限，`max_seq_len` 和 `max_batch_size` 必须根据 LPDDR5 剩余容量精确计算。
- Jetson 的 DLA 可卸载部分卷积/MoE，但 attention 通常仍在 GPU 上执行。
- 尽量使用 `gemm_plugin`、`gpt_attention_plugin` 等插件，避免 TensorRT 原生算子 fallback。
- 如果显存仍不足，可尝试 `--weight_streaming` 或 `--remove_input_padding`。

### 16.2.7 源码阅读路线图

按以下顺序阅读 TensorRT-LLM 源码，可快速建立从 build 到 runtime 的完整认知：

```
Step 1: 命令入口
  tensorrt_llm/commands/build.py   # trtllm-build 参数与流程

Step 2: Python Builder
  tensorrt_llm/builder.py          # Builder 类、create_network、create_builder_config
  tensorrt_llm/network.py          # Network / net_guard / 插件注册

Step 3: 模型定义
  tensorrt_llm/models/             # PretrainedConfig / PretrainedModel / ForCausalLM
  tensorrt_llm/quantization/       # QuantAlgo / QuantMode

Step 4: C++ Executor 入口
  cpp/include/tensorrt_llm/executor/executor.h
  cpp/tensorrt_llm/executor/executor.cpp

Step 5: 调度与 KV Cache
  cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp
  cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp
  cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h
  cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp

Step 6: Plugin 与 Kernel
  cpp/include/tensorrt_llm/plugins/api/tllmPlugin.h
  cpp/tensorrt_llm/plugins/common/plugin.h
  cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h
  cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/
```

---

## 16.3 llama.cpp 源码级分析 ★

### 16.3.1 仓库布局

llama.cpp 是社区最活跃的端侧 LLM 推理框架之一，核心特点是**纯 C/C++、跨平台、GGUF 量化生态、多后端调度**。仓库主要结构如下：

| 目录 | 作用 | 关键文件 |
|------|------|---------|
| `src/` | llama 核心实现 | `llama-kv-cache.h/.cpp`、`llama-model-loader.h/.cpp`、`llama-quant.h/.cpp`、`llama-sampler.h/.cpp`、`llama-grammar.cpp` |
| `ggml/` | 张量计算库与后端抽象 | `include/ggml.h`、`include/ggml-backend.h`、`include/gguf.h`、`src/ggml-backend.cpp`、`src/ggml-backend-reg.cpp` |
| `gguf-py/` | GGUF 转换与量化工具 | `convert_hf_to_gguf.py` |
| `examples/` | 可执行程序源码 | `main/main.cpp`、`server/server.cpp`、`quantize/quantize.cpp` |
| `common/` | 公共 CLI 工具代码 | `common.cpp`、`sampling.cpp` |
| `models/` | 部分模型下载与测试脚本 | - |

### 16.3.2 GGUF 格式与量化 ★

**GGUF（Georgi Gerganov Universal Format）** 是 llama.cpp 自研的二进制模型格式，替代了早期的 GGML 格式。它把模型元数据（hyperparameters、tokenizer、rope 参数等）和权重张量统一存储在一个文件中，并支持按张量选择不同的数据类型。

**`llama.cpp/ggml/include/gguf.h`** 定义了 GGUF 的 C API：

```c
#define GGUF_MAGIC   "GGUF"
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_COUNT,
};

struct gguf_init_params {
    bool no_alloc;
    struct ggml_context ** ctx;
};

struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
int64_t gguf_get_n_kv(const struct gguf_context * ctx);
int64_t gguf_get_n_tensors(const struct gguf_context * ctx);
const char * gguf_get_tensor_name(const struct gguf_context * ctx, int64_t tensor_id);
enum ggml_type gguf_get_tensor_type(const struct gguf_context * ctx, int64_t tensor_id);
bool gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);
```

GGUF 文件结构：

```
[Magic: "GGUF"]
[Version: uint32]
[n_tensors: int64]
[n_kv: int64]
[KV pairs: key, type, value...]
[Tensor infos: name, n_dims, dims[], type, offset...]
[Padding]
[Tensor data blob]
```

**`llama.cpp/src/llama-quant.cpp`** 实现了模型量化逻辑。核心入口是 `llama_model_quantize()`，参数由 `llama_model_quantize_params` 定义（见 `llama.cpp/include/llama.h`），关键字段包括 `nthread`、`ftype`、`output_tensor_type`、`token_embedding_type`、`allow_requantize`、`quantize_output_tensor`、`pure`、`dry_run`、`imatrix`、`tt_overrides`、`prune_layers` 等。

llama.cpp 支持的常见量化类型：

| 类型 | 每权重 bit | 说明 | 适用场景 |
|------|-----------|------|---------|
| **Q4_0** | 4.5 | 旧版 4-bit，精度一般 | 兼容性测试 |
| **Q4_K_M** | 4.5 | K-quant 混合，精度较好 | 7B/13B 端侧首选 |
| **Q5_K_M** | 5.5 | 精度更高，体积稍大 | 对精度敏感的边缘 |
| **Q8_0** | 8.5 | 接近 FP16 精度 | 内存充裕时 |
| **IQ4_XS / IQ4_NL** | ~4.4 | Importance-aware 量化 | 配合 imatrix |
| **FP16 / FP32** | 16/32 | 无压缩 | 调试/小模型 |

**量化命令示例**：

```bash
# 1. 安装 llama.cpp 转换依赖
pip install -r llama.cpp/requirements/requirements-convert.txt

# 2. 将 HuggingFace 模型转换为 GGUF（FP16）
python llama.cpp/gguf-py/gguf/scripts/convert_hf_to_gguf.py \
  --input-dir ./Llama-2-7b-hf \
  --outfile ./llama-2-7b-f16.gguf \
  --outtype f16

# 3. 量化为 Q4_K_M
./llama-quantize ./llama-2-7b-f16.gguf ./llama-2-7b-q4_k_m.gguf Q4_K_M

# 4. 使用 imatrix 进行 importance-aware 量化（精度更高）
./llama-imatrix -m ./llama-2-7b-f16.gguf --output-file imatrix.dat
./llama-quantize ./llama-2-7b-f16.gguf ./llama-2-7b-iq4_xs.gguf IQ4_XS imatrix.dat
```

### 16.3.3 KV Cache 管理 ★

llama.cpp 的 KV Cache 由 `llama_kv_cache` 类管理，位于 **`llama.cpp/src/llama-kv-cache.h/.cpp`**。它实现了页式/槽式缓存、序列操作（删除/复制/移动/保留）、连续与非连续槽位查找等。

核心数据结构 `slot_info` 保存了每个 micro-batch 在 KV Cache 中的位置映射；关键方法包括 `prepare()`（分配 slot）、`find_slot()`（查找连续/非连续槽位）、`apply_ubatch()`（写入 KV）、以及 `seq_rm/cp/keep/add/div()`（序列级增删改）。

关键设计点：

1. **cell-based 存储**：KV Cache 被划分为等长 cell，每个 token 在每个层上占用一个 cell。
2. **slot_info 映射**：`prepare()` 为每个 micro-batch 找到可存放的 cell 位置；`idxs` 数组给出 token 到 cell 的映射。
3. **序列操作**：`seq_cp` 支持 KV 复制（用于 beam search），`seq_rm` / `seq_keep` 支持动态增删序列。
4. **连续与非连续槽位**：`find_slot(cont=true)` 用于需要连续 KV 的 kernel 优化路径。

**滑动窗口注意力（SWA）** 由 **`llama.cpp/src/llama-kv-cache-iswa.h/.cpp`** 处理。它维护了两个 `llama_kv_cache` 实例：一个给非 SWA 层，一个给 SWA 层，并通过 `llama_kv_cache_iswa` 对外暴露统一的 `llama_memory_i` 接口，提供 `get_base()` 和 `get_swa()` 分别访问两套缓存。

### 16.3.4 模型加载 ★

**`llama.cpp/src/llama-model-loader.h/.cpp`** 负责从 GGUF 文件读取元数据、创建张量、加载权重。核心结构 `llama_model_loader` 包含 `llama_tensor_weight`（`idx` 源文件索引、`offs` 文件偏移、`tensor` 指针）、`weights_map`（按层排序的权重表）、`load_all_data()`（支持 mmap 与 direct I/O 加载）、`get_weight()` / `require_weight()` 等。

加载流程：

1. `gguf_init_from_file()` 解析 GGUF header 和 tensor info。
2. `llama_model_loader` 根据 metadata 创建 `ggml_tensor`，并建立 `weights_map`。
3. `load_all_data()` 按 backend buffer 类型分批读取权重，支持 mmap 和 direct I/O。
4. 权重根据 `tensor_buft_overrides` 被放置到对应 backend buffer（CPU / CUDA / Metal / Vulkan 等）。

公共 API 在 `llama.cpp/include/llama.h` 中：

```c
LLAMA_API struct llama_model * llama_model_load_from_file(
    const char * path_model,
    struct llama_model_params params);

LLAMA_API struct llama_context * llama_init_from_model(
    struct llama_model * model,
    struct llama_context_params params);
```

### 16.3.5 Sampling ★

llama.cpp 的采样系统分为两层：

1. **C API 层**：`llama.cpp/include/llama.h` 定义了 `llama_sampler` 接口和内置采样器。
2. **实现层**：`llama.cpp/src/llama-sampler.h/.cpp` 定义了 `llama_sampler_chain` 等内部结构；`llama.cpp/src/llama-grammar.cpp` 实现了 GBNF 语法约束。

**采样器接口**（`llama.h`）定义了 `llama_sampler_i` 回调集合：`name`、`accept`、`apply`、`reset`、`clone`、`free`、`backend_init`；同时提供 `llama_sampler_chain_init/add/sample` 以及 `llama_sampler_init_top_k/top_p/min_p/temp/dist/grammar` 等内置采样器。

**`llama.cpp/src/llama-sampler.h`** 中的 `llama_sampler_chain` 是实际运行链，内部持有 `std::vector<info> samplers`、`std::vector<llama_token_data> cur` 和采样计时统计。

**语法约束**（`llama.cpp/src/llama-grammar.cpp`）：llama.cpp 使用类 GBNF（Gerganov BNF）格式定义输出约束，解析器会逐 token 判断候选是否满足语法。源码中实现了 UTF-8 解码、hex 解析、name/number/char 解析等基础工具函数。

### 16.3.6 多后端调度 ★

llama.cpp 的跨平台能力来自 **GGML backend 抽象层**。`ggml_backend` 负责 buffer 分配、张量拷贝、算子执行；不同硬件（CUDA、Metal、Vulkan、OpenCL、SYCL、CANN 等）各自实现 backend。

**`llama.cpp/ggml/include/ggml-backend.h`** 定义了核心接口：buffer type / buffer / backend / device / reg 的 opaque handle，`ggml_backend_buft_alloc_buffer`、`ggml_backend_tensor_copy`、`ggml_backend_synchronize`、`ggml_backend_graph_compute`，以及 `ggml_backend_dev_type`（CPU/GPU/IGPU/ACCEL/META）等枚举。

**`llama.cpp/ggml/src/ggml-backend-reg.cpp`** 是 backend 注册中心，在 `ggml_backend_registry` 构造函数中通过 `GGML_USE_CUDA`、`GGML_USE_METAL`、`GGML_USE_VULKAN`、`GGML_USE_SYCL`、`GGML_USE_OPENCL`、`GGML_USE_CANN`、`GGML_USE_CPU` 等编译宏注册对应后端。

llama.cpp 在构建计算图时，会根据每个算子的 `backend_supports_op` 结果，把张量分配到合适的 backend；不支持的算子自动回退到 CPU。用户可以通过 `LLAMA_CUDA_VISIBLE_DEVICES`、`GGML_VK_VISIBLE_DEVICES` 等环境变量控制后端选择。

### 16.3.7 典型命令与源码阅读路线图

**常用命令**：

```bash
# 交互式聊天
./llama-cli -m ./llama-2-7b-q4_k_m.gguf -p "Hello," -n 128 -ngl 999

# 启动兼容 OpenAI 的 HTTP 服务
./llama-server -m ./llama-2-7b-q4_k_m.gguf -ngl 999 --host 0.0.0.0 --port 8080

# 模型量化
./llama-quantize ./model-f16.gguf ./model-q4_k_m.gguf Q4_K_M

# 查看模型信息
./llama-gguf-split --list ./model.gguf
```

**源码阅读路线**：

```
Step 1: GGUF 格式
  ggml/include/gguf.h
  ggml/src/gguf.cpp

Step 2: 量化
  include/llama.h  # llama_model_quantize_params
  src/llama-quant.cpp

Step 3: 模型加载
  src/llama-model-loader.h
  src/llama-model-loader.cpp

Step 4: KV Cache
  src/llama-kv-cache.h
  src/llama-kv-cache.cpp
  src/llama-kv-cache-iswa.h
  src/llama-kv-cache-iswa.cpp

Step 5: 采样与语法
  include/llama.h  # llama_sampler_i
  src/llama-sampler.h
  src/llama-sampler.cpp
  src/llama-grammar.cpp

Step 6: 多后端
  ggml/include/ggml-backend.h
  ggml/src/ggml-backend.cpp
  ggml/src/ggml-backend-reg.cpp

Step 7: CLI 入口
  examples/main/main.cpp
  examples/server/server.cpp
```

---

## 16.4 MNN（平头哥/阿里生态）★★

### 16.4.1 架构：Converter / Interpreter / Backend

MNN 是阿里巴巴开源的轻量级深度学习推理框架，定位与 TensorFlow Lite、ONNX Runtime Mobile 类似，但在**阿里/平头哥生态、ARM/RISC-V 后端、端侧模型压缩**上有独特优势。其核心架构分为三层：

```
训练框架模型（TensorFlow / PyTorch / ONNX / Caffe）
                    │
                    ▼
        ┌─────────────────────┐
        │     Converter       │  图优化、量化、算子融合
        │  (tools/converter/) │  生成 .mnn 模型
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │    Interpreter      │  会话管理、内存池、调度
        │   (source/core/)    │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │      Backend        │  CPU / OpenCL / Vulkan / Metal / NPU / RISC-V
        │  (source/backend/)  │
        └─────────────────────┘
```

| 层级 | GitHub 路径 | 主要职责 |
|------|------------|---------|
| **Converter** | `tools/converter/` | 模型解析、图优化、量化转换 |
| **Interpreter** | `source/core/` | Session、Tensor、Schedule、内存管理 |
| **Backend** | `source/backend/` | 各硬件后端注册与算子实现 |
| **Express IR** | `express/` / `source/expr/` | 前端表达式 / 中间表示 |
| **Schema** | `schema/` | 模型序列化格式定义 |
| **Python API** | `pymnn/` | Python 绑定与工具 |
| **LLM Runtime** | `transformers/` | 大模型/扩散模型运行时 |

### 16.4.2 量化与混合精度

MNN 提供离线量化工具链，支持对称/非对称量化、per-channel / per-tensor、INT8 / FP16 混合精度。关键流程：

1. **模型转换**：用 `MNNConvert` 把 ONNX / TensorFlow / PyTorch 模型转为 `.mnn`。
2. **校准（Calibration）**：喂入代表性数据集，统计激活 min/max，生成量化参数。
3. **量化模型生成**：生成 INT8/FP16 混合的 `.mnn`。
4. **运行时精度回退**：对精度敏感的层保留 FP16，其余用 INT8。

```bash
# 转换 ONNX 模型为 MNN
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN

# 离线量化（需要校准数据集）
./quantized.out model.mnn quant.mnn config.json

# Python API 量化
python -m MNN.tools.mnnquant \
  --src_model model.mnn \
  --dst_model model_int8.mnn \
  --config config.json
```

### 16.4.3 RISC-V / NPU backend

MNN 对平头哥玄铁 RISC-V 和端侧 NPU 有较好的支持。后端抽象使得接入新硬件 NPU 时，只需要在 `source/backend/` 下实现：

- `**Backend**` 类：buffer 分配、算子注册、图执行。
- `**Execution**` 类：单个算子的具体实现或调用 NPU driver。
- `**Creator**` 工厂：根据算子类型创建 Execution。

玄铁 RISC-V 后端通常利用 RVV（RISC-V Vector）扩展做向量化，NPU 后端则把支持的子图下放到芯片驱动执行，不支持的子图回退到 CPU。

### 16.4.4 源码阅读路线

```
Step 1: 模型格式
  schema/

Step 2: 转换器
  tools/converter/
  tools/quantization/

Step 3: 核心运行时
  source/core/Interpreter.cpp
  source/core/Session.cpp
  source/core/Schedule.cpp

Step 4: 后端抽象
  source/backend/
  source/backend/cpu/
  source/backend/opencl/
  source/backend/vulkan/

Step 5: 平头哥/玄铁相关后端
  source/backend/  中查找 RISC-V / custom NPU 实现
```

---

## 16.5 其他框架 overview

### 16.5.1 ONNX Runtime / ONNX Runtime GenAI

**ONNX Runtime（ORT）** 是微软开源的跨平台推理引擎，通过 **Execution Provider（EP）** 接入不同硬件：CPU、CUDA、TensorRT、DirectML、OpenVINO、QNN、CoreML、Rockchip NPU 等。ORT 在边缘场景的优势是模型格式通用（ONNX）、工具链成熟、社区庞大。

**ONNX Runtime GenAI** 是专为生成式 AI 设计的上层 API，封装了 LLM 的 tokenization、KV Cache、采样、多轮对话等逻辑，开发者只需关注 `Generator` 和 `GeneratorParams`。

```bash
# 使用 ONNX Runtime GenAI 的 model builder 生成量化模型
python -m onnxruntime_genai.models.builder \
  -m meta-llama/Llama-2-7b-hf \
  -o ./llama2_7b_onnx \
  -p int4 \
  -e cuda

# Python 推理
import onnxruntime_genai as og
model = og.Model("./llama2_7b_onnx")
tokens = model.generate("Hello,")
```

**适用场景**：需要一次开发、多硬件部署；已经在 ONNX 生态中的 CV/NLP 模型。

### 16.5.2 OpenVINO / OpenVINO GenAI

**OpenVINO** 是 Intel 推出的推理优化工具包，支持 Intel CPU / GPU / NPU，也支持 ARM 等第三方硬件。它通过模型优化（MO / OVC）把 PyTorch / ONNX / TensorFlow 转换为 IR（Intermediate Representation），并在运行时做图优化、量化、算子融合。

**OpenVINO GenAI** 是面向 LLM 的高层 API，提供 `LLMPipeline`，封装了 prompt 处理、tokenization、KV Cache、采样。

```bash
# 导出 HuggingFace 模型为 OpenVINO IR
optimum-cli export openvino \
  --model meta-llama/Llama-2-7b-hf \
  --task text-generation-with-past \
  --weight-format int4 \
  ./llama2_7b_ov

# C++ / Python 使用 ov::genai::LLMPipeline
```

**适用场景**：Intel AIPC、工业边缘网关、需要利用 Intel NPU 低功耗特性的场景。

### 16.5.3 Qualcomm QNN

**Qualcomm Neural Network（QNN）SDK** 是高通为 Hexagon NPU / DSP / GPU 提供的专用推理栈。它把模型转换为 `.dlc`（Deep Learning Container），并通过 QNN runtime 在 Snapdragon 手机、车载平台、XR 设备上运行。

典型转换流程：

```bash
# 1. PyTorch / ONNX -> ONNX 优化
python -m onnxsim model.onnx model_sim.onnx

# 2. ONNX -> QNN DLC（使用 qnn-onnx-converter）
qnn-onnx-converter --input_network model_sim.onnx \
  --output_path model.dlc \
  --input_dim input 1,224,224,3

# 3. 量化（PTQ）
qnn-quantize-override --input_dlc model.dlc --output_dlc model_int8.dlc \
  --input_list raw_list.txt

# 4. 在设备上运行
qnn-net-run --model model_int8.dlc --backend libQnnHtp.so
```

**适用场景**：Snapdragon 手机/平板、Snapdragon Ride 车载、XR 眼镜等高通平台。

### 16.5.4 Rockchip RKNN / RKNN-LLM

**RKNN** 是瑞芯微为其 NPU 提供的模型转换与运行时工具链。RKNN Toolkit2 支持 PyTorch / ONNX / TensorFlow / TFLite 到 `.rknn` 的转换，支持 INT8 / FP16 量化。

对于 LLM，瑞芯微推出了 **RKNN-LLM**，专门面向 RK3588 / RK3576 / RK3578 等平台的大语言模型部署。

```python
# RKNN 转换示例（Python API）
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform='rk3588', optimization_level=3)
rknn.load_onnx(model='model.onnx')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('model.rknn')

# RKNN-LLM 大模型转换（命令行）
rkllm-build --model_path ./Llama-2-7b-hf \
            --target_platform rk3588 \
            --quantization int8 \
            --output ./llama2_7b.rkllm
```

**适用场景**：低成本边缘盒子、智能摄像头、NVR、教育平板、小型机器人。

### 16.5.5 MLC-LLM

**MLC-LLM** 是 CMU / OctoML 等社区推动的端侧 LLM 编译部署框架，基于 Apache TVM。它的核心思想是**把 LLM 编译成针对目标硬件的高效 runtime**，支持 CUDA、Metal、Vulkan、OpenCL、ROCm 以及多种 NPU。

```bash
# 使用 mlc_llm 转换并编译模型
mlc_llm convert_build meta-llama/Llama-2-7b-hf --quantization q4f16_1 -o dist

# 启动服务
mlc_llm serve dist/Llama-2-7b-hf-q4f16_1-MLC --port 8080
```

**适用场景**：需要把 LLM 编译到手机 GPU / NPU 上，追求极致 kernel 优化；研究属性强。

### 16.5.6 ExecuTorch

**ExecuTorch** 是 Meta 推出的 PyTorch 端侧推理解决方案，目标是让 PyTorch 模型能够直接部署到手机、AR/VR、IoT 设备。它通过 `torch.export` 捕获计算图，再经过 lowering、量化、编译，生成在目标设备上运行的 `.pte` 文件。

```bash
# 导出并编译模型
python -m examples.portable.scripts.export --model_name=llama2

# 运行
./cmake-out/executor_runner --model_path llama2.pte
```

**适用场景**：已经在 PyTorch 生态中的移动端/可穿戴设备模型；希望用同一套 PyTorch 工具链完成训练到部署。

### 16.5.7 TensorFlow Lite / LiteRT

**TensorFlow Lite（TFLite）** 是 Google 的端侧推理框架，社区和生态最为广泛。2024 年起 Google 将 TFLite 改名为 **LiteRT**，但文件格式和 API 基本保持一致。LiteRT 支持 ARM CPU、GPU delegate、Edge TPU、Hexagon DSP、Apple Neural Engine 等。

```bash
# 转换 SavedModel 为 TFLite
python convert_tflite.py \
  --saved_model_dir ./model \
  --output_file model.tflite \
  --optimization DEFAULT \
  --representative_dataset dataset.py

# 使用 delegate 在 GPU/NPU 上运行
interpreter = tf.lite.Interpreter(
    model_path="model.tflite",
    experimental_delegates=[tf.lite.experimental.load_delegate('libgpu_delegate.so')]
)
```

**适用场景**：移动端 CV、语音、推荐模型；需要最广泛硬件支持和社区资源的项目。

---

## 16.6 框架选型决策树

### 16.6.1 核心权衡维度

边缘框架选型需要在四个维度上做权衡：

| 维度 | 问题 | 影响 |
|------|------|------|
| **硬件锁定** | 是否必须使用 NVIDIA / Qualcomm / Intel / 平头哥等特定平台？ | 决定候选框架范围 |
| **延迟 vs 吞吐** | 场景是单用户低延迟，还是多路并发高吞吐？ | 影响 batch 策略、KV Cache 大小、是否启用 continuous batching |
| **模型类型** | CNN / Transformer / LLM / 多模态？ | 不同框架对不同结构的优化程度差异巨大 |
| **工程成熟度** | 是否有现成工具链、示例、社区支持？ | 直接影响落地周期和长期维护成本 |

### 16.6.2 决策树

```
开始
  │
  ├── 目标硬件是 NVIDIA Jetson / GPU / DLA 吗？
  │     ├── 是 → 是否愿意接受闭源 runtime 并追求极致延迟？
  │     │           ├── 是 → TensorRT-LLM（或 TensorRT for CNN/Vision）
  │     │           └── 否 → ONNX Runtime + TensorRT EP（保留一定可移植性）
  │     └── 否 → 继续
  │
  ├── 目标硬件是 Qualcomm Snapdragon / Hexagon NPU 吗？
  │     ├── 是 → QNN（或 ONNX Runtime + QNN EP）
  │     └── 否 → 继续
  │
  ├── 目标硬件是 Intel CPU / GPU / NPU 吗？
  │     ├── 是 → OpenVINO / OpenVINO GenAI
  │     └── 否 → 继续
  │
  ├── 目标硬件是 Rockchip RK3588 / RK3576 吗？
  │     ├── 是 → RKNN / RKNN-LLM
  │     └── 否 → 继续
  │
  ├── 目标硬件是平头哥玄铁 RISC-V / 阿里 AIoT 生态吗？
  │     ├── 是 → MNN
  │     └── 否 → 继续
  │
  ├── 需要部署端侧 LLM，且希望跨平台、社区活跃？
  │     ├── 是 → llama.cpp（CPU/GPU 通用） / MLC-LLM（编译到 GPU/NPU）
  │     └── 否 → 继续
  │
  ├── 模型来自 PyTorch，目标是移动端/可穿戴设备？
  │     ├── 是 → ExecuTorch
  │     └── 否 → 继续
  │
  └── 默认选择 → ONNX Runtime / TensorFlow Lite（生态最广、文档最丰富）
```

### 16.6.3 典型场景推荐

| 场景 | 推荐框架 | 理由 |
|------|---------|------|
| **Jetson 机器人实时 LLM** | TensorRT-LLM | kernel 级优化、DLA 卸载、延迟最低 |
| **旗舰手机本地 LLM** | QNN / MLC-LLM / llama.cpp | 分别对应高通/跨平台/通用路径 |
| **AI PC（Intel）本地助手** | OpenVINO GenAI | NPU 低功耗、工具链成熟 |
| **低成本摄像头/门禁** | RKNN | 硬件成本低、工具链完善 |
| **平头哥玄铁 AIoT** | MNN | 阿里生态、RISC-V 后端支持 |
| **跨硬件快速原型** | ONNX Runtime GenAI | 一次 ONNX，多 EP 切换 |
| **已有 PyTorch 移动端模型** | ExecuTorch / LiteRT | 与训练生态无缝衔接 |

### 16.6.4 常见误区

1. **误区一：认为开源框架一定比厂商 SDK 好**。厂商 SDK 通常对自家 NPU 的算子支持和内存排布做了深度优化，开源框架反而可能因为 fallback 导致性能大跌。
2. **误区二：只看模型转换成功，不看 runtime 性能**。很多框架能转换模型，但实际运行时大量算子落在 CPU，导致延迟不可接受。
3. **误区三：忽视量化校准**。边缘量化不是简单把 FP32 转成 INT8，而是需要代表性数据集做校准，否则精度会崩。
4. **误区四：把云侧的 continuous batching 直接搬到边缘**。边缘内存和功耗有限，通常需要静态 batch 或很小的动态 batch。
5. **误区五：低估 SDK 版本耦合**。边缘项目必须把芯片、BSP、SDK、模型转换工具链的版本矩阵写入文档并锁定。

---

## 16.7 小结与面试高频考点

### 16.7.1 关键结论

- **边缘框架的核心矛盾是兼容性 vs 性能**：跨平台框架易用但可能无法发挥 NPU 峰值性能；厂商 SDK 性能高但锁定生态。
- **TensorRT-LLM 的 engine 路径仍是 NVIDIA 边缘的最优解**，Builder / Executor / KVCacheManager / GPTAttentionPlugin 是其四大核心。
- **llama.cpp 是端侧 LLM 的事实标准**，GGUF 格式、量化工具链、KV Cache 管理、多后端调度构成了其护城河。
- **MNN 在阿里/平头哥生态中有独特价值**，Converter / Interpreter / Backend 三层架构与 RISC-V/NPU 后端值得关注。
- **框架选型必须先定硬件**，再考虑量化、算子支持、工具链成熟度，最后才是延迟/吞吐优化。

### 16.7.2 面试高频考点 ★

1. **TensorRT-LLM 的 build 流程是什么？**  
   答：`commands/build.py` 解析参数 → `builder.py` 创建 `Network` 和 `BuilderConfig` → 调用 `Builder.build_engine` 生成序列化 engine → `Executor` 加载运行。

2. **TensorRT-LLM 的 KV Cache 如何管理？**  
   答：`KVCacheBlock` + `BlockManager` + `KVCacheManager`，页式管理，支持 prefix caching；调度器在 `capacityScheduler.cpp` 中，chunked prefill 在 `microBatchScheduler.cpp` 中。

3. **llama.cpp 的 GGUF 格式有什么优势？**  
   答：自包含元数据与张量、支持按张量选择量化类型、版本可控（当前 V3）、mmap 友好、社区生态广。

4. **llama.cpp 的 KV Cache 如何支持 SWA？**  
   答：`llama_kv_cache_iswa` 维护两个 `llama_kv_cache` 实例，一个给普通层，一个给 SWA 层，对外统一实现 `llama_memory_i` 接口。

5. **llama.cpp 如何实现跨平台？**  
   答：通过 `ggml_backend` 抽象层，`ggml-backend-reg.cpp` 在编译期注册 CUDA / Metal / Vulkan / OpenCL / CANN / CPU 等后端，运行时按算子支持自动分配。

6. **MNN 的三层架构是什么？**  
   答：Converter 负责图优化与模型转换；Interpreter 负责会话与调度；Backend 负责具体硬件执行。

7. **边缘部署 LLM 为什么要用 INT4/INT8 量化？**  
   答：边缘共享内存容量有限（4–64 GB），FP16 7B 模型权重约 14 GB，加上 KV Cache 和激活很容易 OOM；量化可把权重压缩 2–4 倍。

8. **如何为高通平台选择框架？**  
   答：优先 QNN，次选 ONNX Runtime + QNN EP；复杂 custom op 需要评估 CPU fallback 成本。

### 16.7.3 延伸阅读建议

- 本书 Part 15：复习边缘硬件约束与选型矩阵。
- 本书 Part 17：学习边缘 KV Cache 压缩、模型轻量化、VLM 部署等实战优化。
- TensorRT-LLM 官方文档：`docs/source/deployment-guide/`、`docs/source/features/kvcache.md`。
- llama.cpp 官方文档：`docs/backend.md`、`docs/quantization.md`、`gguf-py/README.md`。
- MNN 官方仓库：https://github.com/alibaba/MNN

---

> **一句话总结**：边缘推理框架没有银弹。对 NVIDIA 平台要读懂 TensorRT-LLM 的 Builder / Executor / KVCacheManager / Plugin 链条；对端侧 LLM 要掌握 llama.cpp 的 GGUF / 量化 / KV Cache / 多后端；对国产/阿里生态要熟悉 MNN 的三层架构。最终选型永远从硬件白名单和工具链成熟度出发，而不是从框架热度出发。
