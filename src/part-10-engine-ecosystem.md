# Part 10：推理引擎生态深度对比

> **面向角色**：需要做技术选型的架构师、技术负责人  
> **目标**：深入理解 vLLM、SGLang、TensorRT-LLM、TGI、LMDeploy 的差异与适用边界

---

## 10.1 生态全景

```
开源推理引擎谱系:

┌─────────────────────────────────────────────────────────────┐
│  HuggingFace Transformers (最基础，慢，但兼容性最好)           │
│       ↓                                                     │
│  TGI (Text Generation Inference) — Rust + Python             │
│       ↓                                                     │
│  vLLM — 社区最大，PagedAttention，通用推理                    │
│       ↓                                                     │
│  SGLang — RadixAttention，高吞吐，编译优化                     │
│       ↓                                                     │
│  TensorRT-LLM — NVIDIA 官方，闭源 C++，极致优化               │
└─────────────────────────────────────────────────────────────┘

国产/特定硬件:
  - LMDeploy (TurboMind) — 商汤，国产 GPU 友好
  - MindSpore Lite / MindIE — 华为昇腾
  - PPL-LLM / LightLLM — 其他中文社区方案
```

---

## 10.2 vLLM

### 定位
**通用型开源推理引擎**，社区最大、模型支持最广、生态最成熟。

### 核心优势
- **模型支持最全**：几乎支持所有 HF 模型，新模型落地最快
- **PagedAttention**：KV Cache 页式管理先驱
- **V1 Engine**：新架构支持 DP、PD 分离、EEP 等高级特性
- **易用性高**：pip install 即可，OpenAI API 兼容
- **社区活跃**：bug fix 快，issue/PR 多

### 核心劣势
- **性能不是最优**：相比 TRT-LLM/SGLang，纯延迟/吞吐不是顶级
- **Python overhead**：即使 V1 也难以完全消除
- **内存占用**：PagedAttention 的 block 粒度有内部碎片
- **大集群扩展**：DP/EEP 还在快速演进中

### 最佳场景
- 快速验证新模型
- 通用推理服务（不追求极致性能）
- 需要灵活二次开发
- 多模态/长文本（社区支持最快）

### 关键参数示例
```bash
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --max-num-seqs 256 \
  --max-model-len 32768 \
  --quantization fp8 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8
```

---

## 10.3 SGLang

### 定位
**高性能结构化推理引擎**，在吞吐和长 prompt 复用场景表现突出。

### 核心优势
- **RadixAttention**：前缀树式 KV Cache，复用效率高于 vLLM
- **Overlap 调度**：CPU/GPU 并行，降低调度开销
- **torch.compile 深度优化**：减少 Python overhead
- **PD 分离 + HiCache 完整实现**：Mooncake、NIXL、3FS 集成
- **多模态/长文本优化**：Vision/RoPE/长上下文专项优化

### 核心劣势
- **模型支持略少**：新模型落地速度不如 vLLM
- **学习曲线陡峭**：代码量大（scheduler.py 4300+ 行），模块复杂
- **生态工具链**：监控/调试工具不如 vLLM 成熟
- **稳定性**：快速迭代中，部分高级特性（HiCache）仍在开发

### 最佳场景
- 高吞吐服务（chat API、RAG）
- 长 system prompt 场景（大量前缀复用）
- 需要 PD 分离/HiCache 的大规模部署
- 对延迟不极端敏感、追求成本效率

### 关键参数示例
```bash
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --dp 4 \
  --max-running-requests 256 \
  --enable-torch-compile \
  --radix-cache \
  --enable-hicache \
  --disaggregation-mode prefill  # P 节点
```

---

## 10.4 TensorRT-LLM

### 定位
**NVIDIA 官方闭源推理引擎**，追求极致性能，绑定 NVIDIA 生态。

### 核心优势
- **极致性能**：kernel 级优化，通常延迟/吞吐优于开源方案 10-30%
- **FP8/FP4 原生支持**：与 Blackwell/Hopper 深度集成
- **CUDA Graph / In-flight Batching**：成熟高效
- **多模态优化**：对 NVIDIA 视觉模型优化好
- **生产稳定性**：大厂背书，bug 相对较少

### 核心劣势
- **闭源**：无法修改核心逻辑，遇到 bug 只能等官方
- **NVIDIA 锁定**：不支持 AMD/国产 GPU
- **模型支持滞后**：新模型适配慢于 vLLM/SGLang
- **部署复杂**：需要编译 model engine，版本兼容性严格
- **生态封闭**：与 vLLM/SGLang 的模型格式不互通

### 最佳场景
- 延迟敏感型生产服务（如实时对话）
- 已锁定 NVIDIA 硬件
- 有充足时间做模型编译和调优
- 对稳定性要求高于灵活性

### 典型工作流
```bash
# 1. 转换/编译模型
trtllm-build \
  --checkpoint_dir ./deepseek-v3-fp8 \
  --output_dir ./deepseek-v3-engine \
  --dtype float16 \
  --tp_size 8 \
  --pp_size 2

# 2. 启动服务
trtllm-serve ./deepseek-v3-engine
```

---

## 10.5 TGI (Text Generation Inference)

### 定位
**HuggingFace 生态的推理服务**，强调 HF 兼容性和特色功能。

### 核心优势
- **HF 原生集成**：与 transformers/datasets 无缝衔接
- **特色功能**：
  - 内置 watermarking（文本水印）
  - 详细 generation 参数控制
  - 较好的 LoRA adapter 支持
- **Rust runtime**：部分性能优于纯 Python
- **容器化友好**：官方 Docker 镜像成熟

### 核心劣势
- **性能一般**：不如 vLLM/SGLang/TRT-LLM
- **生态规模小**：社区贡献和模型支持少于 vLLM
- **功能演进慢**：在 PagedAttention/投机解码上跟进较慢

### 最佳场景
- HuggingFace 重度用户
- 需要 watermarking 等特色功能
- 快速原型验证
- 中小型部署（非千卡集群）

---

## 10.6 LMDeploy

### 定位
**国产推理引擎**，TurboMind C++ runtime，强调低延迟和国产硬件兼容。

### 核心优势
- **TurboMind C++ 引擎**：低延迟，高吞吐
- **国产 GPU 友好**：华为昇腾、摩尔线程、海光 DCU 支持较好
- **量化支持丰富**：AWQ/GPTQ/FP8 均有优化
- **部署简单**：PyTorch 风格 API

### 核心劣势
- **国际生态弱**：海外模型/社区支持不如 vLLM
- **高级特性**：PD 分离/HiCache 等前沿特性跟进较慢
- **文档/社区**：中文为主，英文资料较少

### 最佳场景
- 国产 GPU 部署
- 国内合规环境
- 需要低延迟 C++ runtime
- 与 OpenMMLab 生态结合

---

## 10.7 边缘侧推理引擎 ★

边缘推理与云侧推理的约束差异巨大：内存以 MB/GB 计、功耗受限、工具链与芯片 BSP 强耦合、算子支持决定模型能否落地。本节把主流边缘框架按绑定关系分类，并给出与云侧引擎的选型边界。详细源码级分析见 **Part 16**。

### 边缘框架分类

| 类别 | 代表框架 | 硬件绑定 | 开放程度 | 典型场景 |
|------|---------|---------|---------|---------|
| **NVIDIA 锁定生态** | TensorRT、TensorRT-LLM | NVIDIA GPU / Jetson / DLA | 半开放（runtime 闭源） | 机器人、工业视觉、高端边缘盒子 |
| **跨平台通用框架** | llama.cpp、ONNX Runtime GenAI、OpenVINO、MNN | CPU / GPU / 部分 NPU | 开源 | 手机、PC、AIoT、快速原型 |
| **芯片厂商专用 SDK** | Qualcomm QNN、Rockchip RKNN / RKNN-LLM | 各自 NPU | 私有工具链为主 | 手机、车载、低成本摄像头、国产替代 |

### 边缘引擎选型矩阵

| 框架 | 主要硬件 | 量化支持 | 多模型并发 | 易用性 | 最佳场景 |
|------|---------|---------|-----------|--------|---------|
| **TensorRT / TensorRT-LLM (Jetson)** | NVIDIA GPU / Jetson / DLA | FP8/FP16/INT8/INT4/W4A16 | 中（显存决定） | 中 | 高端边缘、延迟敏感、机器人 |
| **ONNX Runtime GenAI** | CPU / GPU / NPU（EP 扩展） | INT8/FP16/INT4 | 中 | 高 | 跨平台 LLM、Windows Copilot+ PC |
| **llama.cpp** | CPU / CUDA / Metal / Vulkan / OpenCL / 国产后端 | GGML 量化（Q4/Q5/Q8/IQ） | 低-中 | 高 | 端侧 LLM、快速部署、嵌入式 |
| **MNN** | ARM / x86 / RISC-V / NPU / GPU | INT8/FP16/混合精度 | 中 | 中 | AIoT、平头哥玄铁、移动端 |
| **RKNN / RKNN-LLM** | Rockchip NPU | INT8/FP16 / W8A8 | 低 | 中 | 低成本边缘盒子、摄像头、工控 |
| **Qualcomm QNN** | Hexagon NPU / Snapdragon | INT8/FP16 | 中 | 中 | 手机、车载、XR、低功耗 LLM |
| **OpenVINO / OpenVINO GenAI** | Intel CPU / GPU / NPU | INT8/FP16/NNCF | 中 | 高 | Intel AIPC、工业质检、边缘服务器 |

### 主流框架一句话定位

**TensorRT / TensorRT-LLM on Jetson**：NVIDIA 官方闭源方案，把 HuggingFace 模型编译成 TensorRT engine，在 Jetson Orin/Thor 上延迟最低；代价是编译流程重、版本与 CUDA/Drive 强耦合、新模型适配慢。适合预算充足、已锁定 NVIDIA 的高端边缘场景。

**ONNX Runtime GenAI**：微软推出的跨平台 LLM runtime，基于 ONNX Runtime 的 Execution Provider 机制，可在 DirectML/CUDA/CPU 甚至 Qualcomm NPU 上运行；模型需先导出为 ONNX GenAI 格式，适合需要一份模型跑多平台的 Windows/边缘场景。

**llama.cpp**：社区最活跃的端侧 LLM 推理框架，纯 C/C++ 实现，支持 GGUF 格式和极丰富的量化方案（Q4_K_M、IQ4_XS 等）。部署简单到一条命令，是 PC、树莓派、嵌入式设备跑 7B/13B 模型的首选原型工具。

**MNN**：阿里开源的轻量推理引擎，针对 ARM/RISC-V/NPU 做了大量图优化和内存优化，在 AIoT 和移动端 CV/NLP 任务中成熟稳定；LLM 支持通过 MNN-LLM 扩展，社区以中文为主。

**RKNN / RKNN-LLM**：瑞芯微为其 NPU 提供的私有工具链，将 ONNX/PyTorch 模型转换为 RKNN 格式。RKNN-LLM 针对大模型做了 W8A8/INT8 适配，典型落地是 RV1126/RK3588 等低成本盒子，工具链版本与芯片 BSP 强绑定。

**Qualcomm QNN**：高通 Hexagon NPU 的推理 SDK，面向手机、车载、XR 设备。模型需转换为 DLC 格式，HMX/HVX 指令集优化可带来极高能效比；缺点是生态封闭、调试工具链重、不同 Snapdragon 平台兼容性差异大。

**OpenVINO / OpenVINO GenAI**：Intel 官方推理工具包，OpenVINO GenAI 是面向 LLM 的高阶 API。在 Intel CPU/iGPU/NPU（Meteor Lake/Arrow Lake）上优化到位，NNCF 量化工具链成熟；适合 AIPC 和已有 Intel 边缘服务器的场景。

### 云侧引擎 vs 边缘引擎

| 维度 | 云侧（vLLM / SGLang / TRT-LLM datacenter） | 边缘（本节框架） |
|------|------------------------------------------|----------------|
| **目标硬件** | 多卡 GPU / 高功耗服务器 | 单卡/无卡 / 低功耗设备 |
| **优化目标** | 高吞吐、Continuous Batching、PD 分离 | 低延迟单 batch、低功耗、静态形状 |
| **模型格式** | HuggingFace / Safetensors / engine | TensorRT engine / RKNN / QNN DLC / GGUF / OV IR |
| **部署形态** | 服务化 API、容器、K8s | SDK + 模型 + 本地运行时 |
| **选择时机** | 有稳定网络、算力充足、需要服务多用户 | 离线、隐私敏感、网络/功耗/成本受限 |

**关键原则**：边缘选型的第一约束不是性能，而是**目标平台是否支持、模型能否跑通、工具链是否成熟**。一个能在目标 NPU 上跑通的私有 SDK，往往比一个开源但算子缺失的框架更有工程价值。

---

## 10.8 横向对比矩阵

| 维度 | vLLM | SGLang | TensorRT-LLM | TGI | LMDeploy |
|------|------|--------|--------------|-----|----------|
| **延迟** | 中 | 中-低 | **最低** | 中 | 低 |
| **吞吐** | 高 | **很高** | 高 | 中 | 高 |
| **易用性** | **高** | 中 | 低 | 高 | 中 |
| **模型支持** | **最广** | 较广 | 窄 | 广 | 中等 |
| **灵活性** | **高** | 中 | 低 | 中 | 中 |
| **硬件锁定** | 无 | 无 | **NVIDIA 锁定** | 无 | 国产友好 |
| **社区活跃度** | **最高** | 高 | 中（官方主导）| 中 | 中（中文）|
| **前沿特性** | 快 | **最快** | 慢 | 慢 | 中等 |
| **生产稳定性** | 中 | 中-高 | **高** | 高 | 中 |
| **适合规模** | 全规模 | 中大型 | 中大型 | 中小 | 中大型 |

---

## 10.9 选型决策树

```
开始
  │
  ├── 是否必须使用 NVIDIA 之外硬件？
  │     ├── 是 → 国产 GPU → LMDeploy / SGLang
  │     └── 否 → 继续
  │
  ├── 是否追求极致延迟且愿意闭源锁定？
  │     ├── 是 → TensorRT-LLM
  │     └── 否 → 继续
  │
  ├── 是否 HuggingFace 生态重度用户且需要 watermarking？
  │     ├── 是 → TGI
  │     └── 否 → 继续
  │
  ├── 是否高吞吐、长 prompt、需要 PD 分离/HiCache？
  │     ├── 是 → SGLang
  │     └── 否 → 继续
  │
  └── 默认选择 → vLLM（通用、社区最大、最稳）
```

---

## 10.10 生产组合建议

| 场景 | 推荐引擎 | 理由 |
|------|---------|------|
| 通用 LLM API | vLLM | 最稳、最广、文档最全 |
| 高吞吐 RAG/chat | SGLang | RadixAttention 复用长 context |
| 实时低延迟 | TensorRT-LLM | kernel 级优化 |
| HF 生态 + 水印 | TGI | 原生集成 |
| 国产 GPU | LMDeploy / SGLang | 硬件适配 |
| 千卡 MoE | vLLM Wide-EP / SGLang DP+EP | 大规模并行 |

---

## 10.11 迁移成本

| 迁移方向 | 成本 | 主要工作 |
|----------|------|---------|
| vLLM → SGLang | 中 | 参数/调度策略重新调优，监控适配 |
| vLLM → TRT-LLM | 高 | 重新编译 engine，量化转换，API 适配 |
| SGLang → vLLM | 低-中 |  mostly 参数调整，调度行为差异 |
| 任何 → TGI | 低 | API 兼容性好 |
| 任何 → LMDeploy | 中 | 国产 GPU 适配，部分 API 差异 |
