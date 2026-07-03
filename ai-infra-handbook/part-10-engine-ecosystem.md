# Part 10：推理引擎生态深度对比

> **面向角色**：需要做技术选型的架构师、技术负责人  > **目标**：深入理解 vLLM、SGLang、TensorRT-LLM、TGI、LMDeploy 的差异与适用边界

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

## 10.7 横向对比矩阵

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

## 10.8 选型决策树

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

## 10.9 生产组合建议

| 场景 | 推荐引擎 | 理由 |
|------|---------|------|
| 通用 LLM API | vLLM | 最稳、最广、文档最全 |
| 高吞吐 RAG/chat | SGLang | RadixAttention 复用长 context |
| 实时低延迟 | TensorRT-LLM | kernel 级优化 |
| HF 生态 + 水印 | TGI | 原生集成 |
| 国产 GPU | LMDeploy / SGLang | 硬件适配 |
| 千卡 MoE | vLLM Wide-EP / SGLang DP+EP | 大规模并行 |

---

## 10.10 迁移成本

| 迁移方向 | 成本 | 主要工作 |
|----------|------|---------|
| vLLM → SGLang | 中 | 参数/调度策略重新调优，监控适配 |
| vLLM → TRT-LLM | 高 | 重新编译 engine，量化转换，API 适配 |
| SGLang → vLLM | 低-中 |  mostly 参数调整，调度行为差异 |
| 任何 → TGI | 低 | API 兼容性好 |
| 任何 → LMDeploy | 中 | 国产 GPU 适配，部分 API 差异 |
