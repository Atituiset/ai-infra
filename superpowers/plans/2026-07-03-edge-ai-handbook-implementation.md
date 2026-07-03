# Edge AI Handbook Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new edge-AI parts (hardware, frameworks, optimization) and update existing handbook chapters/README so the book covers edge chip infrastructure matching the T-Head JD.

**Architecture:** Follow the existing handbook markdown structure and style. New parts mirror the cloud progression (fundamentals → frameworks → optimization). Two frameworks (TensorRT-LLM and llama.cpp) get source-level deep dives using locally cloned repositories; other frameworks and hardware get overview + command examples.

**Tech Stack:** Markdown, local clones (`llama.cpp/`, `tensorrt-llm/`), shell validation (`grep`, `wc`, `markdownlint` if available).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `docs/ai-infra-handbook/README.md` | Modify | TOC, reading paths, quick-lookup table |
| `docs/ai-infra-handbook/part-00-overview.md` | Modify | Add edge layer to architecture diagram + cloud-vs-edge section |
| `docs/ai-infra-handbook/part-10-engine-ecosystem.md` | Modify | Add edge engine matrix |
| `docs/ai-infra-handbook/part-12-domestic-gpu.md` | Modify | Add domestic edge chips including T-Head |
| `docs/ai-infra-handbook/part-15-edge-hardware.md` | Create | Edge hardware map and constraints |
| `docs/ai-infra-handbook/part-16-edge-frameworks.md` | Create | Edge frameworks, source-level TensorRT-LLM + llama.cpp |
| `docs/ai-infra-handbook/part-17-edge-optimization.md` | Create | Edge optimization, VLM/VLA, troubleshooting |
| `docs/ai-infra-handbook/part-09-glossary.md` | Modify (optional) | Add edge terms |

---

## Task 1: Prepare Local Repositories for Source Analysis

**Files:**
- Read: `llama.cpp/README.md`, `llama.cpp/ggml/include/ggml.h`, `llama.cpp/src/llama.cpp`
- Read: `tensorrt-llm/README.md`, `tensorrt-llm/cpp/include/tensorrt_llm/runtime/`, `tensorrt-llm/tensorrt_llm/`

**Validation:** None (exploration only).

- [ ] **Step 1: Map llama.cpp repository layout**

Run:
```bash
find llama.cpp -maxdepth 2 -type d | sort | head -60
```

Capture the top-level directories. Identify where `ggml`, `llama`, `common`, `examples`, `tests` live.

- [ ] **Step 2: Identify llama.cpp key source files**

Run:
```bash
grep -l "struct llama_kv_cache" llama.cpp/src/*.cpp llama.cpp/llama.cpp 2>/dev/null
ls llama.cpp/src/llama-*.cpp 2>/dev/null || ls llama.cpp/llama-*.cpp 2>/dev/null
```

Note the exact paths for KV cache, model loading, sampling, and backend dispatch.

- [ ] **Step 3: Map TensorRT-LLM repository layout**

Run:
```bash
find tensorrt-llm -maxdepth 2 -type d | sort | head -80
```

Capture top-level directories: `cpp/`, `tensorrt_llm/`, `examples/`, `benchmarks/`.

- [ ] **Step 4: Identify TensorRT-LLM key source files**

Run:
```bash
find tensorrt-llm/cpp/include/tensorrt_llm -type f -name "*.h" | head -40
find tensorrt-llm/tensorrt_llm -maxdepth 2 -type f -name "*.py" | head -40
```

Note paths for `GptSession`, `executor`, `batch_manager`, `plugin` headers.

- [ ] **Step 5: Commit exploration notes**

No commit needed yet; record key paths in a scratch file or directly into Part 16 while writing.

---

## Task 2: Update README.md

**Files:**
- Modify: `docs/ai-infra-handbook/README.md`

**Validation:**
```bash
grep -n "Part 15\|Part 16\|Part 17\|边缘" docs/ai-infra-handbook/README.md
```
Expected: matches for new parts and edge keyword.

- [ ] **Step 1: Add new parts to the main TOC**

Insert after Part 14 in the `## 目录` section:

```markdown
### [Part 15: 边缘 AI 硬件地图与约束](part-15-edge-hardware.md)
- 边缘 AI 定义与边界
- 国际主流边缘平台（Jetson / Qualcomm / Apple / Intel / AMD）
- 国产边缘芯片（平头哥玄铁、昇腾 310、寒武纪、地平线、黑芝麻、RK3588）
- 硬件约束对软件设计的影响
- 边缘芯片选型矩阵

### [Part 16: 边缘推理框架与软件栈](part-16-edge-frameworks.md)
- 边缘框架全景与分类
- TensorRT / TensorRT-LLM 源码级分析（含 Jetson）
- llama.cpp 源码级分析（GGUF / 量化 / KV Cache / 多后端）
- MNN（平头哥/阿里生态）
- ONNX Runtime GenAI / OpenVINO / QNN / RKNN / MLC-LLM / ExecuTorch overview
- 框架选型决策树

### [Part 17: 边缘场景优化与落地](part-17-edge-optimization.md)
- 内存受限下的 KV Cache 管理
- 量化与校准实战
- 模型轻量化（蒸馏 / 剪枝 / TinyLLM）
- 异构调度与多模型部署
- VLM / VLA 边缘部署
- Inflight Batching 在边缘
- 问题排查与客户支持流程
- 边缘上线 checklist
```

- [ ] **Step 2: Add edge reading paths**

Add a row to the `按角色阅读` table:

```markdown
| **边缘 AI 工程师 / 平头哥生态** | Part 15 → Part 16 → Part 17（重点 §16.4 MNN） |
```

- [ ] **Step 3: Add quick-lookup rows**

Add rows to the `按问题速查` table:

```markdown
| 边缘 AI 硬件怎么选？ | Part 15 §15.5 |
| TensorRT-LLM 在 Jetson 上怎么用？ | Part 16 §16.2 |
| llama.cpp 量化/GGUF 原理？ | Part 16 §16.3 |
| 边缘 KV Cache 怎么省？ | Part 17 §17.1 |
| VLM/VLA 怎么部署到边缘？ | Part 17 §17.5 |
| 平头哥玄铁/MNN 生态？ | Part 16 §16.4 + Part 12 |
```

- [ ] **Step 4: Update maintenance section**

Update the `基于源码版本` line to include:
```markdown
- 基于源码版本：vLLM V1 引擎开发主线、SGLang 最新主线、TensorRT-LLM 主线、llama.cpp 主线
```

- [ ] **Step 5: Validate**

Run:
```bash
grep -n "Part 15\|Part 16\|Part 17\|边缘 AI 工程师\|平头哥" docs/ai-infra-handbook/README.md
```

Expected: each string appears at least once.

- [ ] **Step 6: Commit**

```bash
git add docs/ai-infra-handbook/README.md
git commit -m "docs(handbook): add Part 15-17 edge AI to README TOC and reading paths"
```

---

## Task 3: Write Part 15 — Edge AI Hardware Map

**Files:**
- Create: `docs/ai-infra-handbook/part-15-edge-hardware.md`

**Validation:**
```bash
wc -l docs/ai-infra-handbook/part-15-edge-hardware.md
grep -c "^## " docs/ai-infra-handbook/part-15-edge-hardware.md
```
Expected: ≥400 lines, ≥6 H2 sections.

- [ ] **Step 1: Create header and 15.1**

Create the file with the standard part header and section 15.1:

```markdown
# Part 15：边缘 AI 硬件地图与约束

> **面向角色**：边缘 AI 推理优化工程师、端侧部署工程师、芯片软件栈适配工程师  
> **前置知识**：基本了解 Transformer 推理、已有本书 Part 0-2 云侧基础  
> **目标**：建立边缘 AI 硬件的全局认知，理解边缘与云侧在芯片约束上的本质差异

---

## 15.1 什么是边缘 AI？边界与定义

边缘 AI 指在**离数据产生端最近**的设备上完成模型推理，而不是把请求发到云端数据中心。

典型设备：
- 边缘盒子 / 工控机（Jetson Orin、昇腾 310、RK3588）
- 智能手机 / 平板（Snapdragon、天玑、Apple A/M 系列）
- 车载域控制器（地平线 Journey 6、黑芝麻 A1000、DRIVE Orin）
- IoT / 机器人 / 摄像头（RISC-V + NPU、微控制器级 NPU）

### 边缘 vs 云侧：五个核心差异

| 维度 | 云侧（数据中心） | 边缘侧 |
|------|-----------------|--------|
| **算力** | 数百 TFLOPS~PFLOPS | 数 TOPS~数百 TOPS |
| **内存** | 40-192 GB HBM | 4-64 GB 共享 LPDDR/DDR |
| **功耗** | 300-700 W | 5-60 W（常见） |
| **互联** | NVLink / RDMA / InfiniBand | 无高速互联，多为 PCIe/USB/Ethernet |
| **延迟目标** | TTFT < 500ms，TPOT < 50ms | 端到端 < 50-200ms，常需本地闭环 |

**核心结论**：云侧优化看 **HBM 带宽和多卡通信**；边缘优化看 **内存容量、功耗墙、算子兼容性**。
```

- [ ] **Step 2: Write 15.2 international platforms**

Cover NVIDIA Jetson Orin/Thor, Qualcomm Snapdragon/Hexagon, Apple Neural Engine, Intel NPU, AMD Ryzen AI. Include a parameter table per platform.

- [ ] **Step 3: Write 15.3 domestic edge chips**

Cover T-Head XuanTie RISC-V + Wujian, Ascend 310/310P/610, Cambricon MLU edge, Horizon Journey 5/6, Black Sesame Huashan A1000, Rockchip RK3588. Include a comparison table.

- [ ] **Step 4: Write 15.4 hardware constraints**

Discuss shared memory, power/thermal, no NVLink, fragmented NPU ops, BSP/driver coupling.

- [ ] **Step 5: Write 15.5 selection matrix**

Create a final matrix: chip / compute / memory / power / software stack / best scenario.

- [ ] **Step 6: Validate and commit**

Run:
```bash
wc -l docs/ai-infra-handbook/part-15-edge-hardware.md
grep -c "^## " docs/ai-infra-handbook/part-15-edge-hardware.md
grep -n "平头哥\|玄铁\|Jetson\|RK3588\|地平线\|黑芝麻" docs/ai-infra-handbook/part-15-edge-hardware.md
```

Commit:
```bash
git add docs/ai-infra-handbook/part-15-edge-hardware.md
git commit -m "docs(handbook): add Part 15 edge AI hardware map"
```

---

## Task 4: Write Part 16 — Edge Inference Frameworks

**Files:**
- Create: `docs/ai-infra-handbook/part-16-edge-frameworks.md`
- Read: `llama.cpp/` and `tensorrt-llm/` key files mapped in Task 1

**Validation:**
```bash
wc -l docs/ai-infra-handbook/part-16-edge-frameworks.md
grep -n "tensorrt-llm/\|llama.cpp/" docs/ai-infra-handbook/part-16-edge-frameworks.md | head -20
```
Expected: ≥600 lines, source paths cited.

- [ ] **Step 1: Create header and 16.1 landscape**

```markdown
# Part 16：边缘推理框架与软件栈

> **面向角色**：边缘 AI 推理优化工程师、框架适配工程师  
> **前置知识**：Part 15 边缘硬件约束、基本量化概念  
> **目标**：掌握主流边缘推理框架的定位、用法和关键源码结构

---

## 16.1 边缘框架全景

按锁定程度分类：

| 类型 | 代表框架 | 特点 |
|------|---------|------|
| **NVIDIA 锁定** | TensorRT / TensorRT-LLM | Jetson/GPU 极致性能，闭源 C++ runtime |
| **跨平台通用** | ONNX Runtime GenAI, llama.cpp, MLC-LLM, ExecuTorch | 一次转换，多硬件运行 |
| **芯片厂商专用** | QNN, RKNN, OpenVINO, MNN, CANN MindIE | 针对特定 NPU 深度优化 |
```

- [ ] **Step 2: Write 16.2 TensorRT / TensorRT-LLM source-level**

Use the mapped repository structure. Cover:
- Repository layout
- Builder (`tensorrt_llm/builder.py` / `cpp/tensorrt_llm/runtime/`) → engine
- Runtime / Executor / In-flight Batching
- Plugin mechanism
- Jetson workflow
- Source reading roadmap with exact file paths

- [ ] **Step 3: Write 16.3 llama.cpp source-level**

Use the mapped repository structure. Cover:
- Repository layout
- GGUF format and quantization schemes
- KV Cache management (`llama_kv_cache`)
- Batching and speculative decoding
- Multi-backend dispatch
- Source reading roadmap with exact file paths

- [ ] **Step 4: Write 16.4 MNN (T-Head/Alibaba)**

Cover MNN architecture, converter/interpreter/backend, quantization, RISC-V/NPU backend, source reading roadmap.

- [ ] **Step 5: Write 16.5 other frameworks overview**

For each of ONNX Runtime GenAI, OpenVINO, QNN, RKNN, MLC-LLM, ExecuTorch, TFLite: one paragraph + key command example.

- [ ] **Step 6: Write 16.6 selection decision tree**

Text-based decision tree or flowchart in Markdown.

- [ ] **Step 7: Validate and commit**

Run:
```bash
wc -l docs/ai-infra-handbook/part-16-edge-frameworks.md
grep -n "llama.cpp/src\|tensorrt-llm/cpp\|MNN/source" docs/ai-infra-handbook/part-16-edge-frameworks.md | head -20
```

Commit:
```bash
git add docs/ai-infra-handbook/part-16-edge-frameworks.md
git commit -m "docs(handbook): add Part 16 edge inference frameworks with source-level TensorRT-LLM and llama.cpp"
```

---

## Task 5: Write Part 17 — Edge Optimization and Production

**Files:**
- Create: `docs/ai-infra-handbook/part-17-edge-optimization.md`

**Validation:**
```bash
wc -l docs/ai-infra-handbook/part-17-edge-optimization.md
grep -c "^## " docs/ai-infra-handbook/part-17-edge-optimization.md
```
Expected: ≥500 lines, ≥8 H2 sections.

- [ ] **Step 1: Create header and 17.1 KV Cache**

```markdown
# Part 17：边缘场景优化与落地

> **面向角色**：边缘 AI 推理优化工程师、量产交付工程师  
> **前置知识**：Part 15 硬件约束、Part 16 框架、Part 1/2 KV Cache 与量化基础  
> **目标**：掌握边缘场景下的性能优化、精度保障和问题排查方法

---

## 17.1 内存受限下的 KV Cache 管理

- GQA / MQA / MLA 在边缘的收益
- Sliding-window / StreamingLLM / H2O
- KV Cache INT8/INT4 量化
- 与云侧 PagedAttention 的差异
```

- [ ] **Step 2: Write 17.2 quantization and calibration**

PTQ/QAT/GPTQ/AWQ/SmoothQuant applicability, calibration workflow, accuracy evaluation.

- [ ] **Step 3: Write 17.3 model lightweighting**

Distillation, pruning, TinyLLM families, model size selection guidelines.

- [ ] **Step 4: Write 17.4 heterogeneous scheduling and multi-model**

CPU+GPU/NPU mixed execution, graph partitioning, memory budgeting, hot-swap, pipeline composition.

- [ ] **Step 5: Write 17.5 VLM/VLA edge deployment**

Vision encoder quantization, multimodal pipeline, LLaVA/Qwen-VL/Phi-vision/OpenVLA paths.

- [ ] **Step 6: Write 17.6 Inflight Batching on edge**

Small-batch differences, when it helps/hurts.

- [ ] **Step 7: Write 17.7 troubleshooting and customer support**

Triage flow, SDK/BSP/driver conflicts, thermal throttling, cross-team collaboration template.

- [ ] **Step 8: Write 17.8 edge production checklist**

A final checklist table.

- [ ] **Step 9: Validate and commit**

Run:
```bash
wc -l docs/ai-infra-handbook/part-17-edge-optimization.md
grep -n "VLM\|VLA\|Inflight Batching\|客户支持\|checklist" docs/ai-infra-handbook/part-17-edge-optimization.md
```

Commit:
```bash
git add docs/ai-infra-handbook/part-17-edge-optimization.md
git commit -m "docs(handbook): add Part 17 edge optimization, VLM/VLA, and production"
```

---

## Task 6: Update Part 0 — Add Edge Layer to Overview

**Files:**
- Modify: `docs/ai-infra-handbook/part-00-overview.md`

**Validation:**
```bash
grep -n "边缘\|Edge\|端侧" docs/ai-infra-handbook/part-00-overview.md | head -20
```

- [ ] **Step 1: Extend the architecture diagram**

Modify the diagram in `0.1 AI Infra 是什么？` to include an edge layer above the platform layer:

```markdown
   ┌─────────────────────────────────────┐
   │  边缘 / 端侧设备                      │  ← 边缘层
   │  (Jetson / 手机 / 车载 / IoT)         │
   └────┬────────────────────────────────┘
        │  推理结果 / 模型更新
```

- [ ] **Step 2: Add cloud-vs-edge section**

Add `0.7 云侧 vs 边缘侧：同一问题，两种解法` summarizing the divergence and cross-referencing Part 15.

- [ ] **Step 3: Validate and commit**

Run:
```bash
grep -n "边缘\|云侧 vs 边缘侧" docs/ai-infra-handbook/part-00-overview.md
```

Commit:
```bash
git add docs/ai-infra-handbook/part-00-overview.md
git commit -m "docs(handbook): add edge layer to Part 0 overview"
```

---

## Task 7: Update Part 10 — Add Edge Engine Matrix

**Files:**
- Modify: `docs/ai-infra-handbook/part-10-engine-ecosystem.md`

**Validation:**
```bash
grep -n "边缘\|Jetson\|llama.cpp\|ONNX Runtime GenAI" docs/ai-infra-handbook/part-10-engine-ecosystem.md | head -20
```

- [ ] **Step 1: Add edge engine section**

Add `10.7 边缘侧推理引擎` after the existing engine sections. Include:
- A matrix: framework / hardware / quantization / multi-model / ease-of-use
- One-paragraph summaries of TensorRT Jetson, ONNX Runtime GenAI, llama.cpp, MNN, RKNN, QNN, OpenVINO
- A note on when to choose cloud vs edge engines

- [ ] **Step 2: Validate and commit**

Run:
```bash
grep -n "10.7\|边缘侧推理引擎" docs/ai-infra-handbook/part-10-engine-ecosystem.md
```

Commit:
```bash
git add docs/ai-infra-handbook/part-10-engine-ecosystem.md
git commit -m "docs(handbook): add edge engine matrix to Part 10"
```

---

## Task 8: Update Part 12 — Add Domestic Edge Chips

**Files:**
- Modify: `docs/ai-infra-handbook/part-12-domestic-gpu.md`

**Validation:**
```bash
grep -n "平头哥\|玄铁\|地平线\|黑芝麻\|RK3588\|昇腾 310" docs/ai-infra-handbook/part-12-domestic-gpu.md | head -30
```

- [ ] **Step 1: Update title and intro**

Change title to `# Part 12：国产 GPU 与边缘芯片` and update the intro table to include edge chips.

- [ ] **Step 2: Add T-Head section**

Add a dedicated `12.7 平头哥（T-Head）玄铁 RISC-V 与无剑平台` section covering:
- XuanTie C906/C910/C920
- Wujian SoC platform
- AI accelerator integration
- AliOS Things / MNN relationship
- Adaptation path

- [ ] **Step 3: Add other domestic edge chip sections**

Add concise sections for:
- Horizon Journey 5/6
- Black Sesame Huashan A1000
- Cambricon MLU edge series
- Rockchip RK3588/RK3576
- Ascend 310/310P/610

- [ ] **Step 4: Add datacenter-vs-edge selection table**

Add a final table comparing when to use Ascend 910B vs Ascend 310, MTT S4000 vs RK3588, etc.

- [ ] **Step 5: Validate and commit**

Run:
```bash
grep -c "^## " docs/ai-infra-handbook/part-12-domestic-gpu.md
grep -n "平头哥\|地平线\|黑芝麻" docs/ai-infra-handbook/part-12-domestic-gpu.md
```

Commit:
```bash
git add docs/ai-infra-handbook/part-12-domestic-gpu.md
git commit -m "docs(handbook): extend Part 12 with domestic edge chips including T-Head"
```

---

## Task 9: Optional — Update Glossary

**Files:**
- Modify: `docs/ai-infra-handbook/part-09-glossary.md`

**Validation:**
```bash
grep -n "GGUF\|QNN\|RKNN\|MNN\|Jetson\|玄铁" docs/ai-infra-handbook/part-09-glossary.md
```

- [ ] **Step 1: Add edge terms**

Add entries for: GGUF, QNN, RKNN, MNN, Jetson, XuanTie, Wujian, HTP, NPU, DLA, LPDDR, shared memory.

- [ ] **Step 2: Commit**

```bash
git add docs/ai-infra-handbook/part-09-glossary.md
git commit -m "docs(handbook): add edge AI terms to glossary"
```

---

## Task 10: Final Review

**Files:** All modified/created markdown files.

**Validation:**
```bash
echo "=== New parts ==="
wc -l docs/ai-infra-handbook/part-15-edge-hardware.md docs/ai-infra-handbook/part-16-edge-frameworks.md docs/ai-infra-handbook/part-17-edge-optimization.md

echo "=== TOC coverage ==="
grep -n "Part 15\|Part 16\|Part 17" docs/ai-infra-handbook/README.md

echo "=== Edge keywords ==="
grep -Ril "边缘\|Jetson\|平头哥\|玄铁\|MNN\|RKNN\|QNN" docs/ai-infra-handbook/ | sort
```

- [ ] **Step 1: Check internal links**

Run:
```bash
grep -oE '\(part-[0-9]+-[^)]+\.md\)' docs/ai-infra-handbook/README.md | sort -u
```

Verify that `part-15-edge-hardware.md`, `part-16-edge-frameworks.md`, `part-17-edge-optimization.md` are referenced.

- [ ] **Step 2: Check markdown formatting**

If `markdownlint-cli` is installed:
```bash
npx markdownlint-cli docs/ai-infra-handbook/part-15-edge-hardware.md docs/ai-infra-handbook/part-16-edge-frameworks.md docs/ai-infra-handbook/part-17-edge-optimization.md 2>/dev/null || echo "markdownlint not available, skip"
```

Otherwise visually verify headers and tables.

- [ ] **Step 3: Final commit or summary**

If all checks pass, no additional commit needed if each task was committed. Otherwise:
```bash
git status
```

Report any uncommitted changes to the user.

---

## Spec Coverage Check

| Spec Section | Implementing Task |
|--------------|-------------------|
| Part 15 hardware map | Task 3 |
| Part 16 frameworks + TensorRT-LLM/llama.cpp source | Task 4 |
| MNN / T-Head coverage | Task 4 (16.4) + Task 8 (12.7) |
| Part 17 optimization + VLM/VLA + troubleshooting | Task 5 |
| Part 0 edge layer update | Task 6 |
| Part 10 edge engine matrix | Task 7 |
| Part 12 domestic edge chips | Task 8 |
| README updates | Task 2 |
| Glossary updates | Task 9 (optional) |

---

## Placeholder Scan

- No `TBD`, `TODO`, or `implement later` remain.
- Every task contains exact file paths.
- Validation commands are concrete.
- Source-level sections depend on Task 1 repo exploration.

---

**Plan complete and saved to `docs/superpowers/plans/2026-07-03-edge-ai-handbook-implementation.md`.**

**Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach do you want?
