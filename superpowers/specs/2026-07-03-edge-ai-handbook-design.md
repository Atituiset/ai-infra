# Edge AI Handbook Supplement — Design Spec

**Date**: 2026-07-03  
**Scope**: Add edge-AI-chip infrastructure content to the AI Infra inference handbook and update existing chapters to remove the datacenter-only bias.  
**Target reader**: Edge-AI inference optimization engineer (the role described in the T-Head job description).  
**Authoring model**: Source-level deep dive for TensorRT-LLM and llama.cpp; overview + command examples for other frameworks.

---

## 1. Background & Motivation

The existing handbook (`docs/ai-infra-handbook/`) is excellent for **cloud/datacenter LLM inference** (vLLM, SGLang, H100 clusters, TP/PP/EP, PD disaggregation, HiCache). However, the T-Head JD asks for **edge AI inference optimization**:

- Edge hardware: Jetson Orin/Thor, Qualcomm Hexagon, Apple Neural Engine, Intel/AMD NPUs, and domestic edge chips (T-Head XuanTie, Horizon, Black Sesame, Cambricon, Rockchip, Ascend 310).
- Edge frameworks: TensorRT on Jetson, ONNX Runtime GenAI, QNN, OpenVINO, RKNN, MNN, llama.cpp, MLC-LLM, ExecuTorch.
- Edge-specific techniques: memory-constrained KV Cache, INT4/INT8 calibration, model distillation, heterogeneous scheduling, multi-model deployment, VLM/VLA on edge.
- Customer support: functional/performance/accuracy triage across hardware and software stacks.

**Gap**: The current handbook contains zero mentions of “edge/边缘”, Jetson, NPU, ONNX, QNN, OpenVINO, RKNN, MNN, or TVM. It cannot support the JD without substantial new content.

---

## 2. Design Goals

1. **Close the edge-infra gap** without diluting the existing cloud focus.
2. **Mirror the existing book structure**: 3 new parts that parallel the cloud “fundamentals → frameworks → optimization” progression.
3. **Provide source-level depth where it matters**: TensorRT-LLM and llama.cpp are the two most representative edge frameworks and are locally cloneable.
4. **Highlight T-Head/Alibaba ecosystem**: MNN and XuanTie RISC-V + NPU receive dedicated coverage.
5. **Update existing chapters** so the handbook feels cohesive rather than bolted-on.
6. **Keep it interview-oriented**: parameter matrices, command snippets, troubleshooting checklists, and source-code roadmaps.

---

## 3. Approaches Considered

| Approach | Structure | Pros | Cons |
|----------|-----------|------|------|
| A. 3 independent parts (recommended) | Part 15 hardware, Part 16 frameworks, Part 17 optimization | Aligns with existing Part 1/2/3/4 cloud structure; easy to jump to a topic | More files and TOC updates |
| B. 1 mega part | `part-15-edge-ai.md` | Fewer files | Too long for source-level analysis; poor readability |
| C. 2 parts | Hardware+Frameworks / Optimization+Cases | Compromise | Couples unrelated topics; awkward boundaries |

**Decision**: Use Approach A.

---

## 4. New Content

### 4.1 Part 15 — Edge AI Hardware Map & Constraints

**File**: `docs/ai-infra-handbook/part-15-edge-hardware.md`

**Target length**: 6,000–8,000 words.

**Outline**:

```
15.1 What is Edge AI? Boundaries and Definitions
  - Edge vs cloud: power, memory, bandwidth, latency, TCO
  - Why the vLLM/SGLang datacenter stack does not move down unchanged

15.2 International Edge Platforms
  - NVIDIA Jetson family (Orin Nano / NX / AGX Orin, Jetson Thor)
  - Qualcomm Snapdragon / Hexagon NPU (QNN stack)
  - Apple Neural Engine
  - Intel NPU (Meteor Lake / Arrow Lake) and AMD Ryzen AI

15.3 Domestic Edge Chips (T-Head focus)
  - T-Head XuanTie RISC-V cores (C906 / C910 / C920) and Wujian SoC platform
  - T-Head AI accelerator lines (Hanguang, etc.) and AliOS Things
  - Huawei Ascend 310 / 310P / 610
  - Cambricon MLU220 / 270 / 290 edge series
  - Horizon Journey 5 / 6
  - Black Sesame Huashan A1000
  - Rockchip RK3588 / RK3576 and RKNN ecosystem

15.4 Hardware Constraints That Shape Software
  - Shared memory (LPDDR5/DDR5) vs HBM
  - Power envelope and thermal throttling
  - No NVLink / limited interconnect
  - Fragmented NPU operator support
  - BSP/driver/SDK version coupling

15.5 Selection Matrix
  - Table: chip / compute / memory / power / software stack / best-fit scenario
```

**Key deliverables**:
- A hardware selection matrix.
- A side-by-side “cloud GPU vs edge SoC” constraints table.
- Notes on T-Head’s RISC-V + NPU positioning.

---

### 4.2 Part 16 — Edge Inference Frameworks & Software Stacks

**File**: `docs/ai-infra-handbook/part-16-edge-frameworks.md`

**Target length**: 8,000–12,000 words.

**Outline**:

```
16.1 Edge Framework Landscape
  - Classification: NVIDIA-locked / cross-platform / chip-vendor-specific
  - Relationship to cloud engines (vLLM/SGLang/TRT-LLM)

16.2 TensorRT / TensorRT-LLM (source-level deep dive)
  - Repository layout (tensorrt-llm/)
  - Builder / Runtime / Plugin architecture
  - In-flight Batching on Jetson and datacenter GPUs
  - Model conversion workflow (HF → ONNX → TRT engine)
  - Key files and code-reading roadmap
  - Jetson-specific deployment notes

16.3 llama.cpp (source-level deep dive)
  - Repository layout (llama.cpp/)
  - GGUF format and quantization schemes (Q4_0, Q4_K_M, Q8_0, IQ quants)
  - KV Cache management, batching, speculative decoding
  - Multi-backend support (CUDA, Metal, Vulkan, SYCL, CPU)
  - Key files and code-reading roadmap

16.4 MNN (T-Head/Alibaba ecosystem)
  - Architecture: Converter / Interpreter / Backend
  - Quantization and mixed precision
  - RISC-V and NPU backends
  - Key files and code-reading roadmap

16.5 Other Frameworks (overview + command examples)
  - ONNX Runtime / ONNX Runtime GenAI
  - OpenVINO / OpenVINO GenAI
  - Qualcomm QNN
  - Rockchip RKNN / RKNN-LLM
  - MLC-LLM
  - ExecuTorch
  - TensorFlow Lite / LiteRT

16.6 Framework Selection Decision Tree
  - Latency-first vs throughput-first
  - NVIDIA-locked vs cross-platform
  - Proprietary NPU vs generic GPU
```

**Key deliverables**:
- Source-code roadmaps for TensorRT-LLM and llama.cpp.
- Command snippets for model conversion/compilation on each framework.
- A framework selection decision tree.

---

### 4.3 Part 17 — Edge Scenario Optimization & Production

**File**: `docs/ai-infra-handbook/part-17-edge-optimization.md`

**Target length**: 6,000–8,000 words.

**Outline**:

```
17.1 KV Cache Management Under Memory Pressure
  - GQA / MQA / MLA on edge
  - Sliding-window attention, StreamingLLM, H2O
  - KV Cache quantization (INT8 / INT4)
  - Differences from datacenter PagedAttention

17.2 Quantization & Calibration in Practice
  - PTQ / QAT / GPTQ / AWQ / SmoothQuant applicability on edge
  - Calibration dataset selection and accuracy evaluation
  - Per-layer sensitivity analysis

17.3 Model Lightweighting
  - Distillation, pruning, NAS for edge
  - Edge-friendly model families: Phi-3/4, Gemma 2/3, Qwen2.5, Llama 3.2
  - When to use a 1B–4B model vs a quantized 7B–13B model

17.4 Heterogeneous Scheduling & Multi-Model Deployment
  - CPU + GPU/NPU mixed execution
  - Graph partitioning and fallback
  - Memory budgeting, model hot-swap, pipeline composition
  - Real-time OS / Android / Linux constraints

17.5 VLM / VLA Deployment on Edge
  - Vision encoder quantization and graph optimization
  - Multimodal pipeline: camera → preprocess → encoder → LLM
  - LLaVA / Qwen-VL / Phi-vision / OpenVLA deployment paths
  - Frame-rate vs latency trade-offs for video streams

17.6 Inflight / Continuous Batching on Edge
  - Small-batch, low-latency differences from cloud
  - When Continuous Batching helps and when it hurts

17.7 Troubleshooting & Customer Support
  - Functional / performance / accuracy triage flow
  - SDK / BSP / driver version conflicts
  - Thermal throttling detection
  - Cross-team collaboration template (algorithm / hardware / customer)

17.8 Edge Production Checklist
```

**Key deliverables**:
- A quantization/calibration workflow.
- A multi-model memory budget worksheet.
- A troubleshooting decision tree.
- An edge production checklist.

---

## 5. Existing Chapter Updates

### 5.1 Part 0 — AI Infra Overview

**File**: `docs/ai-infra-handbook/part-00-overview.md`

**Changes**:
- Extend the分层架构 diagram to include an “Edge / Endpoint Device Layer” between users and the cloud platform.
- Add a short section “0.7 云侧 vs 边缘侧：同一问题，两种解法” summarizing why the two stacks diverge.

### 5.2 Part 10 — Engine Ecosystem Deep Dive

**File**: `docs/ai-infra-handbook/part-10-engine-ecosystem.md`

**Changes**:
- Add a new section “10.7 边缘侧推理引擎” with an overview matrix.
- Add a paragraph comparing cloud engines (vLLM/SGLang/TRT-LLM) with edge engines (TensorRT Jetson, ONNX Runtime GenAI, llama.cpp, MNN, RKNN, QNN, OpenVINO).

### 5.3 Part 12 — Domestic GPU & Heterogeneous Hardware

**File**: `docs/ai-infra-handbook/part-12-domestic-gpu.md`

**Changes**:
- Rename title to “国产 GPU 与边缘芯片” (or keep title and add a large edge section).
- Add sections for T-Head XuanTie/Wujian, Horizon Journey, Black Sesame Huashan, Cambricon edge MLU, Rockchip RK3588, and Ascend 310 edge series.
- Add a “datacenter vs edge” selection table.

### 5.4 README.md

**File**: `docs/ai-infra-handbook/README.md`

**Changes**:
- Add Part 15/16/17 to the TOC.
- Add reading paths for “边缘 AI 工程师” and “平头哥/T-Head 生态面试”.
- Add quick-lookup rows linking KV Cache, quantization, VLM/VLA, and multi-model deployment to the new parts.

---

## 6. Source Repositories

- `llama.cpp/` already cloned at project root (shallow clone).
- `tensorrt-llm/` cloned by user at project root.
- MNN, ONNX Runtime, OpenVINO, RKNN repos referenced via URLs only (no local clone required for overview coverage).

---

## 7. Style & Consistency Requirements

1. Match existing book conventions:
   - `code` for source symbols.
   - **bold** for first occurrence of key concepts.
   - ★ for interview-high-frequency marks.
   - Tables for comparisons.
   - Command blocks with `bash` language tag.
2. Cross-reference existing parts (e.g., “对比 Part 1 中讲的 KV Cache”）.
3. Keep Chinese as the primary language, English for class/file names.
4. Cite source files with relative paths from the cloned repos.

---

## 8. Out of Scope

- Running actual edge hardware benchmarks (no physical boards available).
- Full source-level coverage of every framework (only TensorRT-LLM and llama.cpp go deep).
- Training or fine-tuning content.
- Detailed chip RTL/architecture deep dives (keep at software-visible level).

---

## 9. Success Criteria

1. A reader can open Part 15–17 and understand edge AI infra without prior cloud-only assumptions.
2. A candidate interviewing for the T-Head JD can use Part 15–17 to answer hardware, framework, optimization, and troubleshooting questions.
3. Existing chapters no longer imply AI Infra == datacenter GPUs.
4. README accurately reflects the new scope.

---

## 10. Implementation Order

1. Update README.md TOC and reading paths.
2. Write Part 15 (hardware).
3. Write Part 16 (frameworks) — requires source exploration of `tensorrt-llm/` and `llama.cpp/`.
4. Write Part 17 (optimization and production).
5. Update Part 0 overview diagram.
6. Update Part 10 engine ecosystem matrix.
7. Update Part 12 domestic hardware sections.
8. Self-review for broken links, TOC consistency, and style.
9. (Optional) Add edge-related entries to Part 9 glossary.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| TensorRT-LLM repo is large and NVIDIA-locked | Shallow clone only; focus on edge-relevant files |
| llama.cpp evolves rapidly | Pin to the cloned commit; note version in the part header |
| Edge hardware specs change | Use ranges and “as of 2026” qualifiers |
| Content becomes too long | Strictly follow outlined sections; move deep details into tables or code roadmaps |
| Duplication with existing quantization sections | Cross-reference Part 1/2 and focus on edge-specific differences |

---

**Next step**: Invoke the `writing-plans` skill to produce a detailed implementation plan and begin execution.
