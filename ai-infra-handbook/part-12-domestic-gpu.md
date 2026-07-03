# Part 12：国产 GPU 与异构硬件适配

> **面向角色**：需要在国产或异构硬件上部署推理的工程师  > **目标**：理解华为昇腾、摩尔线程、海光 DCU、AMD 等非 NVIDIA 硬件的适配要点

---

## 12.1 国产/异构 GPU 概览

| 厂商 | 产品 | 架构 | 生态 | 推理支持 |
|------|------|------|------|---------|
| **华为** | Ascend 910B/910C |达芬奇 (DaVinci) | CANN + MindSpore | MindIE / vLLM 昇腾后端 |
| **摩尔线程** | MTT S4000/S5000 | MUSA | 兼容 CUDA | vLLM / SGLang MUSA backend |
| **海光** | DCU Z100/Z200 | GCN/CDNA-like | ROCm 兼容 | vLLM ROCm / LMDeploy |
| **寒武纪** | MLU370/590 | Cambricon | BANG C | 自有推理框架 |
| **AMD** | MI300X/MI325X | CDNA3 | ROCm | vLLM ROCm / SGLang ROCm |
| **Intel** | Gaudi2/Gaudi3 | - | PyTorch/Habana | TGI / vLLM 有限支持 |

---

## 12.2 华为昇腾 (Ascend) ★

### 硬件特性

```
Ascend 910B:
  - 算力: ~320 TFLOPS FP16
  - HBM: 64 GB
  - 互联: HCCS (Huawei Cache Coherence System)
  - 编程: CANN (Compute Architecture for Neural Networks)
```

### 适配路径

```
PyTorch 模型 → torch_npu (昇腾 PyTorch 后端)
         ↓
CANN ATB (Ascend Transformer Boost) / ACL
         ↓
MindIE (Mind Inference Engine) 或 vLLM 昇腾后端
```

### vLLM 昇腾后端

vLLM 通过 plugin 机制支持昇腾：

```bash
# 安装昇腾 vLLM
pip install vllm-ascend

# 启动
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --tensor-parallel-size 8 \
  --device npu
```

### 关键注意事项

1. **算子支持**：部分 custom CUDA kernel 需要改写成 CANN 算子
2. **通信后端**：使用 HCCL 替代 NCCL
3. **量化**：支持 INT8/INT4，FP8 支持较晚
4. **性能**：通常比同代 NVIDIA 低 20-40%，但成本也更低
5. **生态**：文档和社区以中文为主

---

## 12.3 摩尔线程 (Moore Threads) ★

### 硬件特性

```
MTT S4000:
  - 算力: ~25 TFLOPS FP32 (约 A10 级别)
  - 显存: 48 GB
  - 架构: MUSA (类似 CUDA)
```

### 适配路径

摩尔线程通过 **MUSA 兼容层** 降低迁移成本：

```
CUDA 代码 → MUSA 工具链 (muconvert/mucc)
        ↓
MUSA kernel / 驱动
        ↓
MTT GPU
```

### vLLM / SGLang MUSA Backend

```bash
# vLLM MUSA 后端
VLLM_TARGET_DEVICE=musa python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --tensor-parallel-size 2
```

### 关键注意事项

1. **CUDA 兼容度**：MUSA 兼容大部分 CUDA API，但部分高级特性（如 CUDA Graph）支持有限
2. **性能**：当前主流模型可跑通，但大模型（70B+）性能仍有差距
3. **稳定性**：驱动和工具链更新频繁，需要密切跟进版本
4. **适用场景**：中小模型、成本敏感、国产化替代

---

## 12.4 海光 DCU / AMD ROCm ★

### 硬件特性

```
AMD MI300X:
  - 算力: ~1.3 PFLOPS FP16 (Tensor)
  - HBM: 192 GB
  - 互联: Infinity Fabric / xGMI
  - 生态: ROCm (兼容 CUDA 部分 API)

海光 DCU Z100:
  - 算力: ~100 TFLOPS FP16
  - HBM: 32 GB
  - 兼容 ROCm
```

### 适配路径

```
CUDA 代码 → HIP 转换 (hipify)
        ↓
ROCm / HIP runtime
        ↓
AMD GPU / 海光 DCU
```

### vLLM ROCm 支持

```bash
# ROCm 环境
export VLLM_TARGET_DEVICE=rocm

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --tensor-parallel-size 8 \
  --quantization fp8
```

### ROCm 上的 MoE 优化

AMD ROCm 对 vLLM MoE 有专门优化：

```bash
# 启用 tuned GEMM
export VLLM_ROCM_USE_FLASH_ATTN_TRITON=1
export VLLM_ROCM_USE_FP8_FLASH_ATTN=1
```

### 关键注意事项

1. **FP8 支持**：MI300X 支持 FP8，但工具链成熟度不如 NVIDIA
2. **FlashAttention**：ROCm 版 FlashAttention 功能逐步完善
3. **通信**：RCCL 替代 NCCL，大集群稳定性需验证
4. **海光 DCU**：兼容性比 AMD 原生略差，部分 kernel 需定制

---

## 12.5 跨硬件迁移通用 checklist

```
□ 算子支持验证
  - 列出模型用到的所有 custom kernel
  - 检查目标平台是否有对应实现
  
□ 通信后端
  - NVIDIA: NCCL
  - 华为: HCCL
  - AMD/海光: RCCL
  - 摩尔线程: MUSA 集合通信
  
□ 量化格式
  - NVIDIA: FP8/INT8/INT4
  - 昇腾: INT8/INT4 优先
  - AMD: FP8/INT8
  - 摩尔线程: INT8/INT4

□ 性能基准
  - 相同负载下与 NVIDIA 对比
  - 关注 TTFT/TPOT/Throughput 三指标
  
□ 稳定性测试
  - 长时间压测（24h+）
  - 大并发下的 OOM/崩溃
  - 通信异常恢复

□ 成本核算
  - 硬件成本 vs 性能损失
  - 迁移/维护人力成本
  - 国产化合规收益
```

---

## 12.6 面试常见问题

**Q: 国产 GPU 上部署 LLM 的最大挑战是什么？**
- 答：算子生态和通信后端。NVIDIA CUDA 有最成熟的 custom kernel 生态，国产 GPU 需要逐个验证/迁移算子，且集合通信（AllReduce/AllToAll）的稳定性需要长期打磨。

**Q: 昇腾和 CUDA 的主要区别？**
- 答：昇腾使用达芬奇架构和 CANN 软件栈，算子以 CANN 算子为主；CUDA 使用 SM 架构和 CUDA runtime。昇腾对 Transformer 有专门优化（ATB），但整体生态不如 CUDA 丰富。

**Q: 为什么 MI300X 192GB HBM 比 H100 80GB 更适合大模型？**
- 答：192GB 可以放下更大模型或更多 KV Cache，减少 PP/TP 切分，降低通信开销。但实际性能还取决于算力、带宽、kernel 优化程度。

---

## 12.7 本章小结

| 硬件 | 软件栈 | 主要适配工作 | 适合场景 |
|------|--------|-------------|---------|
| NVIDIA | CUDA/NCCL | 标准方案 | 全场景 |
| 华为昇腾 | CANN/HCCL/MindIE | 算子重写、通信适配 | 国产化替代 |
| 摩尔线程 | MUSA | CUDA 兼容层迁移 | 中小模型、成本敏感 |
| AMD/海光 | ROCm/HIP/RCCL | hipify 转换、性能调优 | 大显存、替代 NVIDIA |
| 寒武纪 | BANG C | 自有框架 | 特定项目 |
