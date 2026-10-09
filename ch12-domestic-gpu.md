# 第12章 国产 GPU 与边缘芯片

> **面向角色**：需要在国产或异构硬件上部署推理的工程师  
> **目标**：理解华为昇腾、摩尔线程、海光、AMD 等非 NVIDIA 硬件，以及国产边缘 AI 芯片的适配要点

---

## 12.1 国产/异构 GPU 与边缘芯片概览

| 厂商 | 产品 | 定位 | 架构/核心 | 生态 | 推理支持 |
|------|------|------|-----------|------|---------|
| **华为** | Ascend 910B/910C | 数据中心 | 达芬奇 (DaVinci) | CANN + MindSpore | MindIE / vLLM 昇腾后端 |
| **摩尔线程** | MTT S4000/S5000 | 数据中心/工作站 | MUSA | 兼容 CUDA | vLLM / SGLang MUSA backend |
| **海光** | DCU Z100/Z200 | 数据中心 | GCN/CDNA-like | ROCm 兼容 | vLLM ROCm / LMDeploy |
| **寒武纪** | MLU370/590 | 数据中心 | Cambricon | BANG C | 自有推理框架 |
| **AMD** | MI300X/MI325X | 数据中心 | CDNA3 | ROCm | vLLM ROCm / SGLang ROCm |
| **Intel** | Gaudi2/Gaudi3 | 数据中心 | - | PyTorch/Habana | TGI / vLLM 有限支持 |
| **华为** | Ascend 310 / 310P / 610 | 边缘/车载 | 达芬奇 | CANN Lite / MindSpore Lite | ACL / 昇腾 OM 模型 |
| **寒武纪** | MLU220 / 270-S 等边缘系列 | 边缘 | Cambricon | CNML / BANG C | 自有边缘推理框架 |
| **地平线** | 征程 5 / 征程 6 | 车载/边缘 | BPU Nash/Bayes | 天工开物 / 踏歌 | Horizon 工具链 |
| **黑芝麻** | 华山 A1000 | 车载/边缘 | NeuralIQ | 山海 AI 平台 | 自有 BEV/Transformer 工具链 |
| **瑞芯微** | RK3588 / RK3576 | 边缘/工控 | ARM + NPU | RKNN | RKNN Toolkit2 / Linux SDK |
| **平头哥** | 玄铁 C906/C910/C920 + 无剑平台 | 端侧/AIoT | RISC-V | AliOS Things / MNN | MNN RISC-V 后端 |

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
  - HBM: 32 GB
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

## 12.7 平头哥（T-Head）玄铁 RISC-V 与无剑平台 ★★

### 玄铁 RISC-V 处理器

平头哥开源/授权了多档玄铁（XuanTie）RISC-V 处理器，覆盖端侧 MCU 到边缘 AI：

```
玄铁 C906:
  - 64-bit RISC-V，单/双发射顺序执行
  - 支持 RV64IMAC + 自定义扩展
  - 典型主频 1.0–1.5 GHz，侧重 AIoT / 边缘控制

玄铁 C910:
  - 64-bit RISC-V，超标量乱序执行
  - 支持 RV64GC + 矢量扩展（RVV）
  - 面向中高端边缘、网关、DPU

玄铁 C920:
  - 64-bit RISC-V，增强 AI/矢量能力
  - 支持更高性能 SIMD / NPU 协同
  - 面向 AI 边缘盒子、自动驾驶域控 MCU
```

### 无剑（Wujian）SoC 平台

无剑平台是平头哥提供的 **AIoT 芯片设计平台**，把玄铁 CPU、NPU/DSP、高速接口、安全子系统打包成可配置 SoC 模板，降低芯片设计门槛：

```
无剑 100 / 600 / 900 系列:
  - 内置玄铁 C906/C910/C920
  - 可选 NPU / 矢量 DSP / 安全岛
  - 支持 Linux / RTOS / AliOS Things
  - 提供参考板级支持包（BSP）与 SDK
```

### AI 加速器集成

玄铁 CPU 通常搭配以下加速器做推理：

1. **NPU IP**：部分无剑 SoC 集成第三方或平头哥自研 NPU，执行 INT8/FP16 算子。
2. **矢量 DSP / RVV**：C910/C920 的 RVV 扩展可加速向量运算，适配 MNN 的 RVV backend。
3. **异构调度**：CPU 负责任载编排和前后处理，NPU/DSP 负责 heavy op（Conv、MatMul、Transformer）。

### AliOS Things / MNN 关系

- **AliOS Things**：阿里 IoT 操作系统，已针对玄铁 RISC-V 优化，提供驱动、网络、安全组件。
- **MNN**：阿里开源端侧推理引擎，支持玄铁 RISC-V、ARM、x86 后端，具备 RVV 与 NPU plugin 适配能力。
- 组合使用场景：AliOS Things + 玄铁 C906/C920 + MNN，可在小体积、低功耗设备上跑视觉/语音模型。

### 适配路径与关键参数

```
模型 (ONNX/TFLite/PyTorch) → MNNConvert 转换
                         ↓
             MNN 模型 (.mnn)
                         ↓
      玄铁 RISC-V backend (RVV / NPU plugin)
                         ↓
           无剑 SoC / 玄铁开发板运行
```

关键参数：

| 参数 | 典型值/说明 |
|------|------------|
| CPU | 玄铁 C906 / C910 / C920 |
| OS | AliOS Things / Linux / RT-Thread |
| 推理框架 | MNN / TFLite Micro / 自研 SDK |
| 精度 | INT8 / FP16 为主，部分支持 INT4 |
| 功耗 | 几百 mW 到数 W |
| 典型算力 | 0.5–10 TOPS（依赖是否带 NPU） |

### 命令示例

```bash
# 1. 交叉编译 MNN（玄铁 RISC-V 工具链）
mkdir build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/riscv64-elf.cmake \
  -DMNN_RV=ON \
  -DMNN_OPENMP=ON \
  -DMNN_USE_THREAD_POOL=OFF
make -j$(nproc)

# 2. 转换 ONNX 模型为 MNN
./MNNConvert -f ONNX \
  --modelFile resnet18.onnx \
  --MNNFile resnet18.mnn \
  --bizCode MNN

# 3. 在玄铁开发板运行 benchmark
adb push ./benchmark.out /data/local/tmp/
adb push ./resnet18.mnn /data/local/tmp/
adb shell "cd /data/local/tmp && ./benchmark.out resnet18.mnn 10 0"
```

### 关键注意事项

1. **工具链版本**：玄铁 GCC/LLVM 与 RVV 版本要匹配芯片实现，否则出现非法指令。
2. **NPU plugin 封闭性**：部分无剑 SoC 的 NPU 驱动/工具链不完全开源，需与平头哥或板卡厂商确认 SDK。
3. **算子回退**：MNN 对 RVV 的支持在持续完善，遇到不支持的算子会回退到标量 C，性能可能骤降。
4. **内存布局**：小设备内存有限，建议使用 MNN 的量化与模型裁剪功能。

---

## 12.8 地平线征程 5 / 6

### 硬件特性

地平线征程（Journey）系列主打车载与边缘 ADAS：

```
征程 5 (J5):
  - 算力: ~128 TOPS INT8
  - 架构: BPU Nash (贝叶斯架构)
  - 典型功耗: ~30 W
  - 定位: L2+ / L3 自动驾驶域控

征程 6 (J6):
  - 算力: 80–560 TOPS 不等（J6B/J6E/J6M/J6P 多档）
  - 架构: BPU Bayes 新一代
  - 制程与能效进一步提升
  - 定位: 高阶智驾、舱驾一体
```

### 适配路径

```
PyTorch/ONNX 模型 → 地平线天工开物工具链（OE）
                ↓
      模型量化 / 算子优化 / 编译为 .hbm
                ↓
      征程 5/6 BPU 上推理
```

### 关键注意事项

1. **量化强相关**：BPU 以 INT8 为主，需要对感知模型做 QAT 或 PTQ 量化调优。
2. **算子白名单**：Transformer/BEV 算子在新一代 BPU 上支持更好，老模型需切分或改写。
3. **多传感器预处理**：征程芯片自带 ISP 与多路视频输入，前后处理 pipeline 需同步设计。

---

## 12.9 黑芝麻华山 A1000

### 硬件特性

```
华山 A1000:
  - 算力: ~40–70 TOPS INT8（视配置）
  - CPU: 多核 ARM Cortex-A55
  - 内置 ISP、视频编解码、安全岛
  - 定位: L2+ 自动驾驶域控 / 边缘 AI 盒子
```

### 适配路径

```
ONNX / Caffe 模型 → 黑芝麻山海 AI 平台工具链
                ↓
      图优化、INT8 量化、生成 runtime 模型
                ↓
      华山 A1000 NPU 推理
```

### 关键注意事项

1. **数据流优化**：A1000 强调 BEV/Occupancy 网络，需利用 NPU 的片上缓存减少 DDR 访问。
2. **混合精度**：INT8 为主，部分层支持 FP16，需手动或自动精度调优。
3. **车规与功能安全**：车规部署需满足 ASIL 等级，工具链提供相应的 trace 与校验机制。

---

## 12.10 寒武纪 MLU 边缘系列

### 硬件特性

寒武纪除了数据中心 MLU370/590，也提供边缘侧产品：

```
MLU220 边缘加速模块:
  - 算力: ~16 TOPS INT8
  - 功耗: 8–15 W
  - 接口: M.2 / mPCIe / 小型模组

MLU270-S 等边缘卡:
  - 算力: 几十 TOPS INT8
  - 支持视频编解码与多路分析
```

### 适配路径

```
PyTorch/ONNX/Caffe 模型 → CNML / MagicMind 转换
                       ↓
             离线模型（.cambricon）
                       ↓
             CNRT / 寒武纪边缘推理 runtime
```

### 关键注意事项

1. **软件栈统一**：边缘与数据中心共享 Neuware 生态，模型转换流程类似，但算子版本需对齐。
2. **多路视频**：边缘场景常需并发多路摄像头，注意 NPU 利用率与内存带宽瓶颈。
3. **散热与功耗**：MLU220 功耗较低，可部署在户外盒子；MLU270-S 需注意风冷/散热设计。

---

## 12.11 瑞芯微 RK3588 / RK3576

### 硬件特性

瑞芯微 RK35xx 是国产边缘开发板最常见的 SoC 之一：

```
RK3588:
  - CPU: 四核 A76 + 四核 A55
  - GPU: Mali-G610 MC4
  - NPU: 6 TOPS INT8（支持 INT4/INT8/FP16/BF16/TF32）
  - 视频: 8K 解码 / 4K 编码
  - 接口: PCIe 3.0 / USB3 / SATA / GMAC

RK3576:
  - CPU: 四核 A72 + 四核 A53
  - GPU: Mali-G52 MC3
  - NPU: 6 TOPS INT8
  - 定位: RK3588 的降成本版本，工控/教育/机器人
```

### 适配路径

```
ONNX / TFLite / Caffe 模型 → RKNN-Toolkit2 转换
                          ↓
                RKNN 模型 (.rknn)
                          ↓
                RKNN Runtime → NPU 推理
```

### 命令示例

```bash
# 1. 安装 RKNN-Toolkit2（PC 端，建议 Ubuntu + Python 3.8/3.10）
pip install rknn-toolkit2

# 2. Python 脚本转换模型
python <<'PY'
from rknn.api import RKNN
rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='yolov5s.onnx')
rknn.build(do_quantization=True, dataset='./dataset.txt')
rknn.export_rknn('yolov5s.rknn')
PY

# 3. 推送到板端运行
adb push yolov5s.rknn /data/
adb push rknn_yolov5_demo /data/
adb shell "cd /data && ./rknn_yolov5_demo yolov5s.rknn"
```

### 关键注意事项

1. **算子支持**：RKNN 对 YOLO、ResNet、MobileNet 支持好；复杂 Transformer 需拆分或 CPU fallback。
2. **NPU 利用率**：小 batch / 低分辨率时 NPU 利用率不高，可做 batching 或 pipeline 优化。
3. **驱动与固件**：不同板厂的 BSP 有差异，升级 NPU 驱动后再跑 RKNN。

---

## 12.12 华为昇腾 310 / 310P / 610

### 硬件特性

昇腾 3xx 系列面向边缘推理、AI 盒子、车载及网关：

```
昇腾 310:
  - 算力: ~16 TOPS INT8 / 8 TOPS FP16
  - 功耗: ~8 W
  - 制程: 12 nm
  - 形态: Atlas 200 DK / 边缘模块

昇腾 310P:
  - 算力: 22–64 TOPS INT8（按型号 P3/P4/P7 等）
  - 功耗: 8–35 W
  - 支持更多视频编解码路数
  - 形态: Atlas 300I Pro / Atlas 500 Pro

昇腾 610:
  - 算力: ~200 TOPS INT8
  - 功耗: 约 65 W
  - 定位: 高阶边缘 / 车载域控
```

### 适配路径

```
ONNX / Caffe / MindSpore 模型 → ATC (Ascend Tensor Compiler)
                             ↓
                   离线模型 (.om)
                             ↓
                   ACL / pyACL → 昇腾 310/310P/610 推理
```

### 命令示例

```bash
# 1. 模型转换（以 ONNX 为例）
atc --model=resnet50.onnx \
    --framework=5 \
    --output=resnet50 \
    --soc_version=Ascend310P3 \
    --input_shape="actual_input_1:1,3,224,224" \
    --insert_op_conf=aipp.config

# 2. pyACL 推理（Python）
python <<'PY'
import acl
# 初始化、加载 om、创建 stream、执行、释放资源
PY

# 3. 查看 NPU 状态
npu-smi info
```

### 关键注意事项

1. **ATC 版本与 soc_version 严格绑定**：不同 310P 子型号需指定正确版本，否则转换失败。
2. **AIPP 预处理**：图像模型常用 AIPP 做色域转换/归一化，需在 ATC 时配置。
3. **内存管理**：ACL 需要显式管理 Device/Host 内存与 stream，开发成本高于 PyTorch。
4. **多路并发**：Atlas 300I Pro 等多卡/多核设备可用多线程 + 多 stream 提升吞吐。

---

## 12.13 数据中心 vs 边缘：选型对比

| 维度 | 数据中心芯片 | 边缘芯片 |
|------|-------------|---------|
| **代表型号** | Ascend 910B / MTT S4000 / MI300X / MLU590 | Ascend 310 / RK3588 / 征程 5 / 玄铁 C920 + NPU |
| **峰值算力** | 百 TOPS 到 PFLOPS 级 | 数 TOPS 到数百 TOPS |
| **功耗** | 300–700 W（单卡或模组） | 数 W 到数十 W |
| **显存/内存** | 32–192 GB HBM / 高带宽 | 数 GB LPDDR / 共享内存 |
| **互联方式** | NVLink / HCCS / Infinity Fabric / PCIe Switch | PCIe / USB / MIPI / 板级总线 |
| **部署形态** | 服务器 / 集群 / 智算中心 | 边缘盒子 / 工控机 / 车载域控 / AIoT 设备 |
| **软件栈** | vLLM / MindIE / SGLang / ROCm / CANN | RKNN / ACL / Horizon OE / MNN / 寒武纪 Neuware Edge |
| **模型规模** | 7B–数百 B 大模型、多卡并行 | 数 M–数十 B 小模型、单卡/单芯片 |
| **主要优化目标** | 高吞吐、低 TTFT/TPOT、可扩展 | 低功耗、低延迟、成本、体积、稳定性 |
| **适用场景** | 云端 LLM 服务、训练、大规模推荐 | 智能制造、自动驾驶、安防、机器人、端侧语音/视觉 |

### 选型建议

1. **先确定算力与功耗天花板**：边缘场景先看散热和供电允许多少 W，再反推芯片型号。
2. **优先复用已有生态**：如果团队已用华为 CANN，边缘直接选昇腾 310P；如果已用 RKNN，优先 RK3588。
3. **软件栈成熟度决定落地速度**：边缘芯片的算子支持、量化工具、调试手段差异大，需提前做 POC。
4. **车规/工规合规**：车载必选通过 AEC-Q100/功能安全认证的芯片（征程、昇腾 610、华山 A1000）。
5. **成本与产量**：大批量 AIoT 可考虑玄铁 RISC-V + 无剑平台定制 SoC；小批量验证优先 RK3588 / 昇腾 310 等成熟开发板。

---

## 12.14 本章小结

| 硬件 | 软件栈 | 主要适配工作 | 适合场景 |
|------|--------|-------------|---------|
| NVIDIA | CUDA/NCCL | 标准方案 | 全场景 |
| 华为昇腾 | CANN/HCCL/MindIE | 算子重写、通信适配 | 国产化替代、数据中心/边缘 |
| 摩尔线程 | MUSA | CUDA 兼容层迁移 | 中小模型、成本敏感 |
| AMD/海光 | ROCm/HIP/RCCL | hipify 转换、性能调优 | 大显存、替代 NVIDIA |
| 寒武纪 | BANG C | 自有框架 | 数据中心/特定边缘项目 |
| 地平线征程 | 天工开物/踏歌 | 量化、算子白名单 | 车载 ADAS |
| 黑芝麻 A1000 | 山海 AI 平台 | BEV/Transformer 优化 | 自动驾驶域控 |
| 瑞芯微 RK3588 | RKNN | 模型转换、pipeline 优化 | 边缘盒子、机器人 |
| 平头哥玄铁 | AliOS Things / MNN | 交叉编译、RVV/NPU 适配 | AIoT、端侧定制 SoC |
