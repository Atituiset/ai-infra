# 第24章 AI 编译器与算子开发

> **面向角色**：推理引擎/内核研发工程师、性能优化工程师  
> **前置知识**：第7章（GPU/CUDA 基础）、第20章 §20.3（Kernel 栈编年史）  
> **目标**：建立编译器的分层心智模型（Graph IR → Tensor/Schedule IR → 机器码），掌握主流栈的取舍与引擎集成方式；能按方法论独立开发、融合、验证一个算子

---

## 24.0 时间发展线 ★

```
2013        2017      2018       2019         2020          2022              2023              2024                    2025-
 │           │         │          │            │             │                 │                 │                       │
 Halide      XLA       TVM        Triton /     Ansor(NSDI)   torch.compile     PT2稳定 /         AOTInductor /           tilelang /
 (compute/  (TF)      (OSDI)     MLIR宣布      MLIR(CGO)     (Dynamo+Inductor) TVM Unity /       vLLM piecewise compile  Mirage/MAX
 schedule分离)                   (MAPL)                     + TE FP8          MLC-LLM           成为默认                 百花齐放
```

**元规律**：编译器每十年换一次主战场——2010s 解决"算子手写太慢"（schedule 自动化），2020s 解决"图优化+部署碎片化"（统一 IR），LLM 时代解决"动态 shape + 与 CUDA Graph 共存"。Halide 的 compute/schedule 分离思想贯穿始终。

---

## 24.1 为什么推理引擎需要编译器 ★

手写 kernel 覆盖不了三个爆炸：

1. **Shape 爆炸**：batch/seq 每种组合都要最优 tiling——autotuning 搜索空间巨大；
2. **融合组合爆炸**：norm+quant+GEMM+act 的排列组合数远超人力；
3. **硬件碎片化**：同一模型要落到 NV/AMD/国产/端侧 NPU。

编译器的三层工作：

```
Graph 层：常量折叠 / 算子融合 / 布局变换(nhwc↔ndhw) / 量化节点插入 / 死代码消除
Tensor(Schedule) 层：tiling / 流水 / 向量化 / 内存提升(shared/register) / autotune
后端层：LLVM / PTX / 各厂商 codegen
```

---

## 24.2 主流栈对比 ★

| 栈 | 血统 | 强项 | 弱项 | LLM 推理角色 |
|----|------|------|------|--------------|
| **XLA** | Google, 2017 | TPU 原生、JAX 底座 | 定制性差 | JAX/Gemma 生态 |
| **TVM / Unity(Relax)** | OSDI'18 | 开山之作，Ansor 自动调度，端到云覆盖 | 工程迭代慢 | MLC-LLM 路线 |
| **torch.compile** (Dynamo+Inductor+AOTInductor) | PyTorch 2022 | Python 生态零成本接入，dynamic shape 处理 | 极致性能仍需自定义 op | **vLLM 默认开启 piecewise compile** |
| **TensorRT(-LLM)** | NVIDIA, 闭源 | tactic 搜索 + kernel 全家桶，极致性能 | 锁定 NV、构建流程重 | 生产闭源路线 |
| **MLIR 生态** | LLVM 系, CGO'21 | 统一基础设施（linalg/tensor dialect） | 上层应用需自建 | 各厂商自研编译器地基 |
| **Triton** | OpenAI, MAPL'19 | Python DSL 写接近 CUTLASS 性能的 kernel | 只解决 Tensor 层 | 引擎自定义事实标准 |
| 新势力：Mojo/MAX、tilelang、Mirage | 2023-25 | 更高抽象或超优化搜索 | 生态早期 | 观察 |

---

## 24.3 引擎中的编译实践：两条路线 ★

### 路线 A：piecewise torch.compile + CUDA Graph（vLLM V1 默认）

```
整图 → Dynamo 捕获 → 按 attention 边界切段
     ├─ 非 attention 子图：Inductor 编译（融合 norm/quant/act 等 memory-bound 算子）
     └─ attention 段：调用手写 backend（FlashAttention/FlashInfer）
最终整段用 CUDA Graph 重放消除 launch 开销
```

为什么是当前工程最优解：attention 这类动态 shape 算子交给专用库，其余静态 shape 部分吃满编译红利，CUDA Graph 兜底 launch 开销。

### 路线 B：预编译 kernel 库（SGLang sgl-kernel / FlashInfer JIT）

把高频算子做成预编译二进制包（sgl-kernel），长尾算子运行时 JIT 编译（FlashInfer 按输入特征即时生成/选择 kernel）。TRT-LLM 则是全量离线编译的极端形态。

---

## 24.4 算子开发方法论 ★

### 24.4.1 六步工作流

```
1. Profile 定位：Nsight Compute 找热点
2. Roofline 判断：memory-bound 还是 compute-bound？（决定优化方向）
   - AI = FLOPs / Bytes；推理大多数非 GEMM 算子都是 memory-bound
3. 实现选型：
   - Triton：首选，90% 场景，迭代快、可读性好
   - CUDA+CUTLASS：需要 warp specialization/TMA 等 Hopper 特性、极致 GEMM
   - FlashInfer 复用：attention 类先查库
4. Autotune：block/warp/num_stages 网格搜索（Triton @triton.autotune）
5. 数值验证：vs eager 参考实现，allclose 分层容差
   （采样路径要求更严：top-k 边界 token 对 logit 微扰敏感）
6. 集成检查：custom op 注册 + CUDA Graph 兼容三查
   （无 host sync / 无动态分配逃逸 / shape 固定化）
```

### 24.4.2 CUDA Graph 兼容性（最容易踩的坑）

- kernel 内禁止 `cudaMemcpy` D2H 同步、禁止依赖 CPU 分支；
- 动态 shape 用 padding 到 bucket 或 max shape + mask；
- workspace 用 graph pool 预分配，不能每次 malloc。

---

## 24.5 算子融合分类学 ★

**收益判据一句话：融合省掉的是中间张量的 HBM 往返字节。** 所以：

| 类型 | 形态 | 案例 | 收益来源 |
|------|------|------|----------|
| **垂直融合** | 元素级链挂到 GEMM epilogue | GEMM+bias+GELU、RMSNorm+残差+FP8-quant | 中间激活不落 HBM |
| **水平融合** | 同形小算子合并成批 | 多请求 GEMV 合并、MoE grouped GEMM（cutlass grouped gemm） | 提升算术强度、减少 launch |
| **跨算子重构** | 改算法消掉整类访存 | FlashAttention（online softmax）、fused decode attention | IO-aware 范式 |

案例拆解（面试可画）：
1. **RMSNorm+残差+量化融合**：三次读写变一次，memory-bound 算子吞吐 ~×3；
2. **MoE grouped GEMM**：token→expert 的 scatter 后，每个 expert 的稠密小 GEMM 合并为单 kernel 的分组执行，消除大量小 kernel 启动；
3. **FlashAttention 即终极融合**：把 softmax 这个"需要全行归一化"的算子改写成流式两遍，使 attention 整体可以分块融合——它不是"融合了已有算子"，而是证明了融合的最高形态是**改数学**。

---

## 24.6 论文线汇总 ★

完整对照见 第21章 线 11/12，核心八篇：

| # | 论文/资料 | 出处 | 一句话 |
|---|----------|------|--------|
| 1 | Halide | PLDI 2013 | compute/schedule 分离思想源头 |
| 2 | TVM (`1810.00937`) | OSDI 2018 | 端到端编译栈开山 |
| 3 | Triton (Tillet et al.) | MAPL 2019 | tiled neural computation 的 Python DSL |
| 4 | Ansor (`2002.10967`) | NSDI 2020 | schedule 自动搜索巅峰 |
| 5 | MLIR (`2002.11054`) | CGO 2021 | 可组合编译器基础设施 |
| 6 | PyTorch 2 (`2407.16332`) | 2024 | Dynamo/Inductor 官方论文 |
| 7 | FlashInfer (`2501.01005`) | 2025 | attention runtime 库化运营 |
| 8 | FlashAttention (`2205.14135`) | NeurIPS 2022 | 融合最高形态 = 改数学 |

---

## 24.7 本章小结

- 编译器三层心智模型：**Graph pass / Schedule+autotune / 后端 codegen**；
- LLM serving 的工程答案是 piecewise compile + 专用 attention 库 + CUDA Graph 兜底；
- 算子开发六步法，Roofline 先行，数值验证与 Graph 兼容是交付门槛；
- 融合收益 = 省掉的 HBM 字节；融合的最高境界是重写数学。

### 面试高频问题

1. 为什么 vLLM 不做整图编译而用 piecewise？（动态 shape + attention 专用库已最优）
2. Triton 和 CUDA 怎么选？什么场景必须下 CUDA？
3. 一个 RMSNorm 算子怎么优化？预期收益多少？（roofline 分析 + 融合）
4. CUDA Graph 捕获对算子的约束有哪些？
5. MoE 的 grouped GEMM 解决什么问题？

→ 交叉复习：Kernel 栈历史 第20章 §20.3；FlashInfer/sgl-kernel 在引擎中的位置 第3章 §3.4 / 第4章。
