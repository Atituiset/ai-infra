# 第20章 技术编年史 — AI Infra 演进脉络

> **面向角色**：所有读者；尤其适合技术叙事与架构判断  
> **前置知识**：建议先读第0~8章（知道"是什么"），再来理解"为什么是这个顺序"  
> **目标**：把散落全书的技术放回时间轴，建立因果链——每一次范式跃迁都由"新瓶颈暴露"驱动。能讲清演进逻辑的人，比只会罗列名词的高一档

---

## 20.1 为什么读历史 ★

AI Infra 是一个**瓶颈快速转移**的领域：每当你解决了当前最贵的环节，下一个环节就成为新的最贵，范式随之更替。掌握这条主线，你可以：

1. 判断新技术该归位到哪个瓶颈上（是真创新还是旧瓶新酒）；
2. 预测下一步（当前最贵的是什么？谁在解它？）；
3. 把任何技术点讲成一个有起承转合的故事。

一条总规律贯穿本章：

> **显存容量 → 显存带宽 → 计算通信重叠 → 集群级资源效率 → workload 结构变化**

---

## 20.2 Serving 范式编年史 ★★

### 20.2.0 时间轴总览

```
2019        2022          2023            2024                2025              2026
 │           │             │               │                   │                 │
 Faster     Orca          vLLM            SGLang   Sarathi    DeepSeek-V3       KV Pool 化
 Transformer (OSDI)      (SOSP)          RadixAttn (OSDI)    Wide-EP/R1        llm-d/Dynamo
 静态批      连续批         页式KV           前缀复用   chunked    PD分离成熟         agentic/RL
 (前史)      调度                          DSL→引擎  prefill                      负载主导
                                        DistServe/Splitwise/Mooncake(PD分离三部曲)
```

### 20.2.1 前史：FasterTransformer 与静态批时代（2019-2022）

NVIDIA FasterTransformer 是 GPT-3 时代的推理标配：手写 CUDA/GEMM 优化 + **静态 batch**——凑够一批或等超时，整批一起生老病死。短板批（straggler）拖累全批，GPU 利用率低。当时的共识是"推理就是个 GEMM 优化问题"。

### 20.2.2 Orca (OSDI 2022)：连续批调度的真正源头 ★

Orca 提出两个沿用至今的思想：

1. **Iteration-level scheduling**：调度粒度从"请求"细化到"模型 forward 迭代"，每步都可以让请求进出 batch（即 continuous batching，第1章 §1.3.2 的出处）；
2. **Selective batching**：attention 逐请求处理、GEMM 部分 合并成大批。

有趣的历史细节：Orca 发表于 ChatGPT 问世前两个月，当时 LLM serving 还是冷门方向，影响力有限；ChatGPT 引爆后大家才发现它早已给出答案。**教训：系统论文的价值常常滞后于负载的爆发。**

### 20.2.3 vLLM / PagedAttention (SOSP 2023)：显存革命 ★

vLLM 把 OS 的虚拟内存思想搬进 KV Cache：定长 block + 页表映射，外部碎片趋近于零，写时复制支撑 beam search 共享。配合 continuous batching 开源发布，恰逢其时地成为 ChatGPT 浪潮的第一个标准答案。社区飞轮自此启动：模型支持广度 → 用户量 → 贡献者 → 更快修复。

### 20.2.4 TGI 的先发与失落 ☆

HuggingFace TGI（2023 春发布）其实早于 vLLM 进入生产视野，Rust router + Python worker 的工程质量很高。但输在两点：闭源倾向的核心 kernel 与**迭代速度**——vLLM 以周为单位吸收社区优化，TGI 跟随吃力。生态战争的经典案例：**先发不重要，迭代飞轮才重要**。

### 20.2.5 SGLang：从 DSL 到引擎的转型 (NeurIPS 2024) ★

SGLang 起点（2024 初）是一个**前端 DSL**：用结构化程序表达 prompt 交互（fork/join/select）。真正的转折点是 RadixAttention——把 KV Cache 组织成基数树，任意请求间的公共前缀自动共享。随后团队果断收缩前端、全力做引擎（overlap 调度、zero-overhead scheduler、FlashInfer 后端），完成从"编程语言项目"到"高性能引擎"的身份转换。这段历史解释了 SGLang 至今保留的独特气质：对结构化/复用类优化的执念。

### 20.2.6 Chunked Prefill (Sarathi-Serve, OSDI 2024) ☆

长 prompt 一次 prefill 会阻塞同 batch 的 decode 步（TTFT 与 TPOT 的干扰）。Sarathi-Serve 把 prefill 切块、与 decode 混排在同一迭代里，用"token 预算"平滑每步耗时。今天 vLLM/SGLang 的 `--enable-chunked-prefill`/默认行为皆源于此。

### 20.2.7 PD 分离三部曲 (ISCA/OSDI/FAST 2024-25) ★

三篇论文几乎同时从不同角度论证同一件事——prefill 和 decode 是资源画像相反的两个相，应该物理分开：

| 论文 | 会议 | 关键贡献 |
|------|------|----------|
| **Splitwise** (Microsoft Research India) | ISCA 2024 | Phase splitting + 好put 分析；用 PCIe/NVLink/IB 做 KV 传输的硬件对比 |
| **DistServe** (PKU+UCSD) | OSDI 2024 | 提出 **goodput** 双 SLO 目标函数；P/D 独立扩缩容 |
| **Mooncake** (Moonshot AI) | FAST 2025 | **KV cache 为中心**的全局架构：KV 复用跨请求、全局调度器、RDMA 传输引擎（实测见 第24章 Finding 6）；Kimi 生产验证 |

此后 PD 分离从论文走进生产默认选项（TRT-LLM Dynamo、SGLang PD、llm-d），P:D 配比成为新的容量规划参数。

### 20.2.8 Wide-EP 与 MoE 大规模服务 (DeepSeek, 2025) ★

DeepSeek-V3 技术报告附录公开了自家部署：prefill 集群 EP32、decode 集群 EP144，配 DP attention 解决 MLA 的 KV 重复问题（第24章 Finding 1/2）。SGLang/vLLM 社区在 2025 上半年将其工程化为可复现的 wide-EP 方案。这标志着**并行拓扑第一次由推理侧需求反向定义**（此前 TP/PP 都来自训练侧经验）。

### 20.2.9 KV Pool 化与平台层崛起 (2025-2026) ☆→★

KV cache 从"引擎内部实现细节"升级为**分布式一等公民资源**：LMCache/NIXL 抽象传输与存储，Dynamo 编排多引擎，llm-d 把 KV-aware 调度标准化进 K8s Gateway API。同时 agentic/RL 负载（第19章）成为流量主体，会话粘性、权重热更新、partial rollout 变成引擎必备 API。竞争重心明确上移到平台层——这就是 第12章 存在的理由。

### 20.2.10 因果链小结

```
静态批浪费算力 ──► Orca 连续批调度
KV 碎片浪费显存 ──► PagedAttention
重复前缀浪费计算 ──► RadixAttention
prefill 干扰 decode ──► chunked prefill / PD 分离
单机资源见顶 ──► wide-EP / KV pool / 平台层编排
workload 变长变多轮 ──► 会话级状态管理（进行中…）
```

---

## 20.3 Kernel 栈编年史 ★

```
2017      2019        2021         2022            2023        2024           2025-
CUTLASS开源 cuBLAS/cuDNN Triton开源   FlashAttention  FA-2        FA-3/FlashInfer 引擎自研kernel
(模板GEMM) 黑盒王朝     (Python DSL) (NeurIPS,IO-aware) (并行度重排) (Hopper特性)    MLIR/tilelang
```

- **黑盒时代**：cuBLAS/cuDNN 提供 GEMM/融合算子，但 attention 这种动态 shape、内存密集的算子无处安放；
- **CUTLASS**（2017 开源）：C++ template 把 GEMM 的 tiling/swizzle 参数化，成为后续一切自定义矩阵 kernel 的地基；
- **FlashAttention**（NeurIPS 2022）：IO-aware 的范式宣言——不减少 FLOPs，只减少 HBM 往返（online softmax 两遍改一遍）。训练侧先引爆，推理侧随后跟进；
- **FA-2**（2023）：重排并行度（seq 维并行）、减少非 matmul FLOPs；**FA-3**（NeurIPS 2024）：拥抱 Hopper 特性（TMA 异步拷贝、warp specialization、pingpong 调度），把"跟硬件走"写到极致；
- **Triton**（OpenAI，2021 开源）：用 Python 写接近 CUTLASS 性能的 kernel，大幅降低门槛 → torch.compile 的底座、引擎自定义 op 的首选语言。没有 Triton 就没有后来社区 kernel 的繁荣；
- **FlashInfer**（2024）：统一 page-KV/各种变体 attention 的 runtime 库，被 SGLang 设为默认后端——kernel 从"逐个手写"走向"库化运营"；
- **当前前沿**：引擎自带 kernel 团队（vLLM 自研 attention backend、TRT-LLM 全家桶封闭栈）、编译器生成路线（MLIR 生态、tilelang、Mirage supoptimizer）。

**规律**：kernel 栈的每次跃迁都紧跟新一代硬件的特性暴露（Ampere tensor core → Hopper TMA/warp-spec → Blackwell tcgen05），**软件永远在追赶硬件的抽象泄漏**。

---

## 20.4 量化编年史 ★

| 时间 | 技术 | 意义 |
|------|------|------|
| 2019-2022 | INT8 量化传统 / LLM.int8() | outlier 问题初现，8bit 权激活勉强可用 |
| ICLR 2023 | **GPTQ** | 基于 OBQ 近似，单卡数小时量化 175B；W4A16 权重-only 路线确立 |
| ICML 2023 | **SmoothQuant** | 把激活 outlier 数学迁移进权重，W8A8 可行 |
| MLSys 2024 | **AWQ** | 激活分布感知缩放保护显著权重，W4 精度反超 GPTQ，成为社区默认 |
| 2024 | **FP8 (E4M3/E5M2)** | Hopper 原生 dtype；DeepSeek-V3 用 FP8 完成 671B 训练是里程碑——FP8 从"推理技巧"升格为"训练格式" |
| 2025 | **MXFP4 / NVFP4** | Blackwell 微缩块格式（block scaling）；gpt-oss 发布即原生 MXFP4——**模型出厂即量化** |
| 持续演进 | KV cache 量化 | FP8 KV → INT4 + outlier 通道保留；与稀疏注意力组合 |

**规律一**：量化精度跟着硬件 dtype 走——INT 时代靠软件硬扛 outlier，FP8/FP4 时代硬件直接给格式。
**规律二**：量化的主战场从权重（省显存）转移到 KV cache（省带宽）——因为瓶颈转移了（呼应 §20.1 总规律）。

---

## 20.5 互联标准战争 ☆→★

推理集群的形态将由 scale-up（域内高带宽）与 scale-out（域间以太网）的边界决定，这场标准之争 2023-2025 年白热化：

### NVLink 一条线（封闭但快）

| 代际 | GPU | 单卡带宽 | 备注 |
|------|-----|---------|------|
| NVLink 1 | P100 (2016) | 160 GB/s | 首次替代 PCIe 做 GPU 直连 |
| NVLink 3 | A100 (2020) | 600 GB/s | NVSwitch 全互联拓扑 |
| NVLink 4 | H100 (2022) | 900 GB/s | NVSwitch 二层组网 |
| NVLink 5 | B200/GB200 (2024) | 1.8 TB/s | **NVL72：72 卡一个 scale-up domain**，聚合 130 TB/s |

### 挑战者们

- **Ultra Ethernet Consortium（UEC）**：2023 年中成立（AMD/AWS/Azure/Google/Meta/Microsoft 等），spec 1.0 于 2025-06 发布。目标：改造以太网 RDMA（多路径、乱序容忍、新型拥塞控制）满足 AI 集群需求——守住 **scale-out** 主场；
- **UALink**：AMD/Broadcom/Google/HPE/Intel/Meta/Microsoft 阵营，spec 1.0 于 2025-04 发布。交换式 scale-up fabric，最多 1024 加速器内存池化——试图在 NVIDIA 之外再造一个 scale-up 标准；
- **NVLink Fusion**（2025-05 Computex）：NVIDIA 半开放策略——允许第三方 CPU/ASIC（联发科、Marvell、Fujitsu 等）接入 NVLink domain。开放的是生态，锁定的是协议。

**判断框架**：
- scale-up domain 内：带宽延迟敏感（TP/wide-EP AllToAll）→ NVLink 目前无可替代，UALink 是期权；
- domain 间：PD 分离的 KV 传输、EP 跨域 → 以太网阵营（UEC + RDMA/RoCE）成本与生态占优；
- 真正的问题是 **domain 边界画在哪**：NVL72 把过去要 scale-out 的拓扑收编进 scale-up，这是 2025 年最重要的架构事实（GB200 上跑 wide-EP 与在 8 卡机上跑是完全不同的算法选择）。

---

## 20.6 引擎生态兴衰速写 ☆

| 引擎 | 兴起 | 现状启示 |
|------|------|----------|
| FasterTransformer | 2019-2021 王者 | 技术被继承，产品已谢幕：不做服务化没有未来 |
| TGI | 2023 先发 | 输给迭代速度而非技术 |
| vLLM | 2023 登顶 | 社区飞轮 = 最深的护城河 |
| SGLang | 2024 差异化 | 敢砍第二曲线（DSL）聚焦引擎内核 |
| TRT-LLM | 持续 | 封闭极致性能路线：绑定硬件卖点的商业选择 |
| Dynamo / llm-d | 2025 起 | 引擎商品化后，价值向编排层转移 |

---

## 20.7 大事年表（速查总表）

| 年份 | 事件 | 类别 |
|------|------|------|
| 2017 | Transformer 论文；CUTLASS 开源 | 架构/Kernel |
| 2019 | FasterTransformer | Serving 前史 |
| 2020 | GPT-3；ZeRO | 规模化 |
| 2021 | Megatron-LM TP/PP 训练论文；Triton 开源 | 并行/Kernel |
| 2022 | **Orca (OSDI)**：iteration-level scheduling | Serving |
| 2022 | FlashAttention (NeurIPS)；ChatGPT (11月)；safetensors | Kernel/行业/存储 |
| 2023 | GPTQ；SmoothQuant；llama.cpp/GGUF | 量化/端侧 |
| 2023 | **vLLM (SOSP)** PagedAttention；TGI | Serving |
| 2023 | UEC 成立 | 互联 |
| 2024 | SGLang RadixAttention；FA-3；FlashInfer | Serving/Kernel |
| 2024 | Sarathi-Serve、DistServe (OSDI)；Splitwise (ISCA)；Mooncake 论文；S-LoRA (OSDI) | Serving/多租户 |
| 2024 | o1 预览（test-time compute）；XGrammar；Blackwell 发布 (GTC) | Workload/结构化/硬件 |
| 2024-12 | DeepSeek-V3（MLA/MTP/FP8 训练/wide-EP 部署附录） | 模型+部署 |
| 2025-01 | DeepSeek-R1 开源；UEC spec 1.0 (06)；UALink spec 1.0 (04)；NVLink Fusion (05) | Workload/互联 |
| 2025 | Mooncake (FAST)；veRL (EuroSys Best Paper)；Dynamo (GTC)；llm-d (05) | RL infra/平台层 |
| 2025 | gpt-oss 原生 MXFP4；NSA/DSA 稀疏注意力 | 量化/注意力 |
| 2026 | KV pool / agentic 负载主导；HBM4 与下一代 rack-scale 落地在途 | 进行中 |

> 注：论文发表年份与会议年份可能相差一年，以 arXiv 首发 + 正式会议双轨记录；2026 条目为撰写时点的行业共识预测，时效性更新见各章对应章节。

---

## 20.8 本章小结

三条可以带走的元认知：

1. **瓶颈转移驱动范式更替**：显存碎片（PagedAttention）→ 前缀冗余（RadixAttention）→ 相间干扰（PD 分离）→ 集群资源效率（wide-EP/KV pool）→ workload 结构（agentic）。下一个瓶颈大概率在**会话级状态的分布式管理**与**decode 主导下的能耗**；
2. **软件追赶硬件**：Kernel 栈与量化格式的每个台阶都是新一代硬件特性暴露的结果。看懂 roadmap（HBM4、scale-up 扩大、FP4 普及）就能预判两年后的软件栈；
3. **生态战争赢家通吃迭代速度**：TGI vs vLLM、DSL vs 引擎、封闭 vs 开放互联——历史反复证明，先发优势和单点性能都不如迭代飞轮与生态位置持久。

### 技术叙事模板

> "这个问题的本质是 XX 瓶颈。它在 20XX 年由 YY 系统首先解决，思路是 ZZ；但到了 20XX 年，因为 workload 变成 WW，这个方案的新短板是……所以现在业界的做法是……我们团队在其中的取舍是……"

→ 交叉复习：各技术的原理细节回对应章节；最新进展时效性声明见 第24章 §24.6。
