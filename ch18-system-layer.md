# 第18章 引擎之上 — 网关、调度与集群系统层

> **面向角色**：AI Infra 平台工程师、SRE、K8s 平台研发  
> **前置知识**：第0章-2（引擎内部原理）、基本 Kubernetes 概念  
> **目标**：掌握推理引擎"之上"的系统层——GPU 资源调度、推理网关与请求路由、多租户服务、模型分发、可观测性与集群容错；建立对 2024-2026 "Inference-First Cloud" 技术版图的完整认知

---

## 18.1 为什么引擎不是终点 ★

vLLM/SGLang 解决的是**单副本内**的问题：一个引擎进程如何高效吃满一张（或一组）卡。但生产系统的核心难题几乎都在**副本之间**：

```
用户请求
   │
   ▼
┌───────────────────────────────────────────────────────────┐
│ 推理网关层：鉴权/限流/协议适配/语义缓存/SLO 路由            │  ← 本书此前未覆盖
└───────┬───────────────────────────────────────────────────┘
        │
┌───────▼───────────────────────────────────────────────────┐
│ 路由与编排层：cache-aware 调度 / PD 分离编排 / 弹性伸缩     │  ← 本书此前未覆盖
└───────┬───────────────────────────────────────────────────┘
        │
┌───────▼──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 引擎副本 A (vLLM)     │  │ 引擎副本 B    │  │ D 池 (SGLang)│  ← 第3章/4 已覆盖
│ Scheduler/KV/Executor│  │ ...          │  │ ...          │
└───────┬──────────────┘  └──────┬───────┘  └──────┬───────┘
        │                        │                 │
┌───────▼────────────────────────▼─────────────────▼───────┐
│ K8s 层：DRA / gang scheduling / GPU 共享 / 拓扑亲和        │  ← 本书此前未覆盖
├───────────────────────────────────────────────────────────┤
│ 存储面：模型分发 / 权重加载 / KV Cache 分层存储             │
└───────────────────────────────────────────────────────────┘
```

**行业判断（2025-2026）**：随着 vLLM 与 SGLang 在能力上快速趋同（互相移植对方的核心特性），**单机引擎正在商品化，差异化竞争上移到网关与编排层**——llm-d（Red Hat/Google 等发起）、NVIDIA Dynamo 都押注这一层。理解本章的内容，是 2026 年之后 AI Infra 岗位能力的区分项。

---

## 18.2 GPU 集群资源调度（K8s 层）★

### 18.2.1 Device Plugin 与它的局限

K8s 原生不认识 GPU。`nvidia-device-plugin` 通过 Extended Resource 把 GPU 变成可计数的整数资源（`nvidia.com/gpu: 8`）：

```yaml
resources:
  limits:
    nvidia.com/gpu: 1   # 只能整卡分配，仅此而已
```

**三大局限**：
1. **纯计数器**：不知道 GPU 在哪张 NVSwitch 域里、NUMA 亲和性如何，可能把 TP=8 的 8 个 Pod 调度到跨 PCIe switch 的两台机器上；
2. **无法表达约束**："这 4 张卡必须在同一 NVLink 域"这类需求无法声明；
3. **分配即静态**：不支持运行期重新配置（如 MIG 切分粒度变更需重启 Pod）。

### 18.2.2 DRA：Dynamic Resource Allocation ☆→★

DRA 是 K8s 社区对上述问题的重构（alpha 自 1.26，beta 进入 1.32+，目标取代 device plugin 处理 GPU 类复杂资源）。核心对象：

| 对象 | 角色 |
|------|------|
| `ResourceSlice` | 厂商 driver 上报设备清单及**属性**（拓扑、显存、MIG profile） |
| `ResourceClaim` | 用户声明需求："4 台设备，同一 NVLink domain" |
| `DeviceClass` | 设备类别模板 |
| 调度器内嵌 allocator | **调度时感知拓扑做分配**，而非事后绑定 |

对推理平台的意义：PD 分离、Wide-EP 这类需要**成组、有拓扑约束**的部署终于可以在 API 层表达，而不是靠自研调度器打补丁。

### 18.2.3 Gang Scheduling：Volcano / Kueue ★

- **Volcano**（CNCF）：引入 PodGroup 概念，All-or-nothing 调度。PD 分离架构中 P/D Pod 必须成组就位；RL 训练（见 第19章 §19.4）的 trainer + rollout engine 组更是典型 gang 场景。
- **Kueue**：K8s 原生队列 + 配额管理，负责"排队等待配额"而不是"抢到一半挂起"。适合多团队共享集群的推理平台做公平性。

**选型经验**：单模型常驻服务用原生 scheduler 即可；**多模型混部 + PD 分离 + RL 混训**的平台几乎都要 Volcano 或 Kueue 之一。

### 18.2.4 GPU 共享：MIG / MPS / Time-slicing / vGPU ★

不是所有场景都值得整卡独占。四种共享方式对比：

| 方式 | 隔离层级 | 显存隔离 | 算力 QoS | 故障域 | 适用 |
|------|---------|---------|---------|--------|------|
| **MIG** (A100+) | 硬件分区（最多 7 实例） | ✅ 物理 | ✅ | 互不影响 | 多租户小模型、数据安全要求高 |
| **MPS** | 进程空间共享 | ❌（逻辑限额） | 部分（SM 百分比） | 一损俱损 | 同信任域内提高利用率 |
| **Time-slicing** | 时间片轮转 | ❌ | ❌ | 一损俱损 | 开发测试，生产慎用 |
| **vGPU** | Hypervisor 虚拟化 | ✅ | ✅ | 可隔离 | VM 化企业环境、云桌面 |

**推理场景要点**：
- MIG 对**延迟敏感的小模型多租户**是利器（7B 以下 INT4 单实例放得下），但对 vLLM 这类预分配大块显存的引擎要调低 `gpu_memory_utilization` 到该实例容量内；
- 大模型（70B+/MoE）没有共享问题——整卡甚至整机都是最小单元；
- 共享会破坏 CUDA Graph 重放的可预测性，延迟敏感 SLA 下慎用 MPS/time-slicing。

### 18.2.5 拓扑感知调度 ★

TP=8 的请求**必须**落在同一 NVSwitch 域（GB200 NVL72 内则是同一 scale-up domain），否则 AllReduce 走 IB/PiCle 吞吐崩塌。工程实践：

1. 用 Node Feature Discovery（NFD）上报机器拓扑（GPU-NVLink 邻接矩阵、NIC rail）；
2. 调度器按拓扑标签做亲和（或等 DRA 拓扑感知 GA）；
3. PD 分离还要加一层：P 池与 D 池之间保证 RDMA 直连路径（同 spine 或 rail-optimized 拓扑），KV 传输带宽直接决定 TTFT（参考 第2章 §2.3、第6章 Finding 6 的 87-190 GB/s 实测）。

---

## 18.3 推理网关与请求路由 ★

### 18.3.1 协议面：OpenAI-compatible 事实标准

2023 年之后，`/v1/chat/completions` 成为事实标准 API——所有主流引擎（vLLM/SGLang/TGI/TRT-LLM）都提供兼容层。网关层的协议职责：

- **SSE 流式**：token 级 chunked 输出，注意 `finish_reason`、usage 字段在最后一个 chunk；
- **工具调用流式**：function call 参数是增量 JSON，需要 partial JSON 解析（见 §18.5.2）；
- **Chat template 归属**：模板（jinja）由谁渲染？裸引擎模式由客户端渲染 `/v1/completions`，chat 模式由引擎渲染 `/v1/chat/completions`。**多模型平台上模板地狱是真实痛点**：每个模型的 system/tool 格式都不同，网关通常需要维护 per-model 渲染策略与回归测试集。

### 18.3.2 缓存、限流与公平性

- **Semantic cache**：对相同/相似 prompt 直接返回缓存答案。收益大（省整个 prefill+decode），风险是语义误命中与个性化失效——只适合 FAQ 型负载，且必须带 TTL 与命中率监控；
- **Token-based rate limiting**：按 input/output token 而非请求数限流（一次请求成本差异可达千倍）；令牌桶按 TPM/RPM 双维度；
- **多租户公平性**：大客户长 prompt 不能饿死小客户——按租户加权公平排队（WFQ），配合引擎侧的 priority/preemption（vLLM V1 priority scheduling）实现端到端分级。

### 18.3.3 Cache-aware / SLO-aware Routing ★

这是 2025 年推理网关最重要的演进。传统 L7 LB 按连接数轮询，对 LLM 是灾难——它完全无视 prefix cache：

```
轮询路由：  会话 A 第 5 轮 → 副本 3（prefix 全 miss，重新 prefill）
cache-aware：会话 A 第 5 轮 → 副本 1（radix tree 命中前 30k token）
                              → TTFT 从秒级降到百毫秒级
```

关键概念与系统：

| 系统/概念 | 核心思路 |
|-----------|----------|
| **Goodput**（DistServe 提出） | 同时满足 TTFT 与 TPOT 双 SLO 约束的有效吞吐。路由和扩缩容都应以 goodput 为目标函数，而非裸 QPS |
| **Session affinity / sticky routing** | 同一会话哈希到同一副本，最大化 radix/prefix 命中（代价是负载可能倾斜，需与最少负载策略混合） |
| **SGLang router** | cache-aware 负载均衡器：维护各副本 RadixTree 的近似视图做匹配度感知路由 |
| **NVIDIA Dynamo** | KV-aware routing + PD 编排，把 vLLM/TRT-LLM/SGLang 当作可插拔 backend |
| **llm-d + Gateway API Inference Extension** | 把 EPP（Endpoint Picker，基于 prefix hash 精确匹配）标准化进 K8s Gateway API，社区路线 |

**技术叙事**：负载均衡的目标函数从「均衡连接」→「均衡 QPS」→「最大化 goodput」的三级跳，本质是因为 LLM 服务中**请求之间不再独立**（prefix cache 使历史状态有价值），这是与传统微服务最本质的区别。

### 18.3.4 多模型服务形态

- **单 endpoint 多模型**：网关按 `model` 字段路由到不同副本池，统一鉴权/计量/计费；
- **灰度与回滚**：模型也是软件——canary 流量切分、A/B 评测指标挂钩 goodput 而非成功率；
- **Serverless GPU**（Modal/RunPod 类）：冷启动 = 镜像拉取 + 权重加载 + CUDA Graph 捕获，权重加载是大头（见 §18.6），快照/预热是核心竞争力。

---

## 18.4 多租户 LoRA 服务 ☆→★

**问题**：平台上有几百上千个领域微调 adapter，每个单独部署一份 base model 成本不可接受。目标是：**一份 base model 副本同时服务 N 个 LoRA**。

| 系统 | 发表 | 核心技术 |
|------|------|----------|
| **S-LoRA** | OSDI 2024 | Unified Paging：KV cache 与 adapter 权重统一分页管理，adapter 权重常驻 host memory 按需换入；异构批（不同 adapter 的请求同 batch） |
| **Punica** | arXiv 2023 | BGMV kernel：把不同 rank 的多个 LoRA GEMV 融合成一个 kernel，消除小矩阵低效 |
| **LoRAX** | Predibase 开源 2024 | 生产化实现：adapter 热加载 + 分层换入换出，支撑万级 adapter |

**工程要点**：
- adapter 切换成本决定调度策略——热门 adapter 常驻显存、长尾走 host memory swap（类似 OS 页面调度）；
- 与 continuous batching 结合时，batch 内允许不同 request 绑定不同 adapter（SGLang/V1 均已支持 LoRA batching）；
- 收益量级：S-LoRA 论文声称同等 GPU 上服务的 adapter 数量提升数十倍，吞吐损失个位数百分比。

---

## 18.5 结构化输出基础设施 ★

Agent 与工具调用时代，**输出必须是合法 JSON/语法树**不再是可选项。这催生了引擎内的一个新子系统：constrained decoding。

### 18.5.1 Constrained Decoding 原理

```
用户给 schema/grammar
      │
      ▼
编译期（一次性）：JSON Schema → CFG → FSM/token mask 缓存
      │
      ▼
解码期（每步）：根据当前状态查 mask → 不合法 token 的 logits 置 -inf → 采样
```

代表实现对比：

| 实现 | 出处 | 特点 |
|------|------|------|
| **Outlines** | dottxt (.txt) | 学术开创者之一；FSM 编译，早期版本编译慢 |
| **Guidance / llguidance** | Microsoft | llguidance 高性能 Rust 实现，被多家采纳 |
| **XGrammar** | SGLang 团队等，2024 | CFG 预处理 + adaptive token mask 缓存，编译快数量级；**已被 SGLang/vLLM/MLC 采用为默认** |

**为什么 XGrammar 快**：传统方法对每个上下文状态重新计算合法 token 集；XGrammar 把语法编译为可复用的自适应 mask 序列，大部分步骤只做查表。

### 18.5.2 Function Calling 的工程现实

- **格式分裂**：Hermes style、OpenAI strict mode、各家私有 token 约定……网关/引擎需要 per-model parser 注册表；
- **流式 partial JSON**：工具参数要在生成中途就开始下发（前端体验），需要容错的增量 JSON 解析器；
- **结构化通道趋势**：gpt-oss（2025-08）的 harmony 格式把 analysis/final/tool 通道编码进输出序列——结构化输出从"外挂约束"走向"模型原生格式"，引擎必须原生解析。

### 18.5.3 性能账

- mask 计算/应用发生在采样热路径上，劣质实现可直接吃掉 10%+ TPOT；
- batch 内 constrained 与 unconstrained 请求混跑时，kernel 选择分支会影响 CUDA Graph 复用；
- 实践：优先用引擎内置实现（SGLang xgrammar backend / vLLM guided decoding），压测带 schema 的真实负载而非只测自由文本。

---

## 18.6 模型分发与存储面 ☆

### 18.6.1 权重格式简史

| 格式 | 时间 | 关键点 |
|------|------|--------|
| PyTorch `.bin` (pickle) | — | **反序列化即可执行任意代码**，供应链攻击重灾区 |
| **safetensors** | HF, 2022 | 无代码执行、header 与张量分离、天然支持 mmap 零拷贝加载，现为主流 |
| **GGUF** | llama.cpp, 2023 | 单文件自描述（元数据+量化张量），面向端侧分发 |

### 18.6.2 分发链路优化

大规模集群每天拉取数百 GB × 数十节点的权重，HF Hub 直连必然成为瓶颈：

```
对象存储 (S3/OSS/HF)
      │  P2P 分发（Dragonfly 等，节点间互传）
      ▼
节点本地 NVMe 缓存（跨 Pod 复用）
      │  safetensors mmap / lazy loading
      ▼
进程地址空间 → 引擎按需触页
```

- **hf_transfer / Xet 后端**：HF 侧的多线程下载与内容定义分块去重（相似模型版本间只传 delta）；
- **弹性伸缩的隐性成本**：扩容一个 70B FP8 副本 ≈ 加载 ~70GB 权重，NVMe 顺序读也要数秒到数十秒——**HPA 扩容速度的上限往往是存储而非 GPU**；
- 预热手段：DaemonSet 预拉镜像与权重、调度前 hook 预热 page cache。

---

## 18.7 基准测试标准化 ☆

第8章 讲了 benchmark 方法论，这里补**行业标准**维度：

- **MLPerf Inference**（MLCommons）：数据中心/边缘两大类，四种场景（SingleStream/MultiStream/Server/Offline），closed division（固定预处理保可比）vs open division（自由优化秀肌肉）。GPT-J（2023）起纳入 LLM，后续 Llama-2/3 系列成为主力 loadgen 模型。看 MLPerf 提交可以了解头部厂商的真实调优水平；
- **Artificial Analysis**：第三方对 API 提供商的横评（延迟/吞吐/价格），选云服务商时的参考系；
- **Goodput 曲线**：学术界（DistServe）推动的呈现方式——x 轴 RPS，y 轴满足双 SLO 的比例，比单点 throughput 更接近采购决策视角。

---

## 18.8 可观测性标准化 ☆

- **OpenTelemetry GenAI semantic conventions**：`gen_ai.system`、`gen_ai.request.model`、token usage 等属性逐步统一（截至 2025 仍 experimental，但方向明确）。接入 OTel 后 LLM trace 可以与既有 APM 体系合流；
- **推理特有黄金指标**（在 第8章 §8.3 基础上网关侧补充）：
  - prefix/radix cache 命中率（路由质量的核心反馈信号）
  - preemption/retract 次数（过载先行指标）
  - 每 token 成本（$ / 1M tokens，分模型分租户）
  - queue time 与 TTFT 分解（排队 vs 计算）
- **trace 关联链**：request id → 网关路由决策 → 引擎 hop → kernel timeline（Nsight/DCGM），故障定位时能逐层下钻。

---

## 18.9 集群故障语义与容错 ★

### 18.9.1 GPU 不是可靠硬件

公开数据点：Meta 披露 Llama 3 级别训练（16k H100）平均**约每 3 小时遭遇一次意外中断**，其中 GPU 相关硬件故障占比最高。推理集群规模虽小，但要建立同样的心智模型：**故障是常态，预算内运维**。

XID 错误速查（完整表见 NVIDIA 文档，DCGM 自动采集）：

| XID | 含义 | 处置 |
|-----|------|------|
| 48 | double-bit ECC error | 隔离 GPU，检查 HBM |
| 63/64 | ECC page retirement | 观察频率，频发则替换 |
| 79 | GPU fell off the bus | 通常驱动/硬件问题，重启节点 |
| 94/95 | contained ECC error（MIG 内） | MIG 故障域隔离的价值所在 |

### 18.9.2 推理服务的容错设计

- **KV Cache 是易失状态**：副本挂掉 = 该副本上全部会话的前缀缓存蒸发。因此：
  - 会话粘性路由必须有**粘性失效**预案（重路由后首请求 TTFT 尖刺是预期行为）；
  - 关键业务可开启 KV 分层存储（HiCache/LMCache，第2章 §2.3.4）让 KV 在节点外存活；
- **优雅驱逐**：K8s drain 前先停止接收新请求 → 排空存量（或迁移会话）→ 退出。给引擎配 preStop hook；
- **优先级抢占链**：离线批处理 < 在线对话 < 关键业务，通过引擎 priority + K8s PriorityClass 两级实现；
- **混沌演练**：定期 kill 引擎副本、断 RDMA 链路、制造 XID，验证路由收敛与 SLO 恢复时间。

---

## 18.10 非 NVIDIA 硅片全景与能耗维度 ☆

第12章 覆盖了国产与边缘芯片，这里补国际非 NVIDIA 数据中心阵营的**架构路线分析**：

| 厂商 | 路线 | 核心取舍 |
|------|------|----------|
| **Groq LPU** | 确定性数据流 + SRAM-only（无 HBM），权重流水线式流过芯片阵列 | 延迟极低且**可预测**（无 cache miss 抖动）；代价：容量受 SRAM 限制，大模型需大量芯片组网，批处理受限 → 单位算力成本高 |
| **Cerebras WSE** | 晶圆级集成（WSE-3：90 万核，片上 44GB SRAM），Weight Streaming 从外部 MemoryX 流式供权重 | 免除 HBM 瓶颈，宣称极高 decode 速度；同样受限于"权重不在本地"的批处理上限 |
| **Tenstorrent** | RISC-V 核心 + 以太网原生互联，开放路线 | 软件生态尚在建设期 |

共同哲学：**放弃 HBM 容量换取确定性延迟与带宽** → 在"低延迟小 batch"细分场景（实时语音 agent 等）有独特价值，但在高吞吐大 batch 主战场难以撼动 GPU 经济学。

**能耗维度**（第13章 成本模型的物理底层）：
- 新指标：tokens/joule、tokens/watt——当集群进入 10 万卡级别（~150MW+），**电力而非芯片**成为扩张第一约束；
- 行业应对：选址跟电走（水电/核电 PPA）、液冷普及、以及推理侧特有的"每 token 能耗优化"（量化、speculative decoding 的能耗账）。

---

## 18.11 本章小结

| 层 | 关键技术 | 一句话 |
|----|----------|--------|
| K8s 资源层 | DRA / Volcano / Kueue / MIG / 拓扑感知 | 让"成组、有拓扑约束的 GPU"可被正确分配 |
| 网关层 | OpenAI-compatible / 限流 / cache-aware 路由 / goodput | 请求不再独立，路由必须懂 KV cache |
| 租户层 | S-LoRA / Punica / LoRAX | 一份 base model 服务千级 adapter |
| 输出层 | XGrammar / Outlines / partial JSON | 结构化输出成为引擎内置子系统 |
| 存储面 | safetensors / P2P / mmap | 扩容速度的上限常在存储 |
| 可观测 | OTel GenAI / 黄金指标 | cache 命中率与 preemption 是先行指标 |
| 容错 | XID 分类 / 粘性失效 / 混沌 | 故障是常态，设计故障半径 |
| 硅片 | Groq/Cerebras/Tenstorrent | 用 HBM 换确定性的另一条路线 |

### 高频问题

1. Device plugin 和 DRA 的本质区别？（计数器分配 vs 拓扑感知声明式分配）
2. 为什么 L7 负载均衡对 LLM 不适用？怎么改？（prefix cache 使请求相关 → session affinity + cache-aware + goodput 目标）
3. MIG 和 MPS 怎么选？（隔离性/故障域/QoS 三角度）
4. Constrained decoding 为什么会拖慢推理？业界怎么优化？（mask 在采样热路径 → XGrammar 编译期缓存）
5. 一个 70B 副本扩容要多久？瓶颈在哪？（权重分发与加载）

→ 交叉复习：引擎内部调度见 第2章 §2.1；KV 分层见 第2章 §2.3.4；benchmark 方法见 第8章 §8.1。
