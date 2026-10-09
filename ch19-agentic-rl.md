# 第19章 Agentic 与 RL 时代的推理负载

> **面向角色**：推理平台架构师、RL Infra 工程师、推理引擎研发  
> **前置知识**：第1章（KV Cache / 指标）、第3章（调度 / 投机解码）、第12章（网关与路由）  
> **目标**：理解 2024-2026 年 workload 的三次变迁（Chat → RAG → Agent/Reasoning）如何重构推理基础设施的设计假设；掌握 RL post-training 基础设施（rollout engine + weight sync）这一 SGLang/vLLM 社区最大的增量场景

---

## 19.1 Workload 三次变迁 ★

推理 infra 的每个设计决策都建立在"负载长什么样"的隐含假设上。2023 年以来这个假设被改写了三次：

| 维度 | Chatbot 时代 (2023) | RAG / 长上下文时代 (2023H2-2024) | **Agent / Reasoning 时代 (2024H2-)** |
|------|--------------------|----------------------------------|-----------------------------------|
| 典型 prompt | 数百 token | 数千~数十万 token | 数千~数万，**多轮累积** |
| 典型输出 | 数百 token | 数百 token | 数千~数万（长 CoT）/ 中间夹工具调用 |
| 会话结构 | 单轮为主 | 单轮 + 长 context | **多轮长会话**（数十次往返） |
| prefix reuse | 低 | 中（文档块复用） | **极高**（历史轮次全部可复用） |
| 瓶颈相 | prefill/decode 均衡 | prefill 重 | **decode 占比持续上升** |
| 敏感指标 | TTFT | TTFT（prefill 优化） | TTFT + TPOT 双敏感，会话级体验 |
| 状态生命周期 | 请求级 | 请求级 | **会话级**（分钟~小时） |

**核心结论**：
1. decode 成为主体 → 引擎重心从 prefill 吞吐转向大 batch decode 效率（overlap 调度、投机解码回归）；
2. prefix reuse 从"锦上添花"变成**成本结构的决定项**；
3. 状态从请求级升级为会话级 → 路由、缓存、容错的所有假设都要重写（第12章 §12.3.3）。

---

## 19.2 Reasoning 模型与 Test-time Compute ★

### 19.2.1 范式本身

o1（OpenAI，2024-09 预览）与 DeepSeek-R1（2025-01 开源权重）确立了新扩展轴：**推理时多想，胜过训练时更大**。学术界同步给出理论支撑——test-time compute 最优分配研究（Snell et al., 2024）表明小模型 + 更多思考可以胜过大模型 + 直接回答。

对 infra 的直接后果：

```
传统 chat：    prompt 1k → output 0.5k     （decode 占比 ~33%）
reasoning：   prompt 2k → thinking 8k+ → output 1k
                          └── decode 占比 >80%，单请求生成 token ×10
```

### 19.2.2 对引擎的连锁影响 ★

1. **decode-bound 化**：集群算力预算从 prefill 向 decode 倾斜；PD 分离的 P:D 配比要重新调（D 侧需求上升）；
2. **并行采样（n>1）成常态**：best-of-N / self-consistency 让同一 prompt 的 N 个 decode 共享 KV → radix/prefix 复用价值放大，n-way 并行是引擎必答题；
3. **投机解码价值回归**：输出越长，speculative decoding 的绝对收益越大。EAGLE-3、MTP（DeepSeek 原生 MTP 权重）成为 reasoning 负载的标配（衰减数据见 第24章 Finding 4——收益随 batch size 下降，需按负载实测）;
4. **Thinking budget 控制**：Qwen3 等 model family 引入可开关/限长的思考模式——本质是把"算力换质量"的旋钮暴露给应用层，网关需要按租户/场景路由到不同 budget 配置；
5. **成本结构**：单请求 token 消耗上升 1-2 个数量级 → 第10章 的 ROI 模型中，KV cache 命中率与稀疏注意力（NSA/DSA，第3章 §3.4.4）从优化项升级为盈亏线。

---

## 19.3 Agent 会话的 KV Cache 生命周期管理 ★

### 19.3.1 问题重述

一次 agent 任务 = 一个长会话内嵌几十次工具调用：

```
turn 1: [system][tools][user]           → assistant + tool_call
turn 2: [system][tools][user][a1][tool_result] → assistant + tool_call
...
turn 30: [... 全部历史 ...]              → final answer
```

每一轮的前缀 = 上一轮的全部内容。若路由不当，每轮都全量重新 prefill——30 轮 × 10k token 的会话，浪费的 prefill 算力是命中缓存时的数十倍。

### 19.3.2 三层应对

**① Session Affinity / Sticky Routing**（网关层，见 第12章 §12.3.3）
- 按 session id 一致性哈希到引擎副本；
- 权衡：粘性导致负载倾斜（长会话副本越来越重），需要 cache-aware 与 least-load 混合策略，并设置粘性 TTL。

**② 会话暂停与恢复**（引擎层）
- 工具执行期间（秒级~分钟级），会话的 KV 占着显存等结果——高并发下这是最贵的闲置；
- vLLM sleep mode（`sleep()` 卸载权重/KV、`wake_up()` 恢复）最初为 RL rollout 内存相位切换设计，同样适用于 agent 场景的低峰回收；
- 更精细的做法：把不活跃会话的 KV offload 到 CPU/SSD（HiCache/LMCache 分层，第3章 §3.3.4），活跃会话留在 HBM。

**③ KV 生命周期的语义升级**
- 缓存条目从「请求结束即弃」变为「会话结束才弃」→ 需要 session 级 TTL 与显式失效 API；
- 计费与配额也要跟着变：按会话的累计 KV 占用计费开始出现。

### 19.3.3 实战参数速查（SGLang 为例）

```bash
# radix cache 是会话复用的基石（默认开启）
--disable-radix-cache            # 仅调试用，生产勿关
--page-size 64                   # 大页减少页表开销，利于长前缀匹配
--mem-fraction-static 0.9        # 留给 KV pool 的比例，agent 负载建议保守些
# 会话保持相关由 router 层控制（cache-aware LB）
```

---

## 19.4 RL Post-training Infra ★★（本章重点）

### 19.4.1 从 RLHF 到 RLVR

| 阶段 | 时间 | 方法 | infra 特征 |
|------|------|------|-----------|
| RLHF（InstructGPT） | 2022 | PPO + reward model | trainer 为主，生成量小 |
| GRPO（DeepSeekMath） | 2024 | 去 critic 的组相对优势 | 同一 prompt 采样一组 → **生成量开始爆炸** |
| RLVR（R1 及后继） | 2025 | 可验证奖励（代码/数学自动判分）+ 长 CoT | rollout 成为时间与成本大头 |

### 19.4.2 系统形态：Trainer + Rollout Engine

RL 循环的四类角色与两种部署形态：

```
        ┌────────────┐
        │  Sampler    │  同一 prompt 采 N 条轨迹
        ▼            │
┌───────────────┐   │   ┌──────────────┐   ┌─────────────┐
│ Rollout Engine │◄──┘   │ Reward / Verifier │  │ Replay Buffer │
│ (SGLang/vLLM) │──────►│ (规则/RM)          ├─►│ (轨迹存储)    │
└──────▲────────┘       └──────────────────┘   └──────┬──────┘
       │ weight sync                                   │
┌──────┴───────────────────────────────────────────────▼──┐
│ Trainer (FSDP/Megatron)：计算优势 → 梯度更新 → 新权重      │
└──────────────────────────────────────────────────────────┘
```

- **Collocated（共置）**：trainer 和 engine 共享同一组 GPU，训练阶段把引擎 sleep 掉腾出显存，rollout 阶段唤醒——内存相位切换，省卡但有空转；
- **Disaggregated（分离）**：独立 rollout 集群，通过网络同步权重——吞吐高但要解决权重分发带宽与 off-policy 窗口问题。

代表框架（截至 2026 初）：

| 框架 | 背景 | 要点 |
|------|------|------|
| **veRL** (HybridFlow) | ByteDance+CUHK，EuroSys'25 Best Paper | 把 RL 数据流抽象为 hybrid controller（单进程编程模型）；支持 FSDP/Megatron trainer + vLLM/SGLang rollout |
| **OpenRLHF** | 社区开源先行者 | Ray 编排 + vLLM rollout |
| **slime** | THU/Zhipu 系 | 以 SGLang 为一等公民，面向 agentic RL（长会话、工具调用训练） |
| **AReaL** | Ant+清华 | 异步 RL：rollout 与训练完全解耦流水线化 |

### 19.4.3 Rollout 为什么必须复用 SGLang/vLLM

RL 采样本质上就是推理服务的高并发压测：同一 prompt 采 N 条、temperature 高、输出长。自研 sampler 无法追平 continuous batching + radix cache + speculative decoding 的工程积累。因此 2025 年后的共识是：**rollout engine = 生产推理引擎本身**。这反过来要求推理引擎暴露训练友好的 API：

- 批量生成 + 指定 n/temperature/logprobs 返回；
- 多轮工具调用式采样（agentic RL 需要 env-in-the-loop）；
- partial rollout：超长轨迹在工具等待点暂停、跨 step 续跑；
- 权重热更新接口（下一节）。

### 19.4.4 Weight Sync ★

训练侧更新后的权重如何进入 rollout engine？三条路径：

| 方式 | 机制 | 适用 |
|------|------|------|
| **NCCL broadcast** | trainer 进程组直接把分片 broadcast 给 engine 的 NCCL group | veRL 默认路线之一；跨机高效，需要打通通信域 |
| **CUDA IPC** | 同节点内 GPU 显存零拷贝传递 | collocated 形态，最快但仅限单机 |
| **CPU/文件中转** | 存 safetensors 再 `update_weights_from_disk` | 简单通用，慢一个量级，适合原型 |

SGLang 对应 API：`update_weights_from_tensor`（进程内/IPC）、`update_weights_from_distributed`（NCCL）。要点：**只传 delta 且与 CUDA Graph 兼容**——权重 shape 不变的原地更新无需重建 graph，这是 RL 场景能高频换权的前提。

**Off-policy 窗口**：disaggregated 架构下 engine 用旧权重采样期间，trainer 已经前进若干步 → 需要 importance sampling 校正（PPO clip / GRPO ratio 天然容忍一定 staleness），窗口大小是吞吐-有效性的调节旋钮。

### 19.4.5 Async RL 与 Partial Rollout ☆

同步 RL（generate → wait all → train）中 rollout 长尾拖累整体利用率。异步化（AReaL/slime 路线）：

- rollout worker 持续从 replay buffer 取任务，与 trainer 流水线重叠；
- 陈旧度（staleness）显式管理：限制最大落后步数 + importance ratio 裁剪；
- **partial rollout**：agent 轨迹中的环境交互（沙箱执行、外部 API）耗时不定，允许轨迹挂起并在恢复后续跑——SGLang 的会话暂停能力（§19.3.2）在此直接复用。

### 19.4.6 训推混部的资源经济学

- RL 集群的 GPU 时间典型分布：rollout 占比常超过 50%，且随 agent 化继续上升；
- 在线 serving 与 RL rollout 混池：白天保在线 SLA、夜间/低峰跑 rollout，或用 MIG/partition 隔离小规模 rollout；
- 与 第10章 的 ROI 模型衔接：RL 后训练的成本核算单位应从"GPU 时"改为"每条轨迹的 rollout token 成本"。

---

## 19.5 协议层：MCP 与 Agent 接口标准 ☆

**MCP（Model Context Protocol）**：Anthropic 于 2024-11 开源的模型-工具连接标准，2025 年获 OpenAI/Google 等采纳，成为 agent 生态事实标准之一。

- 三原语：Tools（模型可调用）、Resources（可读取上下文）、Prompts（模板）；
- Transport：本地 stdio / 远程 Streamable HTTP（替代早期 HTTP+SSE 方案）；
- **对推理 infra 的意义**：工具生态标准化 → agent 会话的工具调用次数进一步暴增 → 服务端会话保持、流式部分 JSON（第12章 §12.5.2）、工具结果注入上下文的 token 计量，全部成为网关/引擎的一等需求。

与 OpenAI function calling 的关系：MCP 解决"工具从哪来"（生态协议），function calling 解决"工具怎么调"（模型输出格式），二者正交且正在融合（各家已支持 MCP 工具直连）。

---

## 19.6 对引擎与平台的综合影响

| 设计维度 | 旧假设（Chat） | 新假设（Agent/Reasoning） |
|----------|---------------|--------------------------|
| 调度目标 | 请求数公平 | 会话级 goodput + cache 命中率 |
| 缓存粒度 | 请求级 prefix | 会话级生命周期 + TTL + 失效 API |
| 路由 | 负载均衡 | sticky + cache-aware + 故障半径控制 |
| 解码优化 | prefill 吞吐 | 大 batch decode + 投机解码 + overlap |
| 输出处理 | 自由文本 | 结构化约束原生内置（XGrammar 等） |
| API 面 | generate | + 权重热更新 / 会话暂停恢复 / n-way 采样 |
| 邻居系统 | 无 | RL trainer 共享资源池 |

### 高频问题

1. Reasoning 模型为什么让 decode-bound 问题更严重？引擎层面如何应对？
2. Agent 多轮会话下，TTFT 的第一影响因素是什么？（上一轮 KV 是否还在原副本）
3. veRL 为什么选 SGLang/vLLM 做 rollout 而不是自己写 sampler？
4. Weight sync 三种方式的取舍？CUDA IPC 为什么最快、局限是什么？
5. Collocated vs disaggregated RL 的内存相位切换怎么实现？（sleep/wake up）
6. MCP 改变了推理服务的哪些假设？

→ 交叉复习：投机解码衰减数据 第24章 Finding 4；HiCache 分层 第3章 §3.3.4；sticky routing 第12章 §12.3.3。
