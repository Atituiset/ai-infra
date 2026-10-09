# 第5章 训练 Infra 速览 — 训推平台工程师的最小知识集

> **面向角色**：训推平台工程师（本书目标读者）、推理 Infra 工程师  
> **前置知识**：第3章 §3.2（TP/PP/EP）、第19章（RL infra）  
> **目标**：不做训练算法，但必须懂训练系统的**资源账、容错账和数据管道**——这是"训推平台"岗位与纯推理岗的分界线。深度刻意控制：够自测、够混部设计、够模型交付对接

---

## 5.0 时间发展线 ★

```
2016        2017         2019            2020       2021      2022                2023              2024                  2025
 │           │            │               │          │         │                   │                 │                     │
激活重计算   混合精度      Megatron TP /   ZeRO       FSDP      Alpa自动并行 +      长上下文CP         万卡生产体系 +        agentic RL
(sublinear  (Apex)       GPipe PP        (ZeRO-123) (PyTorch) TransformerEngine    (Ring Attention    TorchTitan +          训推一体
 memory)                                 DeepSpeed             (FP8训练起步)        / Ulysses)         异步ckpt(MegaScale)   (→第19章)
```

---

## 5.1 为什么推理工程师要懂训练 ★

三个硬理由：

1. **混部**：在线 serving 与离线训练共享 GPU 池（白天推理高峰、夜间训练/RL rollout），不懂训练负载画像无法做资源规划；
2. **RL post-training**：rollout 就是推理服务（第19章 §19.4），trainer 的显存相位切换、weight sync 都要求你懂训练侧在干什么；
3. **模型交付**：engineer 要接 checkpoint → 转换 → 部署的管道，必须看懂数值格式与分片方式。

---

## 5.2 显存账：训练比推理贵在哪 ★

以 7B 模型 + AdamW + 混合精度为例（每参数字节数）：

| 状态 | 字节/参数 | 7B 合计 |
|------|----------|---------|
| 权重 (BF16) | 2 | 14 GB |
| 梯度 (BF16) | 2 | 14 GB |
| 主权重 (FP32 master) | 4 | 28 GB |
| Adam m/v 动量 (FP32) | 8 | 56 GB |
| **静态小计** | **16** | **112 GB** |
| 激活值 | 与 batch×seq×hidden 成正比 | 剩余全部 |

对比推理（仅权重 14GB + KV cache）：**训练的静态状态是推理的 8 倍**——这就是为什么训练要 ZeRO/FSDP 切分而推理通常不用。

- **激活值优化 = 重计算（activation recomputation/checkpointing）**：前向不存中间激活、反向时重算，用 ~30% 额外计算换数倍激活显存；
- 长上下文下激活爆炸 → CP（context parallelism）登场。

---

## 5.3 并行策略：从训练视角回看 第3章 ★

推理篇讲的 TP/PP/EP 全部源自训练，但训练版多了两个维度：

| 维度 | 训练特有点 |
|------|-----------|
| **DP + ZeRO** | 推理的 DP 只是复制；训练的 DP 必须切状态——ZeRO-1 切优化器 / -2 加梯度 / -3 加权重（等价 FSDP） |
| **PP 调度** | GPipe micro-batch 流水 + bubble 问题 → PipeDream/1F1B（前后向交错）→ Megatron interleaved schedule |
| **TP** | 前向+反向各有一次 AllReduce（推理只有前向），通信压力翻倍 → 必须 NVLink 域内 |
| **CP/SP** | 序列维切分处理超长上下文：Ring Attention（KV 分块环形传递）、DeepSpeed Ulysses（attention head 维切分 + all-to-all） |
| **EP/MoE** | 训练侧还要解决 expert 负载漂移（loss balancing），推理只管调度 |

**记忆锚点**：训练 = 推理的通信 ×2 + 状态切分 + 流水调度。理解了 第3章，训练并行只需补这三个增量。

---

## 5.4 数值格式与混合精度 ☆

- **BF16 vs FP16**：FP16 需要 loss scaling 防下溢；BF16 指数位与 FP32 相同、免 scaling，是训练默认；
- **FP8 训练**：Transformer Engine 引入 per-tensor/per-block scaling + 两套格式（E4M3 前向 / E5M2 反向梯度）；DeepSeek-V3 把它推进到细粒度 tile-wise 量化 + 高精度累加，证明 671B 可 FP8 训练；
- **主权重恒为 FP32**：所有低精度格式的更新都累积到 FP32 master copy——这条不变式贯穿混合精度历史。

---

## 5.5 Checkpoint 与容错：有效训练时间经济学 ★

### 5.5.1 故障是常态

- Meta（Llama 3 报告）：16384 张 H100 上平均**约每 3 小时一次意外中断**，GPU 硬件问题占大头；
- Google PaLM 报告：类似量级的 flakiness 数据。
- 结论：万卡集群的可用性（availability）与 MFU 同等重要，**有效吞吐 = MFU × 时间可用率**。

### 5.5.2 工程三板斧

1. **异步 checkpoint**：checkpoint 写盘与训练重叠（PyTorch Distributed Checkpoint / 显存快照 + 后台持久化），同步方案动辄分钟级暂停；
2. **快速恢复**：分片 ckpt 直接按新拓扑恢复（弹性重分配）；优化器状态与权重同存；
3. **自动故障处置**：XID 监控（第12章 §12.9）→ 自动隔离节点 → auto-resume。人工介入每小时的代价 = 数千 GPU 时。

---

## 5.6 数据管道 ☆

预训练吞吐的隐形天花板常在数据面：

```
原始语料( CommonCrawl 等 )
  → 清洗（启发式 + 模型过滤）
  → 去重（精确 hash + MinHash LSH 近似去重）
  → shuffle + tokenize（Rust tokenizer 多进程，吞吐需 > 训练消耗）
  → 分片存储（parquet/jsonl shards）
  → 流式读取（WebDataset / StreamingDataset 类，随机化 + 预取）
```

- FineWeb 类流水线证明数据质量工程对最终模型质量的影响不亚于架构；
- **Infra 视角的自检指标**：dataloader 是否让 GPU 等待（GPU starvation 在 trace 里表现为 forward 之间的空隙）；tokenize 吞吐是否匹配集群消耗速度（十万卡级每秒数百亿 token）。

---

## 5.7 框架版图与选型 ★

| 栈 | 定位 | 适用 |
|----|------|------|
| **Megatron-LM** (+NeMo) | NVIDIA 官方极致性能，TP/PP/EP 全家桶 | 大厂预训练主力 |
| **DeepSpeed** | ZeRO 系统化 + 易用性 | 中大规模、学术与工业通用 |
| **FSDP / TorchTitan** | PyTorch 原生路线（DTensor、async tensor parallel） | PyTorch 生态默认选择，TorchTitan 是官方参考训练框架 |
| **Colossal-AI / Alpa 系** | 自动并行探索 | 研究与特定场景 |
| **veRL/slime/AReaL** | RL post-training（复用上述 trainer + 推理引擎） | 见 第19章 |

生产参考数据点：MegaScale（NSDI'24 Best Paper）披露字节万卡训练的 MFU 55%+ 及全套故障/拥塞治理；Llama 3 报告给出 16k H100、4D 并行的完整配置。**读这两篇胜过十篇二手总结**。

---

## 5.8 训推混部的资源经济学 ★

衔接 第10章/19：

- **时间维混部**：夜间低谷跑训练/RL，白天保在线 SLA——需要 gang scheduling 快速整组腾退（Volcano/Kueue，第12章 §12.2.3）；
- **空间维混部**：同一节点训练与 rollout 分时复用（collocated RL 的 sleep/wake，第19章 §19.4.2）或 MIG 隔离小规模实验；
- **核算口径统一**：训练按"有效 token 成本"、推理按"per-token 服务成本"，混部平台需要一个统一的 GPU 秒计价层。

---

## 5.9 论文线汇总 ★

完整对照见 第21章 线 10，核心八篇：

| # | 论文 | 出处 | 一句话 |
|---|------|------|--------|
| 1 | Training Deep Nets with Sublinear Memory Cost (`1604.06174`) | 2016 | 激活重计算起源 |
| 2 | Mixed Precision Training (`1710.03740`) | ICLR 2018 | 混合精度范式 |
| 3 | Megatron-LM (`1909.08053` / `2104.04473`) | 2019 / SC'21 | TP 与 PP interleaved |
| 4 | GPipe (`1811.06965`) | 2019 | micro-batch 流水线 |
| 5 | ZeRO (`1910.02054`) | SC 2020 | 状态分片三段论 |
| 6 | Ring Attention (`2310.01889`) / Ulysses (`2309.14509`) | 2023 | 长上下文 CP 双路线 |
| 7 | MegaScale (`2402.15627`) | NSDI'24 Best Paper | 万卡生产全栈披露 |
| 8 | The Llama 3 Herd of Models (`2407.21783`) | 2024 | 4D 并行 + 故障预算实录 |

---

## 5.10 本章小结

- 训练显存账：**16 B/param 静态状态**是推理的 8 倍，ZeRO/FSDP 因此存在；
- 训练并行 = 推理并行 + DP 状态切分 + PP 流水调度 + CP 序列切分；
- 有效训练时间 = MFU × 可用率，异步 ckpt + 自动故障处置是核心工程；
- 数据管道的 Infra 自检：别让 GPU 等 token；
- 与推理的交汇点就是 第19章：rollout 即推理服务。

### 高频问题

1. 7B 模型单卡训不动的原因？各状态多大？怎么切？
2. ZeRO-1/2/3 各切什么？和 FSDP 什么关系？
3. 为什么 BF16 取代了 FP16？loss scaling 解决什么？
4. 万卡训练一周会坏多少次？怎么保证有效时间？
5. 训推混部的调度难点？（gang + 相位切换 + 优先级）

→ 交叉复习：并行原理 第3章 §3.2；RL rollout 第19章 §19.4；调度基础设施 第12章 §12.2。
