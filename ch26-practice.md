# 第26章 动手实践手册（Lab Manual）

> **面向角色**：所有读者；尤其适合攒项目作品集、以及"读完了但没摸过"的工程师  
> **前置知识**：按各 Lab 的回指章节而定  
> **目标**：把全书概念变成可复现的实验。每个 Lab 按「目标 → 步骤 → 你应该观察到什么 → 思考题」组织——**观察点比命令重要**：命令会过时（以各工具 `--help` 为准），但"预期看到什么现象"是稳定的

## 26.0 使用约定

**硬件档位标记**：

| 标记 | 含义 |
|------|------|
| `[CPU]` | 笔记本/CPU 可跑（小模型 INT4） |
| `[1×24G]` | 单张 24GB 卡（RTX 3090/4090/A10） |
| `[2×GPU]` | 两卡以上（PD 分离等拓扑实验） |

**环境约定**：vLLM 与 SGLang 各建一个独立 Python 环境（依赖冲突频繁），版本钉死并记录在案；所有 Lab 的原始数据（命令、输出、曲线截图）建议沉淀成一篇复盘笔记——这就是项目作品集。

**Lab 与章节对照总表**：

```
Track A 度量基础     L0 冒烟  L1 压测画曲线  L2 看 batch 动态        → 第1章/8
Track B 内核观察     L3 显存账  L4 缓存命中  L5 投机解码实测          → 第2章/3/4
Track C 优化技术     L6 量化   L7 chunked prefill  L8 迷你 PD 分离   → 第2章/22
Track D 系统层       L9 Triton kernel  L10 结构化输出  L11 路由  L12 混沌 → 第18章/24
```

---

## Track A：度量基础

### L0 冒烟：把两个引擎跑起来 `[1×24G]`

**目标**：建立本地实验床；理解 OpenAI-compatible 接口的最小面。

```bash
# vLLM 侧
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.85 --max-model-len 8192
# SGLang 侧（另开环境）
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --mem-fraction-static 0.85

# 最小流式请求
curl http://localhost:8000/v1/chat/completions -d '{
  "model": "...", "stream": true,
  "messages": [{"role":"user","content":"用三句话解释 KV Cache"}]
}'
```

**你应该观察到**：
1. 启动日志里权重加载 → CUDA Graph 捕获两个阶段，后者占启动时间的相当比例（对应 第3章 §3.4.2）；
2. SSE 输出是一串 `data:` chunk，最后一个带 `finish_reason` 和 usage（对应 第18章 §18.3.1）；
3. 两个引擎的显存占用都远大于权重本身——差额就是 KV pool 预分配。

**思考题**：为什么两个引擎都要在服务前预分配 KV pool，而不是按需 malloc？（提示：碎片化正是 vLLM 论文要解决的起点）

### L1 压测：画出你的第一条 RPS-Latency 曲线 `[1×24G]`

**目标**：亲手复现 第8章 §8.1 的方法论；理解吞吐和延迟为什么是一对矛盾。

```bash
# 以 vLLM bench_serving 为例（SGLang 有同名工具）
python benchmarks/benchmark_serving.py \
  --backend vllm --dataset-name random --random-input-len 512 \
  --random-output-len 256 --num-prompts 200 --request-rate 4
# 固定其他条件，request-rate 从 1 扫到饱和（如 1/2/4/8/16）
```

**你应该观察到**：
1. 低速率时 TTFT/TPOT 平坦；接近饱和点两者同时陡增——拐点就是容量上限；
2. output_len 不变时，提高并发能线性提升总吞吐直到某个平台期；
3. 把 input-len 加倍后再扫一遍，TTFT 曲线整体上移且拐点提前（prefill 成本）。

**产出物**：一张双轴图（x=RPS，y=TTFT p99 & TPOT p99），并在图上标出你定义的 goodput 工作区（如 TTFT<2s 且 TPOT<100ms）。这张图值得放进简历。

**思考题**：如果业务 SLA 是 TPOT<50ms，你会把日常运行水位定在容量的百分之多少？为什么不是 90%？

### L2 观察 Continuous Batching 的动态 `[1×24G]`

**目标**：看见调度器（第1章 §1.3.2 / 第3章 §3.2）。

做法：开 DEBUG 日志或 metrics 端点，用脚本先发 1 个长输出请求，5 秒后追加 3 个短输入请求。

**你应该观察到**：
1. 新请求不等旧请求结束即进入 batch（对比静态批处理的行为）；
2. running/pending 队列长度随时间变化；若持续加压，出现 preemption/retract 事件（第3章 §3.2.3）——记录触发时刻的 KV 占用水位；
3. metrics 里 batch size 是随时间波动的，不是一个常数。

**思考题**：preemption 发生后，被抢占请求重新执行时的第一个 token 延迟由什么决定？（recompute vs swap 的选择逻辑在哪段源码里？）

---

## Track B：引擎内核观察

### L3 验证 KV Cache 显存账 `[1×24G]`

**目标**：用 第1章 §1.4.3 的公式做一次预测-实测闭环。

```text
预测: KV per token = 2(n_layer) × n_kv_head(head_dim) × dtype_bytes
例: Qwen2.5-7B, 28 层, 4 KV 头 ×128, FP16 → 28×4×128×2×2 ≈ 57KB/token
```

步骤：起服务后逐步增大 `--max-num-seqs` 或发送长输入，从 metrics 记录 KV pool 占用，反推每 token 实际字节，与预测对比。

**你应该观察到**：实测 ≈ 预测（±10%，来自 block 取整粒度——block_size 会造成尾块浪费）；换成 GQA 模型与 MHA 模型对比，体会 §1.1.4 的意义。

**思考题**：block_size=16 与 64，哪个对长文本更省？哪个对短对话更省？页内碎片的账怎么算？

### L4 Prefix/Radix Cache 命中率实验 `[1×24G]`

**目标**：量化 第2章 §2.1.2 / 第4章 §4.2 的复用收益。

设计两组流量：A 组共享同一 system prompt（长前缀 + 短问题）；B 组完全随机 prompt。分别压测并记录 TTFT 与缓存命中率指标（两引擎 metrics 都有暴露）。

**你应该观察到**：A 组 TTFT 显著低于 B 组，且差距随 system prompt 长度增长；SGLang 侧可在日志看到 radix tree 的匹配长度。

**思考题**：如果把 A/B 两组流量混在一个实例上，A 组的命中率会受什么影响？（这正是 第19章 §19.3 sticky routing 要解决的问题——先在这里制造痛点。）

### L5 投机解码实测：负载决定一切 `[1×24G]`

**目标**：复现 第6章 Finding 4 与 §2.6.5 选型表的核心结论。

```bash
# 以支持 EAGLE/MTP 的模型配置投机解码，分别在两类负载下压测:
# 负载1: HumanEval 类代码补全（高可预测性）
# 负载2: 创意写作随机 prompt（低可预测性）
# 另加 ngram 草稿跑同样两组（零成本对照组）
```

**你应该观察到**：
1. 代码负载接受率高、加速明显；创意写作接受率掉到接近无收益；
2. 大 batch（提高并发）后投机解码收益衰减甚至转负——draft 验证挤占了主模型算力；
3. ngram 在代码/RAG 负载上"白捡"的加速可能超预期。

**思考题**：给定你的生产负载画像（P19 §19.1 的表格），你会在哪个位置开/关投机解码？依据是哪条观察？

---

## Track C：优化技术

### L6 量化全流程：AWQ vs GPTQ `[1×24G]`

**目标**：走通 第22章 §22.2 的完整管道，拿到自己的精度-速度数据点。

```bash
# 1. 用 llm-compressor/AutoAWQ 类工具量化同一个 7B 模型（W4A16）
#    校准集准备 256 条: 一半通用对话 + 一半你的领域数据
# 2. 精度回归: 用 lm-eval 类 harness 跑固定子集(gsm8k/humaneval 各取一)
# 3. 性能: 用 L1 同款方法压测 BF16 vs W4 两个版本
```

**你应该观察到**：
1. W4 权重体积 ÷4，decode 吞吐提升明显（带宽-bound），prefill 提升有限；
2. 通用基准掉 1 点以内；换掉校准集再量化一次，对比领域子集的差异——亲眼看一次校准集漂移；
3. 两种算法（AWQ/GPTQ）在同基准上的差异通常小于运行噪音，体会"工程上差异不重要、机制上要分清"。

**思考题**：KV cache 保持 FP16 时，W4 化权重的模型在长上下文下的显存瓶颈转移到了哪里？

### L7 Chunked Prefill 开关对比 `[1×24G]`

**目标**：复现 Sarathi-Serve（§2.6 时间轴 / 第20章 §20.2.6）解决的问题。

设计：混合负载 = 持续的小请求流 + 周期性插入 32K 长 prompt。分别在关闭/开启 chunked prefill 下采集小请求的 TPOT p99。

**你应该观察到**：关闭时长 prompt 到来瞬间，在线小请求的步进时间出现尖刺（被大 prefill 阻塞）；开启后尖刺被削平成多个小台阶，TPOT p99 明显改善——这就是"用 token 预算平滑每步耗时"的直观形态。

**思考题**：chunk 大小设得过小的代价是什么？（提示：prefill 总时长与 kernel 效率）

### L8 迷你 PD 分离 `[2×GPU]`

**目标**：在单机上搭最小 P/D 拓扑（对应 第2章 §2.3 / 第20章 §20.2.7）。

做法：用引擎自带的 PD 分离模式（SGLang 示例配置最直观）或两进程 + KV transfer 后端，P 进程绑卡 0、D 进程绑卡 1，中间走本机 NVLink/PCIe。压测长输入短输出负载，与合部署基线对比 TTFT 分布。

**你应该观察到**：
1. 合部署时 prefill 尖峰期间 decode 请求 TPOT 抖动；分离后 D 侧 TPOT 平稳；
2. P:D 配比对长输入负载敏感——试着改变配比找到你这组参数下的最优；
3. 观察 KV 传输耗时在 TTFT 中的占比（传输带宽 ÷ KV 大小可以手算核对，参考 第6章 Finding 6 的量级）。

**思考题**：如果你的 KV 传输只有 PCIe 16GB/s，多大输入长度以上 PD 分离才划算？列出你的推导。

---

## Track D：系统层

### L9 手写 Triton Kernel：RMSNorm 三部曲 `[1×24G]`

**目标**：走完 第24章 §24.4 六步工作流的一次完整循环。

```text
1. Profile: 用 PyTorch profiler 找出 eager RMSNorm 的耗时与访存量
2. Roofline 手算: 读写 bytes / FLOPs → 确认 memory-bound（AI ≪ 机器 ridge 点）
3. 实现: Triton 版 RMSNorm（一行行对着公式写，注意 reduce 的写法）
4. Autotune: BLOCK_SIZE 网格搜索
5. 数值验证: vs eager 参考, allclose(atol=1e-3); 故意引入 fp16 累加 bug 观察误差形态
6. 集成: 注册 custom op, 检查 CUDA Graph 兼容性
```

**你应该观察到**：好的 Triton 实现达到 eager 数倍性能、接近 torch.compile 水平；autotune 前后差距可达 2×；数值 bug 在 atol=1e-3 下现形。

**思考题**：把 RMSNorm+残差+FP8-quant 三步融成一个 kernel，理论上省多少 HBM 字节？用你 L9 profile 出的实际数字算（对应 §24.5 的判据）。

### L10 结构化输出的代价 `[1×24G]`

**目标**：测量 constrained decoding 在热路径上的开销（第18章 §18.5）。

做法：同一批 prompt，分别用自由生成 / JSON Schema 约束（引擎 guided decoding 或 xgrammar 后端）/ 复杂嵌套 schema 三种模式压测，比较 TPOT 与吞吐。

**你应该观察到**：简单 schema 开销很小（mask 查表），复杂嵌套 schema 首个请求有明显编译预热、后续摊薄——亲眼确认 XGrammar 设计要解决的正是这个编译成本。

**思考题**：如果网关层 80% 请求都是同一种 schema，你能想到什么缓存策略？

### L11 写一个 Cache-Aware Router `[CPU]`

**目标**：实现 第18章 §18.3.3 的最小可行版，理解路由为什么必须懂缓存。

做法：模拟 3 个引擎副本，各自维护一个 radix tree 近似视图（可用字典模拟）；写两种策略——纯轮询 vs session 哈希粘性；回放一批多轮对话 trace，统计各自的模拟命中率与副本负载方差。

**你应该观察到**：粘性策略命中率大幅领先，但负载方差也更大——长会话副本越来越重。然后做混合策略（粘性 + 过载迁移阈值），找到你自己的一组平衡参数。

**思考题**：某副本挂掉时，它上面的会话状态去哪了？你的 router 需要什么接口才能把损失降到最低？（引出 L12）

### L12 混沌演练：会话粘性失效 `[2×GPU]`

**目标**：验证 第18章 §18.9 的故障语义设计。

做法：在 L11/L4 的基础上，压测中 kill 掉承载活跃会话的副本，观测：(a) 该会话重路由后的首个请求 TTFT；(b) 全局命中率恢复曲线；(c) 若开启 KV 分层存储（HiCache/LMCache 配置），对比状态丢失后的表现差异。

**你应该观察到**：无外置 KV 时首请求 TTFT 出现数量级尖刺（全量 re-prefill）；有分层存储时尖刺显著缓和——"KV 是易失状态"从一句话变成一次体感。

**思考题**：把这次演练写成一篇 postmortem（现象→根因→修复→预防），格式参考 SRE 行业模板。

---

## 26.x 学习路径建议

| 你的情况 | 建议 |
|----------|------|
| 学生/转行 | L0→L1→L3→L6（一条最小作品集线） |
| 在职推理工程师 | 直奔 Track B/C，补齐没有亲手做过的事 |
| 快速出成果 | L1 的曲线 + L5 的对比表 + L9 的 kernel，三个最能聊 |

> 维护约定：引擎 flag 与目录名会漂移，本章以"实验设计与观察点"为稳定层；每次升级引擎版本时只需重验命令，不需重写实验。
