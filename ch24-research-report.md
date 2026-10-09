# 第24章 Deep Research 研究报告

> **生成方式**：107-agent 多阶段深度检索 → 25 篇文章 → 120 条声明 → 3-vote 对抗验证 → 7 条高置信度声明 + 16 条被否决声明  
> **日期**：2026-07-02  
> **状态**：补充材料，与第0~8章互为印证

---

## 24.0 研究方法

本报告通过以下流程生成：

```
5 个搜索角度 → 5 个并行 WebSearch
       ↓
25 篇来源抓取（去重）
       ↓
120 条可验证声明提取
       ↓
25 条声明进入 3-vote 对抗验证
       ↓
9 条通过 → 语义去重 → 7 条最终声明
16 条被否决（多数 refute）
```

**来源质量分布**：primary (论文/官方文档) 14 篇，secondary (技术博客) 6 篇，blog 5 篇，forum 1 篇。

---

## 24.1 核心发现

### Finding 1：MLA 模型在 TP 下 KV Cache 8× 重复 ★★★

**声明**：DeepSeek MLA 模型在 Tensor Parallelism 下存在 KV Cache 8 倍重复存储问题，根源是其单 KV head 架构无法按 head 维度切分。DP Attention（数据并行注意力）和 DP+EP（数据并行+专家并行）是生产环境的解决方案。

**置信度**：HIGH

**详情**：
- MLA 将所有 K/V 压缩至单一潜在向量 c^KV（维度 512），由 128 个注意力头共享
- 因此每个 TP rank 必须存储完整 KV cache（576 元素/token = 512 latent + 64 RoPE）
- TP=8 时造成 **8 倍重复存储**
- vLLM 源码 `mla_attention.py` 中 `self.num_kv_heads = 1` 可直接验证

**解决方案**：DP Attention 按请求（而非按 head 维度）分区 KV cache，每个 GPU 只存储分配给它的请求的 cache → 约 8× 显存效率提升。

**与 handbook 对应**：补充 第3章 §3.2.1（TP）和 §3.4.3（MLA）的讨论。

**来源**：
- vLLM RFC #16037: "If Tensor Parallelism is used in an MLA model, we duplicate the KV cache across GPUs, wasting memory"
- AMD ROCm MoE Playbook: 源码层面验证
- vLLM v0.11.0 expert_parallel_deployment 文档

---

### Finding 2：Wide-EP 架构 ★★★

**声明**：Wide-EP 是 vLLM 为 MoE 模型设计的核心并行架构：Expert (MoE) 层在所有 EP rank 上分片（EP_SIZE = TP × DP），Attention 层按 DP rank 复制（若 TP > 1 则进一步 TP 分片）。

**置信度**：HIGH

**详情**：

```
Wide-EP = DP × TP

Expert (MoE) Layers:  Sharded across ALL EP ranks (size = TP × DP)
Attention Layers:     Replicated across DP ranks
                      If TP > 1 → further TP-sharded within each DP group
```

**关键洞察**：这不是简单的 TP+EP 叠加，而是**对 Attention 和 MoE 层采用不同的并行策略**。这是由 MLA 的特殊架构驱动的——Attention 层不能通过 TP 高效分片（Finding 1），因此需要用 DP 替代。

**与 handbook 对应**：补充 第3章 §3.2.3（EP）和 第7章 §7.2.4（KV Connector）。

**来源**：
- vLLM v0.11.0 官方文档 expert_parallel_deployment 页面
- vLLM RFC #16037
- AMD ROCm MoE Playbook

---

### Finding 3：DP+EP vs TP+EP 性能基准 ★★

**声明**：在 8× AMD MI300X 上运行 DeepSeek-R1（8192/1024 tokens），DP=8+EP 在 1024 并发下达到 7,114 tok/s——比 TP=8+EP 高 47%；但 TP=8+EP 在 64 低并发下 TTFT 比 DP 低 80%。

**置信度**：MEDIUM（单源数据、预发布版本）

**详情**：

| 策略 | 低并发 (64) TTFT | 高并发 (1024) 吞吐 |
|------|-------------------|---------------------|
| TP=8 + EP | **低 80%** ✅ | 4,839 tok/s |
| DP=8 + EP | 较高 | **7,114 tok/s (+47%)** ✅ |

**结论**：不存在单一最优解——低并发用 TP（延迟优先），高并发用 DP（吞吐优先）。

**Caveats**：
- 仅来自 AMD 单源技术博客
- vLLM 0.11.1rc3 是预发布版本
- Moreh 独立测试在相同硬件上获得更高吞吐（9,440-21,225 tok/s），该数据可能保守

**与 handbook 对应**：补充 第3章 §3.2 和 第23章 §23.3（千卡部署场景设计题）。

**来源**：
- AMD ROCm MoE Playbook: vLLM 0.11.1rc3 + ROCm 7.0

---

### Finding 4：MTP 投机解码的生产环境衰减 ★★

**声明**：MTP 在单节点小规模（16× H200）可获 +60.8% 吞吐提升，但在 128-GPU 大规模生产环境仅剩 +14.2%。FastMTP 通过自蒸馏将多 token 接受率从 2% 恢复至 36%。

**置信度**：MEDIUM

**详情**：

| 规模 | 配置 | 加速比 |
|------|------|--------|
| 小规模 | 16× H200, 2 并发/rank, 65K/4K tokens | **+60.8%** |
| 大规模生产 | 128× H200, 128 并发/rank, 2K/100 tokens | **+14.2%** |
| AMD MI300X | 并发 1→64 | 2.11× → 1.25× |

**根本原因**：
- 高并发下 GPU 已经吃饱 → draft model 的额外计算反而变成负担
- 原生 MTP 的 draft 接受率随 K 增加急剧下降：70% (K=1) → 11% (K=2) → 2% (K=3)
- LMSYS 建议：GPU 接近容量时使用更小的 draft size

**FastMTP 的改进**：
- 自蒸馏微调 + 语言感知词表压缩
- 接受率恢复：70→81% (K=1), 11→56% (K=2), 2→36% (K=3)
- 平均加速比 2.03× (K=3)，7 个 benchmark

**Caveats**：
- 大规模测试中 MTP 和 overlap scheduling 效果捆绑报告
- FastMTP 仅在 MiMo-7B-RL 单模型上评估
- FastMTP 为 arXiv 预印本

**与 handbook 对应**：关键补充 第3章 §3.6（投机解码）—— 补充了 MTP 在大规模生产中的衰减数据。

**来源**：
- LMSYS 博客 (2025-07-17): SGLang MTP 分析
- arXiv:2509.18362: FastMTP
- AMD ROCm MTP 博客

---

### Finding 5：Megatron-LM TP 通信模型 ★★★

**声明**：标准 Transformer 层前向传播恰好需要 2 次 all-reduce（MLP + Self-Attention），每 micro-batch 每层每设备通信量为 `8bsh(t-1)/t` 元素。

**置信度**：HIGH

**详情**：

```
前向: f → AllReduce(g) → g → AllReduce(f)  (共 2 次)
反向: 同理 2 次
总计: 4 次 all-reduce
每次 ring-allreduce: 2(t-1)/t × bsh 元素
总通信: 4 × 2(t-1)/t × bsh = 8bsh(t-1)/t 元素
```

**适用限制**：
- 仅适用于标准 GPT/BERT 式 Transformer
- Encoder-decoder 的 cross-attention 会增加额外 all-reduce
- 单位为元素（非字节），需乘以 dtype 位宽

**与 handbook 对应**：补充 第3章 §3.2.1（TP）的通信开销分析。

**来源**：
- Megatron-LM 论文 (Narayanan et al., SC'21, 同行评审顶会)

---

### Finding 6：Mooncake RDMA 传输性能 ★★★

**声明**：Mooncake（Moonshot AI, FAST '25 最佳论文）的 RDMA KV Cache 传输引擎：
- 拓扑感知路径选择：优先本地 NUMA/PCIe 交换机 NIC
- 16 KB 粒度切片：多 NIC 并行聚合
- 性能：4×200 Gbps → **87 GB/s** (87% 利用率)；8×400 Gbps → **190 GB/s** (47.5% 利用率)
- vs TCP：分别 2.4× 和 4.6×

**置信度**：HIGH

**关键设计**：
1. 传输请求分解为 16 KB 切片
2. 不同切片通过不同 NIC 并行传输
3. 优先选择与 GPU 同 NUMA 节点的 NIC

**与 handbook 对应**：补充 第3章 §3.3.3（Mooncake KV Transfer）和 第8章 §8.5.2（Mooncake 传输后端）。

**来源**：
- USENIX FAST '25 最佳论文 (Qin et al., 2025 年 2 月)

---

### Finding 7：SGLang HiCache 三级缓存架构 ★★★

**声明**：HiCache 在 RadixAttention 上扩展了 HiRadixTree 作为跨三级的页表结构：
- L1: GPU HBM
- L2: CPU 主机内存
- L3: 外部存储/分布式内存（Mooncake、3FS、NIXL）

中央 cache 控制器通过 prefetch 和 write-back 策略编排数据移动。

**置信度**：HIGH

**关键设计**：
- HiRadixTree 作为跨三级的统一页表
- 每个树节点记录 KV cache 所在位置（GPU/CPU/L3/多层同时）
- 中央 cache 控制器决策数据何时 prefetch/write-back

**与 handbook 对应**：补充 第3章 §3.3.4（HiCache）和 第8章 §8.5.3（HiCache 集成）。

**来源**：
- LMSYS 官方博客 (2025-09-10)
- SGLang 官方文档
- GitHub PR #7369 (HiRadixTree C++ 重写) + #10192 (PD-HiCache)

---

## 24.2 被否决的声明（及原因）

25 条进入验证，16 条被否决（多数 refute）。以下是值得关注的几条：

| 声明 | 投票 | 否决原因 |
|------|------|---------|
| "Overlap scheduling 消除 GPU 空闲" | 0-3 ✗ | verifier 认为 overlap 减少而非消除空闲 |
| "FP8 是最鲁棒的量化方法" | 0-3 ✗ | verifier 指出不同任务/模型最适用不同量化方法 |
| "SGLang 用 FlashAttention-3 (prefill) + FlashInfer (decode)" | 0-3 ✗ | verifier 指出 kernel 选择是配置层面，不是硬编码 |
| "4-bit 量化中小模型严重退化" | 0-3 ✗ | verifier 引用最新工作表明退化程度取决于方法 |
| "EP 在 expert density < 1% 时应禁用" | 0-3 ✗ | 与官方文档矛盾 |
| "HiCache 降低 TTFT 56-84%" | 0-3 ✗ | verifier 认为数据来源不独立（厂商营销） |

**启示**：互联网信息需要交叉验证——即使是看似权威的来源。

---

## 24.3 开放问题

以下问题在研究中未能得到满意答案，是潜在的进阶研究方向：

1. **DP+EP 中 AllToAll 和 AllReduce 的联合调度**：两种通信如何 overlap？是否有工作通过计算-通信 overlap 隐藏 EP 的 AllToAll 开销？

2. **HiCache 在不同负载下的表现**：长文本 vs 高并发短文本场景下的缓存命中率和延迟 profile 如何？

3. **Mooncake + HiCache 组合**：Mooncake 的 RDMA 传输引擎和 HiCache 的分布式 L3 存储是否可以组合使用？接口上是否有冲突？

4. **MTP 在 MoE+MLA 模型上的特殊性**：DeepSeek-V3/R1 的 MLA 压缩 KV cache 是否影响 MTP draft token 的质量？与 Dense 模型有本质差异吗？

---

## 24.4 对 Handbook 的关键补充

| Handbook 章节 | 原内容 | Research 补充 |
|---------------|--------|---------------|
| 第3章 §3.2.1 (TP) | TP 切分方式 | **Finding 5**: Megatron-LM TP 通信量公式 |
| 第3章 §3.2.3 (EP) | EP 切分方式 | **Finding 2**: Wide-EP 架构（Attention DP + Expert EP）|
| 第3章 §3.4.3 (MLA) | MLA 压缩原理 | **Finding 1**: TP 下 8× KV Cache 重复 + DP Attention 解决方案 |
| 第3章 §3.6 (投机解码) | Eagle/MTP 原理 | **Finding 4**: MTP 大规模衰减数据 (+60.8% → +14.2%) |
| 第3章 §3.3.3 (Mooncake) | 概念介绍 | **Finding 6**: Mooncake 实测性能 (87-190 GB/s) |
| 第3章 §3.3.4 (HiCache) | 三级缓存概念 | **Finding 7**: HiRadixTree 页表 + 中央 controller |
| 第23章 §23.3 (场景题) | 千卡 MoE 部署 | **Finding 3**: TP+EP vs DP+EP 实测数据 |

---

## 24.5 完整来源列表

| # | URL | 质量 | 角度 |
|---|-----|------|------|
| 1 | [Anatomy of vLLM](https://blog.vllm.com.cn/2025/09/05/anatomy-of-vllm.html) | secondary | 架构 |
| 2 | [Inside SGLang](https://blog.sugiv.fyi/inside-sglang-anatomy-high-performance-structured-llm-inference-system) | blog | 架构 |
| 3 | [Mini-SGLang (LMSYS)](https://lmsys.org/blog/2025-12-17-minisgl/) | primary | 架构 |
| 4 | [IJCAI 2025 Quantization Survey](https://www.ijcai.org/proceedings/2025/902) | primary | 量化 |
| 5 | [FlashInfer Issue #897](https://github.com/flashinfer-ai/flashinfer/issues/897) | forum | 量化 |
| 6 | [LMSYS MTP Blog](https://www.lmsys.org/blog/2025-07-17-mtp/) | primary | 投机解码 |
| 7 | [FastMTP (arXiv)](https://ar5iv.labs.arxiv.org/html/2509.18362) | primary | 投机解码 |
| 8 | [NVIDIA Quantization Guide](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/) | secondary | 量化 |
| 9 | [AMD MTP Benchmark](https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html) | secondary | 投机解码 |
| 10 | [AMD vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) | primary | 并行 |
| 11 | [vLLM Expert Parallel Docs](https://docs.vllm.ai/en/v0.11.0/serving/expert_parallel_deployment.html) | primary | 并行 |
| 12 | [Megatron-LM Paper](https://arxiv.org/pdf/2104.04473v1) | primary | TP |
| 13 | [NVIDIA Megatron-Core Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/parallelism-guide.html) | primary | 并行 |
| 14 | [MoE Infra Guide](https://introl.com/blog/mixture-of-experts-moe-infrastructure-scaling-sparse-models-guide) | blog | MoE |
| 15 | [vLLM RFC #16037 (Wide-EP)](https://github.com/vllm-project/vllm/issues/16037) | primary | 并行 |
| 16 | [Mooncake (FAST '25)](https://www.usenix.org/conference/fast25/presentation/qin) | primary | PD |
| 17 | [SGLang HiCache (LMSYS)](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) | primary | HiCache |
| 18 | [vLLM PR #12957](https://github.com/vllm-project/vllm/pull/12957) | primary | PD |
| 19 | [Mooncake HiCache Design](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html) | primary | HiCache |
| 20 | [vLLM vs TRT-LLM vs SGLang Bench](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/) | blog | 对比 |
| 21 | [Alibaba Cloud Inference Guide](https://developer.aliyun.com/article/1686693) | blog | 对比 |
| 22 | [Top 6 Inference Runtimes 2025](https://www.marktechpost.com/2025/11/07/comparing-the-top-6-inference-runtimes-for-llm-serving-in-2025/) | blog | 对比 |
| 23 | [LLM Inference Metrics](https://bentoml.com/llm/llm-inference-basics/llm-inference-metrics) | secondary | 指标 |
| 24 | [vLLM vs SGLang Throughput](https://tensorfuse.io/blog/llm-throughput-vllm-vs-sglang) | blog | 对比 |
| 25 | [Baidu AI Infra Article](https://developer.baidu.com/article/detail.html?id=6897592) | blog | 架构 |

---

## 24.6 时效性更新（2026-08）

> 本报告基于 2026-07-02 的资料生成。以下条目针对 2026 年 7-8 月的新进展做增量修正，不影响原 Finding 的历史结论。

**1. Finding 1（MLA TP 8× 重复）的适用范围**：
- 结论对 DeepSeek-V3/V3.1 仍成立（MLA 单 KV head 架构未变）。
- DeepSeek-V3.2-Exp 引入 DSA 后，KV 参与计算的方式变为 top-k 选择 + 压缩块，8× 重复的表述不再完全适用；V4 采用 CSA + HCA 混合注意力，KV Cache 布局进一步变化（详见 第18章 §18.1.1）。
- 工程含义：DP Attention / Wide-EP 仍是 V3 系列生产首选；V4 需要按新版引擎（SGLang `deepseek_v4_backend` / vLLM DSpark 支持）重新评估并行度。

**2. Finding 4（MTP 大规模衰减）的后续**：
- FastMTP（arXiv:2509.18362）已进入主流引擎；SGLang 主线另有 dFlash（in-filling 投机）、vLLM 支持 DFlash + 动态投机 + Suffix Decoding（第7章 §7.7）。
- 原结论"大规模下 MTP 收益衰减"未被推翻，但"衰减后仍有 +14.2%"这一量级在更新引擎 + 更优 draft 策略下有望改善，生产选型时应以本地 benchmark 为准。

**3. 被否决声明中有两条需要重新审视**：
- "4-bit 量化中小模型严重退化"：当时被否决；但 V3.2/V4 时代低比特 + 稀疏注意力的组合成为主流，小模型 INT4 部署质量高度依赖量化方法（AWQ/GPTQ/FP8 混合），原否决理由仍然成立，只是讨论语境已切换。
- "HiCache 降低 TTFT 56-84%"：厂商营销数据的质疑不变；SGLang 主线新增 KV Canary（第8章 §8.10），可用于独立复现 HiCache 类缓存命中率数据。

**4. 开放问题更新**：
- 新增研究方向：稀疏注意力（NSA/DSA）下如何与 Radix/Block Cache 协同（top-k 索引缓存、被跳过块的复用）；D-LLM（扩散式 LLM）的调度与投机解码；KV Offload/Tiering 的带宽-延迟权衡（第9章 §9.2.3）。
- 原问题 1（AllToAll/AllReduce 联合调度）在 Wide-EP + EPLB（第8章 §8.10）落地后有了工程答案，但理论最优调度仍未闭合。
