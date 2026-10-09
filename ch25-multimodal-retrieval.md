# 第25章 多模态生成与检索服务 — 扩散模型 Serving 与 Embedding/Reranker 基础设施

> **面向角色**：多模态/生成式 AI 平台工程师、RAG 平台工程师  
> **前置知识**：第1章（指标）、第2章（调度）、第18章（系统层）  
> **目标**：补齐 LLM 文本生成之外的两类生产负载——扩散模型（图像/视频）serving 的独立调度模式，与 embedding/reranker 检索链路的 infra 设计。读完可搭建完整的"生成 + 理解"混合推理平台

---

## 25.0 时间发展线 ★

```
2021        2022                2023                    2024                  2025-
 │           │                   │                       │                     │
 DALL-E      Latent Diffusion    SDXL / TensorRT加速      DiT范式(Sora报告)      视频生成基础设施
 (闭源时代)  (CVPR, 开源引爆)    Consistency/LCM蒸馏      DMD蒸馏+DeepCache     流式扩散 / 步级批调度
                                 (少步数化开始)            block cache            生产工作流编排
```

---

## 25.1 扩散模型 Serving：为什么不能直接搬 vLLM ★

文本自回归和扩散去噪的调度模型完全不同：

| 维度 | LLM serving | 扩散模型 serving |
|------|-------------|-----------------|
| 计算结构 | 自回归，每步 1 token，KV 递增 | 固定 N 步去噪（20~50 步），每步全图 forward |
| 中间状态 | KV Cache（必须驻留显存） | 全部中间 latent 可丢弃、可重算 |
| batch 组成 | 连续批处理动态进出 | 同请求的 N 步天然串行；不同请求间静态/动态批 |
| 显存画像 | 权重 + KV 随长度增长 | 权重（SDXL ~10GB）+ 少量激活，短平快 |
| 加速主轴 | KV 管理、批调度 | **步数压缩**（蒸馏）、缓存复用、分辨率分级 |

### 25.1.1 三大优化轴 ★

**① 蒸馏减步（最有效）**
- Consistency Models → LCM：把几十步压到 4~8 步；
- DMD 类分布匹配蒸馏：1~4 步逼近原模型质量；
- Infra 含义：步数 ×N 减少 = 吞吐 ×N，比任何调度技巧都值钱。

**② 跨请求缓存**
- 相同 prompt/seed 前缀的去噪轨迹高度重叠 → DeepCache（跨步复用深层特征）、block/step-level caching；
- 工程形态类似 LLM 的 prefix cache：以 prompt hash 为键。

**③ 调度与资源池化**
- 步级批调度：把不同请求的同一步聚合成大 batch forward；
- UNet/DiT 权重池化 + 多租户隔离（同 第18章 MIG 思路）；
- 视频/实时生成需要流式输出（边去噪边出低清帧渐进细化），延迟预算按首帧时间（类比 TTFT）管理。

### 25.1.2 生产栈速览 ☆

- **TensorRT for Diffusion / torch.compile**：图捕获 + 融合，收益 2×+；
- **ComfyUI/工作流引擎**：DAG 编排成为事实上的"扩散模型网关"，生产化需加队列与配额层；
- **视频生成集群**：长序列 latent + 多阶段（T2I→I2V→插帧）流水，本质是 GPU 编排问题——P:D 分离思想同样适用（条件编码 vs 迭代去噪分离）。

---

## 25.2 Embedding / Reranker Serving ★

RAG 平台的另一半 infra，负载画像与生成完全相反：

| 维度 | Embedding（双塔） | Reranker（交叉编码） |
|------|------------------|---------------------|
| 输入/输出 | 短文本 → 定长向量 | query+doc 对 → 相关性分 |
| 延迟要求 | 高 QPS、毫秒级 | 召回后精排，可放宽 |
| 计算 | 一次 encoder forward | doc 数 × forward |
| 部署要点 | 动态批 + 向量归一化融合 | 截断策略 + batch 内 padding 浪费控制 |

### 25.2.1 向量检索层 ★

```
写入：encoder 服务 → 向量 → ANN 索引
检索：query 向量化 → ANN top-k（HNSW 图遍历 / IVF 聚类）→ reranker 精排
```

| 组件 | 代表 | 要点 |
|------|------|------|
| ANN 库 | HNSW（TPAMI'18 经典）、FAISS、CAGRA（GPU） | HNSW 内存换延迟；GPU 版适合十亿级 |
| 向量数据库 | Milvus/Qdrant/pgvector | 过滤+向量混合查询、标量分区 |
| Late interaction | ColBERT（SIGIR'20） | token 级向量 + MaxSim，精度/成本介于双塔与交叉编码之间 |

**Infra 自检三问**：召回 QPS 与 encoder 容量是否匹配？索引内存是否超预算（HNSW ≈ 向量数 ×(dim×4 + M×2×8) 字节量级）？reranker 是否只对 top-k 生效（k 是精度-成本旋钮）？

---

## 25.3 经典模型 Serving 一页图 ☆

LLM 之外仍有海量传统 ML/DL 推理需求，一张表定位：

| 栈 | 场景 | 与本书关系 |
|----|------|-----------|
| **ONNX Runtime** | 跨平台 DL 推理交换格式 | 模型交付管道的中间格式常客 |
| **OpenVINO** | Intel CPU/iGPU | 传统企业推理主力 |
| TF Serving / Triton (IS) | 静态批时代的遗产与新容器 | 第20章 §20.2.1 前史的现代延续 |
| sklearn/XGBoost 服务化 | 表格类特征模型 | 网关计量体系可直接复用 |

---

## 25.4 论文线汇总 ★

完整对照见 第21章 线 13/14，核心八篇：

| # | 论文 | 出处 | 一句话 |
|---|------|------|--------|
| 1 | Latent Diffusion (`2112.10752`) | CVPR 2022 | 潜空间扩散，开源引爆点 |
| 2 | DiT (`2212.09748`) | ICCV 2023 | Transformer 骨干扩散，Sora 基石 |
| 3 | Consistency Models (`2303.01469`) | 2023 | 少步生成的理论起点 |
| 4 | LCM (`2310.04378`) | 2023 | latent 空间一致性蒸馏实用化 |
| 5 | DMD (`2311.18828`) | 2023-24 | 分布匹配一步蒸馏 |
| 6 | DeepCache (`2312.00858`) | 2023 | 跨步特征缓存 |
| 7 | Sentence-BERT (`1908.10084`) | EMNLP 2019 | 双塔语义检索范式 |
| 8 | ColBERT (`2004.12832`) | SIGIR 2020 | late interaction 精排 |

---

## 25.5 本章小结

- 扩散 serving 的三大轴：**蒸馏减步 > 缓存复用 > 调度池化**，与 LLM 的"KV 管理 > 批调度"优先级恰好镜像；
- embedding/reranker 是高 QPS 小算力负载，别用 LLM 引擎硬套；
- "生成 + 理解"混合平台 = vLLM/SGLang 池 + 扩散池 + 检索池，统一到 第18章 的网关计量与路由体系。

### 高频问题

1. 为什么 continuous batching 不能直接用于扩散模型？
2. 扩散模型提速的第一杠杆是什么？（步数蒸馏，数量级 > 工程优化）
3. HNSW 的内存怎么估？什么时候上 GPU 检索？
4. RAG 链路里 reranker 放在哪一层？k 怎么定？

→ 交叉复习：prefix cache 思想 第2章 §2.1.2 / 第4章 §4.2；MIG 隔离 第18章 §18.2.4；网关计量 第18章 §18.3。
