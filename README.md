# AI Infra 推理引擎知识全书

> **📖 在线阅读：<https://atituiset.github.io/ai-infra/>**

面向**训推平台 / 推理引擎研发工程师**的 LLM 推理基础设施知识全书：从核心概念、vLLM/SGLang 源码精读，到生产部署、边缘 AI、Agentic/RL 负载与动手 Lab，共 27 章。

## 内容结构

| 卷 | 内容 | 章节 |
|----|------|------|
| I 全景与入门 | AI Infra 全景图、核心概念、GPU/CUDA 基础 | 第0、1、7章 |
| II 推理引擎进阶 | 高性能推理架构、模型压缩、训练 Infra、编译器与算子 | 第2、22、23、24章 |
| III 源码精读 | vLLM V1 引擎、SGLang RadixAttention 源码解剖 | 第3、4章 |
| IV 生产与系统层 | 生产部署运维、成本 ROI、安全合规、网关与集群调度 | 第8、13、14、18章 |
| V 边缘与端侧 | 国产 GPU、边缘硬件、推理框架、边缘优化落地 | 第12、15、16、17章 |
| VI 生态与前沿 | 引擎生态对比、部署案例、Agentic/RL、编年史、论文地图、多模态 | 第10、11、19、20、21、25章 |
| VII 成长：自测与实践 | 实战自测与自检清单、Deep Research 报告、动手 Lab | 第5、6、26章 |

## 本地构建

本书使用 [VitePress](https://vitepress.dev/zh/) 构建：

```bash
npm install
npm run docs:dev        # 热更新预览 http://localhost:5173
npm run docs:build      # 构建静态站点到 .vitepress/dist
npm run docs:preview    # 预览构建产物
```

## 部署

推送到 `master` 分支后，`.github/workflows/deploy-docs.yml` 自动构建并发布到 GitHub Pages。

## License

MIT
