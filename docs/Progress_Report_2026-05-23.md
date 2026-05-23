# ZeroShotVDR 项目进展报告

> **报告日期**：2026-05-23  
> **项目周期**：2025.5.8 – 2025.6.9（当前已进入第 16 天 / 共 33 天）  
> **当前阶段**：Phase 5（报告撰写与答辩准备，进行中）

---

## 一、项目概况

ZeroShotVDR（零样本视觉文档检索）是一个基于 ColPali（Vision Language Model + Late Interaction）的页级视觉文档检索系统。项目面向 MMLongBench 数据集中的 DocumentQA 子集，目标是在不依赖 OCR 的情况下，直接从文档页面图像中检索与文本查询最相关的页面。

项目分为五个阶段，当前已完成前四个阶段的核心工作，正在进行 Phase 5 的论文撰写与答辩准备。

---

## 二、总体进度

| 阶段 | 时间 | 状态 | 完成度 |
|------|------|------|--------|
| Phase 1：文献调研与环境准备 | 5.8 – 5.12 | 已完成 | 100% |
| Phase 2：基础系统实现 | 5.13 – 5.22 | 已完成 | 100% |
| Phase 3：基础评测与调优 | 5.23 – 5.25 | 已完成 | 100% |
| Phase 4：进阶方法研究与实现 | 5.26 – 6.2 | 已完成 | 100% |
| Phase 5：报告撰写与答辩准备 | 6.3 – 6.9 | 进行中 | ~30% |

**关键时间节点**：
- Phase 4 全部实验矩阵与里程碑报告已于 **2026-05-20** 完成
- 当前距最终答辩还有约 **17 天**

---

## 三、各阶段详细进展

### Phase 1：文献调研与环境准备

**完成内容**：
- 深入阅读 ColPali、ColBERT 核心论文及 MMLongBench 数据集文档
- 搭建完整开发环境：Conda (Python 3.10) + uv (依赖管理) + 2× NVIDIA RTX 3090 (24 GB)
- 下载 ColPali-v1.3 模型权重（base ~5.87 GB + LoRA adapter ~114 MB）至项目内缓存
- 下载 MMLongBench 元数据包与 DocumentQA 图像包，完成数据集探索与统计
- 编写环境验证脚本（`scripts/command/check_env.py`）与模型加载测试（`scripts/test_model_load.py`）
- 形成《MMLongBench 数据集探索笔记》（`docs/MMLongBench_Dataset_Notes.md`）

### Phase 2：基础系统实现

**完成内容**：

按照五层架构（数据接入 → 索引 → 检索 → 评测 → 配置支撑）实现了完整的 ColPali-based 页级检索管线：

| 模块 | 文件 | 核心功能 |
|------|------|----------|
| 数据契约 | `src/zeroshot_vdr/contracts.py` | `Page`、`Query`、`RetrievalResult`、`RelevanceJudgment` dataclass 及稳定 ID 构造体系 |
| 数据接入 | `src/zeroshot_vdr/data/` | DocumentQA 适配器、语料聚合器，支持从原始图片路径恢复稳定页面身份 |
| 索引层 | `src/zeroshot_vdr/indexing/` | ColPali 页面编码器（`PageEncoder`）、逐页独立存储（`IndexStore`），支持断点续建 |
| 检索层 | `src/zeroshot_vdr/retrieval/` | 查询编码器、MaxSim 相似度计算、检索流水线（候选召回→精排→Top-k 排序） |
| 评测层 | `src/zeroshot_vdr/evaluation/` | Recall@k / Precision@k / MRR / nDCG@k 四项指标，支持按子任务/长度分组评测 |

**关键工程决策**：
- 每页独立存储为 `.pt` 文件，天然支持增量追加和变长 patch 数
- MaxSim 默认使用 L2 归一化点积（等价于余弦相似度）
- 评测协议采用 query-scoped candidate retrieval（查询在其 `candidate_page_ids` 范围内排序）

### Phase 3：基础评测与调优

**Step 3.1 — 全量评测**：

在 MMLongBench DocumentQA 全部 3 个子任务（`longdocurl`、`mmlongdoc`、`slidevqa`）× 6 个长度档位（K4–K128）上完成了稳定基线全量评测：

| 项目 | 数值 |
|------|------|
| 查询总数 | 15,577 |
| 文档数 | 3,653 |
| 页面数 | 87,922 |
| 索引大小 | ~88.6 GB |
| 索引构建时间 | 6,689 s（~1h52m） |
| 检索总时间 | 1,103 s（~18m23s） |
| 平均延迟 | 0.071 s/query |
| P95 延迟 | 0.138 s/query |

**Step 3.2 — 结果分析**：

- 发现并修复两个关键问题：
  1. **页面身份不稳定**：旧 `doc_id/page_idx` 契约在长上下文场景下不稳定，改为从原始图片路径重建稳定页面身份
  2. **无效 ground truth 污染**：1,192 条查询（7.65%）缺少有效页级标注，已单独标记为 `no_ground_truth` 并从主比较口径中剥离
- 主比较口径（14,385 条有效标注查询）上的基线指标：

| k | Recall | Precision | MRR | nDCG |
|---|--------|-----------|-----|------|
| 1 | 0.3342 | 0.4403 | 0.5838 | 0.4403 |
| 5 | 0.7250 | 0.2041 | 0.5838 | 0.5881 |
| 10 | 0.8517 | 0.1218 | 0.5838 | 0.6325 |

- 坏例分析：真实 bad case 2,636 条（18.32%），主要集中在 slidevqa 长上下文场景（K32–K128）
- 失败模式分类：大候选集 miss_top10、多页部分召回、邻页混淆

**Step 3.3 — 方向决策**：

基于失败分析，正式选择**方向 B（查询自适应两阶段粗精检索）**作为 Phase 4 主线：
- 与坏例模式直接对应（大候选集 miss_top10）
- 与现有实现距离近（已有 mean-pooled page view 和预留的候选策略接口）
- 创新性与实现风险平衡良好

### Phase 4：进阶方法研究与实现

**核心改进思路**：在不破坏 Phase 3 baseline 的前提下，实现 query-adaptive two-stage coarse-to-fine retrieval：

```text
Query.candidate_page_ids → mean-pool coarse retrieval
    → adaptive top-N selection → optional neighbor expansion
    → full MaxSim rerank → final top-k results
```

**10 个 Stage 全部完成**：

| Stage | 内容 | 关键产出 |
|-------|------|----------|
| Stage 0 | 固化 Phase 3 baseline，建立 valid-only 主比较口径 | Phase 4 兼容 schema 回填 |
| Stage 1 | 新增 Phase 4 配置与 `advanced/` 目录 | 配置项 `retrieval.phase4` |
| Stage 2 | page_id 解析与邻页工具 | `neighbors.py` + 单元测试 |
| Stage 3 | adaptive top-N 选择逻辑 | `two_stage.py` 中的自适应函数 |
| Stage 4 | fixed top-N TwoStageRetriever | 最小闭环 + 全量 fixed top-N 结果 |
| Stage 5 | MeanPoolCache 接入 | 批量构建避免 OOM，87,090 页 cache 仅 ~26 MB |
| Stage 6 | adaptive top-N 接入运行时 | `phase4_adaptive` 全量结果 |
| Stage 7 | neighbor expansion 接入运行时 | `phase4_adaptive_neighbors` 全量结果 |
| Stage 8 | 落地评测脚本 | `scripts/run/run_phase4_eval.py` + `main.py` 统一入口 |
| Stage 9 | per-query trace 与 slice 分析 | `phase4_trace.jsonl`、`slice_metrics.csv`、`bucket_metrics.csv` |
| Stage 10 | 完整消融与里程碑报告 | 9 组全量对比实验 |

**核心代码位置**：
- 两阶段检索：`src/zeroshot_vdr/advanced/two_stage.py`
- 邻页扩展：`src/zeroshot_vdr/advanced/neighbors.py`
- Mean-pool 缓存：`src/zeroshot_vdr/advanced/mean_pool_cache.py`
- 评测入口：`scripts/run/run_phase4_eval.py`
- 统一 CLI：`main.py`
- 单元测试：`tests/phase4/`（4 个测试文件）

### Phase 5：报告撰写与答辩准备

**已完成**：
- [x] Phase 4 里程碑报告（`docs/Milestone_Report_Phase4.md`）
- [x] 脚本目录重组（`scripts/run/` + `scripts/command/`）
- [x] 统一命令入口（`main.py`）

**待完成**：
- [ ] 实验报告 PDF（NeurIPS 模板，英文，正文 8-9 页）
- [ ] 答辩 PPT
- [ ] 最终 README.md（快速开始 + 复现指引）
- [ ] 代码清理与最终打包提交

---

## 四、初步成果

### 4.1 Phase 3 稳定基线（14,385 valid-only queries）

| 指标 | 值 |
|------|-----|
| Recall@1 | 0.3342 |
| Recall@5 | 0.7250 |
| Recall@10 | 0.8517 |
| MRR | 0.5838 |
| nDCG@10 | 0.6325 |
| Avg Latency | 0.0716 s/query |
| P95 Latency | 0.1384 s/query |

### 4.2 Phase 4 进阶方法完整消融结果

| Method | Recall@10 | nDCG@10 | Avg Latency (s) | P95 Latency (s) | Avg Rerank Candidates |
|--------|----------:|--------:|----------------:|----------------:|----------------------:|
| Phase 3 Full MaxSim (baseline) | 0.8517 | 0.6325 | 0.0716 | 0.1384 | 32.7 |
| Fixed Top-32 + MaxSim | 0.8482 | 0.6308 | 0.0794 | 0.1434 | 19.0 |
| Fixed Top-64 + MaxSim | 0.8513 | 0.6325 | 0.0889 | 0.1800 | 26.8 |
| Fixed Top-128 + MaxSim | 0.8517 | 0.6326 | 0.0907 | 0.1989 | 32.2 |
| Adaptive + MaxSim | 0.8482 | 0.6308 | 0.0790 | 0.1417 | 19.0 |
| **Adaptive + Neighbor + MaxSim** | **0.8523** | **0.6325** | **0.0796** | **0.1437** | **19.8** |
| Fixed Top-64 + Cache | 0.8513 | 0.6325 | 0.0650 | 0.1178 | 26.8 |
| Adaptive + Cache | 0.8482 | 0.6308 | 0.0592 | 0.0847 | 19.0 |
| **推荐 Adaptive + Neighbor + Cache** | **0.8523** | **0.6325** | **0.0600** | **0.0858** | **19.8** |

> 加粗行为**当前推荐方法**

### 4.3 核心发现

1. **质量保持**：推荐方法（Adaptive + Neighbor + Cache）在 Recall@10 上略优于 Phase 3 全量 MaxSim baseline（0.8523 vs 0.8517），nDCG@10 持平（0.6325）
2. **效率提升**：平均延迟从 0.0716 s/query 降至 0.0600 s/query（下降 ~16%），P95 延迟从 0.1384 s 降至 0.0858 s（下降 ~38%）
3. **存储收益**：MeanPoolCache 仅需 ~26 MB，远小于原始 patch index 的 ~88.6 GB
4. **最大改善切片**：`slidevqa/K64` 与 `slidevqa/K128`（Phase 3 阶段的最困难场景）
5. **轻微 trade-off**：`longdocurl/K128` 与 `mmlongdoc/K128` 上存在轻微质量回落，但换来了显著的时延下降

### 4.4 工程成果总结

| 维度 | 数值 |
|------|------|
| 核心代码量 | `src/zeroshot_vdr/` 下 7 个子包，~15 个模块文件 |
| 进阶模块 | `advanced/` 下 4 个模块（two_stage, neighbors, mean_pool_cache, profiling） |
| 单元测试 | 4 个 Phase 4 测试文件 + 早期 Step 2.x 测试 |
| 运行脚本 | 4 个核心评测/分析脚本 + 4 个辅助命令脚本 |
| 文档产出 | 2 份里程碑报告 + 1 份数据集笔记 + 11 份修订记录 + 项目计划 v9 |
| 评测产物 | 9 组 Phase 4 全量结果 + 1 组 Phase 3 稳定基线 + 分 slice/bucket 分析 |

---

## 五、下一步计划（Phase 5，截至 6.9）

### 5.1 实验报告撰写（预计 3–4 天）

按照 NeurIPS 模板撰写英文实验报告（正文 8-9 页）：

| 章节 | 负责 | 内容要点 | 预计耗时 |
|------|------|----------|----------|
| Introduction | 共同 | 任务背景、长文档页级检索难点、本项目贡献 | 0.5 天 |
| Related Work | 共同 | ColPali、ColBERT、VLM 文档检索综述 | 0.5 天 |
| Method | 成员 A | Stable baseline 架构、two-stage retrieval、mean-pool cache、neighbor expansion | 1 天 |
| Experiments | 成员 B | Phase 3 baseline、Phase 4 主消融表、cache 对比、slice 分析 | 1 天 |
| Analysis | 成员 B | slidevqa 长上下文收益、K128 trade-off、局限性 | 0.5 天 |
| Conclusion | 共同 | 质量-效率结论与后续方向 | 0.5 天 |

### 5.2 答辩 PPT 制作（预计 1–2 天）

- 方法设计思路图示化（two-stage pipeline 流程图）
- 实验亮点一页摘要对比表
- 创新性阐述（与已有工作的差异化）

### 5.3 代码整理与提交（预计 1–2 天）

- [ ] 补齐最终 `README.md`（快速开始、复现实验指引）
- [ ] 统一剩余 docstring、type hints
- [ ] 复核 `uv sync` / `.venv` 一键复现说明
- [ ] 清理调试产物，打包最终代码
