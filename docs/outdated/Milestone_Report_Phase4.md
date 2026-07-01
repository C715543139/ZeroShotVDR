# 阶段四里程碑报告

## 1. 概述

本文档将 [docs/ZeroShotVDR_Phase4_Iterative_Development_Guide.md](docs/ZeroShotVDR_Phase4_Iterative_Development_Guide.md) 从“迭代开发指导”转化为“阶段四已完成里程碑报告”，汇总 ZeroShotVDR 在 Phase 4 的工程落地、实验结果与验收结论。

Phase 4 的核心目标是：在不破坏 Phase 3 stable baseline 的前提下，实现 query-adaptive two-stage coarse-to-fine retrieval，并在 14,385 条 valid-only 查询上完成可复现的质量与效率对比。

最终落地的主流程为：

```text
Query.candidate_page_ids
        ↓
作为 query-specific candidate universe
        ↓
mean-pool coarse retrieval
        ↓
fixed / adaptive top-N
        ↓
optional neighbor expansion
        ↓
full MaxSim rerank
        ↓
final top-k results
```

当前推荐方法为：

**Adaptive + Neighbor + MeanPoolCache**

它在 valid-only 主口径下达到：

- Recall@10 = 0.8523
- nDCG@10 = 0.6325
- Avg latency = 0.0600 s/query
- P95 latency = 0.0858 s/query

相较 Phase 3 valid-only baseline，整体质量持平略优，同时时延明显下降，因此可作为 Phase 4 的主结论方法。

---

## 2. 指南到落地的阶段映射

下表将开发指导中的 Stage 0-10 对应到本阶段已经完成的真实落地点。

| Stage | 指南目标 | 实际落地 | 状态 |
| --- | --- | --- | --- |
| Stage 0 | 固化 Phase 3 baseline | 将阶段三 stable run 回填为 Phase 4 兼容 schema，形成 valid-only 对照目录 `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/` | 已完成 |
| Stage 1 | 新增 Phase 4 配置与目录 | `config/default.yaml` 新增 `retrieval.phase4` 配置，`src/zeroshot_vdr/advanced/` 目录创建 | 已完成 |
| Stage 2 | page_id 解析与邻页工具 | `src/zeroshot_vdr/advanced/neighbors.py`，以及 `tests/phase4/test_neighbors.py` | 已完成 |
| Stage 3 | adaptive top-N 工具函数 | `src/zeroshot_vdr/advanced/two_stage.py` 中的 adaptive top-N 选择逻辑，`tests/phase4/test_adaptive.py` | 已完成 |
| Stage 4 | fixed top-N TwoStageRetriever | `TwoStageRetriever` 最小闭环落地，`tests/phase4/test_two_stage.py`，以及 `phase4_fixed_top32` / `phase4_fixed_topn` / `phase4_fixed_top128` 全量结果 | 已完成 |
| Stage 5 | MeanPoolCache | `src/zeroshot_vdr/advanced/mean_pool_cache.py`，全量 cache 目录 `outputs/cache/mean_pool_full_20260520_rerun/`，并修复 full cache 构建 OOM | 已完成 |
| Stage 6 | 接入 adaptive top-N | `phase4_adaptive` 与 `phase4_adaptive_cache_full_20260520` 全量结果 | 已完成 |
| Stage 7 | 接入 neighbor expansion | `phase4_adaptive_neighbors` 与 `phase4_adaptive_neighbors_cache_full_20260520` 全量结果 | 已完成 |
| Stage 8 | 新增评测脚本 | `scripts/run/run_phase4_eval.py`，并由 `main.py` 统一调度 | 已完成 |
| Stage 9 | trace 与 slice 分析 | `phase4_trace.jsonl`、`slice_metrics.csv`、`bucket_metrics.csv`、`trace_summary.json`，以及 `scripts/analyze_phase4_trace.py` | 已完成 |
| Stage 10 | 完整消融与报告 | fixed top-N、adaptive、adaptive+neighbor 的 valid-only 全量矩阵已完成，本报告即该阶段产物 | 已完成 |

---

## 3. 工程落地概览

### 3.1 两阶段检索主线

Phase 4 的核心实现位于：

- `src/zeroshot_vdr/advanced/two_stage.py`
- `src/zeroshot_vdr/advanced/neighbors.py`
- `src/zeroshot_vdr/advanced/profiling.py`

这条主线完成了三个关键约束：

1. `candidate_page_ids` 被严格当作 query-specific universe，而不是最终候选直接返回。
2. coarse 阶段只使用 mean-pooled page view，rerank 阶段只在 coarse / expanded candidates 上执行 full MaxSim。
3. Phase 3 baseline 默认行为不变；Phase 4 路径通过独立配置和独立脚本入口触发。

### 3.2 MeanPoolCache 与 full-scale 稳定化

缓存实现位于：

- `src/zeroshot_vdr/advanced/mean_pool_cache.py`

在 full run 过程中，原始 cache 构建曾因一次性 materialize 全量页面均值而膨胀到约 24 GB RSS。后续改为批量构建后，full cache 已可稳定在 87,090 页范围内完成构建与复用。

当前 full cache 产物：

- 路径：`outputs/cache/mean_pool_full_20260520_rerun/`
- `scope_num_pages = 87090`
- `embedding_dim = 128`
- `dtype = torch.float16`
- 目录大小约 26 MB

对比之下，原始 patch index 约 87 GB，因此 Stage 5 的工程收益非常明确。

### 3.3 脚本重组与统一入口

当前仓库执行入口已统一为：

- `scripts/run/`：实验运行脚本
- `scripts/command/`：环境、检查、清理、进度
- `main.py`：统一 CLI 入口

对应代表性脚本：

- `scripts/run/run_step3_eval.py`
- `scripts/run/run_step3_analysis.py`
- `scripts/run/run_phase4_eval.py`
- `scripts/run/run_phase4_full.sh`
- `scripts/command/check_env.py`
- `scripts/command/run_step3_clean.py`
- `scripts/command/check_phase4_progress.sh`
- `scripts/analyze_phase4_trace.py`

这一步虽然不直接改变指标，但显著降低了后续 full run、复盘和报告整理的运行摩擦。

### 3.4 测试与验证

Phase 4 对应测试文件已经补齐：

- `tests/phase4/test_neighbors.py`
- `tests/phase4/test_adaptive.py`
- `tests/phase4/test_two_stage.py`
- `tests/phase4/test_mean_pool_cache.py`

其中，cache batching 回归测试已经明确验证通过，并用于约束 full cache 构建不再一次性爆内存。

---

## 4. 实验设置

### 4.1 评测范围

- 数据集：MMLongBench DocumentQA
- 子任务：`longdocurl`、`mmlongdoc`、`slidevqa`
- 长度档位：`K4`、`K8`、`K16`、`K32`、`K64`、`K128`
- 主比较口径：**14,385 条 valid-only queries**

Phase 3 valid-only 对照目录为：

- `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/`

Phase 4 全量结果目录为：

- `outputs/eval_reports/phase4_fixed_top32/`
- `outputs/eval_reports/phase4_fixed_topn/`（fixed top-64）
- `outputs/eval_reports/phase4_fixed_top128/`
- `outputs/eval_reports/phase4_adaptive/`
- `outputs/eval_reports/phase4_adaptive_neighbors/`
- `outputs/eval_reports/phase4_fixed_topn_cache_full_20260520/`
- `outputs/eval_reports/phase4_adaptive_cache_full_20260520/`
- `outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/`

### 4.2 对比方法

主消融矩阵对齐开发指导 Stage 10：

1. Phase 3 Full MaxSim baseline
2. Fixed Top-32 + MaxSim
3. Fixed Top-64 + MaxSim
4. Fixed Top-128 + MaxSim
5. Adaptive + MaxSim
6. Adaptive + Neighbor + MaxSim

此外，为了完整验证 Stage 5 的收益，本报告额外纳入三条 full cache 运行：

1. Fixed Top-64 + Cache
2. Adaptive + Cache
3. Adaptive + Neighbor + Cache

### 4.3 指标与分析产物

主表使用 `summary.json` / `run_summary.json` 中的总体指标：

- Recall@1 / Recall@5 / Recall@10
- MRR
- nDCG@10
- Avg latency
- P95 latency
- Avg rerank candidates

切片表使用 `slice_metrics.csv` 口径；长候选集分桶分析使用 `bucket_metrics.csv` 口径。

---

## 5. 主结果

### 5.1 主消融结果

| Method | Valid Queries | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Avg Latency (s) | P95 Latency (s) | Avg Rerank Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 3 Full MaxSim | 14385 | 0.3342 | 0.7250 | 0.8517 | 0.5838 | 0.6325 | 0.0716 | 0.1384 | 32.7 |
| Fixed Top-32 + MaxSim | 14385 | 0.3340 | 0.7228 | 0.8482 | 0.5828 | 0.6308 | 0.0794 | 0.1434 | 19.0 |
| Fixed Top-64 + MaxSim | 14385 | 0.3344 | 0.7247 | 0.8513 | 0.5839 | 0.6325 | 0.0889 | 0.1800 | 26.8 |
| Fixed Top-128 + MaxSim | 14385 | 0.3344 | 0.7250 | 0.8517 | 0.5839 | 0.6326 | 0.0907 | 0.1989 | 32.2 |
| Adaptive + MaxSim | 14385 | 0.3340 | 0.7228 | 0.8482 | 0.5828 | 0.6308 | 0.0790 | 0.1417 | 19.0 |
| Adaptive + Neighbor + MaxSim | 14385 | 0.3342 | 0.7247 | 0.8523 | 0.5838 | 0.6325 | 0.0796 | 0.1437 | 19.8 |

从 no-cache 主消融看，Phase 4 的结论比较清晰：

- `Fixed Top-32` 和 `Adaptive` 都能显著降低 rerank candidate 数，但质量会有小幅回落。
- `Fixed Top-128` 基本恢复到 Phase 3 baseline 的质量，但时延比 baseline 更差，因此不是好的最终方案。
- `Adaptive + Neighbor` 在 no-cache 情况下取得了全矩阵中最高的 Recall@10，同时 nDCG@10 与 baseline 基本持平，是 Phase 4 主线中最有代表性的质量版本。

### 5.2 Cache 带来的收益

| Method | Recall@10 | nDCG@10 | No-cache Avg Latency (s) | Cache Avg Latency (s) | No-cache P95 (s) | Cache P95 (s) | 质量变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fixed Top-64 | 0.8513 | 0.6325 | 0.0889 | 0.0650 | 0.1800 | 0.1178 | 与 no-cache 等价 |
| Adaptive | 0.8482 | 0.6308 | 0.0790 | 0.0592 | 0.1417 | 0.0847 | 与 no-cache 等价 |
| Adaptive + Neighbor | 0.8523 | 0.6325 | 0.0796 | 0.0600 | 0.1437 | 0.0858 | 与 no-cache 等价 |

Stage 5 的关键结论是：**cache 没有改变质量，却显著降低了 coarse 阶段开销和端到端时延**。因此最终推荐方法不是单纯的 `adaptive_neighbors`，而是 `adaptive_neighbors + mean-pool cache`。

### 5.3 长候选集 K128 分桶结果

| Method | Recall@10 | nDCG@10 | Avg Latency (ms) | P95 Latency (ms) | Avg Rerank Candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 3 Full MaxSim | 0.6818 | 0.3902 | 109.6 | 189.2 | 81.1 |
| Fixed Top-64 + Cache | 0.6827 | 0.3902 | 96.4 | 136.8 | 64.0 |
| Adaptive + Neighbor + Cache | 0.6882 | 0.3916 | 71.5 | 97.4 | 34.8 |

这组结果说明，Phase 4 的主要收益集中在长候选集场景：推荐方法不仅把 K128 bucket 的平均 rerank candidates 从 81.1 降到 34.8，而且把 P95 latency 从 189.2 ms 降到 97.4 ms，同时 Recall@10 与 nDCG@10 还有小幅提升。

---

## 6. 重点切片分析

下表沿用 `slice_metrics.csv` 口径，对齐开发指导中要求重点关注的 5 个切片。对照方法为 Phase 3 valid-only baseline，比较对象为最终推荐方法 `adaptive_neighbors + mean-pool cache`。

| Slice | Phase 3 Recall@10 | Phase 4 Recall@10 | Δ Recall@10 | Phase 3 nDCG@10 | Phase 4 nDCG@10 | Δ nDCG@10 | Phase 3 Avg Latency (ms) | Phase 4 Avg Latency (ms) | Phase 3 Avg Rerank | Phase 4 Avg Rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slidevqa/K128` | 0.5597 | 0.5788 | +0.0191 | 0.2528 | 0.2597 | +0.0069 | 143.2 | 68.4 | 135.2 | 35.5 |
| `slidevqa/K64` | 0.6036 | 0.6113 | +0.0076 | 0.2736 | 0.2754 | +0.0018 | 89.0 | 69.6 | 67.2 | 33.9 |
| `slidevqa/K32` | 0.6533 | 0.6533 | +0.0000 | 0.2956 | 0.2956 | +0.0000 | 67.1 | 66.1 | 33.2 | 31.6 |
| `mmlongdoc/K128` | 0.9022 | 0.8871 | -0.0152 | 0.6887 | 0.6809 | -0.0078 | 88.3 | 70.0 | 56.4 | 34.3 |
| `longdocurl/K128` | 0.9584 | 0.9540 | -0.0043 | 0.7597 | 0.7563 | -0.0035 | 113.5 | 77.4 | 55.3 | 34.6 |

可以看到，Phase 4 的改进并不是所有 hardest slices 都同时提升，而是表现出比较明确的 trade-off：

- 对 `slidevqa/K64` 和 `slidevqa/K128`，推荐方法同时提升了质量和效率，是最核心的正向结果。
- 对 `longdocurl/K128` 和 `mmlongdoc/K128`，推荐方法以极小到中等的质量回落换来了明显的时延与候选规模下降。
- `slidevqa/K32` 基本保持不变，说明 neighbor 与 cache 至少没有伤害中等长度场景。

因此，Phase 4 的整体结论应表述为：

**推荐方法在总体指标上优于或持平于 baseline，主要收益集中在 `slidevqa` 长上下文切片；对另一些 K128 切片则呈现“轻微质量换显著效率”的 trade-off。**

---

## 7. 验收结论

对照开发指导 Stage 10 的验收条件，本阶段结论如下：

### 7.1 工程验收

- [x] Phase 3 baseline 默认行为未被破坏
- [x] valid-only 主表查询数固定为 14,385
- [x] `fixed_topn`、`adaptive`、`adaptive_neighbors` 三种方法都可通过 `run_phase4_eval.py` 运行
- [x] 每个实验均输出 `summary.json`、`slice_metrics.csv`、`bucket_metrics.csv`、`phase4_trace.jsonl`
- [x] Phase 4 对应测试文件与分析脚本已经落地

### 7.2 算法质量验收

Phase 3 valid-only baseline：

- Recall@10 = 0.8517
- nDCG@10 = 0.6325

最终推荐方法 `adaptive_neighbors + mean-pool cache`：

- Recall@10 = 0.8523
- nDCG@10 = 0.6325

因此：

- [x] 满足硬性质量标准：Recall@10 与 nDCG@10 都没有低于 baseline 0.005 以内阈值
- [x] 满足理想质量标准中的一部分：Recall@10 高于 Phase 3 baseline，nDCG@10 基本持平
- [x] `slidevqa/K64` 与 `slidevqa/K128` 这两个 hardest slices 出现了明确改善

### 7.3 效率验收

- [x] Avg rerank candidates 明显低于 full universe
- [x] 长候选集 K128 bucket 的 P95 latency 明显下降
- [x] MeanPoolCache 体量显著小于原始 patch index

推荐方法在 K128 bucket 上把 P95 latency 从 189.2 ms 降到 97.4 ms，下降幅度超过 20% 目标；因此，Phase 4 的效率目标也可以视为达成。

---

## 8. 后续工作

尽管 Phase 4 已完成，但仍有几项工作属于最终报告前的必要收口：

1. 将本报告中的主表与切片表整理为最终论文和答辩材料可直接复用的图表版本。
2. 在正式报告里明确说明 `slidevqa` 长上下文收益与 `longdocurl` / `mmlongdoc` K128 trade-off 的原因与边界。
3. 决定是否将方向 A 的索引压缩作为补充实验，或只放入 future work。
4. 完成最终 `README.md`、复现命令与提交包整理。

---

## 9. 本阶段交付物

- Phase 4 核心代码：`src/zeroshot_vdr/advanced/`
- Phase 4 评测脚本：`scripts/run/run_phase4_eval.py`
- 统一命令入口：`main.py`
- 结果目录：`outputs/eval_reports/phase4_*`
- Full cache：`outputs/cache/mean_pool_full_20260520_rerun/`
- 阶段三 valid-only 对照：`outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/`
- 本报告：`docs/Milestone_Report_Phase4.md`

至此，Phase 4 已经从“开发指导”状态切换为“工程与实验均完成”的里程碑状态，项目下一阶段可以正式转入最终论文、答辩材料和提交包的整理。