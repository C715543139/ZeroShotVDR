# Milestone Report Phase4 Revision v1

## 1. 修订目的

本次修订用于将 [docs/ZeroShotVDR_Phase4_Iterative_Development_Guide.md](docs/ZeroShotVDR_Phase4_Iterative_Development_Guide.md) 从“阶段四开发指导文档”转写为 [docs/Milestone_Report_Phase4.md](docs/Milestone_Report_Phase4.md) 这一份基于真实实现与真实实验产物的里程碑报告。

核心目标不是复述开发计划，而是回答三件事：

1. Stage 0-10 是否真的已经落地。
2. Phase 4 相较 Phase 3 是否带来了可接受的改进。
3. 哪些结论可以直接进入最终实验报告，哪些仍应保留为 future work。

---

## 2. 数据来源

本次报告转写使用的依据分为三类。

### 2.1 指导与历史文档

- [docs/ZeroShotVDR_Phase4_Iterative_Development_Guide.md](docs/ZeroShotVDR_Phase4_Iterative_Development_Guide.md)
- [docs/Milestone_Report_Phase3.md](docs/Milestone_Report_Phase3.md)
- [docs/Project_Plan.md](docs/Project_Plan.md)

### 2.2 代码实现

- `src/zeroshot_vdr/advanced/two_stage.py`
- `src/zeroshot_vdr/advanced/neighbors.py`
- `src/zeroshot_vdr/advanced/mean_pool_cache.py`
- `src/zeroshot_vdr/advanced/profiling.py`
- `scripts/run/run_phase4_eval.py`
- `scripts/analyze_phase4_trace.py`
- `main.py`

### 2.3 实验产物

阶段三对照：

- `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/run_summary.json`
- `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/slice_metrics.csv`
- `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/bucket_metrics.csv`

阶段四主结果：

- `outputs/eval_reports/phase4_fixed_top32/summary.json`
- `outputs/eval_reports/phase4_fixed_topn/summary.json`
- `outputs/eval_reports/phase4_fixed_top128/summary.json`
- `outputs/eval_reports/phase4_adaptive/summary.json`
- `outputs/eval_reports/phase4_adaptive_neighbors/summary.json`
- `outputs/eval_reports/phase4_fixed_topn_cache_full_20260520/summary.json`
- `outputs/eval_reports/phase4_adaptive_cache_full_20260520/summary.json`
- `outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/summary.json`

配套分析产物：

- 各目录下的 `slice_metrics.csv`
- 各目录下的 `bucket_metrics.csv`
- 各目录下的 `trace_summary.json`
- 各目录下的 `phase4_trace.jsonl`

---

## 3. 转写原则

本次从 guide 转成 report 时，采用了以下写法约束：

### 3.1 只写真实完成项

开发指导中存在两条路线：

- 方向 A：查询感知的自适应索引压缩
- 方向 B：查询自适应的两阶段粗精检索

本次报告只将方向 B 写为“已完成”。方向 A 虽然在规划中被提出，但并未在当前仓库里形成正式实现或正式实验矩阵，因此在报告中只能写成“暂缓 / future work”，不能伪装为阶段成果。

### 3.2 主表与切片表分离口径

报告中的总体主表使用 `summary.json` / `run_summary.json` 的总体指标；切片分析表使用 `slice_metrics.csv` 口径。

这样做有两个原因：

1. 总体结果最适合直接反映 valid-only 主结论。
2. 切片分析需要复用 Phase 4 trace 分析链路生成的同 schema 表格，方便与阶段三 backfill 对齐。

### 3.3 结果优先于叙述

原 guide 以“下一步要做什么”为主；转写后的 report 以“已经交付了什么、结果如何、是否达标”为主，具体体现为：

- Stage 0-10 全部改写为已落地映射表
- 增加工程落地位置说明
- 增加主结果表、cache 对比表、关键切片表
- 增加对验收条件的逐项判断

### 3.4 明确 trade-off，而不是强行渲染全面提升

Phase 4 的真实表现不是“所有子任务和所有档位全面变好”，而是：

- 总体 Recall@10 持平略优
- 总体 nDCG@10 基本持平
- 总体时延显著下降
- `slidevqa/K64` 与 `slidevqa/K128` 提升明显
- `longdocurl/K128` 与 `mmlongdoc/K128` 存在轻微到中等质量回落

因此，报告必须把它写成“可接受且有价值的质量-效率优化”，而不是夸大成“全面性能提升”。

---

## 4. 形成的关键结论

写入 [docs/Milestone_Report_Phase4.md](docs/Milestone_Report_Phase4.md) 的核心结论如下：

1. Stage 0-10 已在当前仓库完成落地，其中 Stage 5 还包含一次重要的 full-cache OOM 修复。
2. 推荐方法为 `adaptive_neighbors + mean-pool cache`。
3. 在 14,385 条 valid-only 查询上，推荐方法达到 Recall@10 = 0.8523、nDCG@10 = 0.6325、Avg latency = 0.0600 s/query、P95 latency = 0.0858 s/query。
4. 相较 Phase 3 valid-only baseline，推荐方法在总体质量上持平略优，在效率上明显更好。
5. 最有代表性的提升来自 `slidevqa` 长上下文切片，而非所有 K128 切片同时提升。

---

## 5. 仍需注意的边界

这份 report 已足以作为阶段四里程碑材料，但仍有两个边界需要在后续总报告中继续说明：

1. 方向 A 尚未进入正式实验矩阵，应保留到 future work，而不是并入当前主贡献。
2. 切片层面的收益与损失并不完全均匀，因此最终论文需要解释为什么 `slidevqa` 受益更大，以及为什么某些 K128 切片选择了“轻微质量换显著效率”的 trade-off。

---

## 6. 结果

本次 revision 完成后，仓库已经具备一份与 Phase 3 报告风格一致、但内容完全基于 Phase 4 真实实现和真实结果的里程碑报告：

- [docs/Milestone_Report_Phase4.md](docs/Milestone_Report_Phase4.md)

它可以直接作为最终实验报告与答辩材料中“改进方法”部分的事实基础文档使用。