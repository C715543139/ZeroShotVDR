# Project Plan Revision v2

## 1. 修订目的

本次修订用于将 [docs/Project_Plan.md](docs/Project_Plan.md) 从“Phase 4 尚未开始”的旧状态，同步到当前仓库已经完成 Phase 4 工程落地与 full valid-only 实验的真实状态。

触发原因有三点：

1. `src/zeroshot_vdr/advanced/` 已经不再是预留目录，而是包含 `two_stage.py`、`neighbors.py`、`profiling.py`、`mean_pool_cache.py` 的正式实现目录。
2. 脚本结构已经从旧的 `scripts/*.py` 平铺布局，重组为 `scripts/run/`、`scripts/command/` 与 `main.py` 统一入口。
3. Phase 4 的 valid-only 全量实验矩阵、cache 版本、trace/slice/bucket 产物已经齐备，计划文档若仍写成“未开始”，将直接误导后续报告整理。

---

## 2. 修订依据

本次同步基于以下真实仓库状态与产物：

### 2.1 代码实现

- `src/zeroshot_vdr/advanced/two_stage.py`
- `src/zeroshot_vdr/advanced/neighbors.py`
- `src/zeroshot_vdr/advanced/mean_pool_cache.py`
- `src/zeroshot_vdr/advanced/profiling.py`
- `scripts/run/run_phase4_eval.py`
- `scripts/run/run_phase4_full.sh`
- `main.py`

### 2.2 测试与命令入口

- `tests/phase4/test_neighbors.py`
- `tests/phase4/test_adaptive.py`
- `tests/phase4/test_two_stage.py`
- `tests/phase4/test_mean_pool_cache.py`
- `scripts/command/check_env.py`
- `scripts/command/run_step3_clean.py`
- `scripts/command/check_phase4_progress.sh`

### 2.3 实验结果目录

- `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/`
- `outputs/eval_reports/phase4_fixed_top32/`
- `outputs/eval_reports/phase4_fixed_topn/`
- `outputs/eval_reports/phase4_fixed_top128/`
- `outputs/eval_reports/phase4_adaptive/`
- `outputs/eval_reports/phase4_adaptive_neighbors/`
- `outputs/eval_reports/phase4_fixed_topn_cache_full_20260520/`
- `outputs/eval_reports/phase4_adaptive_cache_full_20260520/`
- `outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/`
- `outputs/cache/mean_pool_full_20260520_rerun/`

---

## 3. 主要修订内容

### 3.1 顶部同步说明改版

将文档顶部的 v8 同步说明更新为 v9，明确以下事实：

- Phase 4 已完成，而不是“尚未开始”。
- `advanced/` 模块已经落地。
- 脚本目录已经重组。
- 当前推荐方法为 `adaptive_neighbors + mean-pool cache`。
- Phase 5 仍剩论文、PPT 与最终打包工作。

### 3.2 项目结构树同步到当前仓库

项目结构部分做了三类修订：

1. 将 `advanced/` 从占位目录改为真实文件列表。
2. 将 `scripts/` 结构改写为 `scripts/run/` 与 `scripts/command/`。
3. 将 `main.py`、`Milestone_Report_Phase4.md` 与新的 revision 文件纳入树状结构。

同时，`outputs/eval_reports/{run_name}/` 补入了 Phase 4 实际常见产物：

- `summary.json`
- `slice_metrics.csv`
- `bucket_metrics.csv`
- `phase4_trace.jsonl`
- `trace_summary.json`

### 3.3 环境与脚本章节重写

原文档仍以旧路径 `scripts/env.sh`、`scripts/check_env.py`、`scripts/run_step3_eval.py` 为主。当前已经统一改写为：

- Linux 激活入口：`source scripts/command/env.sh`
- 推荐执行方式：`python main.py ...`
- 直连脚本路径：`scripts/run/` 与 `scripts/command/`

并新增了对 `main.py` 分组命令、Phase 4 评测入口与辅助分析脚本的说明。

### 3.4 Phase 4 章节从“计划”改为“完成状态”

这是本次修订的核心：

- 删除“尚未开始”“预留目录”等旧表述。
- 保留方向 A 为暂缓项，避免误写成已完成。
- 将主线方向 B 按 Stage 0-10 映射成已落地表。
- 增加当前落地代码位置。
- 增加 14,385 条 valid-only 查询上的主结果表。
- 明确当前结论与推荐方法。

同步写入的主结论为：

- Phase 3 valid-only baseline：Recall@10 = 0.8517，nDCG@10 = 0.6325，Avg latency = 0.0716 s
- 推荐方法 `adaptive_neighbors + mean-pool cache`：Recall@10 = 0.8523，nDCG@10 = 0.6325，Avg latency = 0.0600 s

因此，Phase 4 在总体质量上持平略优，在效率上明显更优，可视为可接受且有价值的优化。

### 3.5 Phase 5 与里程碑勾选同步

原文 Phase 5 仍把大量前置工作当作“待做”。修订后将其改为“基于已有结果做最终收口”，并同步调整：

- 脚本重组与 `main.py` 统一入口已经完成
- `docs/Milestone_Report_Phase4.md` 已作为 Phase 5 输入准备就绪
- Milestone 1-4 改为已完成状态

---

## 4. 修订策略

本次修订遵循以下原则：

1. 只把真实已落地的内容写成“已完成”。
2. 不把方向 A 等未实现内容包装成已完成工作。
3. 结果表优先使用实验产物中的 `run_summary.json` / `summary.json`，避免凭记忆填数。
4. 路径说明以当前仓库实际入口为准，而不是保留历史脚本路径。

---

## 5. 结果

修订后的 [docs/Project_Plan.md](docs/Project_Plan.md) 已能正确反映当前项目状态：

- 前三阶段为已完成状态
- Phase 4 为已实现、已评测、已形成主结论的阶段
- Phase 5 为最终报告与答辩材料整理阶段

这份修订后的计划文档可直接作为后续 Phase 4 里程碑报告、最终实验报告和答辩材料的总导航文档使用。