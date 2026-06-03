# 评审问题应对 Checklist

> **项目**：ZeroShotVDR
> **目的**：把可能被评审/答辩问到的设计问题预先整理成结构化清单，并附上每条对应的代码、trace、表格、测试位置。便于答辩时快速跳转到证据。
> **最后更新**：2026-06-03

---

## 总原则

任何设计问题，建议按下面四步答：

1. **"为什么需要"** → 引用 Phase 3 failure analysis（`docs/Milestone_Report_Phase3.md` §5.3）三类失败 + `page_id` 不稳定问题；
2. **"为什么这样设计"** → 引用论文贡献点 + 代码契约；
3. **"为什么是当前这些超参"** → 引用 trace 实证 + `tests/phase4/test_adaptive.py` 单测；
4. **"代价和局限是什么"** → 引用论文 `## Limitations` 与 `## Where the Trade-off Appears` 段。

---

## 1. 为什么在 ColPali 之上还要做两阶段？ColPali 本身不就是为 VDR 设计的吗？

**回答要点**：

- **ColPali 的 MaxSim 是贵在 inference、不是贵在 indexing**。`S_max(Q,P) = Σᵢ maxⱼ ⟨q̂ᵢ, p̂ⱼ⟩` 是 O(m·r·d) per page，patch 越多越贵。`docs/NJUProject_VDR.md` §2 明确要求"在提升检索效果的同时，还需综合考虑索引构建、存储成本与推理开销"。
- Phase 3 baseline 量化了成本：K128 universe 平均 81.1 页，Avg latency 0.0716 s/query、P95 0.1384 s/query（论文 Table 1/2）。这在 K128 桶里几乎全部是 MaxSim 重复算。
- **替换模型不是作业允许的创新空间**——作业要求"方法设计必须为原创"（`NJUProject_VDR.md` §3.2），所以"换一个更小的 backbone"不构成我们的方法。围绕 ColPali 做检索策略 + cache 是合规的、且对原模型零侵入。
- 因此决定**保留 ColPali 的 MaxSim 作为最终打分器**，把它前面加一道便宜的 candidate controller。

**引用**：

- 代码：`src/zeroshot_vdr/retrieval/pipeline.py:165-167`（`query.candidate_page_ids` 透传）、`src/zeroshot_vdr/advanced/two_stage.py:38-44`（文档串注释）。
- 论文：`main.tex` §3.2 "The index is stored as independent page files…"、§3.3 "We assume that page images remain the retrieval unit and that ColPali's patch-level evidence is valuable"。

---

## 2. 为什么是 mean-pool，而不是 random projection / top-k patch / CLS token / 其他降维方法？

**回答要点**：

- **mean-pool 是 ColPali 自己就有"页向量"的最朴素实现**：把整页所有 patch 的 L2-normalized 向量求平均，再 normalize，得到的页向量与 query 的 mean 向量做 dot product 就是 coarse 分数。代码 8 行内可以写完、零额外训练。
- **它故意做得"弱"**，因为粗筛不需要判别力：只要求"top 段比 bottom 段显著高出"，不需要它排序与 MaxSim 一致。论文里 `### Why Adaptive Coarse-to-Fine Helps` 段说"mean-pooled representation is intentionally weak… Its role is to estimate which pages are plausible enough to deserve the expensive scorer"。
- **为什么不用 CLS / last-token / Q-Former 之类的"页向量"**：所有这些都需要重新训练，违反 zero-shot 约束；且要重新训就要在 MMLongBench 上做 training-set，等于在评测集上做适配。
- **实证上 mean-pool 排序已经够尖锐**：在最终 run 14,385 条 query 中，K128 universe=128 的 top1–top32 平均 margin 是 0.3960（`outputs/.../phase4_trace.jsonl` 的 `coarse_margin` 字段），最小的也 0.0916 远高于 0.035 的扩张阈值——说明 mean-pool 排序完全足以承担"top-32 候选"任务。

**引用**：

- 代码：`src/zeroshot_vdr/advanced/two_stage.py:135-148`（`score_mean_pool`、`mean_pool_query`）。
- Trace：见 Phase 4 trace JSONL 的 `coarse_margin` 字段。
- 测试：`tests/phase4/test_two_stage.py:374-376`（验证 trace 字段非空）。

---

## 3. 为什么 candidate 上限是 32/128、ratio 是 0.2、margin 是 0.035？这四个数怎么来的？

**回答要点**（按重要程度排序）：

- **为什么 max=128**：因为 K128 universe 平均 81.1，最大可达 177（`run_summary.json` 里的 `candidate_stats.max=177`）。max=128 是"如果整段都判别不出，就让 MaxSim 跑一遍"，避免在最坏情况下把候选剪到几乎不含答案。
- **为什么 min=32**：32 是个能稳定覆盖 top1 + 周围邻页的量级。trace 显示 K128 universe=128 的 query，top-1–top-32 平均 margin 0.3960，最小 0.0916，全部远高于 0.035——**实证 32 在我们的分布上就是够的**。再小则覆盖率下降，再大则失去剪枝意义。
- **为什么 base_ratio=0.2**：universe × 0.2 是想"取前 1/5 段"，作为"分数够尖锐时的默认预算"。对 K128=128 来说，0.2×128=26 钳到 32，正好是 min。对 K64=64，0.2×64=13 钳到 32。
- **为什么 flat_margin=0.035**：这是个**经验阈值**——它和 ColPali mean-pool 分数尺度（0~1，K128 上 top1 通常 0.2~0.6，bottom 通常 <0.05）一起凭 trace 调出来的。它足够小，正常分布不会触发扩张；又足够大，能在真出现"top 段彼此都挤在一起"时翻倍兜底。

**承认局限**：这四个数是**手设的、不是学出来的**。论文 `## Limitations` 已写明："the adaptive rule is hand-designed. It is motivated by the score distribution and validated through ablations, but it is not learned." 在 zero-shot 课程设定下不能训练，所以这反而是必要的。

**引用**：

- 配置：`config/default.yaml` 第 31-34 行（`min_candidates=32 / max_candidates=128 / base_ratio=0.2 / flat_margin=0.035`）。
- 单元测试：`tests/phase4/test_adaptive.py::test_sharp_distribution_no_expand` / `test_flat_distribution_expand` / `test_respects_min_bound` / `test_respects_max_bound` / `test_expand_not_exceed_universe`——11 条单测把四条分支都覆盖了。
- 真实 trace：本次 14,385 条 query 中扩张触发 0 次的统计。

---

## 4. 为什么 margin 用 top1–top_base_n 的差，而不是相邻差或方差？

**回答要点**：

- 相邻差（gap）描述的是"局部抖动"，对噪声敏感；top1–topN 描述的是"前 32 名整体的高度"，对整体形状更鲁棒。
- 论文 `## Adaptive candidate selection` 段明确写"flat coarse score distributions may require more candidates to avoid dropping relevant evidence"——判断"分布是否平坦"就是看 top 段整体落差，不是看相邻。
- 实证上相邻差控制不好就会因为 mean-pool 分数里偶尔一个噪声大值而误触发；top1–topN 差是单一标量，更稳定。
- 论文还说"a flat-score margin of 0.035"——这就是 top1–topN 的差值阈值。

**引用**：

- 代码：`src/zeroshot_vdr/advanced/two_stage.py:66-69`（`margin = sorted_scores[0] - sorted_scores[base_n - 1]`）。
- 论文：`main.tex` line 122-124 那一段。

---

## 5. 为什么要做 Neighbor Expansion？为什么不是扩展所有 coarse 页，而是只扩展前 8 个 seed？

**回答要点**：

- **失败模式数据驱动**：Phase 3 失败分析 §5.3 写明三类失败：① 大候选集 miss_top10、② 多页部分召回、③ **邻页混淆**。邻页扩展直接对应第 ③ 类——粗筛在 slide 1 找到了答案，但 slide 0/2 也含部分证据。
- **为什么是 ±1 window（neighbor_window=1）**：单页 window 是"页面级证据连续性"的最保守选择。Slide deck 的"上一页继续讲解 / 下一页进入下一节"通常发生在 ±1 之内。论文：`main.tex` line 128 "expands around the top coarse seeds by adding pages within a one-page window when those pages are present in the query-specific universe"。
- **为什么只扩前 8 个 seed**：扩 8 × 2 = 16 个邻页加到 32 上就是 48，仍然在 max=128 内安全；如果扩全部 32，邻居数 64，总数 96，会立刻让 K128 失去剪枝收益。论文：`main.tex` line 131 "The implementation therefore expands only around the top seed pages."
- **实证**：trace 中 `avg_neighbor_added=0.81`（run_summary），说明大部分 query 的"前 8 周围 ±1"几乎没有"答案"或"邻页证据"在 universe 内可加。这反过来印证 coarse 32 候选已经覆盖大部分答案，少量扩出来的是补漏。

**引用**：

- 代码：`src/zeroshot_vdr/advanced/neighbors.py:62-90`（`expand_neighbors`）。
- 论文：`main.tex` line 126-131 整段 + Table 4 的 slidevqa/K128 提升 +0.0191。

---

## 6. 为什么需要 Mean-Pool Cache？它解决了什么、改变了什么？

**回答要点**：

- **它改变的是 coarse 阶段 I/O，不改变 ranking 质量**。论文明确写："The cache changes coarse-stage efficiency, not ranking quality, because the same mean vectors are used."
- 不加 cache 也要做 mean-pool；加了 cache 只是把"每 query 重新从 patch index 加载 + reduce_mean" 改为"每 query 直接读 fp16 矩阵"。索引大小从 88.6 GB 降到 ~26 MB（论文 line 135），是 **3,400 倍** 的存储压缩。
- **关键证据：cache/no-cache 同方法的质量完全相同**。论文 Table 3：

  | Method           | R@10   | nDCG@10 |
  | ---------------- | ------ | ------- |
  | Adaptive         | 0.8482 | 0.6308  |
  | Adaptive + Cache | 0.8482 | 0.6308  |

  数字完全一致——说明 cache 是纯工程优化，对学术结论零影响。

- **还有一个工程故事**：第一次构造 cache 时一次性把所有页面的 mean-pool 张量在内存里 material 出来 OOM；后来改成 batched 构造（`docs/Milestone_Report_Phase4.md` §3.2）才跑通。这说明这个 cache 不只是"省时"，还直接决定了 full-scale 评估能不能跑完。

**引用**：

- 代码：`src/zeroshot_vdr/advanced/mean_pool_cache.py`。
- 论文：`main.tex` line 132-135 + Table 3。
- 文档：`docs/Milestone_Report_Phase4.md` §3.2。

---

## 7. 为什么 Paper 里说 "fix top-128 反而比 baseline 慢"，不觉得奇怪吗？

**回答要点**：

- 不奇怪。Fix top-128 的流程是：① 跑 mean-pool 粗筛（O(n·d) 一次）→ ② 取 top-128 → ③ 对这 128 页跑完整 MaxSim（O(m·r·d) per page）。K128 universe 平均 81.1，所以 128 ≈ 81.1——**剪掉的 0 页**。
- 与 baseline 相比，Fix top-128 等于"baseline + 多一遍 mean-pool 粗筛"——多了一笔固定开销，自然就比 baseline 慢。
- 这是论文 `### Where the Trade-off Appears` 段最有力的一笔：固定常数法在长上下文里**既不省力又不显著提质**；只有"自适应"才能按 query 真正需要把预算降下来。这是方法存在的核心理由。

**引用**：

- 论文：`main.tex` line 165-168 段 "Top-128 recovers baseline quality, but it is slower than the baseline because it adds a coarse pass while still reranking almost the same number of pages"。
- 表 1 数据：Top-128 Avg 0.0907 vs Full MaxSim 0.0716。

---

## 8. 为什么 candidate universe 一定要用 query.candidate_page_ids，不能用 index_store 按 doc_id 全查？

**回答要点**：

- 这两个不等价，原因是**page_id 跨 (subtask, length) 不唯一**。MMLongBench 同一 `doc_name` 可能在 K4/K8/.../K128 多个档位下被切成不同子区间（page47~page144 之类）。如果按 `query.doc_id` 在 index_store 里查，可能召回错档位的页（K4 的 doc 也可能命中 K128 子集中恰好 doc_name 相同的另一段）。
- 用 `query.candidate_page_ids`（由 `DocumentQAAdapter._build_candidate_page_ids` 在样本级显式枚举）就锁死了"该 query 实际看到的页"。`src/zeroshot_vdr/advanced/two_stage.py:185-210` 的 `resolve_candidate_universe` 优先级 2 就是这个字段。
- 这是**协议层决策**，不是 trick。论文把它放在 `### Task Definition and Protocol` §3.1 单列一段（"Each query also carries `candidate_page_ids`, which fixes the query-scoped candidate universe"），与 valid-only 协议并列。
- 没了这条，对比就是不公平的——比如 Adaptive 可能因为偷看了 baseline 没看到的页而"赢"。

**引用**：

- 代码：`src/zeroshot_vdr/data/adapters.py:178-198`（`_build_candidate_page_ids`）、`src/zeroshot_vdr/advanced/two_stage.py:185-210`（`resolve_candidate_universe`）。
- 论文：`main.tex` line 73-77 那一段。

---

## 9. 为什么 mmlongdoc/K128 / longdocurl/K128 反而掉点？这是 bug 还是预期？

**回答要点**：

- 是**预期内 trade-off**，不是 bug。论文 `### Where the Trade-off Appears` 段已经写明：
  - mmlongdoc/K128: R@10 0.9022 → 0.8871（-0.0152）
  - longdocurl/K128: R@10 0.9584 → 0.9540（-0.0043）
- **为什么掉**：长文档 + 长 web 文档的"答案证据"是**分散**在多页的（不是邻页、不是模板化的），mean-pool 粗筛把"分散的非相邻页"rank 偏低了；neighbor ±1 window 又补不回分散的页。
- **为什么可以接受**：
  1. 主表 overall R@10 是 +0.0006，整体不输 baseline；
  2. 在 K128 桶内，反而 **+0.0064 R@10** 提升（论文 Table 4）；
  3. 课程 advanced task 的评分（`NJUProject_VDR.md` §3.2）允许"在保持较好效果的前提下显著降低计算与存储开销"——效率维度上的提升足以补偿。
- **怎么进一步缓解**：multi-page aggregation（`### Failure Modes` 段明示的下一步），但这超出 zero-shot 单 query 范围。

**引用**：

- 论文：`main.tex` line 196-201 段 + `## Discussion` 段。
- 表数据：论文 Table 5 五条 slice。

---

## 10. 为什么报告里全程说 "query-scoped candidate retrieval"，是不是回避了真正的全局检索？

**回答要点**：

- 不是回避，是**任务定义**。`NJUProject_VDR.md` §3 明确给的样本结构就是 query-specific 候选集（DocumentQA 的 `page_list` 字段）；MMLongBench 也只在 DocumentQA 子集上提供页级 ground truth。
- **如果做全局检索**，candidate universe 是全 87,922 页的 patch index，平均 K128 也要跑 128 页 MaxSim 一次，但**评测 ground truth 仍然是 query 内**——这会导致大量"正确的文档页"被误判为"错的"，因为评测不奖励跨文档命中。
- 论文 `## Limitations` 第一条已经写明："the reported numbers should not be read as document-level retrieval from all pages in the corpus"。这是一个**自觉的实验边界声明**，不是缺陷。
- 进阶任务文档 `NJUProject_VDR.md` §3.2 强调"在保持较好效果的前提下显著降低计算与存储开销"——这本来就该在 query-scoped 设定下优化预算，而不是去解决"先找文档"那一步。

**引用**：

- 论文：`main.tex` `## Limitations` 段第一句。
- 文档：`docs/Milestone_Report_Phase3.md` §4.1 1,192 条无效标注的处理。

---

## 11. 为什么 Phase 3 还要做 page_id 重建？这跟 Phase 4 两阶段有什么关系？

**回答要点**：

- **没有任何方法上的关系**，但**没有它就 Phase 4 也不能成立**。原因：
  - Phase 3 早期 page_id 走"样本内 doc_name + enumerate(page_list)"——同一 `doc_name` 在 K4/K8/.../K128 多档位下会冲突；K128 子集的 `page_idx=0` 物理上指 page47，不是 page0。
  - 这样预测的 page_id 和 ground truth 的 page_id 字符串虽然可能长得像，但**指向不同的源页**，Recall/Precision 都会系统性错算。
  - Phase 3 stable run 修复后，Recall@10 从 0.6721 提升到 0.8517（`docs/Milestone_Report_Phase3.md` §4.3）——这 18 个百分点的差距其实就是 page_id 对齐的差距。
- **对 Phase 4 的直接依赖**：
  1. mean-pool cache 用 page_id 当 key；如果 page_id 不稳定，cache 命中率直接崩塌；
  2. 邻页扩展用 `parse_page_id` 抽 page_idx ± 1；如果 page_id 反映的不是源页号，扩展出去的不是物理上"上一页/下一页"；
  3. trace JSONL 的 `coarse_margin` 等字段需要 page_id 跨 query 可比，否则 trace 分析毫无意义。
- 论文里这段写在 §3.1 `### Task Definition and Protocol`，与 valid-only 协议并列，强调"是协议决策，不是算法改进"。

**引用**：

- 代码：`src/zeroshot_vdr/contracts.py:60-90`（`build_page_id`、`build_page_id_from_image`、`extract_source_doc_id`）、`src/zeroshot_vdr/data/adapters.py:178-210`。
- 论文：`main.tex` line 73-77 那一段 + line 32-34 "These corrections are not the main algorithmic contribution, but they are essential for trustworthy comparison."

---

## 12. 为什么 valid-only 协议？15,577 → 14,385 砍掉 1,192 条，会不会让结果偏乐观？

**回答要点**：

- 不会偏乐观；反而**更悲观（更保守）**。
- 1,192 条被砍是因为它们的 `ans_page_list` 是空、`[-1]`、含 `0`、或越界（`docs/Milestone_Report_Phase3.md` §4.1）。当前指标实现会把空 relevant set 记为"完美召回"，所以不剔除的话 Recall / MRR 会被**虚高**。
- 论文 §3.1 段明确写："These cases are excluded from the main comparison because an empty relevant set would otherwise make recall and MRR misleading."
- 保留在附录的 full disclosure 数字（论文附录 Table 6/7）就是为了让评审可以反过来检查：valid-only 数字**不高于** full disclosure。比较 baseline：full R@10=0.8630 vs valid-only R@10=0.8517——valid-only 更小，正是因为它没"作弊"。
- 论文 `## Limitations` 也写明了"1,192 queries with unusable page-level supervision are excluded from the main table"，把这一限制显式说出来。

**引用**：

- 论文：`main.tex` line 67-71 段。
- 文档：`docs/Milestone_Report_Phase3.md` §4.1/§4.3。

---

## 13. Latency 数字 0.0600 / 0.0858 是怎么测的？硬件换了还能复现吗？

**回答要点**：

- 论文 `### Implementation` 段写明："Experiments were run on Ubuntu with two NVIDIA RTX 3090 GPUs with 24 GB memory each"。
- 数字是**单 query 平均**，不包含索引构建阶段。`scripts/analyze_phase4_trace.py` 聚合 `trace.total_ms` 得到 avg/P95（论文 Table 2 的"Avg Lat."列）。
- 论文 `## Limitations` 第四段已经写明："latency numbers are hardware-dependent; the relative trends are more important than the absolute seconds/query on another machine"。换硬件绝对值会变，**但 relative 趋势**（cache 比 no-cache 快 25%，adaptive 比 baseline 在 K128 上快 35%）会保持。
- 复现命令论文 `## Reproducibility Notes` 都给了路径：`outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/`，配置 `adaptive_neighbors + min_candidates=32 + max_candidates=128 + base_ratio=0.2 + flat_margin=0.035 + neighbor_window=1 + neighbor_seed_n=8 + use_mean_pool_cache=true`。

**引用**：

- 论文：`main.tex` line 153-156 段（Implementation）、line 234-235 段（Limitations）、line 287-291 段（Reproducibility Notes）。
- 配置：`config/default.yaml` 的 `retrieval.phase4` 整段。

---

## 14. 如果我换一份数据，方法还能 work 吗？

**回答要点**：

- **方法本身不带数据集先验**：
  - mean-pool 是 ColPali-v1.3 输出的标准操作，换数据不需要改；
  - adaptive top-N 的超参（0.2 / 0.035 / 32 / 128）只跟"分数分布"有关，不跟 page 数有关；
  - 邻页 ±1 假设"答案多在邻页"，这是文档类数据的常见先验；如果换非文档类（如 NIAH 的"找一张特定子图"），邻页扩展应关掉。
- **zero-shot 约束**：整个方法没有在 MMLongBench 上做训练；超参是用 trace 调出来的，但**调超参不接触 ground truth**，调的是 mean-pool 分数的形状。
- **承认限制**：论文 `## Limitations` 段最后一条说"the adaptive rule is hand-designed. It is motivated by the score distribution and validated through ablations, but it is not learned"——这是诚实声明。如果数据集分布极偏（比如 coarse 分数完全无法区分 top/bottom），margin 阈值需要重调。

**引用**：

- 论文：`main.tex` `## Limitations` 整段、`## Conclusion` 最后一句。

---

## 15. 整体口径模板

评审常问的元问题"**你这个方法到底是干什么的**"，可以这样收束：

> 我们不是要替换 ColPali，而是在 ColPali 不变的前提下，把"对多少页跑 MaxSim"这件事做得**对每个 query 自适应**：先用 ColPali 自己的 mean-pool 当 cheap candidate controller；用 coarse 分数的形状（top1–topN 的 margin）决定预算应该小（32）还是大（64 或更多）；再对粗筛出的前 8 个 seed 做 ±1 邻页扩展以防邻页证据漏网；最后把所有工程加速（mean-pool cache、cache batched 构造、Phase 3 stable page_id 协议）落到位。
>
> 结果是：在 valid-only 14,385 query 上，R@10 持平或略升（+0.0006），nDCG@10 不变（0.6325），**P95 时延从 138.4 ms 降到 85.8 ms，降幅 38%**；K128 桶内 R@10 提升 +0.0064 同时 P95 减半。在 slidevqa/K128 这类视觉重复最严重的子集上 R@10 直接 +0.0191。
>
> 整个方法 0 训练、对原模型零侵入，所有设计取舍都有对应的 trace 数字和单元测试托底。

---

## 答辩准备 Checklist

| #   | 问题                        | 准备好的证据                                                       |
| --- | --------------------------- | ------------------------------------------------------------------ |
| 1   | 为什么要两阶段              | Phase 3 失败分析 §5.3 + 论文 §3.3 + ColPali MaxSim 复杂度          |
| 2   | 为什么要 mean-pool          | zero-shot 约束 + trace margin 实证                                 |
| 3   | 4 个超参怎么来              | trace 数字 + `test_adaptive.py` 11 条单测                          |
| 4   | 为什么是 top1–topN margin   | 论文原文 "flat coarse score distributions"                         |
| 5   | 为什么 neighbor ±1 + 8 seed | 失败模式 + 论文 line 126-131 + run_summary avg_neighbor_added=0.81 |
| 6   | 为什么要 cache              | Table 3 cache/no-cache 同质量 + 88.6 GB → 26 MB 压缩               |
| 7   | Fix-128 为什么更慢          | Table 2 数字 + "Where the Trade-off Appears" 段                    |
| 8   | 为什么用 candidate_page_ids | 协议决策 + doc_name 跨档冲突 + 两阶段依赖                          |
| 9   | mmlongdoc/K128 为什么掉     | 论文 "Where the Trade-off Appears" + Limitations 显式声明          |
| 10  | 为什么不全局检索            | `NJUProject_VDR.md` §3 + 论文 Limitations 第一条                   |
| 11  | page_id 重建的必要性        | 0.6721 → 0.8517 实证 + cache key 依赖 + 邻页 ±1 依赖               |
| 12  | valid-only 协议             | full/valid-only 数字对比 + 空 relevant set 虚高问题                |
| 13  | 时延可复现性                | Reproducibility Notes 段 + Limitations 段                          |
| 14  | 跨数据集泛化                | Limitations 段 + zero-shot 无训练声明                              |
| 15  | 一句话总结                  | 见 §15 整体口径模板                                                |

每条都已经在项目里有具体代码、trace、表格、测试对应——答辩时随时能跳到对应文件打开验证。
