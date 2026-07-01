# ZeroShotVDR 最终报告

## 摘要

ZeroShotVDR 面向零样本视觉文档检索（Zero-Shot Visual Document Retrieval, VDR）：给定自然语言查询，在视觉丰富的长文档候选页面中返回最相关的证据页。项目以 ColPali-v1.3 为基础，在 MMLongBench DocumentQA 上实现了稳定页级检索 baseline，并进一步实现查询自适应两阶段检索方法。最终系统使用 mean-pool 页面向量进行粗检索，按查询分数分布自适应选择候选数，对高置信候选扩展邻页，再用完整 ColPali MaxSim 做精排；同时用 MeanPoolCache 复用粗检索页面向量。

在 14,385 条具有有效页级标注的 valid-only DocumentQA queries 上，最终 `Adaptive + Neighbor + MeanPoolCache` 方法达到 Recall@10 = 0.8523、nDCG@10 = 0.6325；Phase 3 Full MaxSim baseline 为 Recall@10 = 0.8517、nDCG@10 = 0.6325。最终方法在基本保持检索质量的同时，将平均延迟从 0.0716 s/query 降至 0.0600 s/query，将 P95 延迟从 0.1384 s/query 降至 0.0858 s/query，并将平均 full MaxSim rerank 页面数从 32.7 降至 19.8。长候选集场景收益更明显：候选全集规模桶 K128 上，P95 延迟从 189.2 ms 降至 97.4 ms，Recall@10 从 0.6818 提升至 0.6882。

## 1. 项目背景与目标

传统文档检索通常依赖 OCR 文本和文本索引，但视觉丰富文档中的关键信息经常来自表格结构、图表坐标、版式关系、幻灯片布局和图文组合。纯文本管线容易丢失这些视觉线索。ColPali 将页面图像编码为 patch-level embeddings，并通过 late interaction MaxSim 在查询 token 和页面 patch 之间进行细粒度匹配，适合 OCR-free 页级检索。

本项目的目标有两个层次：

1. 完成稳定的 ColPali-based baseline：包括 MMLongBench DocumentQA 数据适配、页面索引、查询编码、MaxSim 检索、指标计算和结果分析。
2. 完成进阶改进：在不训练新模型的零样本设定下，减少完整 MaxSim 计算开销，并尽量保持或提升检索质量。

最终系统不是重新训练视觉检索模型，而是在 ColPali baseline 外围构建更稳健的评测协议和更高效的 query-adaptive reranking 策略。

## 2. 任务定义与评测协议

### 2.1 Query-scoped 页级检索

MMLongBench DocumentQA 的每条样本包含查询文本、候选页面列表 `page_list` 和答案页标注 `ans_page_list`。本项目将每条 query 的候选页面集合记为 `C(q)`，将答案页集合记为 `G(q)`。所有主实验都在 `C(q)` 内排序页面，目标是在 top-k 中尽可能召回 `G(q)`。

这意味着本报告中的结果是 query-scoped candidate retrieval，而不是从全语料所有页面中先找文档再找页面的 global corpus retrieval。这个边界与 DocumentQA 数据格式一致，也保证 baseline 和 Phase 4 方法在相同候选全集上比较。

### 2.2 稳定页面身份

项目早期分析发现，若直接使用样本内 `page_list` 枚举位置构造页面 ID，不同长度档位和样本之间会出现页面身份不稳定，导致预测页面和 ground truth 页面的语义不一致。最终实现统一使用源图像路径恢复稳定页面身份，格式为：

```text
{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}
```

对应实现位于 `src/zeroshot_vdr/contracts.py` 和 `src/zeroshot_vdr/data/adapters.py`。`DocumentQAAdapter` 从图片路径中解析 source document 和 source page index，为 `Page`、`Query.candidate_page_ids` 和 `RelevanceJudgment` 使用同一套 `page_id` 契约。

### 2.3 Valid-only 主口径

原始 full run 中共有 15,577 条 queries，其中 1,192 条存在空标注、非法页码或越界答案页，无法形成可靠页级 ground truth。当前指标函数对空 relevant set 会按信息检索惯例返回完美召回或完美 MRR；如果把这些样本混入主表，会系统性抬高结果。因此主比较只使用 14,385 条 valid-only queries。

主指标为 Recall@k、Precision@k、MRR、nDCG@k，`k` 取 1、3、5、10。最终报告重点分析 Recall@10、nDCG@10、平均延迟、P95 延迟和平均 full MaxSim rerank 页面数。

## 3. 系统实现

### 3.1 Baseline: Full MaxSim

Baseline 使用 ColPali-v1.3 分别编码页面图像和查询文本。页面侧为 patch embeddings，查询侧为 token embeddings。给定 query embedding `Q` 和 page embedding `P`，MaxSim 分数为：

```text
S(Q, P) = sum_i max_j <q_i, p_j>
```

baseline 对 query 的全部 `candidate_page_ids` 执行完整 MaxSim，并按分数降序返回 top-k 页面。代码结构如下：

- `src/zeroshot_vdr/data/`：MMLongBench DocumentQA 适配与语料构建。
- `src/zeroshot_vdr/indexing/`：ColPali 页面编码和逐页独立索引存储。
- `src/zeroshot_vdr/retrieval/`：查询编码、MaxSim 打分和检索流水线。
- `src/zeroshot_vdr/evaluation/`：Recall、Precision、MRR、nDCG 计算。
- `main.py`：统一命令入口，封装 Step 3、Phase 4 和 command 子命令。

逐页独立存储相比单一巨大张量更利于按稳定 `page_id` 追踪、恢复和审计，也为 Phase 4 的 query-scoped 候选裁剪提供了清晰边界。

### 3.2 Query-adaptive two-stage retrieval

最终方法位于 `src/zeroshot_vdr/advanced/`，核心流程为：

```text
Query.candidate_page_ids
 -> mean-pool coarse ranking
 -> adaptive top-N selection
 -> neighbor expansion
 -> full MaxSim rerank
 -> top-k results
```

Mean-pool coarse retrieval 将每页 patch embeddings 平均为一个页面向量，将查询 token embeddings 平均为一个查询向量，然后用向量点积进行粗排序。该阶段不能替代 MaxSim 的细粒度匹配，但足以快速判断哪些页面值得进入精排。

Adaptive top-N 根据候选全集大小和 coarse 分数分布选择 rerank 候选数。最终配置为：

- `min_candidates = 32`
- `max_candidates = 128`
- `base_ratio = 0.20`
- `flat_margin = 0.035`

当候选全集较小，系统保留全部页面；当候选全集较大且 coarse 分数分布明显，系统显著减少进入 full MaxSim 的页面数；当分数分布较平坦，系统扩大候选以降低漏召风险。

Neighbor expansion 解决长文档和幻灯片中常见的邻页混淆。最终方法对 coarse 排名前 8 个 seed 页面扩展前后 1 页，且只加入当前 query 的 `candidate_page_ids` 中实际存在的页面。这样可以补回相邻证据页，同时避免退化为全量 rerank。

### 3.3 MeanPoolCache

MeanPoolCache 将 mean-pooled 页面向量缓存到磁盘，减少每个 query 粗检索阶段重复读取 patch index 并求均值的开销。当前 cache 目录为：

```text
outputs/cache/mean_pool_full_20260520_rerun/
```

缓存元信息：


| 项目             |        数值 |
| ------------------ | ------------: |
| 缓存页数         |      87,090 |
| embedding dim    |         128 |
| dtype            |     float16 |
| cache 文件总大小 | 约 26.95 MB |

MeanPoolCache 不改变排序质量；它只加速 coarse 阶段。最终 MaxSim 精排仍依赖完整 patch index，因此本方法主要优化推理时 rerank 成本，而不是完全替代视觉 patch 索引。

## 4. 实验设置

### 4.1 数据集范围

实验使用 MMLongBench DocumentQA，包含三个子任务和六个长度档位：

- 子任务：`longdocurl`、`mmlongdoc`、`slidevqa`
- 长度档位：K4、K8、K16、K32、K64、K128
- 主比较：14,385 条 valid-only queries

Valid-only query 分布如下：


| Subtask      |   K4 |   K8 |  K16 |  K32 |  K64 | K128 | Total |
| -------------- | -----: | -----: | -----: | -----: | -----: | -----: | ------: |
| `longdocurl` |   84 |  771 |  914 | 1060 | 1141 | 1153 |  5123 |
| `mmlongdoc`  |   53 |  410 |  591 |  663 |  707 |  726 |  3150 |
| `slidevqa`   |  930 | 1000 | 1041 | 1047 | 1047 | 1047 |  6112 |
| **Total**    | 1067 | 2181 | 2546 | 2770 | 2895 | 2926 | 14385 |

### 4.2 对比方法

报告比较以下方法：

- Phase 3 Full MaxSim：对 query-scoped 全部候选页做完整 MaxSim。
- Fixed Top-32/64/128 + MaxSim：mean-pool 粗检索后固定选择 top-N 页面做 MaxSim。
- Adaptive + MaxSim：自适应选择候选页面做 MaxSim。
- Adaptive + Neighbor + MaxSim：自适应选择后扩展邻页，再做 MaxSim。
- Cache variants：使用 MeanPoolCache 的 Fixed Top-64、Adaptive、Adaptive + Neighbor。

最终推荐方法为 `Adaptive + Neighbor + MeanPoolCache`。

### 4.3 运行环境与产物

主要 full run 在 Ubuntu + 2x NVIDIA RTX 3090 24GB 环境完成。当前仓库保留了关键评测输出：

- Baseline valid-only schema：`outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/`
- Final run：`outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/`
- MeanPoolCache：`outputs/cache/mean_pool_full_20260520_rerun/`
- ACL 风格论文源码：`report/main.tex`
- 图表产物：`report/figures/`

## 5. 实验结果

### 5.1 主结果与消融


| Method                          |  Recall@10 | Precision@10 |        MRR |    nDCG@10 | Avg Latency | P95 Latency | Avg Rerank |
| --------------------------------- | -----------: | -------------: | -----------: | -----------: | ------------: | ------------: | -----------: |
| Phase 3 Full MaxSim             |     0.8517 |       0.1218 |     0.5838 |     0.6325 |      0.0716 |      0.1384 |       32.7 |
| Fixed Top-32 + MaxSim           |     0.8482 |       0.1211 |     0.5828 |     0.6308 |      0.0794 |      0.1434 |       19.0 |
| Fixed Top-64 + MaxSim           |     0.8513 |       0.1217 |     0.5839 |     0.6325 |      0.0889 |      0.1800 |       26.8 |
| Fixed Top-128 + MaxSim          |     0.8517 |       0.1218 |     0.5839 |     0.6326 |      0.0907 |      0.1989 |       32.2 |
| Adaptive + MaxSim               |     0.8482 |       0.1211 |     0.5828 |     0.6308 |      0.0790 |      0.1417 |       19.0 |
| Adaptive + Neighbor + MaxSim    |     0.8523 |       0.1217 |     0.5838 |     0.6325 |      0.0796 |      0.1437 |       19.8 |
| Fixed Top-64 + Cache            |     0.8513 |       0.1217 |     0.5839 |     0.6325 |      0.0650 |      0.1178 |       26.8 |
| Adaptive + Cache                |     0.8482 |       0.1211 |     0.5828 |     0.6308 |      0.0592 |      0.0847 |       19.0 |
| **Adaptive + Neighbor + Cache** | **0.8523** |   **0.1217** | **0.5838** | **0.6325** |  **0.0600** |  **0.0858** |   **19.8** |

主表说明三个结论：

1. 单纯固定 top-N 存在质量和效率的取舍。Top-32 省计算但轻微掉点；Top-128 接近 baseline 质量但因为增加 coarse 阶段，延迟反而更高。
2. Neighbor expansion 是质量恢复的关键。`Adaptive + MaxSim` 的 Recall@10 为 0.8482，加入邻页后升至 0.8523。
3. MeanPoolCache 是效率收益的关键。最终 cache 版本在保持 no-cache 方法质量不变的前提下，把平均延迟从 0.0796 s/query 降至 0.0600 s/query。

相对 Full MaxSim baseline，最终方法：

- Recall@10 提升 0.0006，nDCG@10 基本持平。
- 平均延迟下降约 16.2%。
- P95 延迟下降约 38.0%。
- 平均 full MaxSim rerank 页面数下降约 39.4%。

### 5.2 长候选集 K128 桶

长候选集更能体现两阶段检索价值。这里的 K128 指候选全集规模桶，不是长度档位标签。


| Method                          |  Recall@10 |    nDCG@10 | Avg Latency | P95 Latency | Avg Rerank |
| --------------------------------- | -----------: | -----------: | ------------: | ------------: | -----------: |
| Full MaxSim                     |     0.6818 |     0.3902 |    109.6 ms |    189.2 ms |       81.1 |
| Fixed Top-64 + Cache            |     0.6827 |     0.3902 |     96.4 ms |    136.8 ms |       64.0 |
| **Adaptive + Neighbor + Cache** | **0.6882** | **0.3916** | **71.5 ms** | **97.4 ms** |   **34.8** |

最终方法在 K128 桶中将 P95 延迟降低约 48.5%，将 full MaxSim rerank 页面数降低约 57.1%，同时 Recall@10 增加 0.0064。这说明自适应候选控制主要在大候选全集上发挥作用；短候选集本身计算量小，粗筛带来的空间有限。

### 5.3 Slice 级结果


| Slice             | Base R@10 | Final R@10 | Delta R@10 | Base nDCG | Final nDCG | Base ms | Final ms |
| ------------------- | ----------: | -----------: | -----------: | ----------: | -----------: | --------: | ---------: |
| `slidevqa/K128`   |    0.5597 |     0.5788 |    +0.0191 |    0.2528 |     0.2597 |   143.2 |     68.4 |
| `slidevqa/K64`    |    0.6036 |     0.6113 |    +0.0076 |    0.2736 |     0.2754 |    89.0 |     69.6 |
| `slidevqa/K32`    |    0.6533 |     0.6533 |    +0.0000 |    0.2956 |     0.2956 |    67.1 |     66.1 |
| `mmlongdoc/K128`  |    0.9022 |     0.8871 |    -0.0152 |    0.6887 |     0.6809 |    88.3 |     70.0 |
| `longdocurl/K128` |    0.9584 |     0.9540 |    -0.0043 |    0.7597 |     0.7563 |   113.5 |     77.4 |

Slice 结果体现了最终方法的主要适用边界。收益集中在长 SlideVQA：幻灯片页面模板重复、相邻页面语义连续，neighbor expansion 能补回 coarse 阶段附近的证据页。`mmlongdoc/K128` 和 `longdocurl/K128` 存在小幅质量回退，说明某些长报告或网页文档的细粒度证据可能被 mean-pool coarse 阶段排到 adaptive cutoff 之外。最终方法的合理结论不是“所有 slice 都优于 baseline”，而是“总体质量基本保持，长候选集效率显著改善，且在长 SlideVQA 上有清晰质量收益”。

## 6. 讨论

### 6.1 为什么两阶段方法有效

Full MaxSim 的成本随候选页面数增长。对每个 query，如果直接对所有候选页面做 token-patch MaxSim，长候选集会产生明显延迟。Mean-pool coarse retrieval 用每页一个向量快速估计候选重要性，再把完整 MaxSim 留给更少的页面。这个设计保留了 ColPali 的细粒度视觉匹配能力，同时减少了不必要的 full interaction。

Adaptive top-N 比固定 top-N 更符合长文档检索特点。候选全集小或 coarse 分布不可靠时，系统保守保留更多页面；候选全集大且 coarse 排名清晰时，系统减少 rerank。Neighbor expansion 则利用文档页面的局部连续性，修复相邻证据页被粗检索错过的问题。

### 6.2 工程修正的重要性

本项目的结果不仅来自方法改进，也依赖两个工程修正：

- Stable `page_id`：从源图像路径恢复页面身份，避免样本内枚举位置导致的 ID 不一致。
- Valid-only protocol：剔除 1,192 条无有效页级监督的 queries，避免空 relevant set 人为抬高 Recall/MRR。

这两个修正决定了后续 baseline 和 Phase 4 的比较是否可信。没有稳定 ID 和有效标注过滤，方法收益会被评测噪声掩盖或夸大。

### 6.3 局限性

1. 主实验是 query-scoped candidate retrieval，不等价于全局语料检索。
2. MeanPoolCache 很小，但完整 patch index 仍然存在，最终 MaxSim 精排仍依赖原始 ColPali 页面 embeddings。
3. Valid-only 是公平主口径，但也意味着 1,192 条无效页级标注样本没有进入主结果。
4. 自适应规则是手工设计，没有在 benchmark 上训练参数，符合零样本设定，但仍可能对部分子任务不最优。
5. 延迟数值依赖硬件环境；跨机器时应更关注相对变化而不是绝对秒数。

## 7. 结论

ZeroShotVDR 完成了一个稳定的 ColPali-based 零样本视觉文档页级检索系统，并在 MMLongBench DocumentQA 上实现了可复现的 valid-only 评测协议。最终 `Adaptive + Neighbor + MeanPoolCache` 方法在 14,385 条有效 queries 上保持 Full MaxSim baseline 的检索质量，同时显著降低平均延迟、尾延迟和 full MaxSim rerank 页面数。长候选集和长 SlideVQA 是收益最明显的场景。

项目的主要经验是：在强视觉检索模型之外，稳定数据契约、可信评测口径和 query-adaptive reranking 策略同样关键。对于零样本 VDR，保留 ColPali 的 patch-level MaxSim 作为最终精排，同时用轻量 coarse stage 和邻页先验控制计算量，是一个保守但有效的质量-效率折中。

## 参考资料

- Faysse et al. ColPali: Efficient Document Retrieval with Vision Language Models. 2024.
- Khattab and Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR 2020.
- Wang et al. MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly. NeurIPS 2025 Spotlight.
- `docs/NJUProject_VDR.md`
- `docs/MMLongBench_Dataset_Notes.md`
- `docs/Project_Plan.md`
- `report/main.tex`
- `outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/`
