# ZeroShotVDR：用于零样本视觉文档检索的查询自适应两阶段检索

> 中文对照版，对应英文 ACL 源文件：`acl_latex/main.tex`。  
> 本文件用于人工核对内容与答辩准备，不作为 ACL 模板编译源文件。

## 摘要

零样本视觉文档检索要求系统在没有任务特定训练的条件下，根据文本查询从视觉丰富的长文档中找出证据页面。仅依赖 OCR 的方法容易丢失版式、图表、图片和表格证据；而 ColPali 这类无 OCR 的 late-interaction 检索器虽然具有较强的页面级匹配能力，但完整 MaxSim 打分的成本较高。本文报告 ZeroShotVDR，一个面向 MMLongBench DocumentQA 的 ColPali 检索系统。我们首先构建稳定基线，包括从原始图像路径重建稳定页面身份，以及剔除不可用页级监督样本的 valid-only 评测协议。随后提出查询自适应两阶段检索：用 mean-pooled 页面向量进行粗检索，用自适应策略选择候选数量，对高排名页面做邻页扩展，再用完整 MaxSim 精排。最终的 Adaptive + Neighbor + MeanPoolCache 系统在 14,385 条有效查询上达到 Recall@10 = 0.8523、nDCG@10 = 0.6325；相比 Full MaxSim 基线的 0.8517 和 0.6325，质量保持或略有提升，同时平均延迟从 0.0716 s/query 降到 0.0600 s/query，P95 延迟从 0.1384 s 降到 0.0858 s。收益最明显的场景是长上下文 SlideVQA 切片，说明自适应粗到精检索能改善大候选集下的质量-效率权衡。

## 1 引言

现实文档往往包含表格、图表、图片、演示幻灯片、多栏排版和跨区域引用。对于页面级检索来说，相关证据不一定能被 OCR 文本完整表达：图表问题可能依赖坐标轴和数值，幻灯片问题可能依赖图文布局，报告页面也可能通过版式组合表达关键信息。因此，本项目关注视觉文档检索，即给定文本查询，直接从页面图像中排序相关页面。

本项目采用零样本设定，不在 MMLongBench 上训练模型，而是使用 ColPali-v1.3 作为无 OCR 的视觉检索器。ColPali 将每个页面编码为 patch-level embedding，并使用类似 ColBERT 的 late interaction / MaxSim 机制匹配查询 token 与页面 patch。这样可以保留细粒度视觉证据，但代价是候选页面越多，需要执行的 MaxSim 计算越多。在长上下文场景下，这一成本会变得明显；同时，相邻页面或视觉上相似的页面也会增加排序难度。

评测使用 MMLongBench DocumentQA 子集。该子集通过 `ans_page_list` 提供答案所在页面标注，通过 `page_list` 给出每个样本的候选页面集合。因此本文评测的是 query-scoped candidate retrieval，而不是在全局语料库中先找文档再找页面。这个边界需要在报告中明确说明，否则容易把结果误读为全局检索结果。

最终系统采用两阶段流程。第一阶段用页面 patch embedding 的均值向量进行低成本粗排；第二阶段只对缩小后的候选集执行完整 ColPali MaxSim。候选规模不是固定 top-N，而是根据查询候选集和粗排分数分布自适应选择；随后对高排名 seed 页面加入一页窗口内的邻页。最后使用 mean-pool cache 避免重复计算粗排页面向量。该设计来自 Phase 3 失败分析：最难的样本集中在大候选集和邻页混淆，尤其是长上下文 SlideVQA。

本文贡献如下：

- 构建了稳定的 ColPali-based 页面检索基线，包括稳定页面身份重建和 14,385 条有效查询上的 valid-only 评测协议。
- 提出了查询自适应两阶段检索方法，将 mean-pool 粗检索、自适应候选选择、邻页扩展和完整 MaxSim 精排结合起来。
- 实验表明最终系统在保持检索质量的同时提升效率，尤其改善长候选集和长上下文 SlideVQA 场景。

## 2 相关工作

### 2.1 无 OCR 的视觉文档检索

传统文档检索通常先 OCR，再对提取文本建索引。该方法在纯文本证据场景中有效，但容易丢失布局、图表、图片和表格结构。ColPali 直接对渲染后的文档页面进行视觉语言编码，不需要单独 OCR。ZeroShotVDR 使用 ColPali 作为基础方法，并研究如何在长候选集下提升其 late-interaction 打分效率。

### 2.2 Late Interaction 检索

Late interaction 在打分阶段保留 token 或 patch 级表示。ColBERT 在文本检索中提出这种机制，通过 MaxSim 让查询 token 与文档 token 细粒度交互。ColPali 将这一思想迁移到视觉文档页面上，用查询 token embedding 匹配页面 patch embedding。优点是表达能力强，缺点是每个候选页面仍需要大量向量比较。本文保留完整 MaxSim 作为最终打分器，但减少进入 MaxSim 的页面数量。

### 2.3 长上下文视觉语言评测

MMLongBench 面向长上下文视觉语言模型评测，包含多个任务族和标准化上下文长度。DocumentQA 最适合本项目，因为 `ans_page_list` 可以直接转化为页级相关性标注。该数据集也暴露出长候选集的检索难点：候选页面越多，系统越需要从视觉上相似的干扰页中找到真实证据页。

## 3 任务与方法

### 3.1 任务定义与评测协议

对于每个查询 q，输入包含一个由 `page_list` 给出的查询特定候选页面集合 C(q)。任务是在 C(q) 内排序页面，使 `ans_page_list` 对应的相关页面 G(q) 尽量靠前。主要指标包括 Recall@k、Precision@k、MRR 和 nDCG@k。

主评测只使用有可用页级监督的查询。原始 full run 包含 15,577 条查询，其中 1,192 条存在空标注、无效标注或越界页码。如果直接混入主表，空 relevant set 会让 recall 和 MRR 出现误导性结果。因此主比较固定使用 14,385 条 valid-only 查询，完整 15,577 条结果仅作为附录级披露。

页面身份也是一个关键问题。早期基于样本局部 doc name 和页序拼接 page_id 的方式在长上下文样本之间不稳定。稳定实现改为从原始图像路径恢复 source document 和 source page index，并使用包含任务族、子任务、长度档位、文档 ID 和页号的规范 page_id。每个 Query 还显式携带 `candidate_page_ids`，用于固定 query-scoped 候选范围。

### 3.2 稳定 ColPali 基线

基线包含五层：MMLongBench DocumentQA 数据适配层、逐页 embedding 索引层、检索层、评测层和配置支持层。每个页面图像由 ColPali-v1.3 编码为 patch embedding 矩阵，每个文本查询编码为 token embedding。对于 L2 归一化后的查询向量和页面 patch 向量，基线使用 MaxSim：

```text
Score(Q, P) = sum_i max_j <q_i, p_j>
```

基线对 C(q) 中所有页面执行完整 MaxSim 并返回 top-k。由于候选范围是 query-scoped，有效查询上的平均候选数为 32.7 页；但 K128 等长候选桶明显更大。

### 3.3 查询自适应两阶段检索

最终方法保留基线的完整 MaxSim 精排器，但改变候选进入 MaxSim 的方式：

```text
query-specific universe
 -> mean-pool coarse ranking
 -> adaptive top-N selection
 -> neighbor expansion
 -> full MaxSim reranking
 -> top-k pages
```

**Mean-pool 粗检索。** 对每个页面，将 patch embedding 求均值得到一个页面向量；查询 token embedding 同样求均值得到查询向量。粗排使用归一化均值向量的点积。相比完整 MaxSim，每个页面只需要一个向量参与计算。

**自适应候选选择。** 固定 top-N 简单但不够稳健。小候选集不需要强剪枝，而分数分布较平的查询需要保留更多候选。最终配置使用 `min_candidates=32`、`max_candidates=128`、`base_ratio=0.2`、`flat_margin=0.035`，并受实际候选集大小限制。

**邻页扩展。** DocumentQA 中相邻页面经常共同承载证据，且 Phase 3 bad case 包含邻页混淆。系统对粗排高排名 seed 页面添加一页窗口内的邻页，最终 full run 使用 `neighbor_seed_n=8` 和 `neighbor_window=1`。该扩展很小，目的是恢复相邻证据，而不是退化回全候选集精排。

**MeanPoolCache。** mean-pooled 页面向量可以在线计算，但全量实验中重复计算会浪费时间。最终系统预计算 87,090 个页面均值向量，embedding 维度为 128，float16 存储，目录约 26 MB；相比原始 patch index 约 88.6 GB，缓存非常小。缓存只改变效率，不改变排序质量。

### 3.4 复杂度收益

设查询 token 数为 m，页面 patch 数为 r，embedding 维度为 d，候选页数为 n。完整基线每个查询的 MaxSim 成本约为 O(nmrd)。两阶段方法增加 O(nd) 的粗排，但只对自适应扩展后的 a(q) 个页面执行完整 MaxSim。因此候选集越大，收益越明显，这与 K128 和 SlideVQA 长上下文实验结果一致。

## 4 实验设置

数据集为 MMLongBench DocumentQA，包含 `longdocurl`、`mmlongdoc`、`slidevqa` 三个子任务，以及 K4、K8、K16、K32、K64、K128 六个长度档位。主评测使用 14,385 条有效查询。稳定基线 full run 索引 87,922 页，patch index 约 88.6 GB。

有效查询在各子任务和长度档位上的分布如下。由于各切片样本数和难度并不均匀，论文同时报告总体指标和重点切片指标。

| Subtask | K4 | K8 | K16 | K32 | K64 | K128 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `longdocurl` | 84 | 771 | 914 | 1060 | 1141 | 1153 | 5123 |
| `mmlongdoc` | 53 | 410 | 591 | 663 | 707 | 726 | 3150 |
| `slidevqa` | 930 | 1000 | 1041 | 1047 | 1047 | 1047 | 6112 |
| Total | 1067 | 2181 | 2546 | 2770 | 2895 | 2926 | 14385 |

质量指标为 Recall@k、Precision@k、MRR 和 nDCG@k，k 取 1、3、5、10。主表重点报告 Recall@10、Precision@10、MRR、nDCG@10。效率指标为平均延迟、P95 延迟和平均完整 MaxSim 精排候选页数。

对比方法包括 Phase 3 Full MaxSim baseline、Fixed Top-32/64/128 + MaxSim、Adaptive + MaxSim、Adaptive + Neighbor + MaxSim，以及 Fixed Top-64、Adaptive、Adaptive + Neighbor 的 cache 版本。最终推荐方法为 Adaptive + Neighbor + Cache。

实现使用 ColPali-v1.3，运行环境为 Ubuntu + 2x RTX 3090 24GB。索引和检索分阶段执行，索引按页独立存储，以支持按页加载、稳定评测和 per-query trace 分析。

## 5 实验结果

### 5.1 主消融实验

| Method | R@10 | P@10 | MRR | nDCG@10 | Avg Lat. | P95 Lat. | Rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 3 Full MaxSim | 0.8517 | 0.1218 | 0.5838 | 0.6325 | 0.0716 | 0.1384 | 32.7 |
| Fixed Top-32 + MaxSim | 0.8482 | 0.1211 | 0.5828 | 0.6308 | 0.0794 | 0.1434 | 19.0 |
| Fixed Top-64 + MaxSim | 0.8513 | 0.1217 | 0.5839 | 0.6325 | 0.0889 | 0.1800 | 26.8 |
| Fixed Top-128 + MaxSim | 0.8517 | 0.1218 | 0.5839 | 0.6326 | 0.0907 | 0.1989 | 32.2 |
| Adaptive + MaxSim | 0.8482 | 0.1211 | 0.5828 | 0.6308 | 0.0790 | 0.1417 | 19.0 |
| Adaptive + Neighbor + MaxSim | 0.8523 | 0.1217 | 0.5838 | 0.6325 | 0.0796 | 0.1437 | 19.8 |

Full MaxSim 基线达到 Recall@10 = 0.8517、nDCG@10 = 0.6325。无 cache 的 Adaptive + Neighbor 将 Recall@10 提升到 0.8523，nDCG@10 基本不变。Fixed Top-32 和 Adaptive 都能把精排候选数降到 19.0，但质量略降；Fixed Top-128 恢复质量，但由于增加了粗排且精排候选数接近基线，延迟更差。邻页扩展是恢复质量的关键组件。

### 5.2 Mean-pool cache 效果

| Method | R@10 | nDCG@10 | Avg Lat. | P95 Lat. |
| --- | ---: | ---: | ---: | ---: |
| Fixed Top-64 | 0.8513 | 0.6325 | 0.0889 | 0.1800 |
| Fixed Top-64 + Cache | 0.8513 | 0.6325 | 0.0650 | 0.1178 |
| Adaptive | 0.8482 | 0.6308 | 0.0790 | 0.1417 |
| Adaptive + Cache | 0.8482 | 0.6308 | 0.0592 | 0.0847 |
| Adaptive + Neighbor | 0.8523 | 0.6325 | 0.0796 | 0.1437 |
| Adaptive + Neighbor + Cache | 0.8523 | 0.6325 | 0.0600 | 0.0858 |

Cache 不改变对应方法的质量，但显著降低延迟。最终方法相比 Phase 3 baseline，Recall@10 从 0.8517 到 0.8523，nDCG@10 保持 0.6325，平均延迟从 0.0716 降到 0.0600 s/query，P95 延迟从 0.1384 降到 0.0858 s。

### 5.3 K128 长候选桶

| Method | R@10 | nDCG@10 | Avg ms | P95 ms | Rerank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full MaxSim | 0.6818 | 0.3902 | 109.6 | 189.2 | 81.1 |
| Fixed Top-64 + Cache | 0.6827 | 0.3902 | 96.4 | 136.8 | 64.0 |
| Adaptive + Neighbor + Cache | 0.6882 | 0.3916 | 71.5 | 97.4 | 34.8 |

K128 桶最能体现方法收益。最终系统将平均精排候选数从 81.1 降到 34.8，将 P95 延迟从 189.2 ms 降到 97.4 ms，同时 Recall@10 和 nDCG@10 也略有提升。

### 5.4 重点切片分析

| Slice | Base R@10 | Final R@10 | ΔR | Base nDCG | Final nDCG | ΔnDCG | Base ms | Final ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slidevqa/K128` | 0.5597 | 0.5788 | +0.0191 | 0.2528 | 0.2597 | +0.0069 | 143.2 | 68.4 |
| `slidevqa/K64` | 0.6036 | 0.6113 | +0.0076 | 0.2736 | 0.2754 | +0.0018 | 89.0 | 69.6 |
| `slidevqa/K32` | 0.6533 | 0.6533 | +0.0000 | 0.2956 | 0.2956 | +0.0000 | 67.1 | 66.1 |
| `mmlongdoc/K128` | 0.9022 | 0.8871 | -0.0152 | 0.6887 | 0.6809 | -0.0078 | 88.3 | 70.0 |
| `longdocurl/K128` | 0.9584 | 0.9540 | -0.0043 | 0.7597 | 0.7563 | -0.0035 | 113.5 | 77.4 |

最终方法在 `slidevqa/K128` 和 `slidevqa/K64` 上同时提升质量和效率。这些切片中有大量视觉上重复的幻灯片页面，mean-pool 粗排能先定位相关区域，邻页扩展能补回相邻证据。但方法并非所有切片都提升：`mmlongdoc/K128` 和 `longdocurl/K128` 有轻微质量下降，换来了明显延迟和候选规模下降。

## 6 讨论

### 6.1 方法为什么有效

当候选集较小时，粗排能省下的工作有限；当候选集很大时，低成本 mean-pool 粗排可以过滤大量干扰页。K128 和 SlideVQA 的结果说明该方法主要适合大候选集场景。幻灯片常有重复模板和相邻主题页面，粗排负责找到可能区域，邻页扩展负责加入连续证据。

### 6.2 权衡在哪里

最终系统没有在所有切片上全面优于基线。`mmlongdoc/K128` 的 Recall@10 从 0.9022 降到 0.8871，`longdocurl/K128` 从 0.9584 降到 0.9540。这说明某些长报告和网页文档中的细粒度证据可能在 mean-pool 粗排中排名较低，被自适应阈值截断。邻页扩展能缓解但不能完全替代全候选 MaxSim。

### 6.3 失败模式

Phase 3 识别了三类失败：大候选集下 miss_top10、多页部分召回、邻页混淆。本文方法主要针对第一类和第三类：减少大候选集上的 full MaxSim 预算，并显式加入高排名 seed 页周围的邻页。多页部分召回仍未彻底解决，因为当前系统仍是独立页面排序，没有显式建模多页证据聚合。

### 6.4 局限性

第一，本文评测的是 query-scoped candidate retrieval，不是全局语料检索，因此结果不应被解读为从全语料页面中完成文档级检索。第二，mean-pool cache 很小，但最终精排仍依赖完整 patch index，完整视觉索引的存储体量仍然较大。第三，valid-only 协议保证了公平比较，但也意味着 1,192 条页级监督不可用的查询没有进入主表。第四，延迟结果与硬件有关，跨机器更应关注相对趋势。

另一个局限是自适应候选选择规则仍是手工设计的。它由分数分布和消融结果驱动，但不是学习得到的控制器。这符合零样本课程设定，避免在 benchmark 上训练，同时也让 reranking budget 能够根据查询级候选结构进行调整。

## 7 结论

ZeroShotVDR 构建了一个稳定的 ColPali-based MMLongBench DocumentQA 页面检索系统，并通过查询自适应两阶段检索改进质量-效率权衡。最终 Adaptive + Neighbor + MeanPoolCache 方法在保持 baseline 检索质量的同时降低平均延迟和尾部延迟，尤其适合长候选集。本文结论应保持保守：自适应粗到精精排是 query-scoped 零样本视觉文档检索中有效的效率改进；主要不足是评测协议仍为 query-scoped、最终精排仍依赖完整 patch index，并且部分长文档切片存在小幅质量回落。

## 附录：补充指标

### A. Phase 3 Full MaxSim baseline valid-only 指标

| k | Recall | Precision | MRR | nDCG |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3342 | 0.4403 | 0.5838 | 0.4403 |
| 3 | 0.6201 | 0.2853 | 0.5838 | 0.5435 |
| 5 | 0.7250 | 0.2041 | 0.5838 | 0.5881 |
| 10 | 0.8517 | 0.1218 | 0.5838 | 0.6325 |

### B. 最终 Adaptive + Neighbor + Cache valid-only 指标

| k | Recall | Precision | MRR | nDCG |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3342 | 0.4403 | 0.5838 | 0.4403 |
| 3 | 0.6198 | 0.2852 | 0.5838 | 0.5433 |
| 5 | 0.7247 | 0.2040 | 0.5838 | 0.5879 |
| 10 | 0.8523 | 0.1217 | 0.5838 | 0.6325 |

### C. 复现路径

- Phase 3 valid-only baseline：`outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/phase4_schema_valid_only/`
- Phase 4 final run：`outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/`
- Mean-pool cache：`outputs/cache/mean_pool_full_20260520_rerun/`
- 英文 ACL 源文件：`acl_latex/main.tex`
- BibTeX 文件：`acl_latex/zeroshotvdrrefs.bib`
