# ZeroShotVDR Step 3.2 修订记录 v1

## 1. 说明

本轮工作从 Step 3.1 的全量输出继续推进到 Step 3.2，目标是完成三类产物：

1. 指标趋势图
2. Recall@10 < 1.0 的 bad case 汇总
3. 基于代表性样本的人工失败分析结论

本轮分析基于以下实际输出目录：

- `outputs/eval_reports/step3_docqa_full_dual3090/`

并新增了自动分析脚本：

- `scripts/run_step3_analysis.py`

脚本当前会生成：

- `analysis/plots/overall_k_curves.png`
- `analysis/plots/subtask_k_curves.png`
- `analysis/plots/length_k_curves.png`
- `analysis/plots/k32_subtask_comparison.png`
- `analysis/all_cases.csv`
- `analysis/bad_cases_all.csv`
- `analysis/bad_case_summary.csv`
- `analysis/representative_bad_cases.csv`
- `analysis/page_id_stability_summary.csv`
- `analysis/step3_2_analysis_summary.md`

---

## 2. 关键结果摘要

本轮全量评测范围为：

- 15,577 queries
- 3,653 docs
- 117,724 pages

运行效率：

- 索引补建耗时：1h13m57s
- 检索耗时：39m5s
- 平均延迟：0.150 s/query
- P95 延迟：0.273 s/query

整体指标：

| k  | Recall | Precision | MRR   | nDCG  |
| -- | ------ | --------- | ----- | ----- |
| 1  | 0.2403 | 0.2217    | 0.4286 | 0.2982 |
| 3  | 0.4627 | 0.1768    | 0.4286 | 0.3933 |
| 5  | 0.5632 | 0.1358    | 0.4286 | 0.4365 |
| 10 | 0.6972 | 0.0883    | 0.4286 | 0.4839 |

以 K32 为主档位的子任务结果：

| Subtask    | Recall@10 | MRR   | nDCG@10 |
| ---------- | --------- | ----- | ------- |
| longdocurl | 0.8965    | 0.5130 | 0.5853  |
| mmlongdoc  | 0.8963    | 0.6081 | 0.6694  |
| slidevqa   | 0.3421    | 0.1072 | 0.1581  |

直接从结果上看，slidevqa 在中长上下文下显著弱于 longdocurl 与 mmlongdoc。

---

## 3. Bad Case 汇总

按 Recall@10 < 1.0 定义 bad case，则共有：

- 6,446 / 15,577 queries
- bad case rate = 41.38%

最集中的 bad case 热点为：

| Subtask    | Length | Bad Case Rate |
| ---------- | ------ | ------------- |
| slidevqa   | K128   | 92.67%        |
| slidevqa   | K64    | 83.74%        |
| mmlongdoc  | K128   | 79.29%        |
| slidevqa   | K32    | 69.92%        |
| mmlongdoc  | K64    | 62.70%        |
| longdocurl | K128   | 50.13%        |

一个表面结论是：长度越长，bad case 越多；slidevqa 与 mmlongdoc 在长上下文下退化最明显。

但本轮人工复核后发现，**这个表面结论不能直接等价理解为“模型本身失败”**。

---

## 4. 本轮最重要发现：当前 page_id 契约在 DocumentQA 长上下文上不稳定

Step 3.2 新增了 `page_id -> image_path` 稳定性检查。检查方式是：

- 对同一 `subtask × length`
- 若同一个 `page_id`
- 在不同原始样本中对应到多个不同 `page_list` 图片路径
- 则记为 unstable

结果显示，这个问题不是零星现象，而是系统性的：

| Subtask    | Length | Unstable Rate |
| ---------- | ------ | ------------- |
| mmlongdoc  | K4     | 100.00%       |
| mmlongdoc  | K8     | 96.63%        |
| mmlongdoc  | K16    | 96.22%        |
| mmlongdoc  | K32    | 94.57%        |
| mmlongdoc  | K128   | 89.78%        |
| slidevqa   | K128   | 87.52%        |
| slidevqa   | K64    | 86.48%        |
| slidevqa   | K32    | 80.70%        |
| longdocurl | K32    | 60.01%        |
| longdocurl | K64    | 56.81%        |
| longdocurl | K128   | 43.92%        |

这说明当前系统里使用的：

- `page_id = {task_family}/{subtask}_{length}/{doc_id}/p{page_idx}`

在 DocumentQA 长上下文样本上实际上不是“稳定页面 ID”，而更像“样本内部上下文位置 ID”。一旦不同 query 对同一 `doc_id` 采样到的 `page_list` 不同，`p73` 这样的 page index 就会绑定到不同图片。

这会带来两个后果：

1. 索引阶段会把不同图片错误折叠到同一个 `page_id`
2. 评测阶段会把本来命中的页面记成 miss，制造假 bad case

---

## 5. 人工复核的三个代表性证据

### 5.1 longdocurl/K128/q728：表面是 miss_top10，实质上是 page_id 冲突

查询：

- `What was the total consumption of rice in 2020?`

人工查看发现：

- 检索出的 top1 图片就是 `RICE PRODUCTION 2020` 页面，页面中直接出现 `Total Consumption 30,517 mt`
- 但当前标注映射到的“relevant page”却落到了另一张 `MONTHLY LOCAL PRODUCTION & IMPORT OF LIVESTOCK FEED FOR 2020`

这不是语义上的 hard negative，而是 `page_id` 绑定图片不稳定导致的假阴性。

### 5.2 mmlongdoc/K128/q871：top1 语义正确，但“相关页”映射到了另一份手册

查询：

- `Tell me all the pages introducing how to reinstall the software.`

人工查看发现：

- top1 图片是 `Reinstalling Software Using Remote Install Mac OS X`，明显与问题匹配
- 但当前分析里关联到的 relevant image 却是另一份 `4K UHD Display User Manual` 的封面页

这说明在 mmlongdoc 中，同一个 `doc_id/page_idx` 可能跨 query 对应到不同原始 PDF 页面，当前契约不再可靠。

### 5.3 slidevqa/K128/q216：top1 页面标题与问题完全对齐，但被记为 miss

查询：

- `On the slide that has "Tech is Move" written across its top, in the diagram, what does the orange line correspond to?`

人工查看发现：

- top1 图片正是标题含 `TECH IS MOVE` 的目标 slide
- 当前 relevant image 却指向另一份 `Q4 2015 vs. Q4 2014` slide

这进一步说明 slidevqa 的长上下文 bad case 中，有相当一部分其实是 label / page_id 对不齐，而非模型没有召回目标页面。

---

## 6. 在承认 page_id 不稳定后的失败类型分类

在本轮人工分析后，当前 Step 3.2 更合理的失败类型应分为三类：

### 6.1 类型 A：上下文混排导致的 page_id / label 不稳定

这是当前最主要的问题，尤其集中在：

- mmlongdoc 全长度档位
- slidevqa 的 K32 / K64 / K128
- longdocurl 的 K32 以上长度档位

这类 bad case 不能简单解释为检索模型失败，因为 ground truth 与索引页本身就可能未对齐。

### 6.2 类型 B：跨页证据导致的 partial recall

这类样本通常表现为：

- 查询需要同时定位两个或更多页面
- 系统命中了部分相关页，但 Recall@10 仍小于 1.0

例如 longdocurl/K128 的数值比较题中，经常需要同时访问：

- 某个行业的 summary page
- 对应的 monthly table page

模型更容易先召回语义接近的 summary/overview 页面，而漏掉另一个补充证据页。

### 6.3 类型 C：高相似版式下的相邻数值页混淆

这一类主要出现在 longdocurl：

- 多个页面都采用高度一致的统计模板
- 只有标题中的行业名或表格项不同
- 查询又常要求精确数字或跨列比较

在这种情况下，模型虽然抓住了“统计表 + 2020 + consumption/import”这类粗粒度语义，
但容易把目标页与相邻行业页混排。

---

## 7. 对当前 Step 3.1 结果的解释边界

因此，当前全量结果可以用于：

- 验证评测脚本性能与吞吐
- 观察不同子任务与长度档位的表面退化趋势
- 初步识别 slidevqa / 长上下文是困难区域

但它**不适合作为严格可信的最终 baseline 数值**，原因是：

- 当前 `page_id` 设计默认 `page_idx` 在同一 `doc_id × length` 下稳定
- 而真实 DocumentQA 长上下文样本并不满足这一前提

换言之，Step 3.1 当前数值更像“工程管线打通后的 provisional baseline”，而不是可直接进入最终报告主表的定稿结果。

---

## 8. Step 3.2 结论与下一步建议

本轮 Step 3.2 已完成以下目标：

- 生成 k-vs-metric 曲线图
- 输出 bad case 汇总表
- 筛选并人工复核代表性样本
- 识别出当前评测里最重要的结构性问题：`page_id` 不稳定

因此，进入后续 Phase 4 或撰写 Milestone 报告前，建议先完成一项修正：

1. 重构 page_id 契约，使其绑定到稳定的原始图片路径或样本级上下文 ID，而不是共享的 `doc_id/page_idx`
2. 重新构建索引并重跑 Step 3.1
3. 仅在修正后版本上继续做 Step 3.2 的最终 bad case 结论与 Phase 4 对比实验

若不先修正这一步，后续任何“baseline vs. 改进方法”的结论都可能掺入 page_id 混淆造成的评测噪声。