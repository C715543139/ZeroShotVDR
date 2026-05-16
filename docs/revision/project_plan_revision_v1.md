# Project Plan 修订记录 v1

## 1. 目的

本修订文档用于记录 2026-05-16 对项目计划文档的核对结果，说明 [docs/Project_Plan.md](docs/Project_Plan.md) 与当前仓库实现之间的主要偏差，以及本轮已经完成的同步内容。

本轮核对主要对照以下实现锚点：

- [src/zeroshot_vdr/contracts.py](src/zeroshot_vdr/contracts.py)
- [src/zeroshot_vdr/data/adapters.py](src/zeroshot_vdr/data/adapters.py)
- [src/zeroshot_vdr/retrieval/pipeline.py](src/zeroshot_vdr/retrieval/pipeline.py)
- [src/zeroshot_vdr/indexing/store.py](src/zeroshot_vdr/indexing/store.py)
- [scripts/run_step3_eval.py](scripts/run_step3_eval.py)
- [scripts/run_step3_analysis.py](scripts/run_step3_analysis.py)
- [docs/Milestone_Report_Phase3.md](docs/Milestone_Report_Phase3.md)
- [docs/revision/step3_direction_revision_v1.md](docs/revision/step3_direction_revision_v1.md)

---

## 2. 发现的主要偏差

### 2.1 环境与运行平台描述过期

原计划文档仍将主要硬件环境写为 Win10 + RTX 4060 Laptop，并默认以 PowerShell 工作流为主。

但当前真实主评测平台已经切换为 Ubuntu + 2x RTX 3090，且 Linux 下的统一激活入口是：

- `source ./env`

因此，原文档如果继续作为当前执行指南，会误导后续使用者选择错误平台和命令入口。

### 2.2 Phase 3 状态与当前完成度不一致

原计划文档仍把 Step 3.1/3.2/3.3 主要写成待办，并保留基于 Win10 + 4060 的长时间估算、K32 主档位等旧表述。

但当前仓库状态已经是：

- stable full run 已完成；
- Step 3.2 分析已完成；
- Step 3.3 方向决策已完成；
- 主汇报口径已切换为 14,385 条有效标注查询；
- K32 不再是唯一主口径。

### 2.3 Phase 4 状态被文档写得过于靠前

原计划文档把 `src/zeroshot_vdr/advanced/`、`patch_pruner.py` 和 `two_stage.py` 写成了现有目录/文件。

但当前仓库中该目录尚未创建。现阶段真实存在的是扩展点，而不是完整的 Phase 4 实现：

- `IndexStore.get_mean_pooled_view()`
- `RetrievalPipeline` 中的 `candidate_strategy="mean_pool_topN"`
- `batched_maxsim_variable()`

### 2.4 页面身份语义仍沿用旧设计

原计划文档默认把稳定 `page_id` 理解为：

- `doc_name -> normalize_doc_id()`
- `enumerate(page_list) -> page_idx`

这已经不符合当前实现。

当前 stable redesign 的真实语义是：

- Page 侧的 `doc_id/page_idx` 从原始图片路径解析 source doc / source page；
- Query 侧保留 sample 级 `doc_id` 便于追踪；
- `page_id` 字符串格式不变，但语义来源已变化。

### 2.5 检索协议描述仍停留在“按 query.doc_id 默认候选”

原计划文档在多个位置把 baseline 写成“默认按 `query.doc_id` 返回文档内页面集合”。

但当前实际协议是：

1. 若 `Query.candidate_page_ids` 非空，优先使用该 sample-specific 候选集合；
2. 仅当该字段为空时，才回退到 `query.doc_id` 对应页面集合；
3. 显式空候选仍返回空结果，不做隐式扩张。

### 2.6 主比较口径描述过期

原计划文档仍把 K32 作为主汇报档位。

当前 Phase 3 分析后，主比较口径已经固定为：

- 14,385 条有效页级标注查询；
- 全部 15,577 条查询仅作为完整披露口径；
- `no_ground_truth` 不再并入真实 bad case。

---

## 3. 本轮已完成的同步

本轮已经直接更新 [docs/Project_Plan.md](docs/Project_Plan.md)，完成以下同步：

1. 将文档顶部硬件环境改为“当前 Ubuntu 双 3090 主平台 + 历史 Win10 4060 开发环境”。
2. 在环境章节明确 Linux 下统一先执行 `source ./env`。
3. 将 Phase 3 改为已完成状态，并填入 stable full run 的实际规模、时间与产物位置。
4. 将 Phase 4 改为“方向已确定但尚未开始实现”，并明确 advanced 目录尚未创建。
5. 将数据契约与数据接入说明改为当前 stable page identity 语义。
6. 将检索层协议改为“优先 `candidate_page_ids`，否则回退 `query.doc_id`”。
7. 将评测协议与汇报口径改为“有效标注主比较口径 + 全量结果完整披露”。
8. 在结构树中补充 Linux 激活入口、Step 3.2 分析脚本、Phase 3 里程碑文档等当前已存在内容。

---

## 4. 当前仍需注意的实现备注

以下内容并不构成这次文档同步的阻塞项，但值得在后续实现或报告中继续留意：

### 4.1 `config/default.yaml` 仍保留 baseline 示例值

当前 [config/default.yaml](config/default.yaml) 里的 `page_id_template` 和 `index.dir` 仍更接近早期 baseline 示例配置。

但 stable full run 的真实关键行为已经主要由代码层实现与 CLI 参数决定：

- 稳定页面身份来自 [src/zeroshot_vdr/contracts.py](src/zeroshot_vdr/contracts.py) 中基于图片路径的 helper；
- 当前有效 full index 位于 `data/processed/index_stable_page_ids/`；
- 这意味着“文档已经同步到当前实现”并不等于“所有历史默认配置都已完全重写”。

### 4.2 Phase 4 扩展点已具备，但主线代码尚未开始

当前可以确认：

- 方向 B 的最小可运行入口具备较低工程风险；
- 方向 A 更适合作为补充实验或未来工作；
- 但仓库里仍没有正式的 `advanced/` 目录和独立 Phase 4 代码文件。

---

## 5. 结论

本轮核对的结论是：

- 原 [docs/Project_Plan.md](docs/Project_Plan.md) 与当前实现之间确实存在明显偏差；
- 偏差主要集中在环境平台、Phase 3 完成度、Phase 4 落地状态、页面身份语义、检索协议和主汇报口径；
- 这些偏差已经在计划文档中完成同步；
- 本文件作为 revision 记录，用于说明“为什么这次要改”和“当前还有哪些实现备注需要在后续继续跟踪”。
