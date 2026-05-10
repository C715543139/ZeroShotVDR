# ZeroShotVDR 核心模块修订建议 v4

## 1. 说明

v4 不再引入新的架构约束，而是针对 v3 已经落入 `Project_Plan.md` 后仍残留的少量表述漂移，做最后一轮收尾。

这一轮关注的不是“设计方向是否正确”，而是“摘要区、导览区、示例区是否已经和权威接口定义完全一致”。如果这些位置继续保留旧表述，后续实现时仍可能出现“正文按新协议，任务分解按旧协议”的偏差。

---

## 2. v4 审查结论

当前 `Project_Plan.md` 的主体设计已经基本闭合，v3 关注的核心问题也大多已被吸收。剩余问题主要集中在三类“尾差”：

1. 个别说明段仍沿用旧版“baseline = 全局全量候选”的表述。
2. Phase 2 的任务摘要和 API 摘要尚未完全同步到最终数据契约。
3. 文档结构中的修订记录展示仍停留在早期状态，没有反映 v2-v4 的存在。

这些问题不会改变整体架构方向，但会影响实现者对默认行为和最终接口的理解，因此值得在本轮直接收尾。

---

## 3. 需要收尾的修改项

### 3.1 收紧 baseline 候选范围的残留旧表述

当前设计已经在检索接口章节中明确：baseline 采用文档内检索，默认候选范围应当是 `query.doc_id` 对应的页面集合，而不是全局语料集合。

因此，凡是仍写着“全量 page_ids”“候选召回 = 全量”的位置，都应同步改写为：

- baseline 候选召回 = 当前文档内全部页面
- 全局候选仅在显式开启 global retrieval 实验配置时启用

这一步的意义不是增加新限制，而是消除同一文档内部对 baseline 定义的自相矛盾。

### 3.2 同步 Phase 2 摘要到最终契约

v3 已经在 `contracts.py` 的权威设计里固定了以下事实：

1. `Query` 需要携带 `raw_doc_name`
2. `RelevanceJudgment` 是正式契约的一部分
3. `RetrievalResult` 包含 `query_id`
4. `RetrievalPipeline.retrieve()` 的摘要签名应包含 `score_batch_size`

因此，Phase 2 的任务拆解和 API 摘要也应同步到这一最终版本，避免实现时优先参考前文摘要而不是后文权威定义。

### 3.3 同步修订记录展示

当前计划文档已经吸收了 v2、v3 的大量内容，并将在本轮继续吸收 v4 的收尾修改。因此：

- 项目结构中的 `docs/revision/` 示例应列出 v1-v4
- 第四章开头不应再只写“根据 v1 重构”，而应表述为“综合吸收 v1-v4 的逐轮修订结果”

这一步主要是让文档本身的历史描述与当前状态一致。

---

## 4. 建议直接回填到计划文档的内容

### 4.1 关于 baseline 候选范围

建议统一替换为：

> Baseline 中候选召回的默认范围为当前 `query.doc_id` 对应文档内的全部页面，而不是全局语料的全部页面。若需进行跨文档或全局语料检索，必须通过显式实验配置启用。

### 4.2 关于 Phase 2 契约摘要

建议统一替换为包含以下要点的摘要：

- `Query(query_id, text, doc_id, raw_doc_name, task_family, subtask, length)`
- `RelevanceJudgment(query_id, page_id, relevance)`
- `RetrievalResult(query_id, page_id, score, rank)`

### 4.3 关于检索 API 摘要

建议将摘要区的 `RetrievalPipeline` 接口同步为最终主协议版本，包括：

- `retrieve(..., candidate_ids=None, score_batch_size=64)`
- `retrieve_text()` 仅为便利包装
- `generate_candidates()` 默认服务于文档内候选集合

### 4.4 关于修订记录展示

建议统一替换为：

> 本章综合吸收 `docs/revision/core_module_revision_v1.md` 至 `docs/revision/core_module_revision_v4.md` 的逐轮修订结果。

---

## 5. 收尾后状态

完成本轮修改后，`Project_Plan.md` 应达到以下状态：

1. baseline 默认协议在全文中不再自相矛盾。
2. 摘要区与权威接口定义区保持一致。
3. 修订历史展示与文档真实演化过程一致。

到这一步，计划文档就可以视为当前阶段的实现基准文档，不再需要为同一批核心问题继续新增 v5 级别的修订约束。
