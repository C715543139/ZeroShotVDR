# ZeroShotVDR 核心模块修订建议 v6

## 1. 说明

v6 延续 v5 的工作方式，属于**事后核实型修订记录**。

这一次核实对象不再是 Step 2.2 的索引层，而是 Step 2.3 的查询编码与检索层。其目标不是提出新的架构方向，而是基于**真实实现代码**和**已通过的 Step 2.3 测试**，回看 `docs/Project_Plan.md` 中关于检索层的接口摘要、行为说明和阶段任务拆解，找出仍然存在的“文档表述与真实行为不一致”之处，并将其作为勘误记录沉淀下来。

本轮修订的直接背景是：在真实实现上运行 `tests/step2.3` 并补充边界测试后，发现 Step 2.3 的若干文档表述仍带有“理想化 API 草案”痕迹，没有完整反映实际实现中的细节语义。

因此，v6 的重点是为后续实现者、测试编写者和 Phase 4 扩展提供一个更可信赖的检索层文档基线。

---

## 2. 审查范围

- `src/zeroshot_vdr/retrieval/encoder.py` — `QueryEncoder` 实现
- `src/zeroshot_vdr/retrieval/scoring.py` — MaxSim 打分实现
- `src/zeroshot_vdr/retrieval/pipeline.py` — `RetrievalPipeline` 实现
- 对照对象：`docs/Project_Plan.md` 中 Step 2.3 相关任务摘要、API 摘要以及 4.3 / 4.5.5 / 4.5.7 的检索层说明
- 验证依据：真实实现下通过的 `tests/step2.3` 测试集（本轮补强后共 41 项）

---

## 3. 已实现接口与文档描述的偏差

### 3.1 `QueryEncoder` 构造函数并非简化的 `QueryEncoder(model)`

**文档旧表述：**

```python
class QueryEncoder:
    def __init__(self, model): ...
```

**实际实现：**

```python
class QueryEncoder:
    def __init__(self, model, processor, device: str = "cuda:0"): ...
```

并额外提供：

```python
@classmethod
def from_pretrained(...) -> "QueryEncoder": ...

@classmethod
def from_page_encoder(cls, page_encoder) -> "QueryEncoder": ...

def encode_batch(self, queries: list[str]) -> torch.Tensor: ...
```

**偏差说明：**

Step 2.3 的早期接口草案把 `QueryEncoder` 简化成了“只依赖模型”的抽象，但真实实现与 `PageEncoder` 一样，将 processor 独立为显式依赖，以便：

1. 在测试中单独 mock 文本预处理逻辑
2. 与页面编码器共享底层 ColPali 模型 / processor
3. 在生产代码中通过工厂方法统一加载模型与处理器

如果文档继续写成 `QueryEncoder(model)`，实现者会误以为 processor 被模型内部封装，从而错误理解构造方式和测试注入方式。

**建议回填：**

将 `QueryEncoder` 的权威签名更新为：

```python
class QueryEncoder:
    def __init__(self, model, processor, device: str = "cuda:0"): ...
    @classmethod
    def from_pretrained(...) -> "QueryEncoder": ...
    @classmethod
    def from_page_encoder(cls, page_encoder) -> "QueryEncoder": ...
    def encode(self, query: str) -> torch.Tensor: ...
    def encode_batch(self, queries: list[str]) -> torch.Tensor: ...
```

并说明：直接构造通常用于测试/注入，生产场景更适合使用工厂方法或从 `PageEncoder` 共享模型。

---

### 3.2 MaxSim 的 `Sim` 语义应明确为“归一化点积”，而不是笼统写成“余弦或点积”

**文档旧表述：**

> `Sim` 通常为余弦相似度或点积

**实际实现：**

```python
def maxsim_score(query_emb, page_emb, norm: bool = True) -> torch.Tensor: ...
def batched_maxsim(query_emb, pages_emb, norm: bool = True) -> torch.Tensor: ...
```

内部默认会先做 L2 归一化，再计算点积。

**偏差说明：**

旧文档的写法对“理论上可以选什么”是成立的，但对“当前项目究竟实现了什么”是不够精确的。真实实现的默认行为是：

- 先做 L2 归一化
- 再做点积
- 因而数值上等价于余弦相似度

这不是“开放待定项”，而是已经落地的具体实现语义，应在文档中明确写出。

**建议回填：**

统一替换为：

> 当前实现默认使用 **L2 归一化后的点积** 作为 `Sim`，其数值等价于余弦相似度；接口保留 `norm=True/False` 以便未来实验关闭归一化。

---

### 3.3 `scoring.py` 实际提供了变长 patch 的回退路径 `batched_maxsim_variable()`

**文档旧表述：**

仅列出：

```python
def maxsim_score(...) -> torch.Tensor: ...
def batched_maxsim(...) -> torch.Tensor: ...
```

**实际实现：**

```python
def batched_maxsim_variable(
    query_emb: torch.Tensor,
    pages_list: list[torch.Tensor],
    norm: bool = True,
) -> torch.Tensor:
    ...
```

**偏差说明：**

虽然 Step 2.3 的 baseline 主路径仍假设同批页面 patch 数一致，但真实实现已经提前准备好了变长 patch 的逐页回退打分路径。这意味着文档若只写 `batched_maxsim()`，会低估当前实现对 Phase 4A 扩展的支持程度。

**建议回填：**

在打分接口摘要中补充 `batched_maxsim_variable()`，并明确其语义：

> 当候选页面 patch 数不一致时，检索层可回退到逐页打分路径，而不强制要求 padding 或统一 patch 数。

---

### 3.4 `RetrievalPipeline.__init__` 的真实签名包含 `processor=None`

**文档旧表述：**

```python
RetrievalPipeline(model, index_store, query_encoder=None, config=None)
```

**实际实现：**

```python
RetrievalPipeline(
    model,
    index_store,
    processor=None,
    query_encoder: QueryEncoder | None = None,
    config: dict | None = None,
)
```

**偏差说明：**

真实实现支持多种构造路径：

1. 直接显式传 `query_encoder`
2. `model` 本身就是 `QueryEncoder`
3. `model` 是 `PageEncoder`，自动提取 `_model/_processor`
4. `model + processor` 组合构造 `QueryEncoder`

因此 `processor` 不是多余参数，而是“允许直接用裸 ColPali 模型接入检索层”的必要入口。文档若不写这一参数，使用者会错误以为只能先手工构造 `QueryEncoder`。

**建议回填：**

将构造签名更新为：

```python
def __init__(self, model, index_store: IndexStore,
             processor=None,
             query_encoder: QueryEncoder | None = None,
             config: dict | None = None): ...
```

并补充一条说明：`query_encoder` 优先级最高，`processor` 仅在直接传裸模型时需要。

---

### 3.5 `candidate_ids=None` 与 `candidate_ids=[]` 的语义必须严格区分

**文档旧表述（容易引发误读）：**

> 若 candidate_ids 为空，则默认使用 `query.doc_id` 对应的文档内页面集合

**实际实现：**

```python
if candidate_ids is None:
    candidate_ids = self.generate_candidates(query, query_emb)

if not candidate_ids:
    return []
```

**真实语义：**

- `candidate_ids is None`：才触发默认候选生成
- `candidate_ids == []`：被视为调用方显式提供的空候选集，直接返回空结果

**偏差说明：**

“为空”这个措辞在自然语言里容易把 `None` 与 `[]` 混在一起，但对真实实现来说两者语义不同。这个差异在本轮补充边界测试时被直接暴露：若按旧文档理解去写测试，会错误断言“传空列表时应回退默认候选生成”。

这不是细枝末节，而是会直接影响：

1. 测试用例设计
2. 上层实验脚本对空候选行为的预期
3. 后续 Phase 4 粗筛结果为空时的处理逻辑

**建议回填：**

将文档统一改写为：

> 仅当 `candidate_ids is None` 时，`retrieve()` 才调用 `generate_candidates()` 生成 baseline 默认候选集；若显式传入 `candidate_ids=[]`，当前实现将其视为“空候选集”，并直接返回空结果。

---

### 3.6 `retrieve_text()` 的便利接口行为应写得更具体

**文档旧表述：**

> 内部构造临时 Query 对象后委托给 `retrieve()`

**实际实现：**

```python
temp_query = Query(
    query_id="adhoc/q000",
    text=text,
    doc_id="",
    raw_doc_name=None,
    task_family="",
    subtask="",
    length="",
)
```

**偏差说明：**

旧表述的大方向正确，但仍缺少两个对调试和测试都有意义的事实：

1. 当前实现中临时查询的 `query_id` 固定为 `adhoc/q000`
2. 文档级元信息字段为空值，不承担 baseline 文档内协议

这些细节不是为了“冻结便利接口的实现细节”，而是为了让调用者理解：`retrieve_text()` 本质上是一个**显式候选集下的 adhoc 查询包装器**，而不是 baseline 主协议入口。

**建议回填：**

在文档中明确：

> `retrieve_text()` 会构造一个临时 `Query` 对象（当前实现中 `query_id="adhoc/q000"`，文档级元信息留空），然后复用 `retrieve()` 的打分与结果组装逻辑。该接口必须由调用方显式提供 `candidate_ids`。

---

### 3.7 `score_candidates()` 的返回值不是裸分数，而是 `(scores, scored_page_ids)`

**文档旧表述：**

```python
def score_candidates(...) -> torch.Tensor:
    """返回 [n_candidates] scores"""
```

**实际实现：**

```python
def score_candidates(
    self,
    query_emb: torch.Tensor,
    candidate_ids: list[str],
    batch_size: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    ...
```

**偏差说明：**

真实实现同时返回：

- `scores`
- `scored_page_ids`

这是合理且必要的：因为候选加载过程中可能发生过滤、回退、顺序重组或异常跳过，调用方不能仅凭输入候选列表假设输出分数与其逐项严格对齐。因此文档若只写“返回分数张量”，就会低估接口的可追溯性设计。

**建议回填：**

将接口描述更新为：

```python
def score_candidates(...) -> tuple[torch.Tensor, list[str]]:
    """返回 (scores, scored_page_ids)。"""
```

并说明：结果组装阶段应使用返回的 `scored_page_ids` 而不是直接复用输入候选列表。

---

### 3.8 “记录单次查询平均延迟”与真实实现不符

**文档旧表述：**

> 记录单次查询平均延迟

**实际实现：**

`retrieve()` 内部用 `time.perf_counter()` 统计的是**当前这一次调用**的耗时，并通过 debug 日志输出；检索层本身并不维护“平均延迟”的累积统计状态。

**偏差说明：**

“平均延迟”意味着：

- 有跨多次调用的状态聚合
- 或者有统一的统计器/评测器负责汇总

而当前检索层并没有承担这个职责。若文档继续写“平均延迟”，会让人误以为 `RetrievalPipeline` 自带 benchmark 状态。

**建议回填：**

将任务描述改为：

> 记录查询耗时。当前实现仅在 `retrieve()` 内统计单次调用耗时并输出日志；若需平均延迟，应在评测层或实验脚本中做聚合。

---

## 4. 实现中良好实践的确认

以下几点与设计目标高度一致，值得在文档中显式保留为“确认项”：

### 4.1 检索层的四阶段流水线抽象已经稳定

真实实现清晰分离了：

1. `encode_query`
2. `generate_candidates`
3. `score_candidates`
4. `_assemble_results`

这说明 Step 2.3 的核心抽象是健康的，后续 Phase 4 的改动应优先替换局部策略，而不是破坏整条流水线。

### 4.2 baseline 主协议与便利接口已经有清晰边界

真实实现中：

- `retrieve(Query)` 承担 baseline 主协议
- `retrieve_text()` 只是显式候选集下的便利包装

这一边界是合理的，文档中应继续强调，避免调用者误把 `retrieve_text()` 当作 baseline 入口。

### 4.3 检索层已经为变长 patch 场景预留了演化空间

虽然 baseline 仍主要依赖同 patch 数批量打分，但 `score_candidates()` 已经在 `read_stacked()` 失败时提供逐页加载和 `batched_maxsim_variable()` 回退路径。这为后续 patch pruning / 变长表示实验保留了兼容空间，是一项值得保留的实现优势。

---

## 5. v6 收尾状态

完成本轮文档同步后，Step 2.3 的设计文档应达到以下状态：

1. **`QueryEncoder`** 的构造方式、工厂方法和批量接口与实现一致。
2. **MaxSim 默认相似度语义** 明确为“L2 归一化后的点积”。
3. **`RetrievalPipeline`** 的构造签名与候选召回语义不再含糊。
4. **`candidate_ids=None` 与 `candidate_ids=[]`** 的行为区别被明确记录，避免再次误写测试或上层逻辑。
5. **`score_candidates()`** 的真实返回值与变长 patch 回退路径被正式写入接口摘要。
6. **延迟记录职责边界** 被纠正为“单次耗时记录在检索层，平均统计由评测层聚合”。

到这一步，`Project_Plan.md` 中关于 Step 2.3 的描述即可视为与当前实现基本对齐的权威参考。

**补充说明：**

本轮 v6 记录主要解决的是**计划文档**与真实实现之间的偏差；代码内个别 docstring 若仍保留旧措辞，可作为后续小规模文档清理任务继续跟进，但不影响本轮对外权威计划文档的对齐完成度。

