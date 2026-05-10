# ZeroShotVDR 核心模块修订建议 v3

## 1. 说明

本次修订基于对最新 `docs/Project_Plan.md` 的第三轮审查形成。

与 v2 相比，v3 不再停留在“问题类型”和“修改方向”层面，而是对仍未闭合的关键点给出更具体的固定建议。这里的“固定”有两个适用范围：

1. 直接影响 baseline 正确性的语义契约。
2. 一旦后续再改会牵动多模块联动的核心接口。

换句话说，v3 不是要把整个设计再细化一遍，而是针对已经识别出的剩余问题，明确哪些代码和接口现在就应该定下来。

---

## 2. v3 审查结论

当前 `Project_Plan.md` 的整体方向已经基本符合预期，但仍有几处关键问题需要收口：

1. DocumentQA 的文档标识来源仍未和真实数据字段完全对齐。
2. baseline 已定义为文档内检索，但 Query 契约和检索接口还没有真正承载这一协议。
3. 索引读取接口中仍混合着核心抽象与 baseline 便利实现。
4. 风险说明里仍残留与当前索引格式不一致的加载表述。
5. 个别评测范围和示例资源还需要再做一次边界收紧。

因此，v3 的重点是把这些点从“原则正确”推进到“接口闭合”。

---

## 3. 需要固定的设计项

### 3.1 固定 DocumentQA 的文档标识来源

这一点应明确写死。

对于当前 Phase 2 的 DocumentQA baseline：

- 原始数据中的来源字段使用 `doc_name`。
- 项目内部统一使用 `doc_id` 作为稳定文档标识。
- `doc_id` 不是假定数据里天然存在的字段，而是由 `doc_name` 归一化得到的内部稳定标识。

建议在设计文档中直接写成：

> 对 DocumentQA 子集，原始样本中的文档标识来源字段为 `doc_name`。为保证 page_id、query_id、索引文件名和结果落盘的一致性，系统内部统一使用归一化后的 `doc_id`。若原始数据未提供独立 `doc_id` 字段，则由 `doc_name` 通过统一规则生成，并在 metadata 中保留 `raw_doc_name -> doc_id` 的映射。

这条表述的作用是把“真实字段”和“内部主键”分开，避免后续实现中混淆。

### 3.2 固定数据契约中的关键字段

当前 `source_subset` 已不足以承载文档内检索、跨子任务评测和多档位对比这三类需求。v3 建议将以下字段视为核心契约的一部分，并明确固定。

建议直接替换为以下表述：

```python
from dataclasses import dataclass


@dataclass
class Page:
    page_id: str
    doc_id: str
    raw_doc_name: str | None
    task_family: str
    subtask: str
    length: str
    page_idx: int
    image_path: str


@dataclass
class Query:
    query_id: str
    text: str
    doc_id: str
    raw_doc_name: str | None
    task_family: str
    subtask: str
    length: str


@dataclass
class RetrievalResult:
    query_id: str
    page_id: str
    score: float
    rank: int


@dataclass
class RelevanceJudgment:
    query_id: str
    page_id: str
    relevance: int
```

这里要固定的不是 dataclass 语法本身，而是以下语义：

1. `Page` 必须能表达页面属于哪个文档、哪个子任务、哪个长度档位。
2. `Query` 必须能表达它属于哪个文档，否则“文档内检索”只是文字说明。
3. `raw_doc_name` 建议显式保留，方便调试、回溯和数据核对。
4. `source_subset` 这种混合字段应尽量退出核心契约，因为它无法稳定承载 v2 中已经建立的命名分层。

### 3.3 固定 ID 构造辅助函数

既然现在已经把命名分层作为正式设计的一部分，那么 ID 构造逻辑不应再散落在 Adapter、GroundTruthLoader 或评测脚本里临时拼接。

建议固定以下辅助函数接口：

```python
def normalize_doc_id(raw_doc_name: str) -> str:
    ...


def build_page_id(
    task_family: str,
    subtask: str,
    length: str,
    doc_id: str,
    page_idx: int,
) -> str:
    ...


def build_query_id(
    task_family: str,
    subtask: str,
    length: str,
    query_index: int,
) -> str:
    ...
```

建议在文档中明确：

- `build_page_id()` 和 `build_query_id()` 是全链路唯一合法的 ID 构造入口。
- `GroundTruthLoader`、`DocumentQAAdapter`、`PageCorpus` 和结果落盘模块不得自行拼接 ID 字符串。

这条约束值得固定，因为它能显著减少隐式不一致。

### 3.4 固定 baseline 检索协议对应的 Query 与 RetrievalPipeline 接口

这是当前最需要收口的一点。

既然 baseline 已经被定义为文档内检索，那么默认检索接口就不能继续把“全量候选”作为隐式默认行为。

建议将关键接口明确收紧为：

```python
class RetrievalPipeline:
    def retrieve(
        self,
        query: Query,
        top_k: int = 10,
        candidate_ids: list[str] | None = None,
        score_batch_size: int = 64,
    ) -> list[RetrievalResult]:
        ...

    def generate_candidates(
        self,
        query: Query,
        query_emb,
        top_n: int | None = None,
    ) -> list[str]:
        ...
```

并在设计文档中明确以下默认规则：

> Baseline 模式下，`retrieve()` 接收 `Query` 对象而不是纯文本字符串。若 `candidate_ids` 为空，则 `generate_candidates()` 默认返回与 `query.doc_id` 对应的全部页面，即文档内候选集合，而不是全局页面集合。仅在显式启用 global retrieval 实验配置时，候选范围才允许扩展为全局语料。

如果还想保留纯文本查询的便利接口，可以额外提供一个非核心包装接口，例如：

```python
def retrieve_text(
    self,
    text: str,
    candidate_ids: list[str],
    top_k: int = 10,
) -> list[RetrievalResult]:
    ...
```

但这个包装接口不应再承担 baseline 的默认协议。

### 3.5 固定 IndexStore 的核心接口与便利接口边界

当前 `read_all()` 的定义前后不完全一致，且它本身更像 baseline 便利函数，而不是长期稳定的核心抽象。

v3 建议将 IndexStore 的核心接口固定为以下几类：

```python
class IndexStore:
    def write_page(self, page_id: str, embedding) -> None: ...
    def read_page(self, page_id: str): ...
    def iter_pages(self, page_ids: list[str] | None = None): ...
    def list_page_ids(self, doc_id: str | None = None) -> list[str]: ...
    def get_mean_pooled_view(self, page_ids: list[str] | None = None): ...
```

其中：

- `write_page` / `read_page` / `list_page_ids` 属于核心接口。
- `iter_pages` 用于兼容变长 patch 的读取路径，应视为正式支持能力。
- `get_mean_pooled_view` 用于 Phase 4 两阶段检索，也属于稳定接口。

对于 `read_all()`，v3 建议不要再把它放在核心接口列表中，而是改成如下定位：

> `read_all()` 或 `read_stacked()` 若保留，应明确标注为 baseline 便利函数，仅在页面表示可安全堆叠时使用，不作为变长 patch 场景下的通用读取接口。

如果要给出更具体的接口，可以写成：

```python
def read_stacked(self, page_ids: list[str]) -> tuple:
    """仅在所有页面 patch 数一致时使用的便利函数。"""
    ...
```

这样比保留一个含义模糊的 `read_all()` 更稳定。

### 3.6 固定 GroundTruthLoader 的 ID 构造签名

当前 `GroundTruthLoader.build_page_id()` 仍只接受 `subset, doc_id, page_idx`，这已经跟 v2 建立的命名分层不一致。

建议直接改为：

```python
@staticmethod
def build_page_id(
    task_family: str,
    subtask: str,
    length: str,
    doc_id: str,
    page_idx: int,
) -> str:
    ...
```

同时建议增加：

```python
@staticmethod
def build_query_id(
    task_family: str,
    subtask: str,
    length: str,
    query_index: int,
) -> str:
    ...
```

需要固定的原则是：Ground truth 侧不得再使用比语料构建侧更弱的主键体系。

### 3.7 固定 Phase 2 baseline 的主评测范围

当前文档已经基本收敛到 DocumentQA，但仍建议再明确一层主评测范围，避免后续把辅助文件或边缘子任务混入主结果。

建议直接写成：

> Phase 2-3 的主评测范围固定为 DocumentQA 中的 `longdocurl`、`mmlongdoc`、`slidevqa` 三个子任务。`text_mmlongdoc` 不纳入主评测表，除非后续确认其图像页面与标注协议可与当前页级检索设置严格对齐。

这条表述虽然不是代码接口，但它会直接影响数据加载、结果统计和报告主表，是值得固定的实验边界。

### 3.8 固定索引加载的风险说明口径

当前风险章节中“使用 `mmap_mode=True` 加载 `.pt`”的说法建议移除，因为它与当前逐页 `.pt` 方案并不一致。

建议替换为：

> 对当前逐页 `.pt` 文件索引，默认采用按页加载或按批加载的方式控制内存占用，不依赖 `mmap_mode`。若后续引入分片化的 `npy`、`safetensors` 或其他支持内存映射的存储格式，再单独评估 memmap 方案。

这个修改虽然小，但可以避免实现阶段误解。

### 3.9 固定最小资源示例与 baseline 资源保持一致

项目结构和数据示例中，建议把 baseline 最小资源路径也统一到 DocumentQA 主线，避免目录示意仍给人以 VRAG 为默认入口的印象。

建议在相关示意中优先展示：

- `0_mmlb_data.tar.gz`
- `5_docqa_image.tar.gz`

而把 `1_vrag_image.tar.gz` 放到“扩展任务资源”或“后续可选下载”位置。

这条修改不会影响实现，但会让计划文本的主线更稳定。

---

## 4. 建议直接替换到设计文档中的关键表述

以下内容可以较直接地替换进 `Project_Plan.md`。

### 4.1 关于 DocumentQA 文档标识

建议替换为：

> DocumentQA 原始样本使用 `doc_name` 表示文档来源。系统内部统一使用归一化后的 `doc_id` 作为稳定文档标识，并保留 `raw_doc_name` 以便回溯。若原始数据未提供独立 `doc_id` 字段，则由 `doc_name` 按统一规则生成，禁止在不同模块中各自临时构造。

### 4.2 关于 Query 契约

建议替换为：

> 由于 baseline 采用文档内检索，`Query` 必须显式携带所属文档标识 `doc_id`，以便默认候选集合能够约束在同一文档页面内。纯文本字符串查询仅作为便利接口存在，不作为 baseline 主协议。

### 4.3 关于 RetrievalPipeline 默认候选范围

建议替换为：

> Baseline 模式下，`RetrievalPipeline.retrieve()` 的默认候选范围是与 `query.doc_id` 对应的页面集合，而不是全局语料集合。只有在显式启用全局检索实验配置时，系统才允许跳出文档内候选范围。

### 4.4 关于 IndexStore 的核心读取能力

建议替换为：

> `IndexStore` 的稳定读取语义为按页读取与按页面列表迭代读取。任何需要返回全量 stacked tensor 的接口均视为 baseline 便利函数，而非变长 patch 场景下的通用接口。

### 4.5 关于索引加载的风险说明

建议替换为：

> 当前逐页 `.pt` 索引默认采用按页或按批加载控制内存占用，不以 `mmap_mode` 作为标准加载方案。若未来切换为支持内存映射的分片格式，再单独引入 memmap 设计。

---

## 5. 优先级建议

### 第一优先级

1. 固定 `doc_name -> doc_id` 的内部转换规则。
2. 固定 `Query` 必须携带 `doc_id`。
3. 固定 `RetrievalPipeline` baseline 默认采用文档内候选集。

这三项若不收口，baseline 协议仍可能停留在文字层，而非接口层。

### 第二优先级

1. 固定 `IndexStore` 的核心接口与便利接口边界。
2. 固定 `GroundTruthLoader` 的完整 ID 构造签名。
3. 固定主评测子任务范围。

这些问题主要影响实现时的一致性和 Phase 4 的扩展成本。

### 第三优先级

1. 修正风险说明中的加载表述。
2. 调整目录与资源示例的主线一致性。

这些问题不会阻塞开发，但有助于减少误读。

---

## 6. 总结

v3 的目标不是继续增加新的抽象层，而是把已经选择的 baseline 协议真正落实到代码契约和默认接口上。

如果本轮修订被吸收，设计文档会在以下三点上明显闭合：

1. 真实数据字段与内部主键的关系被明确固定。
2. 文档内检索不再只是评测说明，而会成为 Query 和 RetrievalPipeline 的默认行为。
3. 核心索引接口与 baseline 便利函数之间的边界被清晰区分。

这三点补齐后，当前设计就不仅是方向上合理，而且会更接近“可以直接照着实现且不容易跑偏”的状态。
