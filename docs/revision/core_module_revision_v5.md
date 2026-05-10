# ZeroShotVDR 核心模块修订建议 v5

## 1. 说明

v5 与 v1-v4 的性质不同。

v1-v4 均属于**前瞻性审查**——在实现完成之前，对设计文档的接口契约、命名体系、模块边界提出预防性修订建议。

v5 是**事后核实**——基于 Step 2.2（索引层）的实际实现代码和 77 项测试用例的运行结果，对照原始设计文档和已有修订建议，逐条确认哪些契约描述与最终实现存在偏差，并给出需要同步回文档的勘误项。

这一轮的意义不是继续扩展架构，而是建立文档与代码之间的一致性基准，为后续 Phase 3/4 的实现者提供可信赖的参考。

---

## 2. 审查范围

- `src/zeroshot_vdr/indexing/store.py` — `IndexStore` 实现
- `src/zeroshot_vdr/indexing/encoder.py` — `PageEncoder` 实现
- 对照对象：`docs/Project_Plan.md` 中 Step 2.2 相关的接口描述，以及 v3 修订建议中的 IndexStore API 草案

---

## 3. 已实现接口与文档描述的偏差

### 3.1 PageEncoder 构造函数缺少必填参数 `processor`

**文档描述（v3 草案）：**

```python
PageEncoder(model, batch_size: int = 4, dtype=torch.float16)
```

**实际实现：**

```python
PageEncoder(
    model,
    processor,                      # 必填，ColPaliProcessor 实例
    batch_size: int = 4,
    dtype: torch.dtype | None = None,
    device: str = "cuda:0",
    storage_dtype: torch.dtype = torch.float16,
)
```

**偏差说明：**

实现将 ColPali 的图像处理器（`ColPaliProcessor`）作为独立的第二位置参数分离出来，而不是让 `model` 内部封装处理器。这是一个有意义的设计决定，可以：

1. 在测试中独立 mock 处理器行为
2. 在多任务场景下允许同一模型配合不同处理器

`dtype` 参数语义也有偏差：原始草案中 `dtype` 控制"推理精度"，实现中 `dtype` 会对模型本身调用 `.to(dtype)` 转换，而独立的 `storage_dtype` 参数才控制落盘精度。原草案把两者合并成一个 `dtype` 参数，语义不够区分。

**建议回填：**

将 PageEncoder 构造函数的权威签名更新为：

```python
class PageEncoder:
    def __init__(
        self,
        model,
        processor,
        batch_size: int = 4,
        dtype: torch.dtype | None = None,   # None 表示不转换模型精度
        device: str = "cuda:0",
        storage_dtype: torch.dtype = torch.float16,  # 独立控制落盘精度
    ): ...
```

并在文档中说明：`processor` 与 `model` 分离是有意设计，便于独立替换和测试。

---

### 3.2 PageEncoder 提供 `from_pretrained` 工厂方法

**文档描述：** 未提及。

**实际实现：**

```python
@classmethod
def from_pretrained(
    cls,
    model_repo: str = "vidore/colpali-v1.3",
    base_repo: str = "vidore/colpaligemma-3b-pt-448-base",
    device: str = "cuda:0",
    dtype: torch.dtype | None = None,
    batch_size: int = 4,
    storage_dtype: torch.dtype = torch.float16,
) -> "PageEncoder": ...
```

**偏差说明：**

`from_pretrained` 是生产环境中构造 `PageEncoder` 的标准入口（内部调用 PEFT patch 验证、加载 ColPali 模型和处理器），但文档中没有提及这一工厂方法的存在。实现者直接看文档会认为只能手动构造。

**建议回填：**

在接口描述中补充：

> `PageEncoder.from_pretrained()` 是生产场景下的推荐构造入口，内部封装了 sitecustomize 补丁验证和模型/处理器的加载。直接构造 `PageEncoder(model, processor, ...)` 主要用于测试和离线场景。

---

### 3.3 PageEncoder.encode_corpus() 额外参数未在文档中体现

**文档描述（v3 草案）：**

```python
def encode_corpus(self, pages: list[Page], store: IndexStore) -> None: ...
```

**实际实现：**

```python
def encode_corpus(
    self,
    pages: list[Page],
    store: IndexStore,
    show_progress: bool = True,
    resume: bool = True,          # 断点续建：跳过已索引的页面
) -> None: ...
```

**偏差说明：**

`resume=True` 是一个有重要语义意义的参数：默认行为是跳过 `store` 中已存在的 `page_id`，支持中断后重启而不重复编码。这一行为没有在文档中说明，会让实现者不清楚"重复调用 encode_corpus 是否安全"。

**建议回填：**

```python
def encode_corpus(
    self,
    pages: list[Page],
    store: IndexStore,
    show_progress: bool = True,
    resume: bool = True,
) -> None: ...
```

并在说明中补充：

> `resume=True` 为默认行为，跳过 `store.list_page_ids()` 中已存在的页面，支持断点续建。重复调用 `encode_corpus` 是安全的，不会重复写入已索引页面。

---

### 3.4 `page_ids.json` 的存储格式与文档描述不符

**文档描述（store.py 文件头注释）：**

```
page_ids.json   # page_id → file-path mapping（暗示 dict）
```

**实际实现：**

```json
["docqa/longdocurl_K4/doc001/p0", "docqa/longdocurl_K4/doc001/p1", ...]
```

`page_ids.json` 存储的是 **有序 JSON 数组**，而不是映射字典。每页的实际文件路径由 `page_id` 通过确定性规则生成（将 `/` 和 `\` 替换为 `_` 后加 `.pt` 扩展名），不需要显式存储映射关系。

**偏差说明：**

原始注释"page_id → file-path mapping"会让读者预期 dict，但实际是 list。这一偏差在测试阶段直接导致了断言失败，需要在文档层面澄清。

**建议回填：**

将存储布局说明更新为：

```
{index_dir}/
├── pages/
│   └── {safe_page_id}.pt    # safe_page_id = page_id 中 '/' 替换为 '_'
├── page_ids.json             # 有序 JSON 数组 [page_id, ...]，记录全局页面顺序
└── index_meta.json           # 元信息：model_name, dim, num_pages, created_at
```

并说明文件路径规则：

> `pages/{safe_page_id}.pt` 中 `safe_page_id = page_id.replace("/", "_").replace("\\", "_")`。`page_ids.json` 维护插入顺序，用于迭代和 baseline 全量候选生成。

---

### 3.5 `index_meta.json` 的创建时机与文档预期不一致

**文档预期：**

`index_meta.json` 在首次 `write_page()` 时自动创建，记录当前索引的元信息。

**实际实现：**

`index_meta.json` **不由 `write_page()` 触发创建**。它只在显式调用 `store.save_meta(model_name, dim)` 后才写入。若未调用 `save_meta()`，`load_meta()` 返回默认值 `{"model_name": "unknown", "dim": 0, ...}`，`index_meta.json` 文件本身不存在。

**偏差说明：**

这是一个设计上的有意决策：因为 `write_page()` 不携带 `model_name` 和 `dim` 参数，无法在写入页面时自动生成元信息，只能由调用方在编码完成后显式记录。然而文档中没有说明这一两步骤流程，让使用者可能遗漏 `save_meta()` 调用。

**建议回填：**

在 IndexStore 接口说明中明确加入 `save_meta` 和 `load_meta`：

```python
class IndexStore:
    def write_page(self, page_id: str, embedding: torch.Tensor) -> None: ...
    def read_page(self, page_id: str) -> torch.Tensor: ...
    def iter_pages(self, page_ids: list[str] | None = None): ...
    def list_page_ids(self, doc_id: str | None = None) -> list[str]: ...
    def get_mean_pooled_view(self, page_ids: list[str] | None = None): ...
    def read_stacked(self, page_ids: list[str]) -> tuple[torch.Tensor, list[str]]: ...
    def save_meta(self, model_name: str, dim: int) -> None: ...   # ← 需补充
    def load_meta(self) -> dict: ...                              # ← 需补充
    @property
    def stats(self) -> dict: ...
```

并说明调用顺序：

> 完整的编码流程应以 `store.save_meta(model_name, dim)` 收尾，以确保 `index_meta.json` 记录当前索引的模型信息和总页数。

---

### 3.6 `get_mean_pooled_view()` 的返回类型与 v3 草案不一致

**v3 草案描述：**

```python
def get_mean_pooled_view(self, page_ids: list[str] | None = None): ...
```

返回类型未显式标注。

**实际实现：**

```python
def get_mean_pooled_view(
    self, page_ids: list[str] | None = None
) -> tuple[torch.Tensor, list[str]]:
```

返回 `(pooled_embeddings [n_pages, dim], page_id_list)`，与 `read_stacked()` 的返回格式保持一致——始终同时返回张量和对应的页面 ID 列表，以避免调用方自行对齐顺序。

**建议回填：**

将接口描述更新为：

```python
def get_mean_pooled_view(
    self, page_ids: list[str] | None = None
) -> tuple[torch.Tensor, list[str]]:
    """
    返回 (pooled_tensor [n_pages, dim], page_ids)。
    pooled_tensor[i] 是 page_ids[i] 对应页面的 patch embeddings 的均值向量。
    """
```

并在文档中统一说明：IndexStore 的批量读取接口（`read_stacked` 和 `get_mean_pooled_view`）均返回 `tuple[Tensor, list[str]]` 而不是裸张量，以保证顺序可追溯。

---

### 3.7 IndexStore 额外提供 `write_batch()` 便利方法

**文档描述：** 未提及。

**实际实现：**

```python
def write_batch(self, page_ids: list[str], embeddings: torch.Tensor) -> None:
    """批量写入，embeddings shape [batch, n_patches, dim]。"""
```

`encode_corpus()` 内部使用的是 `write_batch()` 而不是循环调用 `write_page()`。这使得批量写入路径更清晰，并让 `write_page()` 保持为原子操作。

**建议回填：**

将 `write_batch()` 补充进 IndexStore 接口列表，注明它是 `encode_corpus()` 的内部路径，外部调用者也可直接使用：

```python
def write_batch(self, page_ids: list[str], embeddings: torch.Tensor) -> None:
    """批量写入 [batch, n_patches, dim] 的 embeddings，等价于逐页调用 write_page()。"""
```

---

## 4. 实现中良好实践的确认

以下几点与设计文档的隐式预期一致，值得在文档中显式记录为"确认项"：

### 4.1 page_id 中的 `/` 路径分隔符处理

`page_id` 格式（如 `docqa/longdocurl_K4/doc001/p0`）中包含 `/`，IndexStore 将其替换为 `_` 后作为文件名，确保跨平台安全。文档中应显式说明这一转换规则。

### 4.2 断点续建保证了幂等性

`encode_corpus(resume=True)` 的默认行为确保：重复执行同一编码任务不会导致重复写入，已有条目在重复调用时被安全跳过。这一性质对长文档语料库的增量建索引场景非常重要。

### 4.3 `stats` 属性聚合了实时信息

`store.stats` 返回的 `num_pages` 来自实时加载的 `page_ids.json`，不依赖 `index_meta.json` 的快照计数。这意味着即使从未调用 `save_meta()`，`stats.num_pages` 仍然是准确的。文档中可以用这一点说明 `stats` 与 `load_meta()` 的语义差异。

---

## 5. v5 收尾状态

完成本轮文档同步后，Step 2.2 的接口描述应达到以下状态：

1. **PageEncoder** 的构造签名、工厂方法、`encode_corpus` 参数与实现完全一致。
2. **IndexStore** 的磁盘布局描述（数组 vs 映射）与实现一致。
3. **IndexStore** 的完整公共接口列表包含 `write_batch`、`save_meta`、`load_meta`。
4. **批量读取接口**（`read_stacked`、`get_mean_pooled_view`）的返回类型显式标注为 `tuple[Tensor, list[str]]`。
5. **`index_meta.json`** 的创建时机明确为显式调用 `save_meta()` 而非隐式触发。

v5 之后，设计文档与 Step 2.2 已实现代码之间的已知偏差已全部消除，可作为 Phase 3（检索层）实现的稳定基准。
