# ZeroShotVDR 核心模块修订建议 v7

## 1. 说明

v7 延续 v5、v6 的方式，属于**事后核实型修订记录**。

本轮核实对象从 Step 2.3 的检索层进一步推进到 Step 2.4 的评测层。其目标不是提出新的评测架构，而是基于**真实实现代码**、**已通过的 Step 2.4 测试**以及对 `docs/Project_Plan.md` 的逐段回查，确认评测层中哪些接口描述仍停留在早期草案状态，并将这些偏差整理为正式勘误。

本轮修订的直接背景是：在真实实现上重写并运行 `tests/step2.4` 后，发现 `metrics.py` 与 `ground_truth.py` 的若干计划文档表述，与当前代码已经出现显著偏差。如果不及时修正，将直接影响：

1. Step 2.4 测试的编写方式
2. Phase 3 评测脚本对输入输出格式的理解
3. 后续实验脚本和结果汇总逻辑的接口约定

因此，v7 的重点是把 `Project_Plan.md` 中 Step 2.4 的评测层摘要、API 草案和行为说明，同步到当前真实实现的语义基线。

---

## 2. 审查范围

- `src/zeroshot_vdr/evaluation/metrics.py` — 四项指标与批量聚合实现
- `src/zeroshot_vdr/evaluation/ground_truth.py` — ground truth 加载实现
- 对照对象：`docs/Project_Plan.md` 中 Step 2.4 任务摘要与 4.4 评测层接口说明
- 验证依据：真实实现下通过的 `tests/step2.4` 测试集（本轮补强后共 53 项）

---

## 3. 已实现接口与文档描述的偏差

### 3.1 `compute_all_metrics()` 当前接受的是 `page_id` 列表，而不是 `RetrievalResult` 列表

**文档旧表述：**

```python
def compute_all_metrics(
    retrieval_results: dict[str, list[RetrievalResult]],
    ground_truth: dict[str, set[str]],
    k_values: list[int] = [1, 3, 5, 10],
) -> pd.DataFrame:
    ...
```

**实际实现：**

```python
def compute_all_metrics(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    ...
```

其中 `retrieval_results` 的真实语义是：

```python
{query_id: [retrieved_page_id, ...]}
```

**偏差说明：**

旧文档把批量评测层写成了“直接消费检索对象列表”的形式，这在抽象上看似自然，但当前真实实现更偏向于先在检索层之外把结果规整成 `page_id` 排序列表，再将其喂给评测层。也就是说：

- 原子指标函数只处理纯 Python 原生类型
- 批量聚合函数也延续了这一风格
- `compute_all_metrics()` 不依赖 `RetrievalResult` dataclass

这一点在测试中被直接暴露：若按旧文档传入 `list[RetrievalResult]`，会在内部集合操作中触发类型错误。

**建议回填：**

将权威签名修正为：

```python
def compute_all_metrics(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_values: list[int] | None = None,
) -> pd.DataFrame: ...
```

并明确：评测层当前消费的是**已排序 page_id 列表**，而不是检索结果对象本身。

---

### 3.2 `compute_all_metrics()` 的返回列不仅有 4 项指标，还包含 `n_queries`

**文档旧表述：**

```python
pd.DataFrame : columns=['k', 'Recall', 'Precision', 'MRR', 'nDCG']
```

**实际实现：**

返回列为：

```python
['k', 'Recall', 'Precision', 'MRR', 'nDCG', 'n_queries']
```

**偏差说明：**

当前实现除输出宏平均指标外，还显式记录了每个 `k` 对应的聚合查询数 `n_queries`。这不是附属细节，而是对结果可解释性很有帮助的元信息：

- 能确认统计时实际参与聚合的查询数量
- 有助于排查查询 ID 对齐问题
- 便于分组评测或抽样子集汇报时检查样本量

若文档不写 `n_queries`，调用者会误以为 DataFrame 只包含 4 个指标列，从而在结果表消费或测试断言时产生偏差。

**建议回填：**

把返回列说明更新为：

```python
pd.DataFrame : columns=['k', 'Recall', 'Precision', 'MRR', 'nDCG', 'n_queries']
```

---

### 3.3 `k_values` 当前允许为 `None`，并在内部回退到默认 `[1, 3, 5, 10]`

**文档旧表述：**

```python
k_values: list[int] = [1, 3, 5, 10]
```

**实际实现：**

```python
k_values: list[int] | None = None
```

并在函数内部：

```python
if k_values is None:
    k_values = [1, 3, 5, 10]
```

**偏差说明：**

旧文档把默认值写死在签名层面，真实实现则把默认值延后到函数体内部。两者对大多数调用者来说结果相同，但在接口语义上仍有差异：

- 当前实现允许显式传 `None`
- 文档旧表述看起来像“必须总是 list[int]”

为避免测试与调��说明继续沿用旧签名，应在文档中与实现保持一致。

**建议回填：**

统一改为：

```python
k_values: list[int] | None = None
```

并补充一句：`None` 时使用默认 `[1, 3, 5, 10]`。

---

### 3.4 `GroundTruthLoader.load()` 当前不是 `subset` 单参数接口

**文档旧表述：**

```python
class GroundTruthLoader:
    def load(self, subset: str | None = None) -> dict[str, set[str]]: ...
```

**实际实现：**

```python
class GroundTruthLoader:
    def load(
        self,
        subtasks: list[str] | None = None,
        lengths: list[str] | None = None,
        task_family: str = "docqa",
    ) -> dict[str, set[str]]:
        ...
```

**偏差说明：**

旧文档把评测集过滤抽象为一个 `subset` 字符串参数，这更接近“任务族级别”选择。但当前真实实现已经落地为更贴近 DocumentQA 场景的三参数接口：

- `subtasks`：过滤具体子任务（如 `longdocurl` / `slidevqa`）
- `lengths`：过滤具体长度档位（如 `K32`）
- `task_family`：当前默认也是唯一稳定值 `"docqa"`

这不是命名细节，而是会直接影响：

1. ground truth 测试的调用方式
2. Phase 3 按子任务 × 长度档位批量评测的脚本设计
3. 调用者如何理解“任务族”与“子任务”的层级关系

**���议回填：**

将文档中的 `load()` 签名统一替换为：

```python
def load(
    self,
    subtasks: list[str] | None = None,
    lengths: list[str] | None = None,
    task_family: str = "docqa",
) -> dict[str, set[str]]: ...
```

并明确：当前实现的主过滤维度是 `subtasks` 与 `lengths`，而不是 `subset` 单参数。

---

### 3.5 `GroundTruthLoader` 当前没有暴露 `build_page_id()` / `build_query_id()` 静态 helper

**文档旧表述：**

```python
class GroundTruthLoader:
    @staticmethod
    def build_page_id(...): ...

    @staticmethod
    def build_query_id(...): ...
```

**实际实现：**

`GroundTruthLoader` 当前**没有**上述两个静态方法。

其核心逻辑是：

```python
from zeroshot_vdr.data.adapters import DocumentQAAdapter
...
adapter = DocumentQAAdapter(...)
gt = adapter.build_ground_truth()
```

**偏差说明：**

文档旧表述隐含了“ground truth 层自己负责 ID 构造”的设计，但真实实现已经选择了另一条路径：

- 将 DocumentQA 解析与 ID 构造集中在 `DocumentQAAdapter`
- `GroundTruthLoader` 只负责配置解析、过滤入口和缓存

因此，如果文档继续写静态 helper，会让实现者和测试作者误以为这些方法应当存在，并据此编写不符合真实接口的调用。

**建议回填：**

删除 `GroundTruthLoader` 中关于 `build_page_id()` / `build_query_id()` 的接口承诺，并改为说明：

> 当前实现直接通过 `DocumentQAAdapter.build_ground_truth()` 生成 `{query_id: set[page_id]}`，ID 命名规则由数据接入层统一保证。

---

### 3.6 `GroundTruthLoader` 当前额外提供 `load_by_subtask()`、`load_by_length()` 和 `config`

**文档旧表述：** 未提及。

**实际实现：**

```python
def load_by_subtask(
    self,
    subtask: str,
    lengths: list[str] | None = None,
) -> dict[str, set[str]]: ...

def load_by_length(
    self,
    length: str,
    subtasks: list[str] | None = None,
) -> dict[str, set[str]]: ...

@property
def config(self) -> dict: ...
```

**偏差说明：**

当前真实实现已经为 Phase 3/4 的实验脚本提供了更便捷的过滤入口，但文档仍停留在最简草案层面。尤其是：

- `load_by_subtask()` 非常适合分子任务评测
- `load_by_length()` 非常适合分长度档位评测
- `config` 属性便于外层脚本确认 loader 使用的配置来源

这些接口虽然不是评测层的绝对核心，但已经构成当前公共 API 的一部分，应补入文档。

**建议回填：**

在 `GroundTruthLoader` API 摘要中补充上述三个成员，并注明它们是对 `load()` 的便利包装与只读配置访问入口。

---

### 3.7 默认长度回退逻辑应在文档中明确写出

**文档旧表述：**

仅笼统写“加载 ground truth”，未说明 `lengths=None` 时的真实行为。

**实际实现：**

当 `lengths is None` 时，按如下顺序决定：

1. 读取 `config['data']['length']`
2. 若该配置存在，则使用该单长度或长度列表
3. 若该配置为空，则回退到：

```python
["K4", "K8", "K16", "K32", "K64", "K128"]
```

**偏差说明：**

这个默认回退策略会直接影响评测覆盖范围：

- 若配置文件将 `data.length` 设为某个单长度，`load()` 默认并不会扫全档位
- 若配置为空，才会回退到六个标准长度档位

如果文档不写清楚，测试与实验脚本可能误判默认评测范围。

**建议回填：**

在 `GroundTruthLoader.load()` 的参数说明中补充：

> `lengths=None` 时先读取 `config.data.length`；若仍为空，则使用 `['K4', 'K8', 'K16', 'K32', 'K64', 'K128']` 作为默认长度集合。

---

### 3.8 `compute_metrics_by_group()` 是当前实现中已存在的分组评测接口

**文档旧表述：** 未提及。

**实际实现：**

```python
def compute_metrics_by_group(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    group_fn,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    ...
```

**偏差说明：**

这说明当前评测层已经不只是“单表聚合器”，而是为 Phase 3 中“分子任务、分档位汇报”准备好了直接可用的分组接口。若文档不提及它，会低估当前评测层对实验汇总的支持程度。

**建议回填：**

将 `compute_metrics_by_group()` 一并纳入评测层 API 摘要，并说明：

> 用于按自定义分组维度（如子任务、长度档位）输出分组指标表，适合作为实验汇总层的基础工具。

---

## 4. 实现中良好实践的确认

以下几点与项目设计目标保持一致，值得在文档中明确保留为“确认项”：

### 4.1 原子指标函数已经彻底与数据集实现解耦

`recall_at_k()` / `precision_at_k()` / `mrr()` / `ndcg_at_k()` 当前只依赖：

- `list[str]`
- `set[str]`
- `int`

这很好地实现了文档中“标准化输入、与具体数据集解耦”的目标，应继续作为评测层的稳定边界保留。

### 4.2 Ground truth 层与适配器层职责���离是成立的

虽然 `GroundTruthLoader` 当前没有自己暴露 ID 构造 helper，但它通过 `DocumentQAAdapter.build_ground_truth()` 复用数据接入层的命名规则，避免重复实现解析逻辑。这一职责分层是合理的，不应因文档旧草案而被误判为“接口缺失”。

### 4.3 当前评测层已经具备 Phase 3 所需的基础能力

从真实实现看，评测层已经能够支持：

1. 单 query 列表输入的原子指标计算
2. 全量宏平均聚合 `compute_all_metrics()`
3. 分组聚合 `compute_metrics_by_group()`
4. Ground truth 的按子任务/按长度过滤加载

这说明 Step 2.4 的基础评测能力已经具备扩展到 Phase 3 的条件，文档中应以“已具备的接口能力”而非“草案”来描述它。

---

## 5. v7 收尾状态

完成本轮文档同步后，Step 2.4 的计划文档应达到以下状态：

1. **`metrics.py`** 的批量聚合输入类型与真实实现一致，明确为 `dict[str, list[str]]`。
2. **`compute_all_metrics()`** 的返回列说明与真实实现一致，包含 `n_queries`。
3. **`GroundTruthLoader.load()`** 的过滤接口与真实实现一致，明确使用 `subtasks` / `lengths` / `task_family`。
4. **`GroundTruthLoader`** 不再错误承诺 `build_page_id()` / `build_query_id()` 静态方法。
5. **评测层便利接口**（`load_by_subtask()`、`load_by_length()`、`compute_metrics_by_group()`）被正式写入接口摘要。
6. **默认长度回退策略** 被明确记录，避免评测范围理解再次出现偏差。

到这一步，`Project_Plan.md` 中关于 Step 2.4 的描述即可视为与当前实现基本对齐的参考基线。

**补充说明：**

本轮 v7 主要解决的是**计划文档**与评测层真实代码之间的偏差。后续若 Step 3 评测脚本或结果导出模块继续扩展出新的公共接口，可在 v8 或单独的评测层修订文档中继续补充，但不影响本轮对 Step 2.4 核心接口对齐的完成度。

