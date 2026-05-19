# ZeroShotVDR Phase 4 迭代式开发指导文档

本文档将 Phase 4 开发拆分为多个可独立完成、可测试、可回退的迭代阶段。每一阶段都包含：

```text
1. 本阶段目标
2. 需要修改的文件
3. 具体修改内容
4. 如何检验效果
5. 验收标准
6. 常见风险与排查方式
```

Phase 4 的核心目标是：在不破坏 Phase 3 稳定 baseline 的前提下，实现 **query-adaptive two-stage coarse-to-fine retrieval**。

核心检索流程应为：

```text
Query.candidate_page_ids
        ↓
作为 query-specific candidate universe
        ↓
mean-pool coarse retrieval
        ↓
adaptive top-N selection
        ↓
optional neighbor expansion
        ↓
full MaxSim rerank
        ↓
final top-k results
```

最关键的原则是：

```text
candidate_page_ids 不能被当作最终候选直接返回。
candidate_page_ids 应该被当作当前 query 的候选全集 universe。
```

---

# 0. Phase 4 开发总览

## 0.1 推荐迭代顺序

建议严格按照以下顺序开发：

```text
Stage 0: 建立 Phase 4 开发分支与 baseline 复现
Stage 1: 新增 Phase 4 配置与目录结构，不改变旧逻辑
Stage 2: 实现 page_id 解析与 neighbor 工具函数
Stage 3: 实现 adaptive top-N 工具函数
Stage 4: 实现 TwoStageRetriever 的 fixed top-N 最小版本
Stage 5: 接入 mean-pool cache，降低 coarse 阶段 I/O
Stage 6: 接入 adaptive top-N
Stage 7: 接入 neighbor expansion
Stage 8: 新增 run_phase4_eval.py 评测脚本
Stage 9: 增加 per-query trace 与 slice-level 分析
Stage 10: 运行完整消融实验与 Phase 4 报告
```

每一个 Stage 都应该能单独提交一次 commit，并且通过测试后再进入下一阶段。

---

# 1. Stage 0：建立开发分支与 Phase 3 baseline 复现

## 1.1 本阶段目标

在正式修改 Phase 4 代码之前，先确认当前仓库可以稳定复现 Phase 3 baseline。

这一阶段不做功能开发，只做环境确认、baseline 固化和结果备份。

## 1.2 需要修改的文件

本阶段原则上不修改源码。

建议新增或记录：

```text
outputs/eval_reports/phase3_baseline_recheck/
docs/dev_notes_phase4.md
```

可选新增：

```text
scripts/run_phase3_recheck.sh
```

## 1.3 具体操作

### 1.3.1 创建开发分支

```bash
git checkout main
git pull
git checkout -b phase4-two-stage-retrieval
```

### 1.3.2 运行现有测试

```bash
pytest tests/
```

### 1.3.3 运行 Phase 3 smoke eval

```bash
python scripts/run_step3_eval.py --smoke
```

如果仓库中没有 `--smoke` 参数，则使用当前已有的最小评测参数，例如限制 query 数量或使用小样本配置。

### 1.3.4 运行 Phase 3 valid-only 全量评测

推荐将结果保存到：

```text
outputs/eval_reports/phase3_baseline_recheck/
```

主结果必须基于：

```text
14,385 valid page-labeled queries
```

不要把 1,192 条无效 ground truth query 混入主表。

## 1.4 如何检验效果

检查以下内容：

```text
1. pytest 是否通过
2. smoke eval 是否正常完成
3. valid-only query 数量是否为 14,385
4. Phase 3 Recall@10 是否接近 0.8517
5. Phase 3 nDCG@10 是否接近 0.6325
6. 平均延迟是否接近 0.071s/query
7. P95 延迟是否接近 0.138s/query
```

## 1.5 验收标准

本阶段完成的验收标准：

```text
[ ] tests/ 全部通过，或已确认失败项与 Phase 4 无关
[ ] Phase 3 smoke eval 可以运行
[ ] valid-only query 数为 14,385
[ ] Recall@10 与 0.8517 的差距不超过合理浮动
[ ] nDCG@10 与 0.6325 的差距不超过合理浮动
[ ] baseline 结果已保存，后续 Phase 4 所有实验都能与其对比
```

建议合理浮动范围：

```text
Recall@10: ±0.003
nDCG@10: ±0.003
Latency: 允许因硬件、缓存、磁盘状态有更大浮动
```

## 1.6 常见风险与排查

### 风险 1：valid query 数量不是 14,385

排查：

```text
1. 检查是否错误包含 invalid ground truth
2. 检查是否过滤了没有 page label 的 query
3. 检查是否把 all-query 表当作主表
```

### 风险 2：baseline 指标明显偏离

排查：

```text
1. 检查 index 是否完整
2. 检查 page_id 构建方式是否稳定
3. 检查 candidate_page_ids 是否被正确使用
4. 检查是否使用了不同的模型 checkpoint 或配置
```

---

# 2. Stage 1：新增 Phase 4 配置与目录结构

## 2.1 本阶段目标

建立 Phase 4 的代码位置和配置入口，但不改变 Phase 3 默认行为。

这是一个“无行为变化”的重构阶段。

## 2.2 需要修改的文件

新增：

```text
src/zeroshot_vdr/advanced/__init__.py
src/zeroshot_vdr/advanced/two_stage.py
src/zeroshot_vdr/advanced/neighbors.py
src/zeroshot_vdr/advanced/profiling.py
```

修改：

```text
config/default.yaml
```

可选新增：

```text
docs/dev_notes_phase4.md
```

## 2.3 具体修改内容

### 2.3.1 新增目录

```bash
mkdir -p src/zeroshot_vdr/advanced
touch src/zeroshot_vdr/advanced/__init__.py
touch src/zeroshot_vdr/advanced/two_stage.py
touch src/zeroshot_vdr/advanced/neighbors.py
touch src/zeroshot_vdr/advanced/profiling.py
```

### 2.3.2 新增配置项

在 `config/default.yaml` 的 `retrieval` 下新增：

```yaml
retrieval:
  candidate_strategy: full
  score_batch_size: 32

  phase4:
    enabled: false
    method: fixed_topn

    min_candidates: 32
    max_candidates: 128
    coarse_top_n: 64
    base_ratio: 0.20
    flat_margin: 0.035

    neighbor_window: 0
    neighbor_seed_n: 8

    use_query_scope: true
    use_mean_pool_cache: false
    mean_pool_cache_dir: outputs/cache/mean_pool

    trace_enabled: false
```

注意：

```text
phase4.enabled 必须默认为 false。
```

这样可以保证 Phase 3 baseline 默认不受影响。

## 2.4 如何检验效果

运行：

```bash
pytest tests/
python scripts/run_step3_eval.py --smoke
```

检查：

```text
1. 是否存在 import error
2. default.yaml 是否能正常加载
3. Phase 3 smoke eval 结果是否与修改前一致
```

## 2.5 验收标准

```text
[ ] 新增 advanced/ 目录
[ ] 新增 phase4 配置项
[ ] phase4.enabled 默认为 false
[ ] 不修改 RetrievalPipeline 的默认行为
[ ] pytest 通过
[ ] Phase 3 smoke eval 结果不变
```

## 2.6 常见风险与排查

### 风险：新增配置导致旧代码解析失败

排查：

```text
1. 检查配置读取代码是否严格校验字段
2. 检查 YAML 缩进
3. 检查 retrieval.phase4 是否被错误读取为 retrieval 的默认策略
```

---

# 3. Stage 2：实现 page_id 解析与 neighbor 工具函数

## 3.1 本阶段目标

实现 Phase 4 后续需要的页面 ID 解析与邻页扩展能力。

本阶段只实现工具函数，不接入主检索流程。

## 3.2 需要修改的文件

修改：

```text
src/zeroshot_vdr/advanced/neighbors.py
```

新增测试：

```text
tests/test_phase4_neighbors.py
```

## 3.3 具体修改内容

### 3.3.1 实现 page_id 解析

当前稳定 page_id 格式应类似：

```text
{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}
```

例如：

```text
slidevqa/default_K128/doc_001/p15
```

推荐实现：

```python
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedPageId:
    prefix: str
    page_idx: int


_PAGE_ID_RE = re.compile(r"^(?P<prefix>.+)/p(?P<page_idx>\d+)$")


def parse_page_id(page_id: str) -> ParsedPageId:
    match = _PAGE_ID_RE.match(page_id)
    if match is None:
        raise ValueError(f"Invalid page_id format: {page_id}")

    return ParsedPageId(
        prefix=match.group("prefix"),
        page_idx=int(match.group("page_idx")),
    )


def make_page_id(prefix: str, page_idx: int) -> str:
    return f"{prefix}/p{page_idx}"
```

### 3.3.2 实现邻页扩展

```python
from typing import Iterable


def expand_neighbors(
    coarse_ids: list[str],
    universe_ids: Iterable[str],
    window: int = 1,
    seed_n: int = 8,
) -> list[str]:
    if window <= 0 or seed_n <= 0:
        return list(dict.fromkeys(coarse_ids))

    universe_set = set(universe_ids)
    output: list[str] = []
    seen: set[str] = set()

    def add(pid: str) -> None:
        if pid in universe_set and pid not in seen:
            output.append(pid)
            seen.add(pid)

    for pid in coarse_ids:
        add(pid)

    for pid in coarse_ids[:seed_n]:
        parsed = parse_page_id(pid)
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            neighbor_idx = parsed.page_idx + delta
            if neighbor_idx < 0:
                continue
            add(make_page_id(parsed.prefix, neighbor_idx))

    return output
```

## 3.4 如何检验效果

新增测试：

```python
import pytest

from zeroshot_vdr.advanced.neighbors import (
    parse_page_id,
    make_page_id,
    expand_neighbors,
)


def test_parse_page_id():
    parsed = parse_page_id("slidevqa/default_K128/doc_001/p15")
    assert parsed.prefix == "slidevqa/default_K128/doc_001"
    assert parsed.page_idx == 15


def test_make_page_id():
    assert make_page_id("a/b/c", 3) == "a/b/c/p3"


def test_expand_neighbors_within_universe():
    coarse = ["a/b/doc/p5"]
    universe = ["a/b/doc/p4", "a/b/doc/p5", "a/b/doc/p6"]
    assert expand_neighbors(coarse, universe, window=1, seed_n=1) == [
        "a/b/doc/p5",
        "a/b/doc/p4",
        "a/b/doc/p6",
    ]


def test_expand_neighbors_does_not_leave_universe():
    coarse = ["a/b/doc/p5"]
    universe = ["a/b/doc/p5"]
    assert expand_neighbors(coarse, universe, window=1, seed_n=1) == [
        "a/b/doc/p5"
    ]


def test_expand_neighbors_stable_dedup_order():
    coarse = ["a/b/doc/p5", "a/b/doc/p5", "a/b/doc/p6"]
    universe = ["a/b/doc/p4", "a/b/doc/p5", "a/b/doc/p6"]
    assert expand_neighbors(coarse, universe, window=1, seed_n=2) == [
        "a/b/doc/p5",
        "a/b/doc/p6",
        "a/b/doc/p4",
    ]
```

运行：

```bash
pytest tests/test_phase4_neighbors.py
pytest tests/
```

## 3.5 验收标准

```text
[ ] parse_page_id 能解析稳定 page_id
[ ] invalid page_id 会抛出 ValueError
[ ] expand_neighbors 保留 coarse 原始顺序
[ ] expand_neighbors 去重稳定
[ ] expand_neighbors 不会加入 universe 外页面
[ ] window=0 时不扩展邻页
[ ] pytest tests/test_phase4_neighbors.py 通过
[ ] pytest tests/ 通过
```

## 3.6 常见风险与排查

### 风险：page_id 格式不完全一致

排查：

```text
1. 从真实 index metadata 中抽样打印 page_id
2. 确认末尾是否一定是 /p{page_idx}
3. 如果 doc_id 中包含 /，当前正则仍然可用，因为 prefix 使用 .+ 贪婪匹配
```

---

# 4. Stage 3：实现 adaptive top-N 工具函数

## 4.1 本阶段目标

实现 query-adaptive coarse candidate 数量选择逻辑。

本阶段只实现函数和单元测试，不接入主检索流程。

## 4.2 需要修改的文件

修改：

```text
src/zeroshot_vdr/advanced/two_stage.py
```

新增测试：

```text
tests/test_phase4_adaptive.py
```

## 4.3 具体修改内容

在 `two_stage.py` 中新增：

```python
import torch


def choose_adaptive_top_n(
    scores: torch.Tensor,
    universe_size: int,
    min_n: int = 32,
    max_n: int = 128,
    base_ratio: float = 0.20,
    flat_margin: float = 0.035,
) -> int:
    if universe_size <= 0:
        return 0

    if universe_size <= min_n:
        return universe_size

    base_n = int(round(universe_size * base_ratio))
    base_n = max(min_n, min(base_n, max_n, universe_size))

    sorted_scores = torch.sort(scores.detach().float(), descending=True).values

    if base_n < universe_size:
        margin = float(sorted_scores[0] - sorted_scores[base_n - 1])
        if margin < flat_margin:
            base_n = min(base_n * 2, max_n, universe_size)

    return int(base_n)
```

## 4.4 如何检验效果

新增测试：

```python
import torch

from zeroshot_vdr.advanced.two_stage import choose_adaptive_top_n


def test_adaptive_top_n_empty():
    scores = torch.tensor([])
    assert choose_adaptive_top_n(scores, universe_size=0) == 0


def test_adaptive_top_n_small_universe():
    scores = torch.rand(10)
    assert choose_adaptive_top_n(scores, universe_size=10, min_n=32) == 10


def test_adaptive_top_n_respects_bounds():
    scores = torch.linspace(1.0, 0.0, steps=1000)
    n = choose_adaptive_top_n(
        scores,
        universe_size=1000,
        min_n=32,
        max_n=128,
        base_ratio=0.20,
    )
    assert 32 <= n <= 128


def test_adaptive_top_n_sharp_distribution_no_expand():
    scores = torch.cat([
        torch.tensor([1.0]),
        torch.linspace(0.5, 0.0, steps=127),
    ])
    n = choose_adaptive_top_n(
        scores,
        universe_size=128,
        min_n=32,
        max_n=128,
        base_ratio=0.25,
        flat_margin=0.035,
    )
    assert n == 32


def test_adaptive_top_n_flat_distribution_expand():
    scores = torch.linspace(1.0, 0.98, steps=128)
    n = choose_adaptive_top_n(
        scores,
        universe_size=128,
        min_n=32,
        max_n=128,
        base_ratio=0.25,
        flat_margin=0.035,
    )
    assert n == 64
```

运行：

```bash
pytest tests/test_phase4_adaptive.py
pytest tests/
```

## 4.5 验收标准

```text
[ ] universe_size <= min_n 时返回 universe_size
[ ] top-N 不小于 min_n
[ ] top-N 不大于 max_n
[ ] top-N 不大于 universe_size
[ ] 分数分布尖锐时不扩张
[ ] 分数分布平坦时扩张
[ ] pytest tests/test_phase4_adaptive.py 通过
[ ] pytest tests/ 通过
```

## 4.6 常见风险与排查

### 风险：flat_margin 不适合真实分数范围

排查：

```text
1. 在后续 trace 中记录 top1_score、topN_score、margin
2. 统计不同 task_family / length 下的 margin 分布
3. 根据真实分布调整 flat_margin
```

---

# 5. Stage 4：实现 TwoStageRetriever fixed top-N 最小版本

## 5.1 本阶段目标

实现 Phase 4 最小可用版本：

```text
candidate_page_ids 作为 universe
mean-pool coarse top-N
full MaxSim rerank
无 adaptive
无 neighbor
无 cache
```

这是 Phase 4 的第一个核心功能阶段。

## 5.2 需要修改的文件

修改：

```text
src/zeroshot_vdr/advanced/two_stage.py
```

新增测试：

```text
tests/test_phase4_two_stage.py
```

可选修改：

```text
src/zeroshot_vdr/retrieval/pipeline.py
```

但建议初期不要直接大改 `RetrievalPipeline`，而是通过组合方式复用它。

## 5.3 具体修改内容

### 5.3.1 新增 TwoStageResult dataclass

```python
from dataclasses import dataclass


@dataclass
class TwoStageTrace:
    query_id: str | None
    universe_size: int
    coarse_top_n: int
    expanded_candidate_count: int
    neighbor_added_count: int
    coarse_ms: float
    rerank_ms: float
    total_ms: float
    method: str


@dataclass
class TwoStageOutput:
    results: list
    trace: TwoStageTrace
```

如果仓库已有 `RetrievalResult` 类型，则 `results` 应改成更具体的类型。

### 5.3.2 实现 candidate universe 解析

```python
def resolve_candidate_universe(query, explicit_candidate_ids=None, index_store=None):
    if explicit_candidate_ids is not None:
        return list(explicit_candidate_ids)

    candidate_page_ids = getattr(query, "candidate_page_ids", None)
    if candidate_page_ids:
        return list(candidate_page_ids)

    if index_store is None:
        raise ValueError(
            "index_store is required when query.candidate_page_ids is missing"
        )

    return index_store.list_page_ids(
        doc_id=getattr(query, "doc_id", None),
        task_family=getattr(query, "task_family", None),
        subtask=getattr(query, "subtask", None),
        length=getattr(query, "length", None),
    )
```

### 5.3.3 实现 mean-pool coarse scoring

推荐复用：

```text
IndexStore.get_mean_pooled_view(page_ids)
```

粗筛逻辑：

```python
import torch
import torch.nn.functional as F


def mean_pool_query(query_emb: torch.Tensor) -> torch.Tensor:
    query_mean = query_emb.mean(dim=0)
    return F.normalize(query_mean, dim=-1)


def score_mean_pool(query_emb: torch.Tensor, page_means: torch.Tensor) -> torch.Tensor:
    query_mean = mean_pool_query(query_emb)
    page_means = F.normalize(page_means, dim=-1)
    return page_means @ query_mean
```

### 5.3.4 实现 fixed top-N coarse_select

```python
def select_topn_by_scores(
    page_ids: list[str],
    scores: torch.Tensor,
    top_n: int,
) -> list[str]:
    if len(page_ids) == 0:
        return []

    top_n = min(top_n, len(page_ids))
    top_indices = torch.topk(scores, k=top_n).indices.tolist()
    return [page_ids[i] for i in top_indices]
```

### 5.3.5 实现 TwoStageRetriever.retrieve()

伪代码：

```python
import time


class TwoStageRetriever:
    def __init__(
        self,
        base_pipeline,
        index_store,
        coarse_top_n: int = 64,
        method: str = "fixed_topn",
    ):
        self.pipeline = base_pipeline
        self.index_store = index_store
        self.coarse_top_n = coarse_top_n
        self.method = method

    def retrieve(self, query, top_k: int = 10, candidate_ids=None):
        t0 = time.perf_counter()

        query_emb = self.pipeline.encode_query(query.text)

        universe_ids = resolve_candidate_universe(
            query=query,
            explicit_candidate_ids=candidate_ids,
            index_store=self.index_store,
        )

        coarse_start = time.perf_counter()
        page_means = self.index_store.get_mean_pooled_view(universe_ids)
        coarse_scores = score_mean_pool(query_emb, page_means)
        coarse_ids = select_topn_by_scores(
            page_ids=universe_ids,
            scores=coarse_scores,
            top_n=self.coarse_top_n,
        )
        coarse_ms = (time.perf_counter() - coarse_start) * 1000

        rerank_start = time.perf_counter()
        results = self.pipeline.retrieve(
            query=query,
            top_k=top_k,
            candidate_page_ids=coarse_ids,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        trace = TwoStageTrace(
            query_id=getattr(query, "query_id", None),
            universe_size=len(universe_ids),
            coarse_top_n=len(coarse_ids),
            expanded_candidate_count=len(coarse_ids),
            neighbor_added_count=0,
            coarse_ms=coarse_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
            method=self.method,
        )

        return TwoStageOutput(results=results, trace=trace)
```

注意：

如果 `pipeline.retrieve()` 不支持传入 `candidate_page_ids` 参数，应改用现有的：

```python
scores, page_ids = pipeline.score_candidates(
    query_emb=query_emb,
    candidate_page_ids=coarse_ids,
)
```

然后复用或实现结果组装逻辑。

## 5.4 如何检验效果

### 5.4.1 单元测试

测试目标：

```text
1. candidate_page_ids 被当作 universe，而不是直接作为最终候选
2. coarse top-N 后候选数量小于 universe
3. rerank 阶段只收到 coarse_ids
```

可用 fake pipeline / fake index_store 测试，不必加载真实模型。

示例测试思路：

```python
class FakeIndexStore:
    def get_mean_pooled_view(self, page_ids):
        # 返回可控 embedding
        ...


class FakePipeline:
    def encode_query(self, text):
        ...

    def score_candidates(self, query_emb, candidate_page_ids):
        self.last_candidate_page_ids = candidate_page_ids
        ...
```

运行：

```bash
pytest tests/test_phase4_two_stage.py
pytest tests/
```

### 5.4.2 Smoke eval

新增临时脚本或 notebook，选取少量 query：

```bash
python scripts/run_phase4_eval.py   --method fixed_topn   --coarse-top-n 64   --max-queries 100   --valid-only
```

如果此时还没有 `run_phase4_eval.py`，可以写一个临时调试脚本：

```text
scripts/debug_phase4_fixed_topn.py
```

## 5.5 验收标准

```text
[ ] TwoStageRetriever 可以初始化
[ ] retrieve() 可以跑通单条 query
[ ] candidate_page_ids 被解析为 universe
[ ] coarse_ids 数量 <= coarse_top_n
[ ] coarse_ids 数量 <= universe_size
[ ] full MaxSim rerank 只处理 coarse_ids
[ ] trace 中包含 universe_size、coarse_top_n、rerank_ms、total_ms
[ ] 单元测试通过
[ ] 100 条 query smoke eval 可以跑通
```

质量指标的临时验收建议：

```text
fixed top-64 在 100 条 query smoke eval 中不应出现大量空结果或异常。
Recall@10 可以低于 baseline，但不能完全失效。
```

## 5.6 常见风险与排查

### 风险 1：`candidate_page_ids` 仍然被原 pipeline 直接返回

这是 Phase 4 最核心风险。

排查方式：

```text
1. 在 fake pipeline 测试中记录 rerank 输入
2. 确认 rerank 输入是 coarse_ids，而不是完整 universe_ids
3. trace 中检查 expanded_candidate_count 是否小于 universe_size
```

### 风险 2：mean-pooled view 维度不匹配

排查：

```text
1. 打印 query_emb.shape
2. 打印 page_means.shape
3. 确认 query_emb 的最后一维等于 page_means 的最后一维
```

---

# 6. Stage 5：接入 MeanPoolCache

## 6.1 本阶段目标

减少每个 query 重复从磁盘读取 patch embedding 并 mean pooling 的开销。

本阶段不改变检索算法，只优化 coarse 阶段效率。

## 6.2 需要修改的文件

新增或修改：

```text
src/zeroshot_vdr/advanced/two_stage.py
src/zeroshot_vdr/advanced/mean_pool_cache.py
```

如果不想新增文件，也可以先把 cache 类放在 `two_stage.py`，稳定后再拆分。

新增测试：

```text
tests/test_phase4_mean_pool_cache.py
```

## 6.3 具体修改内容

### 6.3.1 新增 cache 类

```python
from pathlib import Path
import json
import torch


class MeanPoolCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.page_ids_path = self.cache_dir / "page_ids.json"
        self.embeddings_path = self.cache_dir / "page_means.pt"
        self.meta_path = self.cache_dir / "meta.json"

        self.page_ids: list[str] | None = None
        self.embeddings: torch.Tensor | None = None
        self.id_to_idx: dict[str, int] | None = None

    def exists(self) -> bool:
        return (
            self.page_ids_path.exists()
            and self.embeddings_path.exists()
            and self.meta_path.exists()
        )

    def load(self, map_location="cpu") -> None:
        self.page_ids = json.loads(self.page_ids_path.read_text(encoding="utf-8"))
        self.embeddings = torch.load(self.embeddings_path, map_location=map_location)
        self.id_to_idx = {pid: i for i, pid in enumerate(self.page_ids)}

    def save(self, page_ids: list[str], embeddings: torch.Tensor, meta: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.page_ids_path.write_text(
            json.dumps(page_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        torch.save(embeddings.cpu(), self.embeddings_path)
        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, page_ids: list[str]) -> torch.Tensor:
        if self.embeddings is None or self.id_to_idx is None:
            raise RuntimeError("MeanPoolCache is not loaded")

        indices = [self.id_to_idx[pid] for pid in page_ids]
        return self.embeddings[indices]
```

### 6.3.2 构建 cache

新增方法：

```python
def build_mean_pool_cache(index_store, page_ids: list[str], cache: MeanPoolCache) -> None:
    embeddings = index_store.get_mean_pooled_view(page_ids)
    meta = {
        "num_pages": len(page_ids),
        "embedding_dim": int(embeddings.shape[-1]),
    }
    cache.save(page_ids, embeddings, meta)
```

### 6.3.3 TwoStageRetriever 接入 cache

初始化参数新增：

```python
use_mean_pool_cache: bool = False
mean_pool_cache_dir: str | None = None
```

coarse 阶段改为：

```python
if self.mean_pool_cache is not None:
    page_means = self.mean_pool_cache.get(universe_ids)
else:
    page_means = self.index_store.get_mean_pooled_view(universe_ids)
```

## 6.4 如何检验效果

### 6.4.1 单元测试

测试：

```text
1. cache save 后文件存在
2. cache load 后 page_ids 数量一致
3. cache.get(page_ids) 顺序与输入 page_ids 一致
4. cache 缺失 page_id 时抛出 KeyError
```

运行：

```bash
pytest tests/test_phase4_mean_pool_cache.py
pytest tests/
```

### 6.4.2 效率对比

对同一批 query 分别运行：

```bash
python scripts/run_phase4_eval.py   --method fixed_topn   --coarse-top-n 64   --max-queries 500   --valid-only   --use-mean-pool-cache false
```

和：

```bash
python scripts/run_phase4_eval.py   --method fixed_topn   --coarse-top-n 64   --max-queries 500   --valid-only   --use-mean-pool-cache true
```

比较：

```text
coarse_ms
total_ms
Avg latency
P95 latency
```

## 6.5 验收标准

```text
[ ] cache 可以保存和加载
[ ] cache.get() 保持 page_ids 输入顺序
[ ] cache 版本的检索结果与无 cache 版本一致
[ ] cache 版本 coarse_ms 明显下降
[ ] cache 版本不改变 Recall@10 / nDCG@10
[ ] cache 文件大小显著小于原始 88.6GB patch index
```

建议效率验收：

```text
500 条 query 上，cache 版本的平均 coarse_ms 至少下降 30%。
```

如果磁盘缓存、OS page cache 或数据规模导致下降不明显，应记录原因，但仍可进入下一阶段。

## 6.6 常见风险与排查

### 风险：cache 中 page_id 与 index_store 不一致

排查：

```text
1. 在 meta.json 中记录 index_dir、created_at、num_pages
2. 构建 cache 时保存 page_id checksum
3. 加载 cache 后检查 query universe 中 page_id 是否全部存在
```

---

# 7. Stage 6：接入 adaptive top-N

## 7.1 本阶段目标

将 Stage 3 实现的 `choose_adaptive_top_n()` 接入 TwoStageRetriever。

此阶段算法从 fixed top-N 升级为 adaptive top-N。

## 7.2 需要修改的文件

修改：

```text
src/zeroshot_vdr/advanced/two_stage.py
config/default.yaml
```

新增或扩展测试：

```text
tests/test_phase4_two_stage.py
tests/test_phase4_adaptive.py
```

## 7.3 具体修改内容

### 7.3.1 TwoStageRetriever 增加参数

```python
class TwoStageRetriever:
    def __init__(
        self,
        base_pipeline,
        index_store,
        method: str = "fixed_topn",
        coarse_top_n: int = 64,
        min_candidates: int = 32,
        max_candidates: int = 128,
        base_ratio: float = 0.20,
        flat_margin: float = 0.035,
        ...
    ):
        ...
```

### 7.3.2 根据 method 决定 top-N

```python
if self.method == "fixed_topn":
    selected_n = self.coarse_top_n
elif self.method == "adaptive":
    selected_n = choose_adaptive_top_n(
        scores=coarse_scores,
        universe_size=len(universe_ids),
        min_n=self.min_candidates,
        max_n=self.max_candidates,
        base_ratio=self.base_ratio,
        flat_margin=self.flat_margin,
    )
else:
    raise ValueError(f"Unsupported phase4 method: {self.method}")
```

### 7.3.3 trace 增加 adaptive 信息

建议 trace 增加：

```text
top1_coarse_score
topn_coarse_score
coarse_margin
adaptive_expanded
```

示例：

```python
trace.top1_coarse_score = float(sorted_scores[0])
trace.topn_coarse_score = float(sorted_scores[selected_n - 1])
trace.coarse_margin = trace.top1_coarse_score - trace.topn_coarse_score
trace.adaptive_expanded = selected_n > base_n
```

如果 dataclass 不方便动态加字段，则提前定义好。

## 7.4 如何检验效果

### 7.4.1 单元测试

新增测试：

```text
1. method=fixed_topn 时不调用 adaptive
2. method=adaptive 时 coarse_top_n 根据分数变化
3. adaptive 结果不超过 max_candidates
4. adaptive 结果不超过 universe_size
```

运行：

```bash
pytest tests/test_phase4_adaptive.py
pytest tests/test_phase4_two_stage.py
pytest tests/
```

### 7.4.2 小规模实验

运行：

```bash
python scripts/run_phase4_eval.py   --method fixed_topn   --coarse-top-n 64   --max-queries 500   --valid-only
```

再运行：

```bash
python scripts/run_phase4_eval.py   --method adaptive   --min-candidates 32   --max-candidates 128   --base-ratio 0.20   --flat-margin 0.035   --max-queries 500   --valid-only
```

比较：

```text
Recall@10
nDCG@10
Avg rerank candidates
Avg latency
P95 latency
```

## 7.5 验收标准

```text
[ ] method=adaptive 可以正常运行
[ ] adaptive top-N 不小于 min_candidates
[ ] adaptive top-N 不大于 max_candidates
[ ] adaptive top-N 不大于 universe_size
[ ] trace 中记录 selected_n / margin
[ ] 500 query 小规模实验可以跑通
[ ] adaptive 的 Avg rerank candidates 低于 full universe
[ ] adaptive 的 Recall@10 不出现灾难性下降
```

建议质量验收：

```text
500 query smoke：
Recall@10 相比 Phase 3 full baseline 下降不超过 0.03
nDCG@10 相比 Phase 3 full baseline 下降不超过 0.03
```

注意：小样本波动较大，不能作为最终结论，但可以作为进入下一阶段的门槛。

## 7.6 常见风险与排查

### 风险：adaptive 选得太少，Recall 大幅下降

处理：

```text
1. 提高 min_candidates，例如从 32 提到 64
2. 提高 base_ratio，例如从 0.20 提到 0.30
3. 提高 max_candidates，例如从 128 提到 256
4. 增大 flat_margin，使更多 query 触发扩张
```

### 风险：adaptive 几乎总是选 max_candidates

处理：

```text
1. 降低 flat_margin
2. 降低 base_ratio
3. 按 task_family / length 分析 margin 分布
```

---

# 8. Stage 7：接入 neighbor expansion

## 8.1 本阶段目标

在 coarse top-N 之后加入邻页扩展，提高跨页、多页和邻页混淆样本的鲁棒性。

本阶段从：

```text
mean-pool top-N → MaxSim rerank
```

升级为：

```text
mean-pool top-N → neighbor expansion → MaxSim rerank
```

## 8.2 需要修改的文件

修改：

```text
src/zeroshot_vdr/advanced/two_stage.py
src/zeroshot_vdr/advanced/neighbors.py
config/default.yaml
```

扩展测试：

```text
tests/test_phase4_neighbors.py
tests/test_phase4_two_stage.py
```

## 8.3 具体修改内容

### 8.3.1 TwoStageRetriever 增加 neighbor 参数

```python
neighbor_window: int = 0
neighbor_seed_n: int = 8
```

### 8.3.2 coarse 后扩展

```python
from zeroshot_vdr.advanced.neighbors import expand_neighbors


if self.neighbor_window > 0:
    expanded_ids = expand_neighbors(
        coarse_ids=coarse_ids,
        universe_ids=universe_ids,
        window=self.neighbor_window,
        seed_n=self.neighbor_seed_n,
    )
else:
    expanded_ids = coarse_ids

neighbor_added_count = len(expanded_ids) - len(coarse_ids)
```

### 8.3.3 rerank 使用 expanded_ids

必须确认：

```python
scores, page_ids = self.pipeline.score_candidates(
    query_emb=query_emb,
    candidate_page_ids=expanded_ids,
)
```

或：

```python
results = self.pipeline.retrieve(
    query=query,
    top_k=top_k,
    candidate_page_ids=expanded_ids,
)
```

注意：

```text
rerank 的输入必须是 expanded_ids，而不是 coarse_ids，也不是 universe_ids。
```

## 8.4 如何检验效果

### 8.4.1 单元测试

测试：

```text
1. neighbor_window=0 时 expanded_ids == coarse_ids
2. neighbor_window=1 时加入 p-1 / p+1
3. neighbor expansion 不跨出 universe
4. rerank 输入为 expanded_ids
5. trace 中 neighbor_added_count 正确
```

运行：

```bash
pytest tests/test_phase4_neighbors.py
pytest tests/test_phase4_two_stage.py
pytest tests/
```

### 8.4.2 小规模实验

运行：

```bash
python scripts/run_phase4_eval.py   --method adaptive   --max-queries 500   --valid-only
```

对比：

```bash
python scripts/run_phase4_eval.py   --method adaptive_neighbors   --neighbor-window 1   --neighbor-seed-n 8   --max-queries 500   --valid-only
```

重点比较：

```text
Recall@10
nDCG@10
Avg rerank candidates
neighbor_added_count
slidevqa/K128
mmlongdoc/K128
longdocurl/K128
```

## 8.5 验收标准

```text
[ ] neighbor_window=0 时行为等价于无 neighbor
[ ] neighbor_window=1 时只加入 universe 内邻页
[ ] neighbor expansion 去重且顺序稳定
[ ] rerank 输入为 expanded_ids
[ ] trace 正确记录 neighbor_added_count
[ ] adaptive_neighbors 小规模评测可以跑通
[ ] Avg rerank candidates 增加幅度可控
[ ] Recall@10 或 nDCG@10 不低于 adaptive no-neighbor，或下降极小但有明确效率收益
```

建议验收阈值：

```text
neighbor 后 Avg rerank candidates 相比 no-neighbor 增加不超过 25%
Recall@10 不低于 no-neighbor - 0.005
nDCG@10 不低于 no-neighbor - 0.005
```

如果 neighbor 对部分 slice 有提升，但整体略降，可以保留为可选配置，不作为默认最终方法。

## 8.6 常见风险与排查

### 风险：neighbor 加入无关页面导致 nDCG 下降

处理：

```text
1. 降低 neighbor_seed_n，例如从 8 降到 4
2. 保持 window=1，不要直接用 window=2
3. 只对长候选集合 K64/K128 开启 neighbor
4. 只对 coarse rank 靠前页面开启 neighbor
```

### 风险：neighbor_added_count 为 0

排查：

```text
1. 检查 page_id 解析是否正确
2. 检查 universe 中是否真的包含相邻页
3. 检查 page_idx 是否从 0 或 1 开始；通常不影响，只要 universe 匹配
```

---

# 9. Stage 8：新增 run_phase4_eval.py 评测脚本

## 9.1 本阶段目标

提供一个专门用于 Phase 4 的评测入口，避免破坏 Phase 3 的稳定脚本。

## 9.2 需要修改的文件

新增：

```text
scripts/run_phase4_eval.py
```

可选新增：

```text
scripts/run_phase4_ablation.sh
```

## 9.3 具体修改内容

### 9.3.1 CLI 参数设计

建议支持：

```bash
python scripts/run_phase4_eval.py   --run-name phase4_adaptive_neighbors   --method adaptive_neighbors   --coarse-top-n 64   --min-candidates 32   --max-candidates 128   --base-ratio 0.20   --flat-margin 0.035   --neighbor-window 1   --neighbor-seed-n 8   --valid-only   --max-queries 500   --use-mean-pool-cache   --trace-enabled
```

核心参数：

```text
--method:
  fixed_topn
  adaptive
  adaptive_neighbors

--valid-only:
  只评测 valid page-labeled queries

--max-queries:
  小规模调试用

--trace-enabled:
  输出 per-query trace
```

### 9.3.2 脚本内部流程

```text
1. 读取配置
2. 加载 dataset / queries
3. 过滤 valid-only queries
4. 加载 index_store
5. 加载 base RetrievalPipeline
6. 初始化 TwoStageRetriever
7. 对每个 query 执行 retrieve
8. 收集 metrics
9. 保存 summary.json
10. 保存 metrics.csv
11. 可选保存 phase4_trace.jsonl
```

### 9.3.3 输出目录

建议：

```text
outputs/eval_reports/{run_name}/
  summary.json
  metrics.csv
  slice_metrics.csv
  phase4_trace.jsonl
  config_used.yaml
```

## 9.4 如何检验效果

### 9.4.1 参数解析测试

运行：

```bash
python scripts/run_phase4_eval.py --help
```

应能看到所有 Phase 4 参数。

### 9.4.2 最小运行

```bash
python scripts/run_phase4_eval.py   --run-name debug_phase4_10q   --method fixed_topn   --coarse-top-n 64   --max-queries 10   --valid-only   --trace-enabled
```

检查输出：

```text
outputs/eval_reports/debug_phase4_10q/
  summary.json
  metrics.csv
  phase4_trace.jsonl
```

### 9.4.3 三种方法都能运行

```bash
python scripts/run_phase4_eval.py --method fixed_topn --max-queries 50 --valid-only
python scripts/run_phase4_eval.py --method adaptive --max-queries 50 --valid-only
python scripts/run_phase4_eval.py --method adaptive_neighbors --max-queries 50 --valid-only
```

## 9.5 验收标准

```text
[ ] run_phase4_eval.py --help 正常显示
[ ] fixed_topn 可运行
[ ] adaptive 可运行
[ ] adaptive_neighbors 可运行
[ ] --valid-only 生效
[ ] --max-queries 生效
[ ] 输出 summary.json
[ ] 输出 metrics.csv
[ ] trace-enabled 时输出 phase4_trace.jsonl
[ ] 不修改 run_step3_eval.py 的默认行为
```

## 9.6 常见风险与排查

### 风险：Phase 4 脚本和 Phase 3 脚本的数据加载逻辑不一致

处理：

```text
1. 尽量复用 run_step3_eval.py 中的数据加载函数
2. 如果必须复制逻辑，需要逐项对比 query 数量、valid 数量、candidate_page_ids 数量
3. 在 summary.json 中记录 valid query count
```

---

# 10. Stage 9：增加 trace 与 slice-level 分析

## 10.1 本阶段目标

让 Phase 4 的结果不仅能看全局指标，还能解释：

```text
1. 每条 query 选了多少 coarse candidate
2. neighbor 加了多少页面
3. coarse 阶段花了多久
4. rerank 阶段花了多久
5. 哪些 task_family / length 改善
6. 哪些 slice 变差
```

## 10.2 需要修改的文件

修改：

```text
src/zeroshot_vdr/advanced/profiling.py
src/zeroshot_vdr/advanced/two_stage.py
scripts/run_phase4_eval.py
```

可选新增：

```text
scripts/analyze_phase4_trace.py
```

## 10.3 具体修改内容

### 10.3.1 trace jsonl 字段

每条 query 输出一行：

```json
{
  "query_id": "...",
  "task_family": "...",
  "subtask": "...",
  "length": "K128",
  "method": "adaptive_neighbors",
  "universe_size": 128,
  "coarse_top_n": 64,
  "expanded_candidate_count": 70,
  "neighbor_added_count": 6,
  "coarse_ms": 1.8,
  "rerank_ms": 41.2,
  "total_ms": 43.6,
  "top1_coarse_score": 0.42,
  "topn_coarse_score": 0.38,
  "coarse_margin": 0.04,
  "hit_at_1": false,
  "hit_at_5": true,
  "hit_at_10": true,
  "gt_page_ids": ["..."],
  "pred_page_ids": ["...", "..."]
}
```

### 10.3.2 slice-level 分组

至少按以下维度分组：

```text
task_family
task_family + length
subtask + length
universe_size bucket
```

universe_size bucket 推荐：

```text
K8
K16
K32
K64
K128
other
```

### 10.3.3 输出 slice_metrics.csv

字段：

```text
method
slice_name
num_queries
Recall@1
Recall@5
Recall@10
MRR
nDCG@10
Avg latency
P95 latency
Avg universe size
Avg rerank candidates
Avg neighbor added
```

## 10.4 如何检验效果

运行：

```bash
python scripts/run_phase4_eval.py   --run-name debug_trace_100q   --method adaptive_neighbors   --neighbor-window 1   --neighbor-seed-n 8   --max-queries 100   --valid-only   --trace-enabled
```

检查：

```bash
head outputs/eval_reports/debug_trace_100q/phase4_trace.jsonl
```

检查 JSON 是否能解析：

```bash
python -m json.tool outputs/eval_reports/debug_trace_100q/summary.json
```

如果新增 `analyze_phase4_trace.py`：

```bash
python scripts/analyze_phase4_trace.py   --trace outputs/eval_reports/debug_trace_100q/phase4_trace.jsonl   --out outputs/eval_reports/debug_trace_100q/slice_metrics.csv
```

## 10.5 验收标准

```text
[ ] phase4_trace.jsonl 每行是合法 JSON
[ ] trace 行数等于实际评测 query 数
[ ] 每行包含 query_id、method、universe_size、coarse_top_n、expanded_candidate_count
[ ] 每行包含 coarse_ms、rerank_ms、total_ms
[ ] 每行包含 hit_at_10
[ ] slice_metrics.csv 可以生成
[ ] slice_metrics.csv 至少包含 task_family + length 分组
[ ] 能定位 slidevqa/K128、mmlongdoc/K128、longdocurl/K128 的结果
```

## 10.6 常见风险与排查

### 风险：trace 中 hit_at_10 与 summary 指标不一致

排查：

```text
1. 检查 ground truth page_id 格式是否一致
2. 检查 pred_page_ids 是否是最终排序后的 top-k
3. 检查多 ground truth 页面时 hit 判断是否使用 any-match
4. 检查 valid-only 过滤是否一致
```

---

# 11. Stage 10：完整消融实验与 Phase 4 验收

## 11.1 本阶段目标

完成 Phase 4 的正式实验，比较：

```text
1. Phase 3 Full MaxSim baseline
2. Fixed Top-32 + MaxSim
3. Fixed Top-64 + MaxSim
4. Fixed Top-128 + MaxSim
5. Adaptive + MaxSim
6. Adaptive + Neighbor + MaxSim
```

并输出最终 Phase 4 报告。

## 11.2 需要修改的文件

新增：

```text
docs/Milestone_Report_Phase4.md
```

可选新增：

```text
scripts/run_phase4_ablation.sh
```

## 11.3 具体实验命令

### 11.3.1 Fixed top-N

```bash
python scripts/run_phase4_eval.py   --run-name phase4_fixed_top32   --method fixed_topn   --coarse-top-n 32   --valid-only   --trace-enabled
```

```bash
python scripts/run_phase4_eval.py   --run-name phase4_fixed_top64   --method fixed_topn   --coarse-top-n 64   --valid-only   --trace-enabled
```

```bash
python scripts/run_phase4_eval.py   --run-name phase4_fixed_top128   --method fixed_topn   --coarse-top-n 128   --valid-only   --trace-enabled
```

### 11.3.2 Adaptive

```bash
python scripts/run_phase4_eval.py   --run-name phase4_adaptive   --method adaptive   --min-candidates 32   --max-candidates 128   --base-ratio 0.20   --flat-margin 0.035   --valid-only   --trace-enabled
```

### 11.3.3 Adaptive + Neighbor

```bash
python scripts/run_phase4_eval.py   --run-name phase4_adaptive_neighbors   --method adaptive_neighbors   --min-candidates 32   --max-candidates 128   --base-ratio 0.20   --flat-margin 0.035   --neighbor-window 1   --neighbor-seed-n 8   --valid-only   --trace-enabled
```

## 11.4 主结果表

最终报告中至少包含：

| Method | Valid Queries | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Avg Latency | P95 Latency | Avg Rerank Candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 3 Full MaxSim | 14,385 | - | - | 0.8517 | - | 0.6325 | 0.071s | 0.138s | full universe |
| Fixed Top-32 + MaxSim | 14,385 | - | - | - | - | - | - | - | 32 |
| Fixed Top-64 + MaxSim | 14,385 | - | - | - | - | - | - | - | 64 |
| Fixed Top-128 + MaxSim | 14,385 | - | - | - | - | - | - | - | 128 |
| Adaptive + MaxSim | 14,385 | - | - | - | - | - | - | - | - |
| Adaptive + Neighbor + MaxSim | 14,385 | - | - | - | - | - | - | - | - |

## 11.5 Slice 结果表

重点单独报告：

```text
slidevqa/K32
slidevqa/K64
slidevqa/K128
mmlongdoc/K128
longdocurl/K128
```

每个 slice 至少报告：

```text
Recall@10
nDCG@10
Avg latency
P95 latency
Avg rerank candidates
```

## 11.6 最终验收标准

### 11.6.1 工程验收

```text
[ ] Phase 3 baseline 默认配置不变
[ ] Phase 4 所有方法通过 run_phase4_eval.py 运行
[ ] valid-only 主表 query 数为 14,385
[ ] fixed_topn / adaptive / adaptive_neighbors 都有结果
[ ] 每个实验都有 summary.json、metrics.csv、slice_metrics.csv
[ ] trace-enabled 时有 phase4_trace.jsonl
[ ] tests/ 全部通过
```

### 11.6.2 算法质量验收

以 Phase 3 baseline 为参照：

```text
Phase 3 baseline:
Recall@10 = 0.8517
nDCG@10 = 0.6325
Avg latency ≈ 0.071s/query
P95 latency ≈ 0.138s/query
```

硬性质量标准：

```text
[ ] 最终推荐方法 Recall@10 不低于 0.8517 - 0.005
[ ] 最终推荐方法 nDCG@10 不低于 0.6325 - 0.005
```

理想质量标准：

```text
[ ] Recall@10 或 nDCG@10 至少一项高于 Phase 3 baseline
[ ] slidevqa/K128、mmlongdoc/K128、longdocurl/K128 至少部分 slice 有改善
```

### 11.6.3 效率验收

```text
[ ] Avg rerank candidates 明显低于 full universe
[ ] 长候选集合 K64/K128 的 P95 latency 下降
[ ] mean-pool cache 大小显著小于原始 patch index
```

建议效率目标：

```text
Avg rerank candidates 降低 30% 以上。
K128 slice P95 latency 下降 20% 以上。
```

如果质量基本持平但效率明显提升，可以作为 Phase 4 的主要贡献。

如果质量提升但效率下降，需要解释 neighbor expansion 或 cache miss 的影响。

## 11.7 失败时如何调整

### 情况 1：Fixed Top-32 明显掉点

说明 coarse top-N 太小。

调整：

```text
1. 增加 coarse_top_n 到 64 或 128
2. adaptive 的 min_candidates 从 32 提到 64
```

### 情况 2：Fixed Top-128 仍明显掉点

说明 mean-pool coarse retrieval 本身漏召回严重。

调整：

```text
1. 检查 mean_pool_query 是否归一化
2. 检查 page_means 是否归一化
3. 检查 page_id 与 embedding 对齐
4. 尝试 query token max/mean 混合表示
5. 加 neighbor expansion
```

### 情况 3：Adaptive 质量差于 Fixed Top-64

说明 adaptive 策略过于激进。

调整：

```text
1. 提高 min_candidates
2. 提高 base_ratio
3. 提高 flat_margin
4. 降低扩张触发门槛的依赖，先保守选更多
```

### 情况 4：Neighbor 降低 nDCG

说明邻页引入噪声。

调整：

```text
1. neighbor_seed_n 从 8 降到 4
2. 只对 K64/K128 开启 neighbor
3. window 保持 1，不使用 2
4. 只对 coarse margin 较小的 query 开启 neighbor
```

### 情况 5：Latency 没有下降

排查：

```text
1. 是否仍在 rerank full universe
2. trace 中 expanded_candidate_count 是否小于 universe_size
3. mean-pool cache 是否生效
4. 是否每条 query 都重复读磁盘
5. MaxSim batch size 是否过小
```

---

# 12. 推荐 Commit 顺序

建议每个 Stage 一个或多个小 commit：

```text
commit 1: add phase4 config and advanced package
commit 2: add page_id neighbor utilities with tests
commit 3: add adaptive top-n utility with tests
commit 4: add fixed-topN TwoStageRetriever
commit 5: add phase4 fixed-topN smoke path
commit 6: add mean-pool cache
commit 7: integrate adaptive top-N
commit 8: integrate neighbor expansion
commit 9: add run_phase4_eval.py
commit 10: add trace logging and slice metrics
commit 11: add ablation script
commit 12: add Phase 4 report
```

每个 commit 后至少运行：

```bash
pytest tests/
```

关键 commit 后运行：

```bash
python scripts/run_phase4_eval.py --max-queries 50 --valid-only
```

---

# 13. 最终文档交付清单

Phase 4 完成后，仓库中应至少包含：

```text
src/zeroshot_vdr/advanced/
  __init__.py
  two_stage.py
  neighbors.py
  profiling.py
  mean_pool_cache.py

scripts/
  run_phase4_eval.py
  analyze_phase4_trace.py          # 可选
  run_phase4_ablation.sh           # 可选

tests/
  test_phase4_neighbors.py
  test_phase4_adaptive.py
  test_phase4_two_stage.py
  test_phase4_mean_pool_cache.py   # 如果实现 cache

docs/
  Milestone_Report_Phase4.md
```

---

# 14. 总结

Phase 4 应以“逐步迭代、每步可验证”的方式推进。

最小闭环是：

```text
Stage 1: 配置和目录
Stage 2: neighbor 工具
Stage 3: adaptive 工具
Stage 4: fixed top-N two-stage retriever
Stage 8: run_phase4_eval.py
```

完整闭环是：

```text
fixed top-N
    ↓
mean-pool cache
    ↓
adaptive top-N
    ↓
neighbor expansion
    ↓
trace + slice analysis
    ↓
full valid-only ablation
```

最重要的验收点始终是：

```text
candidate_page_ids 是否被作为 query-specific universe？
rerank 阶段是否真的只处理 coarse / expanded candidates？
valid-only 主表是否严格使用 14,385 条 query？
Phase 3 baseline 是否没有被破坏？
```

只要这四点成立，Phase 4 的开发就能在工程上保持可控，在实验上保持可解释，并且能清楚展示 two-stage retrieval 相比 Phase 3 full MaxSim 的质量与效率 trade-off。
