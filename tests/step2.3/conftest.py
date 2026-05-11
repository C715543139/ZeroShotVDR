"""
Step 2.3 测试的共享夹具。

设计原则：
- 仅依据 docs/Project_Plan.md 中 Step 2.3 的公开契约组织测试。
- 不依赖 retrieval 模块的具体实现细节。
- 使用轻量 mock / stub，确保测试聚焦于接口行为。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zeroshot_vdr.contracts import Query
from zeroshot_vdr.indexing.store import IndexStore
from zeroshot_vdr.retrieval.encoder import QueryEncoder
from zeroshot_vdr.retrieval.pipeline import RetrievalPipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIM = 4
_QUERY_TOKENS = 2
_MODEL_TOKENS = 6
_MODEL_DIM = 8

_PAGE_ID_A0 = "docqa/longdocurl_K4/doc001/p0"
_PAGE_ID_A1 = "docqa/longdocurl_K4/doc001/p1"
_PAGE_ID_A2 = "docqa/longdocurl_K4/doc001/p2"
_PAGE_ID_B0 = "docqa/longdocurl_K4/doc002/p0"

_QUERY_ID_A = "docqa/longdocurl_K4/q001"
_QUERY_ID_B = "docqa/longdocurl_K4/q002"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(index: int, dim: int = _DIM) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    vec[index] = 1.0
    return vec


class _MockBatch(dict):
    """支持 .to(device) 的最小 batch 对象。"""

    def to(self, *args, **kwargs):
        return self


class _MockQueryProcessor:
    """最小 ColPaliProcessor mock，只实现 process_queries()."""

    def process_queries(self, queries: list[str]):
        batch_size = len(queries)
        return _MockBatch(
            input_ids=torch.ones(batch_size, _MODEL_TOKENS, dtype=torch.long),
            attention_mask=torch.ones(batch_size, _MODEL_TOKENS, dtype=torch.long),
        )


class _MockQueryModel:
    """最小 ColPali model mock，返回 [batch, n_tokens, dim] embeddings。"""

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            batch_size = 1
            n_tokens = _MODEL_TOKENS
        else:
            batch_size = int(input_ids.shape[0])
            n_tokens = int(input_ids.shape[1])

        base = torch.arange(
            batch_size * n_tokens * _MODEL_DIM,
            dtype=torch.float32,
        ).reshape(batch_size, n_tokens, _MODEL_DIM)
        return base

    __call__ = forward

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def parameters(self):
        return iter([])


class _FixedQueryEncoder:
    """为 RetrievalPipeline 提供稳定查询向量的 stub。"""

    def __init__(self, query_emb: torch.Tensor):
        self._query_emb = query_emb.clone()
        self.calls: list[str] = []
        self.model = _DummyModel()
        self.processor = _MockQueryProcessor()
        self.device = "cpu"

    def encode(self, query_text: str) -> torch.Tensor:
        self.calls.append(query_text)
        return self._query_emb.clone()


class _DummyModel:
    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def parameters(self):
        return iter([])


# ---------------------------------------------------------------------------
# Runtime construction helpers
# ---------------------------------------------------------------------------


def _make_query_encoder(model, processor) -> QueryEncoder:
    attempts = [
        lambda: QueryEncoder(model, processor, device="cpu"),
        lambda: QueryEncoder(model, processor),
        lambda: QueryEncoder(model, device="cpu"),
        lambda: QueryEncoder(model),
    ]

    last_error: Exception | None = None
    for build in attempts:
        try:
            return build()
        except TypeError as exc:
            last_error = exc
    raise AssertionError(
        f"无法按文档契约构造 QueryEncoder，最后一个错误: {last_error}"
    )


def _make_pipeline(model, index_store, query_encoder, config=None) -> RetrievalPipeline:
    config = config or {"score_batch_size": 2}
    attempts = [
        lambda: RetrievalPipeline(
            model=model,
            index_store=index_store,
            query_encoder=query_encoder,
            config=config,
        ),
        lambda: RetrievalPipeline(model, index_store, query_encoder, config),
        lambda: RetrievalPipeline(model, index_store, query_encoder),
    ]

    last_error: Exception | None = None
    for build in attempts:
        try:
            return build()
        except TypeError as exc:
            last_error = exc
    raise AssertionError(
        f"无法按文档契约构造 RetrievalPipeline，最后一个错误: {last_error}"
    )


# ---------------------------------------------------------------------------
# Fixtures: deterministic embeddings for scoring / retrieval
# ---------------------------------------------------------------------------


@pytest.fixture()
def query_embedding() -> torch.Tensor:
    return torch.stack([_unit(0), _unit(1)])


@pytest.fixture()
def page_embedding_a1() -> torch.Tensor:
    # 完全匹配两个 query token：理论上应获得最高分。
    return torch.stack([_unit(0), _unit(1)])


@pytest.fixture()
def page_embedding_a0() -> torch.Tensor:
    # 仅匹配第一个 token：应低于 a1，高于 a2。
    return torch.stack([_unit(0), _unit(2)])


@pytest.fixture()
def page_embedding_a2() -> torch.Tensor:
    # 与两个 query token 都不对齐：应获得最低分。
    return torch.stack([_unit(2), _unit(3)])


@pytest.fixture()
def page_embedding_b0() -> torch.Tensor:
    # doc002 的唯一页面，只匹配第二个 token。
    return torch.stack([_unit(1), _unit(3)])


# ---------------------------------------------------------------------------
# Fixtures: QueryEncoder
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_query_model() -> _MockQueryModel:
    return _MockQueryModel()


@pytest.fixture()
def mock_query_processor() -> _MockQueryProcessor:
    return _MockQueryProcessor()


@pytest.fixture()
def actual_query_encoder(mock_query_model, mock_query_processor) -> QueryEncoder:
    return _make_query_encoder(mock_query_model, mock_query_processor)


# ---------------------------------------------------------------------------
# Fixtures: IndexStore / RetrievalPipeline
# ---------------------------------------------------------------------------


@pytest.fixture()
def index_dir(tmp_path: Path) -> Path:
    d = tmp_path / "index"
    d.mkdir()
    return d


@pytest.fixture()
def store(index_dir: Path) -> IndexStore:
    return IndexStore(str(index_dir))


@pytest.fixture()
def populated_store(
    store: IndexStore,
    page_embedding_a0: torch.Tensor,
    page_embedding_a1: torch.Tensor,
    page_embedding_a2: torch.Tensor,
    page_embedding_b0: torch.Tensor,
) -> IndexStore:
    store.write_page(_PAGE_ID_A0, page_embedding_a0)
    store.write_page(_PAGE_ID_A1, page_embedding_a1)
    store.write_page(_PAGE_ID_A2, page_embedding_a2)
    store.write_page(_PAGE_ID_B0, page_embedding_b0)
    return store


@pytest.fixture()
def fixed_query_encoder(query_embedding: torch.Tensor) -> _FixedQueryEncoder:
    return _FixedQueryEncoder(query_embedding)


@pytest.fixture()
def pipeline(populated_store: IndexStore, fixed_query_encoder: _FixedQueryEncoder):
    return _make_pipeline(
        model=_DummyModel(),
        index_store=populated_store,
        query_encoder=fixed_query_encoder,
        config={"score_batch_size": 2},
    )


@pytest.fixture()
def sample_query() -> Query:
    return Query(
        query_id=_QUERY_ID_A,
        text="What is the answer in this document?",
        doc_id="doc001",
        raw_doc_name="doc001",
        task_family="docqa",
        subtask="longdocurl",
        length="K4",
    )


@pytest.fixture()
def other_doc_query() -> Query:
    return Query(
        query_id=_QUERY_ID_B,
        text="Find the relevant page in the second document.",
        doc_id="doc002",
        raw_doc_name="doc002",
        task_family="docqa",
        subtask="longdocurl",
        length="K4",
    )

