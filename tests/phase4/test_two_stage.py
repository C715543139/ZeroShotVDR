"""Stage 4 测试：TwoStageRetriever fixed top-N 最小版本。"""

import pytest
import torch

from zeroshot_vdr.advanced.two_stage import (
    TwoStageRetriever,
    TwoStageOutput,
    TwoStageTrace,
    mean_pool_query,
    score_mean_pool,
    select_topn_by_scores,
    resolve_candidate_universe,
)
from zeroshot_vdr.contracts import Query, RetrievalResult


# ===========================================================================
# 测试用 Fake 对象
# ===========================================================================


class FakeIndexStore:
    """可控的 fake IndexStore。"""

    def __init__(self, page_means: torch.Tensor | None = None):
        self._page_means = page_means
        self._page_ids: list[str] = []
        self._score_candidates_calls: list[list[str]] = []
        self._page_embeddings: dict[str, torch.Tensor] = {}

    def set_universe(self, page_ids: list[str], means: torch.Tensor | None = None):
        self._page_ids = list(page_ids)
        if means is not None:
            self._page_means = means

    def get_mean_pooled_view(self, page_ids):
        if self._page_means is None:
            # 生成随机 embeddings
            dim = 128
            means = torch.randn(len(page_ids), dim)
            self._page_means = means
        return self._page_means, page_ids

    def list_page_ids(self, **kwargs):
        return list(self._page_ids)

    def read_stacked(self, page_ids):
        """fake read_stacked 返回随机 embeddings。"""
        dim = 128
        n_patches = 16
        valid_ids = [p for p in page_ids if p in self._page_ids or not self._page_ids]
        if not valid_ids:
            return torch.empty((0,)), []
        # 使用页面在 self._page_ids 中的索引作为 seed，保证确定性
        emb_list = []
        for pid in valid_ids:
            idx = self._page_ids.index(pid) if pid in self._page_ids else 0
            gen = torch.Generator()
            gen.manual_seed(idx + 42)
            emb_list.append(torch.randn(n_patches, dim, generator=gen))
        return torch.stack(emb_list), valid_ids


class FakePipeline:
    """可控的 fake RetrievalPipeline。"""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._encode_query_calls: list[str] = []
        self._score_candidates_calls: list[list[str]] = []

    def encode_query(self, text: str):
        self._encode_query_calls.append(text)
        gen = torch.Generator()
        gen.manual_seed(hash(text) & 0xFFFFFFFF)
        return torch.randn(8, self.dim, generator=gen)  # 8 tokens

    def score_candidates(self, query_emb, candidate_ids, batch_size=None):
        self._score_candidates_calls.append(list(candidate_ids))
        dim = query_emb.shape[-1]
        gen = torch.Generator()
        gen.manual_seed(hash(str(candidate_ids)) & 0xFFFFFFFF)
        scores = torch.rand(len(candidate_ids), generator=gen)
        return scores, candidate_ids

    def _assemble_results(self, query_id, scores, page_ids, top_k):
        sorted_idx = scores.argsort(descending=True)
        results = []
        for rank_0, idx in enumerate(sorted_idx[:top_k].tolist()):
            results.append(
                RetrievalResult(
                    query_id=query_id,
                    page_id=page_ids[idx],
                    score=float(scores[idx].item()),
                    rank=rank_0 + 1,
                )
            )
        return results


# ===========================================================================
# 辅助函数
# ===========================================================================


def make_query(query_id="test/q1", text="test query", candidate_page_ids=None):
    return Query(
        query_id=query_id,
        text=text,
        doc_id="test_doc",
        raw_doc_name=None,
        task_family="docqa",
        subtask="slidevqa",
        length="K128",
        candidate_page_ids=tuple(candidate_page_ids) if candidate_page_ids else (),
    )


# ===========================================================================
# 测试 mean_pool_query / score_mean_pool
# ===========================================================================


class TestMeanPoolHelpers:
    def test_mean_pool_query_shape(self):
        q_emb = torch.randn(8, 128)
        result = mean_pool_query(q_emb)
        assert result.shape == (128,)

    def test_mean_pool_query_normalized(self):
        q_emb = torch.randn(8, 128)
        result = mean_pool_query(q_emb)
        norm = result.norm(p=2)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_score_mean_pool_shape(self):
        q_emb = torch.randn(8, 128)
        page_means = torch.randn(50, 128)
        scores = score_mean_pool(q_emb, page_means)
        assert scores.shape == (50,)


class TestSelectTopN:
    def test_empty(self):
        result = select_topn_by_scores([], torch.tensor([]), 10)
        assert result == []

    def test_top_n_smaller_than_total(self):
        pids = [f"p{i}" for i in range(100)]
        scores = torch.arange(100, 0, -1, dtype=torch.float32)
        result = select_topn_by_scores(pids, scores, top_n=10)
        assert len(result) == 10
        assert result == [f"p{i}" for i in range(10)]

    def test_top_n_larger_than_total(self):
        pids = ["a", "b", "c"]
        scores = torch.tensor([0.3, 0.8, 0.1])
        result = select_topn_by_scores(pids, scores, top_n=10)
        assert len(result) == 3


class TestResolveCandidateUniverse:
    def test_explicit_ids(self):
        q = make_query(candidate_page_ids=["a", "b"])
        result = resolve_candidate_universe(q, explicit_candidate_ids=["x", "y"])
        assert result == ["x", "y"]

    def test_query_candidate_page_ids(self):
        q = make_query(candidate_page_ids=["a", "b", "c"])
        result = resolve_candidate_universe(q)
        assert result == ["a", "b", "c"]

    def test_empty_candidate_page_ids(self):
        q = make_query(candidate_page_ids=[])
        store = FakeIndexStore()
        store.set_universe(["d1", "d2"])
        result = resolve_candidate_universe(q, index_store=store)
        assert result == ["d1", "d2"]

    def test_missing_store_raises(self):
        q = make_query(candidate_page_ids=[])
        with pytest.raises(ValueError, match="index_store is required"):
            resolve_candidate_universe(q)


# ===========================================================================
# 测试 TwoStageRetriever
# ===========================================================================


class TestTwoStageRetriever:
    def test_init(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()
        retriever = TwoStageRetriever(pipeline, store, coarse_top_n=32)
        assert retriever.coarse_top_n == 32
        assert retriever.method == "fixed_topn"

    def test_invalid_method_raises(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()
        with pytest.raises(ValueError, match="不支持的 method"):
            TwoStageRetriever(pipeline, store, method="invalid")

    def test_retrieve_basic(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()

        universe = [f"doc/p{i}" for i in range(50)]
        store.set_universe(universe)

        retriever = TwoStageRetriever(pipeline, store, coarse_top_n=10)
        q = make_query(candidate_page_ids=universe)

        output = retriever.retrieve(q, top_k=5)
        assert isinstance(output, TwoStageOutput)
        assert isinstance(output.trace, TwoStageTrace)
        assert len(output.results) <= 5

    def test_candidate_is_universe_not_result(self):
        """验证 candidate_page_ids 被当作 universe，而 rerank 输入是 coarse_ids。"""
        pipeline = FakePipeline()
        store = FakeIndexStore()

        universe = [f"doc/p{i}" for i in range(100)]
        store.set_universe(universe)

        retriever = TwoStageRetriever(pipeline, store, coarse_top_n=10)
        q = make_query(candidate_page_ids=universe)

        output = retriever.retrieve(q, top_k=5)

        # rerank 只收到 coarse_ids，不是完整 universe
        assert len(pipeline._score_candidates_calls) == 1
        rerank_input = pipeline._score_candidates_calls[0]
        assert len(rerank_input) <= 10  # ≤ coarse_top_n

    def test_coarse_reduces_candidates(self):
        """coarse 后候选数应小于 universe。"""
        pipeline = FakePipeline()
        store = FakeIndexStore()

        universe = [f"doc/p{i}" for i in range(200)]
        store.set_universe(universe)

        retriever = TwoStageRetriever(pipeline, store, coarse_top_n=20)
        q = make_query(candidate_page_ids=universe)

        output = retriever.retrieve(q, top_k=5)

        assert output.trace.universe_size == 200
        assert output.trace.coarse_top_n <= 20

    def test_trace_fields(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()

        universe = [f"doc/p{i}" for i in range(50)]
        store.set_universe(universe)

        retriever = TwoStageRetriever(pipeline, store, coarse_top_n=10)
        q = make_query(candidate_page_ids=universe)

        output = retriever.retrieve(q, top_k=5)
        trace = output.trace

        assert trace.query_id == q.query_id
        assert trace.universe_size == 50
        assert trace.coarse_top_n <= 10
        assert trace.method == "fixed_topn"
        assert trace.coarse_ms >= 0
        assert trace.rerank_ms >= 0
        assert trace.total_ms >= 0

    def test_empty_universe(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()
        store.set_universe([])

        retriever = TwoStageRetriever(pipeline, store, coarse_top_n=10)
        q = make_query(candidate_page_ids=[])

        output = retriever.retrieve(q, top_k=5)
        assert output.results == []
        assert output.trace.universe_size == 0

    # ---- adaptive 测试 ----

    def test_adaptive_method_runs(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()
        universe = [f"doc/p{i}" for i in range(100)]
        store.set_universe(universe)

        retriever = TwoStageRetriever(
            pipeline, store,
            method="adaptive",
            min_candidates=10,
            max_candidates=50,
            base_ratio=0.20,
        )
        q = make_query(candidate_page_ids=universe)
        output = retriever.retrieve(q, top_k=5)

        assert output.trace.method == "adaptive"
        assert output.trace.coarse_top_n >= 10
        assert output.trace.coarse_top_n <= 50

    def test_adaptive_trace_fields(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()
        universe = [f"doc/p{i}" for i in range(100)]
        store.set_universe(universe)

        retriever = TwoStageRetriever(
            pipeline, store, method="adaptive",
        )
        q = make_query(candidate_page_ids=universe)
        output = retriever.retrieve(q, top_k=5)

        trace = output.trace
        assert trace.top1_coarse_score is not None
        assert trace.topn_coarse_score is not None
        assert trace.coarse_margin is not None

    # ---- neighbor 测试 ----

    def test_neighbor_method_runs(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()
        universe = [f"doc/p{i}" for i in range(50)]
        store.set_universe(universe)

        retriever = TwoStageRetriever(
            pipeline, store,
            method="adaptive_neighbors",
            neighbor_window=1,
            neighbor_seed_n=4,
            coarse_top_n=10,
        )
        q = make_query(candidate_page_ids=universe)
        output = retriever.retrieve(q, top_k=5)

        assert output.trace.method == "adaptive_neighbors"
        # neighbor may add pages
        assert output.trace.expanded_candidate_count >= output.trace.coarse_top_n

    def test_neighbor_window_zero_no_expand(self):
        pipeline = FakePipeline()
        store = FakeIndexStore()
        universe = [f"doc/p{i}" for i in range(50)]
        store.set_universe(universe)

        retriever = TwoStageRetriever(
            pipeline, store,
            method="adaptive_neighbors",
            neighbor_window=0,
            coarse_top_n=10,
        )
        q = make_query(candidate_page_ids=universe)
        output = retriever.retrieve(q, top_k=5)

        assert output.trace.neighbor_added_count == 0
        assert output.trace.coarse_top_n == output.trace.expanded_candidate_count
