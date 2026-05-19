"""Phase 4 Two-Stage Retriever: mean-pool coarse → MaxSim rerank。

核心流程:
    Query.candidate_page_ids → universe
    → mean-pool coarse retrieval
    → adaptive top-N selection
    → optional neighbor expansion
    → full MaxSim rerank
    → final top-k results
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from zeroshot_vdr.contracts import Query, RetrievalResult

logger = logging.getLogger(__name__)


# ===========================================================================
# Data Classes
# ===========================================================================


@dataclass
class TwoStageTrace:
    """单条 query 的 two-stage 检索 trace。"""

    query_id: str | None = None
    universe_size: int = 0
    coarse_top_n: int = 0
    expanded_candidate_count: int = 0
    neighbor_added_count: int = 0
    coarse_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    method: str = "fixed_topn"

    # adaptive 相关
    top1_coarse_score: float | None = None
    topn_coarse_score: float | None = None
    coarse_margin: float | None = None
    adaptive_expanded: bool = False


@dataclass
class TwoStageOutput:
    """Two-stage 检索输出。"""

    results: list[RetrievalResult]
    trace: TwoStageTrace


# ===========================================================================
# Adaptive Top-N
# ===========================================================================


def choose_adaptive_top_n(
    scores: torch.Tensor,
    universe_size: int,
    min_n: int = 32,
    max_n: int = 128,
    base_ratio: float = 0.20,
    flat_margin: float = 0.035,
) -> int:
    """根据 coarse 分数分布自适应选择 top-N。

    策略：
    1. 先按 base_ratio 计算基础 N（base_n）
    2. 若 top-1 与 top-base_n 之间 margin < flat_margin，
       说明分数分布过于平坦 → 扩张 base_n × 2

    Parameters
    ----------
    scores : torch.Tensor
        coarse 阶段所有候选页面的分数（1-D）
    universe_size : int
        候选全集大小
    min_n : int
        top-N 下限
    max_n : int
        top-N 上限
    base_ratio : float
        基础比例
    flat_margin : float
        平坦分布判断阈值

    Returns
    -------
    int
        自适应选出的 top-N
    """
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


# ===========================================================================
# Coarse Retrieval Helpers
# ===========================================================================


def mean_pool_query(query_emb: torch.Tensor) -> torch.Tensor:
    """对 query token embeddings 做 mean pooling + L2 normalize。"""
    query_mean = query_emb.mean(dim=0)
    return F.normalize(query_mean, dim=-1)


def score_mean_pool(
    query_emb: torch.Tensor,
    page_means: torch.Tensor,
) -> torch.Tensor:
    """用 mean-pool query 对 mean-pool pages 打分。

    Parameters
    ----------
    query_emb : torch.Tensor
        shape ``[n_tokens, dim]``
    page_means : torch.Tensor
        shape ``[n_pages, dim]``

    Returns
    -------
    torch.Tensor
        shape ``[n_pages]``，每页一个 cosine 分数
    """
    query_mean = mean_pool_query(query_emb)
    page_means = F.normalize(page_means, dim=-1)
    return page_means @ query_mean


def select_topn_by_scores(
    page_ids: list[str],
    scores: torch.Tensor,
    top_n: int,
) -> list[str]:
    """按分数取 top-N page_ids。"""
    if len(page_ids) == 0:
        return []

    top_n = min(top_n, len(page_ids))
    top_indices = torch.topk(scores, k=top_n).indices.tolist()
    return [page_ids[i] for i in top_indices]


# ===========================================================================
# Candidate Universe Resolution
# ===========================================================================


def resolve_candidate_universe(
    query: Query,
    explicit_candidate_ids: list[str] | None = None,
    index_store: Any | None = None,
) -> list[str]:
    """解析 query 的候选全集（universe）。

    优先级：
    1. explicit_candidate_ids（外部传入）
    2. query.candidate_page_ids（query 显式携带的 sample-specific 候选）
    3. index_store 按 doc_id/task_family/subtask/length 查询

    Parameters
    ----------
    query : Query
    explicit_candidate_ids : list[str] | None
    index_store : IndexStore | None

    Returns
    -------
    list[str]
    """
    # 优先级 1: 显式传入
    if explicit_candidate_ids is not None:
        return list(explicit_candidate_ids)

    # 优先级 2: query 携带的 candidate_page_ids
    candidate_page_ids = getattr(query, "candidate_page_ids", None)
    if candidate_page_ids:
        return list(candidate_page_ids)

    # 优先级 3: 从 index_store 查询
    if index_store is None:
        raise ValueError(
            "index_store is required when query.candidate_page_ids is missing "
            "and no explicit_candidate_ids provided"
        )

    return index_store.list_page_ids(
        doc_id=getattr(query, "doc_id", None),
        task_family=getattr(query, "task_family", None),
        subtask=getattr(query, "subtask", None),
        length=getattr(query, "length", None),
    )


# ===========================================================================
# TwoStageRetriever
# ===========================================================================


class TwoStageRetriever:
    """两阶段检索器：mean-pool 粗筛 + MaxSim 精排。

    原则：
    - candidate_page_ids 是 query-specific universe，不是最终候选
    - coarse 阶段在 universe 内用 mean-pool 快速粗筛
    - rerank 阶段只对 coarse 选出的候选做完整 MaxSim

    Parameters
    ----------
    base_pipeline : RetrievalPipeline
        Phase 3 的检索流水线，提供 encode_query / score_candidates
    index_store : IndexStore
        索引存储
    coarse_top_n : int
        固定 top-N（method=fixed_topn 时使用）
    method : str
        方法标识: "fixed_topn"
    """

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

        # 配置参数（后续 Stage 扩展）
        self.min_candidates: int = 32
        self.max_candidates: int = 128
        self.base_ratio: float = 0.20
        self.flat_margin: float = 0.035
        self.neighbor_window: int = 0
        self.neighbor_seed_n: int = 8

    # ------------------------------------------------------------------
    # 主检索接口
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: Query,
        top_k: int = 10,
        candidate_ids: list[str] | None = None,
    ) -> TwoStageOutput:
        """执行两阶段检索。

        Parameters
        ----------
        query : Query
        top_k : int
        candidate_ids : list[str] | None
            显式候选；None 则从 query.candidate_page_ids 解析

        Returns
        -------
        TwoStageOutput
        """
        t0 = time.perf_counter()

        # ---- 阶段 0: 查询编码 ----
        query_emb = self.pipeline.encode_query(query.text)

        # ---- 阶段 0.5: 解析 candidate universe ----
        universe_ids = resolve_candidate_universe(
            query=query,
            explicit_candidate_ids=candidate_ids,
            index_store=self.index_store,
        )

        # ---- 阶段 1: mean-pool coarse retrieval ----
        coarse_start = time.perf_counter()
        page_means, pids = self.index_store.get_mean_pooled_view(universe_ids)

        if page_means.numel() == 0:
            # 空 universe → 直接返回空结果
            trace = TwoStageTrace(
                query_id=getattr(query, "query_id", None),
                universe_size=len(universe_ids),
                method=self.method,
            )
            return TwoStageOutput(results=[], trace=trace)

        coarse_scores = score_mean_pool(query_emb, page_means)

        # 选择 coarse top-N（当前为 fixed）
        selected_n = self.coarse_top_n
        coarse_ids = select_topn_by_scores(
            page_ids=pids,
            scores=coarse_scores,
            top_n=selected_n,
        )
        coarse_ms = (time.perf_counter() - coarse_start) * 1000

        # ---- 阶段 2: full MaxSim rerank ----
        rerank_start = time.perf_counter()
        scores, scored_ids = self.pipeline.score_candidates(
            query_emb=query_emb,
            candidate_ids=coarse_ids,
        )
        # 结果组装
        results = self.pipeline._assemble_results(
            query_id=getattr(query, "query_id", ""),
            scores=scores,
            page_ids=scored_ids,
            top_k=top_k,
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
