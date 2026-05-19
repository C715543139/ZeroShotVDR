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
    # 对齐 dtype：query 可能是 bfloat16，page_means 可能是 float16
    query_mean = query_mean.to(dtype=page_means.dtype)
    page_means = F.normalize(page_means.float(), dim=-1).to(page_means.dtype)
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

    支持三种 method:
    - ``fixed_topn``: 固定 coarse top-N
    - ``adaptive``: 自适应 top-N（根据分数分布）
    - ``adaptive_neighbors``: 自适应 + 邻页扩展

    Parameters
    ----------
    base_pipeline : RetrievalPipeline
        Phase 3 的检索流水线，提供 encode_query / score_candidates
    index_store : IndexStore
        索引存储
    method : str
        "fixed_topn" / "adaptive" / "adaptive_neighbors"
    coarse_top_n : int
        固定 top-N（method=fixed_topn 时使用）
    min_candidates : int
        adaptive 下限
    max_candidates : int
        adaptive 上限
    base_ratio : float
        adaptive 基础比例
    flat_margin : float
        adaptive 平坦阈值
    neighbor_window : int
        邻页窗口大小；0 表示不扩展
    neighbor_seed_n : int
        对前 seed_n 个 coarse page 扩展邻页
    """

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
        neighbor_window: int = 0,
        neighbor_seed_n: int = 8,
    ):
        self.pipeline = base_pipeline
        self.index_store = index_store
        self.method = method
        self.coarse_top_n = coarse_top_n
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.base_ratio = base_ratio
        self.flat_margin = flat_margin
        self.neighbor_window = neighbor_window
        self.neighbor_seed_n = neighbor_seed_n

        # 校验 method
        _valid_methods = {"fixed_topn", "adaptive", "adaptive_neighbors"}
        if method not in _valid_methods:
            raise ValueError(
                f"不支持的 method: {method}，可选: {_valid_methods}"
            )

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
            trace = TwoStageTrace(
                query_id=getattr(query, "query_id", None),
                universe_size=len(universe_ids),
                method=self.method,
            )
            return TwoStageOutput(results=[], trace=trace)

        coarse_scores = score_mean_pool(query_emb, page_means)

        # ---- 选择 coarse top-N ----
        sorted_scores = torch.sort(
            coarse_scores.detach().float(), descending=True
        ).values

        if self.method == "fixed_topn":
            selected_n = self.coarse_top_n
            adaptive_expanded = False
        elif self.method in ("adaptive", "adaptive_neighbors"):
            selected_n = choose_adaptive_top_n(
                scores=coarse_scores,
                universe_size=len(pids),
                min_n=self.min_candidates,
                max_n=self.max_candidates,
                base_ratio=self.base_ratio,
                flat_margin=self.flat_margin,
            )
            # 判断是否触发了扩张
            base_n = int(round(len(pids) * self.base_ratio))
            base_n = max(self.min_candidates, min(base_n, self.max_candidates, len(pids)))
            adaptive_expanded = selected_n > base_n
        else:
            raise ValueError(f"不支持的 method: {self.method}")

        coarse_ids = select_topn_by_scores(
            page_ids=pids,
            scores=coarse_scores,
            top_n=selected_n,
        )

        # ---- Neighbor expansion（method=adaptive_neighbors 时） ----
        neighbor_added_count = 0
        if self.method == "adaptive_neighbors" and self.neighbor_window > 0:
            from zeroshot_vdr.advanced.neighbors import expand_neighbors

            expanded_ids = expand_neighbors(
                coarse_ids=coarse_ids,
                universe_ids=universe_ids,
                window=self.neighbor_window,
                seed_n=self.neighbor_seed_n,
            )
            neighbor_added_count = len(expanded_ids) - len(coarse_ids)
            rerank_ids = expanded_ids
        else:
            rerank_ids = coarse_ids

        coarse_ms = (time.perf_counter() - coarse_start) * 1000

        # ---- 阶段 2: full MaxSim rerank ----
        rerank_start = time.perf_counter()
        scores, scored_ids = self.pipeline.score_candidates(
            query_emb=query_emb,
            candidate_ids=rerank_ids,
        )
        results = self.pipeline._assemble_results(
            query_id=getattr(query, "query_id", ""),
            scores=scores,
            page_ids=scored_ids,
            top_k=top_k,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        # ---- Trace ----
        trace = TwoStageTrace(
            query_id=getattr(query, "query_id", None),
            universe_size=len(universe_ids),
            coarse_top_n=len(coarse_ids),
            expanded_candidate_count=len(rerank_ids),
            neighbor_added_count=neighbor_added_count,
            coarse_ms=coarse_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
            method=self.method,
        )

        # adaptive trace 字段
        if self.method in ("adaptive", "adaptive_neighbors") and len(sorted_scores) > 0:
            trace.top1_coarse_score = float(sorted_scores[0])
            if selected_n <= len(sorted_scores):
                trace.topn_coarse_score = float(sorted_scores[selected_n - 1])
                trace.coarse_margin = float(
                    sorted_scores[0] - sorted_scores[selected_n - 1]
                )
            trace.adaptive_expanded = adaptive_expanded

        return TwoStageOutput(results=results, trace=trace)
