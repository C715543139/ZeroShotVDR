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

import torch


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
