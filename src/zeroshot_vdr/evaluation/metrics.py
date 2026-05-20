"""
指标计算模块。

所有指标函数接受标准化输入 ``(retrieved_page_ids, relevant_page_ids, k)``，
与具体数据集解耦。参数类型均为 Python 原生类型（list / set / int），
不依赖 torch，保持模块轻量化。

指标定义：
- **Recall@k**   = |retrieved[:k] ∩ relevant| / |relevant|
- **Precision@k** = |retrieved[:k] ∩ relevant| / k
- **MRR**        = Mean Reciprocal Rank（首个相关页面的倒数排名均值）
- **nDCG@k**     = Normalized Discounted Cumulative Gain at k
"""

from __future__ import annotations

import math
from typing import List

# Avoid importing pandas at module level — computed only inside compute_all_metrics
# to keep the module lightweight for single-metric calls.


# ============================================================================
# 原子指标
# ============================================================================


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Recall@k = |retrieved[:k] ∩ relevant| / |relevant|。

    若 relevant 为空集，返回 1.0（根据信息检索惯例，无相关文档时召回视为完美）。

    Parameters
    ----------
    retrieved : list[str]
        按排名升序排列的 page_id 列表
    relevant : set[str]
        相关 page_id 集合
    k : int
        截断值

    Returns
    -------
    float
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，得到 {k}")

    if not relevant:
        return 1.0

    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & relevant)
    return hits / len(relevant)


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Precision@k = |retrieved[:k] ∩ relevant| / k。

    Parameters
    ----------
    retrieved : list[str]
    relevant : set[str]
    k : int

    Returns
    -------
    float
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，得到 {k}")

    if not relevant:
        return 0.0

    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & relevant)
    return hits / k


def mrr(
    retrieved: list[str],
    relevant: set[str],
) -> float:
    """Mean Reciprocal Rank：首个相关页面出现排名的倒数。

    若检索结果中无相关页面，返回 0.0。

    Parameters
    ----------
    retrieved : list[str]
    relevant : set[str]

    Returns
    -------
    float
    """
    if not relevant:
        return 1.0

    for rank_1, page_id in enumerate(retrieved, start=1):
        if page_id in relevant:
            return 1.0 / rank_1

    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k。

    使用二元相关性（1/0）和标准对数折损：
        DCG@k = Σ_{i=1}^{k} rel_i / log₂(i+1)
        IDCG@k  = 最优排序下的 DCG@k
        nDCG@k = DCG@k / IDCG@k

    若 relevant 为空集，返回 1.0。

    Parameters
    ----------
    retrieved : list[str]
    relevant : set[str]
    k : int

    Returns
    -------
    float
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，得到 {k}")

    if not relevant:
        return 1.0

    # DCG
    dcg = 0.0
    for i, page_id in enumerate(retrieved[:k], start=1):
        if page_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG（最优情况：前 min(k, |relevant|) 位都是相关页面）
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ============================================================================
# 批量计算
# ============================================================================


def compute_all_metrics(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_values: list[int] | None = None,
) -> "pd.DataFrame":  # noqa: F821
    """批量计算全部指标。

    Parameters
    ----------
    retrieval_results : dict[str, list[str]]
        {query_id: [retrieved_page_id, ...]}，按排名升序
    ground_truth : dict[str, set[str]]
        {query_id: {relevant_page_id, ...}}
    k_values : list[int] | None
        截断值列表；None 使用 [1, 3, 5, 10]

    Returns
    -------
    pandas.DataFrame
        columns = ['k', 'Recall', 'Precision', 'MRR', 'nDCG']
        每行对应一个 k 值的**宏平均**（跨所有查询平均）
    """
    import pandas as pd

    if k_values is None:
        k_values = [1, 3, 5, 10]

    # 对齐查询集合（取交集）
    common_queries = set(retrieval_results.keys()) & set(ground_truth.keys())
    if not common_queries:
        raise ValueError(
            "retrieval_results 和 ground_truth 无共同查询 ID，"
            "请检查 ID 构造是否一致"
        )

    rows: list[dict] = []

    for k in k_values:
        recall_sum = 0.0
        prec_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        n = 0

        for qid in common_queries:
            retrieved = retrieval_results.get(qid, [])
            relevant = ground_truth.get(qid, set())

            recall_sum += recall_at_k(retrieved, relevant, k)
            prec_sum += precision_at_k(retrieved, relevant, k)
            mrr_sum += mrr(retrieved, relevant)
            ndcg_sum += ndcg_at_k(retrieved, relevant, k)
            n += 1

        rows.append({
            "k": k,
            "Recall": recall_sum / n,
            "Precision": prec_sum / n,
            "MRR": mrr_sum / n,
            "nDCG": ndcg_sum / n,
            "n_queries": n,
        })

    return pd.DataFrame(rows)


# ============================================================================
# 分组统计工具
# ============================================================================


def compute_metrics_by_group(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    group_fn,
    k_values: list[int] | None = None,
) -> "pd.DataFrame":  # noqa: F821
    """按自定义分组维度计算指标。

    用于分子任务、分长度档位等分组汇报场景。

    Parameters
    ----------
    retrieval_results : dict[str, list[str]]
    ground_truth : dict[str, set[str]]
    group_fn : callable
        分组函数，签名为 ``(query_id: str) -> str``，
        返回分组标签（如 "longdocurl"、"K32" 等）
    k_values : list[int] | None

    Returns
    -------
    pandas.DataFrame
        columns = ['group', 'k', 'Recall', 'Precision', 'MRR', 'nDCG', 'n_queries']
    """
    import pandas as pd

    if k_values is None:
        k_values = [1, 3, 5, 10]

    frames: list[pd.DataFrame] = []

    common_queries = set(retrieval_results.keys()) & set(ground_truth.keys())

    # 按 group 分组
    groups: dict[str, list[str]] = {}
    for qid in common_queries:
        g = group_fn(qid)
        groups.setdefault(g, []).append(qid)

    for group_label, query_ids in sorted(groups.items()):
        subset_results = {q: retrieval_results[q] for q in query_ids}
        subset_gt = {q: ground_truth[q] for q in query_ids if q in ground_truth}
        df = compute_all_metrics(subset_results, subset_gt, k_values)
        df.insert(0, "group", group_label)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
