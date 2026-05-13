"""
MaxSim 相似度计算模块。

实现 ColBERT-style Late Interaction 打分：
    Score(Q, P) = Σ_i max_j Sim(q_i, p_j)

其中 Sim 为 L2 归一化后的点积（等价于余弦相似度）。

模块独立于检索流水线，便于后续替换或扩展打分函数（如加权 MaxSim、
稀疏 MaxSim 等）。
"""

from __future__ import annotations

import torch


def maxsim_score(
    query_emb: torch.Tensor,
    page_emb: torch.Tensor,
    norm: bool = True,
) -> torch.Tensor:
    """计算单页的 MaxSim 相似度分数。

    Parameters
    ----------
    query_emb : torch.Tensor
        查询 token embeddings，shape ``[n_tokens, dim]``
    page_emb : torch.Tensor
        页面 patch embeddings，shape ``[n_patches, dim]``
    norm : bool
        是否预先 L2 归一化嵌入（使点积等价于余弦相似度）

    Returns
    -------
    torch.Tensor
        标量 score
    """
    query_emb, page_emb = _align_similarity_dtypes(query_emb, page_emb)

    if norm:
        query_emb = _l2_normalize(query_emb)
        page_emb = _l2_normalize(page_emb)

    # [n_tokens, dim] @ [dim, n_patches] → [n_tokens, n_patches]
    sim_matrix = query_emb @ page_emb.T

    # max over patches → [n_tokens]; sum over tokens → scalar
    score = sim_matrix.max(dim=-1).values.sum()

    return score


def batched_maxsim(
    query_emb: torch.Tensor,
    pages_emb: torch.Tensor,
    norm: bool = True,
) -> torch.Tensor:
    """批量计算 MaxSim 相似度分数。

    对一批页面（patches 数相同）同时计算与同一条查询的 MaxSim 分数。

    Parameters
    ----------
    query_emb : torch.Tensor
        查询 token embeddings，shape ``[n_tokens, dim]``
    pages_emb : torch.Tensor
        页面 patch embeddings，shape ``[batch, n_patches, dim]``
    norm : bool
        是否预先 L2 归一化嵌入

    Returns
    -------
    torch.Tensor
        shape ``[batch]``，每个页面一个 score
    """
    query_emb, pages_emb = _align_similarity_dtypes(query_emb, pages_emb)

    if norm:
        query_emb = _l2_normalize(query_emb)
        pages_emb = _l2_normalize(pages_emb)

    # einsum: 'td,bpd->btp'
    #   t = n_tokens, d = dim, b = batch, p = n_patches
    sim_tensor = torch.einsum("td,bpd->btp", query_emb, pages_emb)

    # max over patches → [batch, n_tokens]; sum over tokens → [batch]
    scores = sim_tensor.max(dim=-1).values.sum(dim=-1)

    return scores


def batched_maxsim_variable(
    query_emb: torch.Tensor,
    pages_list: list[torch.Tensor],
    norm: bool = True,
) -> torch.Tensor:
    """对变长 patches 的页面列表逐页计算 MaxSim 分数。

    当各页 patch 数不完全相同时使用此函数（如 Phase 4A patch pruning 后）。

    Parameters
    ----------
    query_emb : torch.Tensor
        查询 token embeddings，shape ``[n_tokens, dim]``
    pages_list : list[torch.Tensor]
        各页面 patch embeddings，每个 shape ``[n_patches_i, dim]``
    norm : bool
        是否预先 L2 归一化嵌入

    Returns
    -------
    torch.Tensor
        shape ``[len(pages_list)]``
    """
    if norm:
        query_emb = _l2_normalize(query_emb)

    scores: list[float] = []
    for page_emb in pages_list:
        aligned_query, aligned_page = _align_similarity_dtypes(query_emb, page_emb)
        if norm:
            aligned_query = _l2_normalize(aligned_query)
            aligned_page = _l2_normalize(aligned_page)
        sim = aligned_query @ aligned_page.T  # [n_tokens, n_patches_i]
        scores.append(sim.max(dim=-1).values.sum().item())

    return torch.tensor(scores, dtype=torch.float32)


# ------------------------------------------------------------------
# 内部辅助
# ------------------------------------------------------------------


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """沿最后一维做 L2 归一化（原地操作安全版本）。

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor
        归一化后的张量（不修改原张量）
    """
    norm = x.norm(p=2, dim=-1, keepdim=True)
    # 避免除零
    norm = norm.clamp(min=1e-12)
    return x / norm


def _align_similarity_dtypes(
    query_emb: torch.Tensor,
    page_emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """为相似度计算对齐 dtype。

    页面向量常以 float16 落盘，查询向量则可能来自 bfloat16 推理。
    在 Windows/CPU 侧做 MaxSim 时，两者必须先提升到同一 dtype。
    """
    common_dtype = torch.promote_types(query_emb.dtype, page_emb.dtype)

    if query_emb.dtype != common_dtype:
        query_emb = query_emb.to(common_dtype)
    if page_emb.dtype != common_dtype:
        page_emb = page_emb.to(common_dtype)

    return query_emb, page_emb
