"""Phase 4 邻页扩展工具：page_id 解析与 neighbor expansion。

稳定 page_id 格式::

    {task_family}/{subtask}_{length}/{doc_id}/p{page_idx}

例如::

    slidevqa/default_K128/doc_001/p15
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# page_id 解析
# ---------------------------------------------------------------------------

# 匹配末尾 "/p{page_idx}"，prefix 为前面的所有内容
_PAGE_ID_RE = re.compile(r"^(?P<prefix>.+)/p(?P<page_idx>\d+)$")


@dataclass(frozen=True)
class ParsedPageId:
    """page_id 解析结果。"""

    prefix: str  # {task_family}/{subtask}_{length}/{doc_id}
    page_idx: int  # 页码（从 0 开始）


def parse_page_id(page_id: str) -> ParsedPageId:
    """将稳定 page_id 解析为 prefix 和 page_idx。

    Parameters
    ----------
    page_id : str
        格式 ``{prefix}/p{page_idx}``

    Returns
    -------
    ParsedPageId

    Raises
    ------
    ValueError
        若 page_id 格式不符合预期
    """
    match = _PAGE_ID_RE.match(page_id)
    if match is None:
        raise ValueError(f"Invalid page_id format: {page_id}")

    return ParsedPageId(
        prefix=match.group("prefix"),
        page_idx=int(match.group("page_idx")),
    )


def make_page_id(prefix: str, page_idx: int) -> str:
    """从 prefix 和 page_idx 构造稳定 page_id。"""
    return f"{prefix}/p{page_idx}"


# ---------------------------------------------------------------------------
# 邻页扩展
# ---------------------------------------------------------------------------


def expand_neighbors(
    coarse_ids: list[str],
    universe_ids: Iterable[str],
    window: int = 1,
    seed_n: int = 8,
) -> list[str]:
    """在 coarse top-N 基础上扩展相邻页面。

    保持 coarse_ids 原始顺序，对前 ``seed_n`` 个页面扩展
    前后各 ``window`` 个邻页。只加入 universe 中存在的页面。
    结果去重且顺序稳定。

    Parameters
    ----------
    coarse_ids : list[str]
        coarse 阶段选出的 top-N page_id 列表（有序）
    universe_ids : Iterable[str]
        候选全集 page_id（邻页必须在此集合内）
    window : int
        每侧扩展窗口大小；0 表示不扩展
    seed_n : int
        对前 seed_n 个 coarse_id 扩展邻页

    Returns
    -------
    list[str]
        扩展后的 page_id 列表（去重，保持顺序）
    """
    if window <= 0 or seed_n <= 0:
        return list(dict.fromkeys(coarse_ids))

    universe_set = set(universe_ids)
    output: list[str] = []
    seen: set[str] = set()

    def _add(pid: str) -> None:
        if pid in universe_set and pid not in seen:
            output.append(pid)
            seen.add(pid)

    for pid in coarse_ids:
        _add(pid)

    for pid in coarse_ids[:seed_n]:
        try:
            parsed = parse_page_id(pid)
        except ValueError:
            continue

        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            neighbor_idx = parsed.page_idx + delta
            if neighbor_idx < 0:
                continue
            _add(make_page_id(parsed.prefix, neighbor_idx))

    return output
