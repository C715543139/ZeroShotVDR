"""Phase 4 进阶方法模块：query-adaptive two-stage coarse-to-fine retrieval。

子模块：
- two_stage: TwoStageRetriever 与 adaptive top-N 工具
- neighbors: page_id 解析与邻页扩展
- mean_pool_cache: mean-pooled embedding 磁盘缓存
- profiling: per-query trace 与 slice-level 分析
"""

from zeroshot_vdr.advanced.two_stage import (
    TwoStageRetriever,
    TwoStageOutput,
    TwoStageTrace,
    choose_adaptive_top_n,
    resolve_candidate_universe,
)
from zeroshot_vdr.advanced.neighbors import (
    parse_page_id,
    make_page_id,
    expand_neighbors,
    ParsedPageId,
)
from zeroshot_vdr.advanced.mean_pool_cache import (
    MeanPoolCache,
    build_mean_pool_cache,
)

__all__ = [
    # two_stage
    "TwoStageRetriever",
    "TwoStageOutput",
    "TwoStageTrace",
    "choose_adaptive_top_n",
    "resolve_candidate_universe",
    # neighbors
    "parse_page_id",
    "make_page_id",
    "expand_neighbors",
    "ParsedPageId",
    # mean_pool_cache
    "MeanPoolCache",
    "build_mean_pool_cache",
]
