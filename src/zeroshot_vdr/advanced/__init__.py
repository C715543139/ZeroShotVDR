"""Phase 4 进阶方法模块：query-adaptive two-stage coarse-to-fine retrieval。

子模块：
- two_stage: TwoStageRetriever 与 adaptive top-N 工具
- neighbors: page_id 解析与邻页扩展
- mean_pool_cache: mean-pooled embedding 磁盘缓存
- profiling: per-query trace 与 slice-level 分析
"""

from zeroshot_vdr.advanced.two_stage import TwoStageRetriever, TwoStageOutput, TwoStageTrace

__all__ = [
    "TwoStageRetriever",
    "TwoStageOutput",
    "TwoStageTrace",
]
