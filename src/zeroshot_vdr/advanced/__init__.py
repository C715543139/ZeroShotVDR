"""Phase 4 进阶方法模块：query-adaptive two-stage coarse-to-fine retrieval。

子模块：
- two_stage: TwoStageRetriever 与 adaptive top-N 工具
- neighbors: page_id 解析与邻页扩展
- mean_pool_cache: mean-pooled embedding 磁盘缓存
- profiling: per-query trace 与 slice-level 分析
"""

# 各子模块在对应 Stage 中逐步实现，导入随实现推进逐步开放。
from zeroshot_vdr.advanced import neighbors  # noqa: F401

# 以下导入在对应模块实现后开放：
# from zeroshot_vdr.advanced.two_stage import TwoStageRetriever, TwoStageOutput, TwoStageTrace  # Stage 4
# from zeroshot_vdr.advanced.profiling import ...  # Stage 9
