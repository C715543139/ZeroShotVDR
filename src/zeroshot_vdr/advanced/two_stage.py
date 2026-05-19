"""Phase 4 Two-Stage Retriever: mean-pool coarse → MaxSim rerank。

核心流程:
    Query.candidate_page_ids → universe
    → mean-pool coarse retrieval
    → adaptive top-N selection
    → optional neighbor expansion
    → full MaxSim rerank
    → final top-k results
"""

# 占位；具体实现在后续 Stage 中添加
