"""检索执行层：查询编码 → 候选召回 → MaxSim 精排 → Top-k 结果组装。"""

from zeroshot_vdr.retrieval.encoder import QueryEncoder
from zeroshot_vdr.retrieval.scoring import maxsim_score, batched_maxsim
from zeroshot_vdr.retrieval.pipeline import RetrievalPipeline

__all__ = [
    "QueryEncoder",
    "maxsim_score",
    "batched_maxsim",
    "RetrievalPipeline",
]
