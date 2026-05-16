"""ZeroShotVDR research package."""

from zeroshot_vdr.contracts import (
    Page,
    Query,
    RelevanceJudgment,
    RetrievalResult,
    build_page_id,
    build_page_id_from_image,
    build_query_id,
    extract_source_doc_id,
    extract_source_page_idx,
    normalize_image_rel_path,
    normalize_doc_id,
)
from zeroshot_vdr.retrieval import (
    QueryEncoder,
    RetrievalPipeline,
    batched_maxsim,
    maxsim_score,
)
from zeroshot_vdr.evaluation import (
    GroundTruthLoader,
    compute_all_metrics,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__: list[str] = [
    "Page",
    "Query",
    "RelevanceJudgment",
    "RetrievalResult",
    "build_page_id",
    "build_page_id_from_image",
    "build_query_id",
    "extract_source_doc_id",
    "extract_source_page_idx",
    "normalize_image_rel_path",
    "normalize_doc_id",
    "QueryEncoder",
    "RetrievalPipeline",
    "batched_maxsim",
    "maxsim_score",
    "GroundTruthLoader",
    "compute_all_metrics",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]