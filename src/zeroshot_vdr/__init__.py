"""ZeroShotVDR research package."""

from zeroshot_vdr.contracts import (
    Page,
    Query,
    RelevanceJudgment,
    RetrievalResult,
    build_page_id,
    build_query_id,
    normalize_doc_id,
)

__all__: list[str] = [
    "Page",
    "Query",
    "RelevanceJudgment",
    "RetrievalResult",
    "build_page_id",
    "build_query_id",
    "normalize_doc_id",
]