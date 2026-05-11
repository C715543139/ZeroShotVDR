"""数据接入层：从 MMLongBench 等数据源构建统一的页面语料与查询集。"""

from zeroshot_vdr.data.adapters import BaseAdapter, DocumentQAAdapter, PDFAdapter
from zeroshot_vdr.data.corpus import PageCorpus

__all__ = [
    "BaseAdapter",
    "DocumentQAAdapter",
    "PDFAdapter",
    "PageCorpus",
]
