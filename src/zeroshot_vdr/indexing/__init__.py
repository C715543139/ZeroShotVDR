"""索引存储层：页面 embedding 的持久化存储，每页独立文件。"""

from zeroshot_vdr.indexing.store import IndexStore
from zeroshot_vdr.indexing.encoder import PageEncoder

__all__ = [
    "IndexStore",
    "PageEncoder",
]
