"""Phase 4 Mean-Pool Cache: 缓存页面的 mean-pooled embedding，减少 coarse 阶段磁盘 I/O。

使用场景：
    每个 query 在 coarse 阶段需要加载大量页面的 patch embeddings
    并做 mean pooling。Cache 将这一步结果持久化到磁盘，后续 query
    直接读取预计算的 mean-pooled vectors。
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class MeanPoolCache:
    """页面 mean-pooled embedding 磁盘缓存。

    存储结构::

        {cache_dir}/
        ├── page_ids.json   # 页面 ID 列表（保持顺序）
        ├── page_means.pt   # mean-pooled embeddings [n_pages, dim]
        └── meta.json       # 元信息（维度、页数、创建时间等）

    Parameters
    ----------
    cache_dir : str | Path
        缓存存储目录
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.page_ids_path = self.cache_dir / "page_ids.json"
        self.embeddings_path = self.cache_dir / "page_means.pt"
        self.meta_path = self.cache_dir / "meta.json"

        self.page_ids: list[str] | None = None
        self.embeddings: torch.Tensor | None = None
        self.id_to_idx: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # 存在性检查
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """检查三个缓存文件是否都存在。"""
        return (
            self.page_ids_path.exists()
            and self.embeddings_path.exists()
            and self.meta_path.exists()
        )

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def save(
        self,
        page_ids: list[str],
        embeddings: torch.Tensor,
        meta: dict | None = None,
    ) -> None:
        """保存 mean-pooled embeddings 到缓存。

        Parameters
        ----------
        page_ids : list[str]
            页面 ID 有序列表
        embeddings : torch.Tensor
            shape ``[n_pages, dim]``
        meta : dict | None
            额外元信息；为 None 时自动生成基础信息
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.page_ids_path.write_text(
            json.dumps(page_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        torch.save(embeddings.cpu(), self.embeddings_path)

        if meta is None:
            meta = {}
        meta.setdefault("num_pages", len(page_ids))
        meta.setdefault("embedding_dim", int(embeddings.shape[-1]))
        meta.setdefault("dtype", str(embeddings.dtype))

        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "MeanPoolCache 已保存: %d 页, dim=%d → %s",
            len(page_ids),
            embeddings.shape[-1],
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self, map_location: str = "cpu") -> None:
        """从缓存加载。

        Parameters
        ----------
        map_location : str
            torch.load 的 map_location 参数

        Raises
        ------
        FileNotFoundError
            若缓存文件不存在
        """
        if not self.exists():
            raise FileNotFoundError(f"MeanPoolCache 不完整: {self.cache_dir}")

        self.page_ids = json.loads(
            self.page_ids_path.read_text(encoding="utf-8")
        )
        self.embeddings = torch.load(
            self.embeddings_path,
            map_location=map_location,
            weights_only=True,
        )
        self.id_to_idx = {pid: i for i, pid in enumerate(self.page_ids)}

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        logger.info(
            "MeanPoolCache 已加载: %d 页, dim=%d ← %s",
            meta.get("num_pages", len(self.page_ids)),
            meta.get("embedding_dim", self.embeddings.shape[-1]),
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, page_ids: list[str]) -> torch.Tensor:
        """按 page_ids 顺序返回对应的 mean-pooled embeddings。

        Parameters
        ----------
        page_ids : list[str]
            请求的页面 ID 列表（保持输入顺序）

        Returns
        -------
        torch.Tensor
            shape ``[len(page_ids), dim]``

        Raises
        ------
        RuntimeError
            若缓存未加载
        KeyError
            若某个 page_id 不在缓存中
        """
        if self.embeddings is None or self.id_to_idx is None:
            raise RuntimeError("MeanPoolCache 未加载，请先调用 load()")

        indices = [self.id_to_idx[pid] for pid in page_ids]
        return self.embeddings[indices]

    def __contains__(self, page_id: str) -> bool:
        """检查 page_id 是否在缓存中。"""
        if self.id_to_idx is None:
            self.load()
        return page_id in self.id_to_idx

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------

    def load_meta(self) -> dict:
        """加载缓存元信息（仅读取 meta.json）。"""
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))


# ===========================================================================
# 构建辅助
# ===========================================================================


def build_mean_pool_cache(
    index_store,
    page_ids: list[str],
    cache: MeanPoolCache,
    meta: dict | None = None,
    batch_size: int = 1024,
) -> None:
    """从 IndexStore 构建 MeanPoolCache。

    Parameters
    ----------
    index_store : IndexStore
    page_ids : list[str]
        要缓存的页面 ID 列表
    cache : MeanPoolCache
        目标缓存对象
    meta : dict | None
        额外元信息
    batch_size : int
        每批处理的页面数，避免全量构建时占用过高内存
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")

    logger.info(
        "开始构建 MeanPoolCache: %d 页 (batch_size=%d)...",
        len(page_ids),
        batch_size,
    )

    mean_chunks: list[torch.Tensor] = []
    cached_page_ids: list[str] = []
    total_batches = (len(page_ids) + batch_size - 1) // batch_size if page_ids else 0

    for batch_index, start in enumerate(range(0, len(page_ids), batch_size), start=1):
        batch_page_ids = page_ids[start:start + batch_size]
        batch_embeddings, batch_pids = index_store.get_mean_pooled_view(batch_page_ids)
        if batch_embeddings.numel() == 0:
            continue

        mean_chunks.append(batch_embeddings.cpu())
        cached_page_ids.extend(batch_pids)

        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            logger.info(
                "MeanPoolCache 构建进度: batch %d/%d (%d 页)",
                batch_index,
                total_batches,
                len(cached_page_ids),
            )

        del batch_embeddings
        gc.collect()

    if mean_chunks:
        embeddings = torch.cat(mean_chunks, dim=0)
    else:
        embeddings = torch.empty((0, 0))

    if meta is not None:
        meta = dict(meta)
        meta.setdefault("build_batch_size", batch_size)

    cache.save(cached_page_ids, embeddings, meta=meta)
    logger.info("MeanPoolCache 构建完成")
