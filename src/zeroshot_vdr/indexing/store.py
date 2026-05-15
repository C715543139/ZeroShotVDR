"""
索引持久化存储。

存储布局::

    {index_dir}/
    ├── pages/
    │   ├── {page_id}.pt    # torch.Tensor [n_patches, dim]
    │   └── ...
    ├── page_ids.json       # [page_id, ...] 有序列表
    └── index_meta.json     # 模型名、维度、时间戳、总页数
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)


class IndexStore:
    """索引持久化存储——按页独立文件。

    核心设计：
    - 每页一个 ``{page_id}.pt`` 文件，天然支持增量追加和变长 patch
    - ``page_ids.json`` 维护全局有序页面列表
    - ``index_meta.json`` 记录索引元信息

    Parameters
    ----------
    index_dir : str
        索引存储根目录
    """

    def __init__(self, index_dir: str):
        self._index_dir = Path(index_dir)
        self._pages_dir = self._index_dir / "pages"
        self._page_ids_file = self._index_dir / "page_ids.json"
        self._meta_file = self._index_dir / "index_meta.json"

        # 延迟创建目录，在首次写入时自动创建
        self._page_ids: list[str] | None = None  # 缓存
        self._page_ids_set: set[str] | None = None

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """确保存储目录存在。"""
        self._pages_dir.mkdir(parents=True, exist_ok=True)

    def _page_path(self, page_id: str) -> Path:
        """获取页面 embedding 文件路径。"""
        # page_id 含 "/" 字符（如 "docqa/longdocurl_K4/4088173/p0"），
        # 将其转为文件系统安全的路径分隔符
        safe_id = page_id.replace("/", "_").replace("\\", "_")
        return self._pages_dir / f"{safe_id}.pt"

    def _write_embedding_file(self, page_id: str, embedding: torch.Tensor) -> None:
        """仅写入页面 embedding 文件，不更新 page_ids 清单。"""
        self._ensure_dirs()
        torch.save(embedding, self._page_path(page_id))

    def _load_page_ids(self) -> list[str]:
        """加载或初始化 page_ids 列表。"""
        if self._page_ids is not None:
            return self._page_ids
        if self._page_ids_file.exists():
            with open(self._page_ids_file, "r", encoding="utf-8") as f:
                self._page_ids = json.load(f)
        else:
            self._page_ids = []
        self._page_ids_set = set(self._page_ids)
        return self._page_ids

    def _save_page_ids(self) -> None:
        """保存 page_ids 列表到文件。"""
        self._ensure_dirs()
        ids = self._load_page_ids()
        with open(self._page_ids_file, "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)

    def _append_page_ids(self, page_ids: list[str]) -> bool:
        """批量追加 page_ids，并在有新增时一次性落盘。"""
        ids = self._load_page_ids()
        ids_set = self._page_ids_set
        if ids_set is None:
            ids_set = set(ids)
            self._page_ids_set = ids_set

        added = False
        for page_id in page_ids:
            if page_id in ids_set:
                continue
            ids.append(page_id)
            ids_set.add(page_id)
            added = True

        if added:
            self._save_page_ids()

        return added

    # ------------------------------------------------------------------
    # 核心写入接口
    # ------------------------------------------------------------------

    def write_page(
        self,
        page_id: str,
        embedding: torch.Tensor,
        update_manifest: bool = True,
    ) -> None:
        """写入单页 embedding → ``{page_id}.pt``。

        Parameters
        ----------
        page_id : str
            稳定的页面标识
        embedding : torch.Tensor
            shape ``[n_patches, dim]``
        """
        self._write_embedding_file(page_id, embedding)

        if update_manifest:
            self._append_page_ids([page_id])

    def write_batch(
        self,
        page_ids: list[str],
        embeddings: torch.Tensor,
        update_manifest: bool = True,
    ) -> None:
        """批量写入页面 embeddings。

        Parameters
        ----------
        page_ids : list[str]
        embeddings : torch.Tensor
            shape ``[batch, n_patches, dim]``
        """
        if len(page_ids) != embeddings.shape[0]:
            raise ValueError(
                f"page_ids 数量 ({len(page_ids)}) 与 embeddings batch 维度 "
                f"({embeddings.shape[0]}) 不匹配"
            )

        new_page_ids: list[str] = []
        for pid, emb in zip(page_ids, embeddings):
            self._write_embedding_file(pid, emb.cpu())
            new_page_ids.append(pid)

        if update_manifest:
            self._append_page_ids(new_page_ids)

    def register_page_ids(self, page_ids: list[str]) -> None:
        """批量注册已存在于磁盘中的页面 ID。"""
        self._append_page_ids(page_ids)

    def has_page(self, page_id: str) -> bool:
        """检查某个 page_id 对应的索引文件是否已存在。"""
        return self._page_path(page_id).exists()

    def recover_page_ids(self, candidate_page_ids: list[str]) -> int:
        """从已存在的页面文件中恢复缺失的 page_ids 清单。

        典型场景是多进程索引构建时，页面文件已写盘，但父进程在
        统一落盘 ``page_ids.json`` 之前被中断。

        Parameters
        ----------
        candidate_page_ids : list[str]
            可能已存在于磁盘上的 page_id 候选列表。

        Returns
        -------
        int
            本次新恢复并写入 manifest 的 page_id 数量。
        """
        existing_ids = set(self._load_page_ids())
        recovered = [
            page_id
            for page_id in candidate_page_ids
            if page_id not in existing_ids and self.has_page(page_id)
        ]

        if not recovered:
            return 0

        self._append_page_ids(recovered)
        return len(recovered)

    # ------------------------------------------------------------------
    # 核心读取接口
    # ------------------------------------------------------------------

    def read_page(self, page_id: str) -> torch.Tensor:
        """读取单页 embedding → ``[n_patches, dim]``。

        Raises
        ------
        FileNotFoundError
            若页面不存在于索引中
        """
        path = self._page_path(page_id)
        if not path.exists():
            raise FileNotFoundError(f"页面不在索引中: {page_id}")
        return torch.load(path, weights_only=True)

    def iter_pages(
        self, page_ids: list[str] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """按页迭代读取，兼容变长 patch 场景。

        Parameters
        ----------
        page_ids : list[str] | None
            要读取的页面列表；None 表示全部页面。

        Yields
        ------
        tuple[str, torch.Tensor]
            ``(page_id, embedding [n_patches, dim])``
        """
        ids = page_ids if page_ids is not None else self._load_page_ids()
        for pid in ids:
            try:
                emb = self.read_page(pid)
                yield pid, emb
            except FileNotFoundError:
                logger.warning("索引文件缺失，跳过: %s", pid)

    @staticmethod
    def _split_page_id(page_id: str) -> tuple[str, str, str, str] | None:
        """解析 page_id 的层级字段。

        page_id 规范格式为 ``{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}``。
        若格式不符合预期，返回 None。
        """
        parts = page_id.split("/")
        if len(parts) != 4:
            return None

        task_family, subtask_length, doc_id, page_part = parts
        if "_" not in subtask_length:
            return None

        subtask, length = subtask_length.rsplit("_", 1)
        return task_family, subtask, length, doc_id

    def list_page_ids(
        self,
        doc_id: str | None = None,
        task_family: str | None = None,
        subtask: str | None = None,
        length: str | None = None,
    ) -> list[str]:
        """列出索引中的页面 ID。

        Parameters
        ----------
        doc_id : str | None
            限定文档；None 表示不按文档过滤。
        task_family : str | None
            限定任务族；None 表示不按任务族过滤。
        subtask : str | None
            限定子任务；None 表示不按子任务过滤。
        length : str | None
            限定长度档位；None 表示不按长度过滤。

        Returns
        -------
        list[str]
            满足所有过滤条件的 page_id 列表。
        """
        ids = self._load_page_ids()
        if all(value is None for value in (doc_id, task_family, subtask, length)):
            return list(ids)

        filtered: list[str] = []
        for pid in ids:
            parsed = self._split_page_id(pid)
            if parsed is None:
                logger.warning("无法解析 page_id，跳过过滤匹配: %s", pid)
                continue

            pid_task_family, pid_subtask, pid_length, pid_doc_id = parsed

            if task_family is not None and pid_task_family != task_family:
                continue
            if subtask is not None and pid_subtask != subtask:
                continue
            if length is not None and pid_length != length:
                continue
            if doc_id is not None and pid_doc_id != doc_id:
                continue

            filtered.append(pid)

        return filtered

    # ------------------------------------------------------------------
    # 视图接口
    # ------------------------------------------------------------------

    def get_mean_pooled_view(
        self, page_ids: list[str] | None = None
    ) -> tuple[torch.Tensor, list[str]]:
        """返回页面的均值池化向量 ``[n_pages, dim]``，供两阶段粗筛使用。

        Parameters
        ----------
        page_ids : list[str] | None
            限定页面；None 表示全部。

        Returns
        -------
        tuple[torch.Tensor, list[str]]
            ``(mean_pooled_embeddings [n_pages, dim], page_id_list)``
        """
        vectors: list[torch.Tensor] = []
        pids: list[str] = []

        for pid, emb in self.iter_pages(page_ids):
            vectors.append(emb.mean(dim=0))
            pids.append(pid)

        if not vectors:
            return torch.empty((0, 0)), []

        return torch.stack(vectors), pids

    # ------------------------------------------------------------------
    # baseline 便利函数（非核心抽象）
    # ------------------------------------------------------------------

    def read_stacked(
        self, page_ids: list[str]
    ) -> tuple[torch.Tensor, list[str]]:
        """返回 ``(stacked_tensor, page_id_list)``。

        仅在所有页面 patch 数一致时使用的便利函数。
        变长 patch 场景请使用 ``iter_pages()``。

        Returns
        -------
        tuple[torch.Tensor, list[str]]
            stacked_tensor shape ``[n_pages, n_patches, dim]``
        """
        valid_pids: list[str] = []
        tensors: list[torch.Tensor] = []

        for pid, emb in self.iter_pages(page_ids):
            valid_pids.append(pid)
            tensors.append(emb)

        if not tensors:
            return torch.empty((0,)), []

        return torch.stack(tensors), valid_pids

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------

    def save_meta(self, model_name: str, dim: int) -> None:
        """保存索引元信息。

        Parameters
        ----------
        model_name : str
            用于生成索引的模型名称
        dim : int
            embedding 维度
        """
        self._ensure_dirs()
        ids = self._load_page_ids()
        meta = {
            "model_name": model_name,
            "dim": dim,
            "num_pages": len(ids),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "storage_format": "per_page_pt",
        }
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load_meta(self) -> dict:
        """加载索引元信息。"""
        if not self._meta_file.exists():
            return {
                "model_name": "unknown",
                "dim": 0,
                "num_pages": 0,
                "created_at": "N/A",
            }
        with open(self._meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def stats(self) -> dict:
        """返回索引统计信息。"""
        ids = self._load_page_ids()
        meta = self.load_meta()

        # 计算存储大小
        total_size = 0
        if self._pages_dir.exists():
            for pid in ids:
                path = self._page_path(pid)
                if path.exists():
                    total_size += path.stat().st_size

        return {
            "num_pages": len(ids),
            "dim": meta.get("dim", 0),
            "total_size_mb": total_size / (1024 * 1024),
            "storage_dir": str(self._index_dir),
            "created_at": meta.get("created_at", "N/A"),
        }

    @property
    def index_dir(self) -> Path:
        """索引根目录路径。"""
        return self._index_dir
