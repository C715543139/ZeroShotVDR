"""
页面语料聚合器：聚合多个 Adapter 产出的 Page，分配稳定 page_id，持久化元信息。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from zeroshot_vdr.contracts import Page
from zeroshot_vdr.data.adapters import BaseAdapter
from zeroshot_vdr.utils import get_project_root, resolve_path

logger = logging.getLogger(__name__)


class PageCorpus:
    """页面语料聚合器。

    职责：
    1. 聚合多个 Adapter 产出的 Page
    2. 确保每个 Page 的 page_id 稳定一致
    3. 持久化 corpus_meta.json
    4. 提供加载和查询接口

    Parameters
    ----------
    config : dict | None
        全局配置字典；为 None 时从 config/default.yaml 加载
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            from zeroshot_vdr.config import load_config

            config = load_config()

        self._config = config
        self._pages: list[Page] = []
        self._page_index: dict[str, Page] = {}  # page_id → Page
        self._doc_index: dict[str, list[str]] = {}  # doc_id → [page_id, ...]
        self._built = False

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def build(self, adapters: list[BaseAdapter]) -> list[Page]:
        """从多个适配器构建统一页面语料。

        遍历所有适配器的 iter_pages()，收集唯一页面（按 page_id 去重）。

        Parameters
        ----------
        adapters : list[BaseAdapter]
            适配器列表（如 [DocumentQAAdapter(...)]）

        Returns
        -------
        list[Page]
            去重后的页面列表
        """
        seen: set[str] = set()
        pages: list[Page] = []

        for adapter in adapters:
            for page in adapter.iter_pages():
                if page.page_id in seen:
                    logger.debug("跳过重复页面: %s", page.page_id)
                    continue
                seen.add(page.page_id)
                pages.append(page)

        # 按 page_id 排序（保证确定性）
        pages.sort(key=lambda p: p.page_id)

        self._pages = pages
        self._page_index = {p.page_id: p for p in pages}

        # 构建文档索引
        self._doc_index.clear()
        for p in pages:
            self._doc_index.setdefault(p.doc_id, []).append(p.page_id)

        # 每个 doc_id 内的页面按 page_idx 排序
        for doc_id in self._doc_index:
            self._doc_index[doc_id].sort(
                key=lambda pid: self._page_index[pid].page_idx
            )

        self._built = True
        logger.info(
            "PageCorpus 构建完成: %d 页面, %d 文档",
            len(pages),
            len(self._doc_index),
        )

        return pages

    def build_from_adapter(self, adapter: BaseAdapter) -> list[Page]:
        """从单个适配器构建（便捷方法）。"""
        return self.build([adapter])

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def pages(self) -> list[Page]:
        """返回所有页面列表。"""
        if not self._built:
            raise RuntimeError("PageCorpus 尚未构建，请先调用 build()")
        return self._pages

    def get_page(self, page_id: str) -> Page | None:
        """按 page_id 查询页面。"""
        return self._page_index.get(page_id)

    def get_doc_pages(self, doc_id: str) -> list[str]:
        """按 doc_id 获取其所有 page_id 列表（按 page_idx 排序）。"""
        return self._doc_index.get(doc_id, [])

    @property
    def num_pages(self) -> int:
        """页面总数。"""
        return len(self._pages)

    @property
    def num_docs(self) -> int:
        """文档总数。"""
        return len(self._doc_index)

    @property
    def doc_ids(self) -> list[str]:
        """所有文档 ID 列表。"""
        return sorted(self._doc_index.keys())

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save_metadata(self, path: str | None = None) -> str:
        """保存 corpus_meta.json。

        Parameters
        ----------
        path : str | None
            输出路径；为 None 时使用配置中的 corpus_meta 路径

        Returns
        -------
        str
            实际写入的文件路径
        """
        if not self._built:
            raise RuntimeError("PageCorpus 尚未构建，请先调用 build()")

        if path is None:
            meta_path = self._config.get("paths", {}).get(
                "corpus_meta", "data/processed/corpus_meta.json"
            )
            path = str(resolve_path(meta_path))

        output_file = Path(path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for p in self._pages:
            records.append(
                {
                    "page_id": p.page_id,
                    "doc_id": p.doc_id,
                    "raw_doc_name": p.raw_doc_name,
                    "task_family": p.task_family,
                    "subtask": p.subtask,
                    "length": p.length,
                    "page_idx": p.page_idx,
                    "image_path": p.image_path,
                }
            )

        meta = {
            "num_pages": len(self._pages),
            "num_docs": len(self._doc_index),
            "pages": records,
            "doc_index": {
                doc_id: page_ids
                for doc_id, page_ids in self._doc_index.items()
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("corpus_meta.json 已保存: %s (%d 页面)", output_file, len(self._pages))
        return str(output_file)

    @classmethod
    def load_metadata(cls, path: str) -> list[Page]:
        """从 corpus_meta.json 加载还原 Page 列表。

        Parameters
        ----------
        path : str
            corpus_meta.json 文件路径

        Returns
        -------
        list[Page]
        """
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        pages = []
        for record in meta.get("pages", []):
            pages.append(
                Page(
                    page_id=record["page_id"],
                    doc_id=record["doc_id"],
                    raw_doc_name=record.get("raw_doc_name"),
                    task_family=record["task_family"],
                    subtask=record["subtask"],
                    length=record["length"],
                    page_idx=record["page_idx"],
                    image_path=record["image_path"],
                )
            )

        return pages

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """返回语料统计信息。"""
        if not self._built:
            return {"built": False}

        by_subtask: dict[str, dict] = {}
        by_length: dict[str, int] = {}

        for p in self._pages:
            s = by_subtask.setdefault(p.subtask, {"pages": 0, "docs": set()})
            s["pages"] += 1
            s["docs"].add(p.doc_id)

            by_length[p.length] = by_length.get(p.length, 0) + 1

        return {
            "built": True,
            "num_pages": len(self._pages),
            "num_docs": len(self._doc_index),
            "by_subtask": {
                k: {"pages": v["pages"], "docs": len(v["docs"])}
                for k, v in by_subtask.items()
            },
            "by_length": by_length,
        }
