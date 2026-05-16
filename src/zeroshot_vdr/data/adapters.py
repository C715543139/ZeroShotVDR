"""
数据集适配器：将不同数据格式转为统一契约（Page / Query / RelevanceJudgment）。

数据来源差异封装在适配器内部，不传播到索引、检索和评测层。
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from zeroshot_vdr.contracts import (
    Page,
    Query,
    RelevanceJudgment,
    build_page_id,
    build_page_id_from_image,
    build_query_id,
    extract_source_doc_id,
    extract_source_page_idx,
    normalize_image_rel_path,
    normalize_doc_id,
)
from zeroshot_vdr.utils import get_project_root

logger = logging.getLogger(__name__)

# MMLongBench 任务族名（数据目录中的小写短名）
TASK_FAMILY = "docqa"

# 长度档位列表
LENGTHS = ["K4", "K8", "K16", "K32", "K64", "K128"]


# ============================================================================
# 适配器基类
# ============================================================================


class BaseAdapter:
    """数据集适配器基类：将不同数据格式转为统一契约。"""

    def iter_pages(self) -> Iterator[Page]:
        """遍历页面。"""
        raise NotImplementedError

    def iter_queries(self) -> Iterator[Query]:
        """遍历查询。"""
        raise NotImplementedError

    def iter_judgments(self) -> Iterator[RelevanceJudgment]:
        """遍历相关性标注。"""
        raise NotImplementedError


# ============================================================================
# DocumentQA 适配器
# ============================================================================


class DocumentQAAdapter(BaseAdapter):
    """MMLongBench DocumentQA 子集适配器。

    输入：mmlb_data/documentQA/ 下的 JSONL 文件（含 page_list + ans_page_list）。
    输出：Page（page_id = "docqa/{subtask}_{length}/{doc_id}/p{idx}"）、
          Query（携带 doc_id 以支持文档内检索协议）、RelevanceJudgment。

    Parameters
    ----------
    data_dir : str
        MMLongBench raw 数据根目录（含 mmlb_data/ 和 mmlb_image/）
    subtasks : list[str] | None
        要加载的子任务列表；None 表示全部（longdocurl, mmlongdoc, slidevqa）
    lengths : list[str] | None
        要加载的长度档位列表；None 表示全部 K4-K128
    """

    def __init__(
        self,
        data_dir: str | None = None,
        subtasks: list[str] | None = None,
        lengths: list[str] | None = None,
    ):
        if data_dir is None:
            data_dir = str(get_project_root() / "data" / "MMLongBench" / "raw")
        self._data_dir = Path(data_dir)
        self._jsonl_dir = self._data_dir / "mmlb_data" / "documentQA"
        self._image_root = self._data_dir / "mmlb_image"

        self._subtasks = subtasks or ["longdocurl", "mmlongdoc", "slidevqa"]
        self._lengths = lengths or LENGTHS

        # 缓存：加载后的样本数据
        self._samples: list[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """加载所有 JSONL 文件到内存。"""
        if self._loaded:
            return

        for subtask in self._subtasks:
            for length in self._lengths:
                jsonl_path = self._jsonl_dir / f"{subtask}_{length}.jsonl"
                if not jsonl_path.exists():
                    logger.warning("JSONL 文件不存在，跳过: %s", jsonl_path)
                    continue

                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        # 注入元信息（文件名已经编码了 subtask/length）
                        sample["_subtask"] = subtask
                        sample["_length"] = length
                        self._samples.append(sample)

        self._loaded = True
        logger.info(
            "DocumentQAAdapter 加载完成: %d 条样本（subtasks=%s, lengths=%s）",
            len(self._samples),
            self._subtasks,
            self._lengths,
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_page_number(image_rel_path: str) -> Optional[int]:
        """从图像相对路径中提取页码。

        支持的模式：
        1. longdocurl/mmlongdoc: ``..._page101.jpg`` → 101
        2. slidevqa: ``...-3-1024.jpg`` → 3（slide 编号在分辨率前）

        Returns
        -------
        int or None
            1-based 页码；无法提取时返回 None
        """
        filename = Path(image_rel_path).stem

        # 模式 1: _page{N} 后缀
        m = re.search(r"_page(\d+)$", filename)
        if m:
            return int(m.group(1))

        # 模式 2: -{N}-{resolution}.{ext} → slidevqa 的 slide 编号
        # 例: "...-want-3-1024" → 3
        m = re.search(r"-(\d+)-\d+$", filename)
        if m:
            return int(m.group(1))

        return None

    def _resolve_image_path(self, image_rel_path: str) -> str:
        """将数据中的相对图像路径解析为绝对路径。"""
        return str(self._image_root / image_rel_path)

    @staticmethod
    def _source_raw_doc_name(image_rel_path: str) -> str:
        normalized_path = normalize_image_rel_path(image_rel_path)
        return Path(normalized_path).parent.name

    @staticmethod
    def _build_candidate_page_ids(
        page_list: list[str],
        subtask: str,
        length: str,
    ) -> tuple[str, ...]:
        candidate_page_ids: list[str] = []
        seen: set[str] = set()
        for fallback_page_idx, rel_path in enumerate(page_list):
            page_id = build_page_id_from_image(
                TASK_FAMILY,
                subtask,
                length,
                rel_path,
                fallback_page_idx=fallback_page_idx,
            )
            if page_id in seen:
                continue
            seen.add(page_id)
            candidate_page_ids.append(page_id)
        return tuple(candidate_page_ids)

    @staticmethod
    def _build_page_number_to_ids(
        page_list: list[str],
        subtask: str,
        length: str,
    ) -> dict[int, list[tuple[str, str]]]:
        page_number_to_ids: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for fallback_page_idx, rel_path in enumerate(page_list):
            source_page_idx = extract_source_page_idx(
                rel_path,
                fallback_page_idx=fallback_page_idx,
            )
            source_doc_id = extract_source_doc_id(rel_path)
            page_id = build_page_id(
                TASK_FAMILY,
                subtask,
                length,
                source_doc_id,
                source_page_idx,
            )
            page_number_to_ids[source_page_idx].append((page_id, source_doc_id))
        return page_number_to_ids

    @staticmethod
    def _build_page_number_map(page_list: list[str]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """构建 pages 的页码映射。

        Returns
        -------
        page_num_to_idx : dict
            {1-based page number → 0-based page_idx}
        idx_to_page_num : dict
            {0-based page_idx → 1-based page number}
        """
        page_num_to_idx: dict[int, int] = {}
        idx_to_page_num: dict[int, int] = {}

        for idx, rel_path in enumerate(page_list):
            pn = DocumentQAAdapter._extract_page_number(rel_path)
            if pn is not None:
                page_num_to_idx[pn] = idx
                idx_to_page_num[idx] = pn

        return page_num_to_idx, idx_to_page_num

    # ------------------------------------------------------------------
    # 契约产出
    # ------------------------------------------------------------------

    def iter_pages(self) -> Iterator[Page]:
        """遍历 DocumentQA 中的所有唯一页面。"""
        self._load()

        seen: set[str] = set()

        for sample in self._samples:
            subtask = sample["_subtask"]
            length = sample["_length"]

            for fallback_page_idx, rel_path in enumerate(sample.get("page_list", [])):
                source_raw_doc_name = self._source_raw_doc_name(rel_path)
                source_doc_id = extract_source_doc_id(rel_path)
                source_page_idx = extract_source_page_idx(
                    rel_path,
                    fallback_page_idx=fallback_page_idx,
                )
                page_id = build_page_id(
                    TASK_FAMILY,
                    subtask,
                    length,
                    source_doc_id,
                    source_page_idx,
                )

                if page_id in seen:
                    continue
                seen.add(page_id)

                image_path = self._resolve_image_path(rel_path)

                yield Page(
                    page_id=page_id,
                    doc_id=source_doc_id,
                    raw_doc_name=source_raw_doc_name,
                    task_family=TASK_FAMILY,
                    subtask=subtask,
                    length=length,
                    page_idx=source_page_idx,
                    image_path=image_path,
                )

    def iter_queries(self) -> Iterator[Query]:
        """遍历 DocumentQA 中的所有查询。"""
        self._load()

        for sample in self._samples:
            subtask = sample["_subtask"]
            length = sample["_length"]
            raw_doc_name = sample.get("doc_name", "")
            doc_id = normalize_doc_id(raw_doc_name)

            # 使用原始 id 的最后数字部分作为 query_index
            raw_id = sample.get("id", "")
            # 尝试从 raw_id 中提取序号（如 "longdocurl_16" → 16）
            query_index = self._parse_query_index(raw_id, subtask)

            query_id = build_query_id(TASK_FAMILY, subtask, length, query_index)
            question = sample.get("question", "")
            candidate_page_ids = self._build_candidate_page_ids(
                sample.get("page_list", []),
                subtask,
                length,
            )

            yield Query(
                query_id=query_id,
                text=question,
                doc_id=doc_id,
                raw_doc_name=raw_doc_name,
                task_family=TASK_FAMILY,
                subtask=subtask,
                length=length,
                candidate_page_ids=candidate_page_ids,
            )

    def iter_judgments(self) -> Iterator[RelevanceJudgment]:
        """遍历 DocumentQA 中的所有相关性标注。

        ans_page_list 给出 1-based 原始页码，我们映射到 page_idx。
        """
        self._load()

        for sample in self._samples:
            subtask = sample["_subtask"]
            length = sample["_length"]
            raw_doc_name = sample.get("doc_name", "")
            doc_id = normalize_doc_id(raw_doc_name)
            page_list = sample.get("page_list", [])
            ans_page_list = sample.get("ans_page_list", [])

            raw_id = sample.get("id", "")
            query_index = self._parse_query_index(raw_id, subtask)
            query_id = build_query_id(TASK_FAMILY, subtask, length, query_index)

            page_number_to_ids = self._build_page_number_to_ids(page_list, subtask, length)
            preferred_doc_id = normalize_doc_id(raw_doc_name)
            seen_pairs: set[tuple[str, str]] = set()

            for ans_page_num in ans_page_list:
                candidates = page_number_to_ids.get(ans_page_num, [])
                if not candidates:
                    continue

                preferred_candidates = [
                    page_id
                    for page_id, source_doc_id in candidates
                    if source_doc_id == preferred_doc_id
                ]
                resolved_candidates = preferred_candidates or [
                    page_id for page_id, _ in candidates
                ]

                for page_id in resolved_candidates:
                    pair = (query_id, page_id)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    yield RelevanceJudgment(
                        query_id=query_id,
                        page_id=page_id,
                        relevance=1,
                    )

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------

    def build_ground_truth(self) -> dict[str, set[str]]:
        """构建 {query_id: {relevant_page_id, ...}} 的 ground truth 字典。"""
        gt: dict[str, set[str]] = {}
        for j in self.iter_judgments():
            gt.setdefault(j.query_id, set()).add(j.page_id)
        return gt

    @staticmethod
    def _parse_query_index(raw_id: str, subtask: str) -> int:
        """从 raw_id 中提取查询序号。

        样例:
            "longdocurl_16" → 16
            "mmlongdoc_42" → 42
            "slidevqa_7"   → 7

        若无法解析，使用 hash 取模生成稳定序号。
        """
        # 尝试剥离 subtask 前缀
        if raw_id.startswith(subtask):
            suffix = raw_id[len(subtask):].lstrip("_")
            if suffix.isdigit():
                return int(suffix)

        # 尝试末尾数字
        m = re.search(r"(\d+)$", raw_id)
        if m:
            return int(m.group(1))

        # 回退：hash
        return hash(raw_id) % 1000000

    @property
    def stats(self) -> dict:
        """返回数据统计信息。"""
        self._load()
        pages = list(self.iter_pages())
        queries = list(self.iter_queries())
        judgments = list(self.iter_judgments())

        # 按子任务统计
        by_subtask: dict[str, dict] = {}
        for p in pages:
            s = by_subtask.setdefault(
                p.subtask,
                {"pages": 0, "queries": 0, "docs": set()},
            )
            s["pages"] += 1
            s["docs"].add(p.doc_id)
        for q in queries:
            s = by_subtask.setdefault(
                q.subtask,
                {"pages": 0, "queries": 0, "docs": set()},
            )
            s["queries"] += 1

        return {
            "total_pages": len(pages),
            "total_queries": len(queries),
            "total_judgments": len(judgments),
            "by_subtask": {
                k: {
                    "pages": v["pages"],
                    "queries": v["queries"],
                    "docs": len(v["docs"]),
                }
                for k, v in by_subtask.items()
            },
        }


# ============================================================================
# PDF 适配器（占位，Phase 2 后续补充）
# ============================================================================


class PDFAdapter(BaseAdapter):
    """PDF 页面渲染适配器（基于 pypdfium2）。

    用于从原始 PDF 文件构建页面语料（非 DocumentQA 路径时使用）。
    Phase 2 当前以 DocumentQA 为优先路径，PDF 适配器预留接口。

    Parameters
    ----------
    pdf_dir : str
        PDF 文件所在目录
    output_dir : str
        渲染图像输出目录
    target_size : tuple[int, int]
        送入 ColPali 前的图像尺寸
    scale : float
        渲染缩放因子
    """

    def __init__(
        self,
        pdf_dir: str,
        output_dir: str,
        target_size: tuple[int, int] = (672, 672),
        scale: float = 2.0,
    ):
        self._pdf_dir = Path(pdf_dir)
        self._output_dir = Path(output_dir)
        self._target_size = target_size
        self._scale = scale
        self._task_family = "pdf"

    def iter_pages(self) -> Iterator[Page]:
        """遍历 PDF 渲染页面。

        .. note::
           当前为占位实现，Phase 2 后续补充完整的 PDF→图像 渲染逻辑。
        """
        # 占位：Phase 2 后续补充
        return
        yield  # 使此方法成为 generator（类型标注兼容）

    def iter_queries(self) -> Iterator[Query]:
        """PDF 适配器无查询。"""
        return
        yield

    def iter_judgments(self) -> Iterator[RelevanceJudgment]:
        """PDF 适配器无标注。"""
        return
        yield
