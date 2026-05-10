"""
Step 2.1 测试的 Pytest 共享夹具。

夹具数据基于真实的 MMLongBench DocumentQA JSONL 格式：
  {"id": ..., "doc_name": ..., "question": ..., "answer": ...,
   "ans_page_list": [...], "answer_sources": [...], "answer_format": ...,
   "page_list": [...], "length": ...}

真实的 DocumentQAAdapter 期望 `data_dir` 包含以下内容：
  data_dir/mmlb_data/documentQA/{subtask}_{length}.jsonl
  data_dir/mmlb_image/{rel_image_path}

夹具在 tmp_path 内创建此目录结构。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 匹配真实 MMLongBench DocumentQA 格式的原始 JSONL 记录
# ---------------------------------------------------------------------------

# ---- longdocurl_K4 记录 ----
# 图片文件名："_page{N}.jpg"，其中 N 是基于 0 的页面索引。
# ans_page_list 使用基于 1 的页码，与提取的 _pageN 值匹配。
LONGDOCURL_K4_RECORDS = [
    {
        "id": "longdocurl_16",
        "doc_name": "4088173",
        "question": "Which organization produced the Aponjon-MAMA project?",
        "answer": "Dnet",
        "ans_page_list": [102],           # 基于1；匹配 _page101 → page_idx 0
        "answer_sources": ["Text"],
        "answer_format": "String",
        "page_list": [
            "longdocurl/4088173/4088173_page101.jpg",
            "longdocurl/4088173/4088173_page102.jpg",
        ],
        "length": 3991,
    },
    {
        "id": "longdocurl_25",
        "doc_name": "4027862",
        "question": "Which miscellaneous crop accounted for the highest CIF value in 2020?",
        "answer": "Onion",
        "ans_page_list": [78],            # 匹配 _page77 → page_idx 1（此记录）
        "answer_sources": ["Layout", "Table"],
        "answer_format": "String",
        "page_list": [
            "longdocurl/4027862/4027862_page76.jpg",
            "longdocurl/4027862/4027862_page77.jpg",
            "longdocurl/4027862/4027862_page78.jpg",
            "longdocurl/4027862/4027862_page79.jpg",
            "longdocurl/4027862/4027862_page80.jpg",
        ],
        "length": 3642,
    },
]

# ---- mmlongdoc_K4 记录 ----
MMLONGDOC_K4_RECORDS = [
    {
        "id": "mmlongdoc_4",
        "doc_name": "0e94b4197b10096b1f4c699701570fbf",
        "doc_type": "Tutorial/Workshop",
        "question": "What range does red color represent in the West Nile Virus chart?",
        "answer": "0-375 miles",
        "ans_page_list": [9],             # 匹配 _page8 → page_idx 0（此记录）
        "answer_sources": ["Chart"],
        "answer_format": "String",
        "page_list": [
            "mmlongbench-doc/0e94b4197b10096b1f4c699701570fbf/"
            "0e94b4197b10096b1f4c699701570fbf_page8.jpg",
            "mmlongbench-doc/0e94b4197b10096b1f4c699701570fbf/"
            "0e94b4197b10096b1f4c699701570fbf_page9.jpg",
        ],
        "length": 3369,
    },
]


# ---------------------------------------------------------------------------
# 目录结构构建器
# ---------------------------------------------------------------------------

_JPEG_STUB = b"\xff\xd8\xff\xe0" + b"\x00" * 12  # 最小的有效 JPEG 文件头


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _make_placeholder_images(image_root: Path, records: list[dict]) -> None:
    """在 image_root 下为 page_list 中的每个条目创建桩 JPEG 文件。"""
    for rec in records:
        for rel_path in rec["page_list"]:
            full = image_root / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            if not full.exists():
                full.write_bytes(_JPEG_STUB)


def _build_data_dir(
    base: Path,
    *,
    longdocurl_records: list[dict] | None = None,
    mmlongdoc_records: list[dict] | None = None,
) -> Path:
    """
    在 ``base`` 下创建 MMLongBench 原始目录结构：

        base/
          mmlb_data/documentQA/
            longdocurl_K4.jsonl
            mmlongdoc_K4.jsonl
          mmlb_image/
            <占位图片>

    返回 ``base``（传递给 DocumentQAAdapter 的 data_dir）。
    """
    docqa_dir = base / "mmlb_data" / "documentQA"
    image_root = base / "mmlb_image"

    if longdocurl_records is not None:
        _write_jsonl(docqa_dir / "longdocurl_K4.jsonl", longdocurl_records)
        _make_placeholder_images(image_root, longdocurl_records)

    if mmlongdoc_records is not None:
        _write_jsonl(docqa_dir / "mmlongdoc_K4.jsonl", mmlongdoc_records)
        _make_placeholder_images(image_root, mmlongdoc_records)

    return base


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture()
def longdocurl_data_dir(tmp_path: Path) -> Path:
    """
    仅包含 longdocurl_K4.jsonl + 占位图片的 data_dir。
    传递给 DocumentQAAdapter(data_dir=..., subtasks=["longdocurl"], lengths=["K4"])。
    """
    return _build_data_dir(
        tmp_path / "mmlb_longdocurl",
        longdocurl_records=LONGDOCURL_K4_RECORDS,
    )


@pytest.fixture()
def mmlongdoc_data_dir(tmp_path: Path) -> Path:
    """
    仅包含 mmlongdoc_K4.jsonl + 占位图片的 data_dir。
    """
    return _build_data_dir(
        tmp_path / "mmlb_mmlongdoc",
        mmlongdoc_records=MMLONGDOC_K4_RECORDS,
    )


@pytest.fixture()
def combined_data_dir(tmp_path: Path) -> Path:
    """
    同时包含 longdocurl_K4 和 mmlongdoc_K4 数据的 data_dir。
    """
    return _build_data_dir(
        tmp_path / "mmlb_combined",
        longdocurl_records=LONGDOCURL_K4_RECORDS,
        mmlongdoc_records=MMLONGDOC_K4_RECORDS,
    )


# ---------------------------------------------------------------------------
# 便捷工具：预构建的原始记录，用于直接参数化
# ---------------------------------------------------------------------------

@pytest.fixture()
def longdocurl_records() -> list[dict]:
    return LONGDOCURL_K4_RECORDS


@pytest.fixture()
def mmlongdoc_records() -> list[dict]:
    return MMLONGDOC_K4_RECORDS
