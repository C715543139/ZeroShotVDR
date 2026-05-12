"""
Step 2.4 测试的共享夹具。

设计原则：
- 仅依据 `docs/Project_Plan.md` 中 Step 2.4 的公开契约组织测试。
- 不依赖 evaluation 模块的具体实现细节。
- Ground truth 夹具使用最小化的 MMLongBench DocumentQA 风格 JSONL 数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Minimal DocumentQA-style raw records
# ---------------------------------------------------------------------------

LONGDOCURL_K4_RECORDS = [
    {
        "id": "longdocurl_0",
        "doc_name": "doc001",
        "question": "Which pages mention the main answer?",
        "answer": "Pages one and three.",
        "ans_page_list": [1, 3, 3],
        "answer_sources": ["Text"],
        "answer_format": "String",
        "page_list": [
            "longdocurl/doc001/doc001_page1.jpg",
            "longdocurl/doc001/doc001_page2.jpg",
            "longdocurl/doc001/doc001_page3.jpg",
        ],
        "length": 3991,
    },
    {
        "id": "longdocurl_1",
        "doc_name": "doc001",
        "question": "Which page contains the chart?",
        "answer": "Page two.",
        "ans_page_list": [2],
        "answer_sources": ["Chart"],
        "answer_format": "String",
        "page_list": [
            "longdocurl/doc001/doc001_page1.jpg",
            "longdocurl/doc001/doc001_page2.jpg",
            "longdocurl/doc001/doc001_page3.jpg",
        ],
        "length": 3991,
    },
]

SLIDEVQA_K8_RECORDS = [
    {
        "id": "slidevqa_0",
        "doc_name": "deck_intro",
        "question": "Which slide contains the summary?",
        "answer": "Slide two.",
        "ans_page_list": [2],
        "answer_sources": ["Figure"],
        "answer_format": "String",
        "page_list": [
            "slideVQA/deck_intro/deck_intro-1-1024.jpg",
            "slideVQA/deck_intro/deck_intro-2-1024.jpg",
            "slideVQA/deck_intro/deck_intro-3-1024.jpg",
        ],
        "length": 8123,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")



def _build_docqa_data_dir(base: Path) -> Path:
    docqa_dir = base / "mmlb_data" / "documentQA"
    _write_jsonl(docqa_dir / "longdocurl_K4.jsonl", LONGDOCURL_K4_RECORDS)
    _write_jsonl(docqa_dir / "slidevqa_K8.jsonl", SLIDEVQA_K8_RECORDS)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def docqa_ground_truth_data_dir(tmp_path: Path) -> Path:
    return _build_docqa_data_dir(tmp_path / "mmlb_eval_docqa")


@pytest.fixture()
def longdocurl_k4_records() -> list[dict]:
    return LONGDOCURL_K4_RECORDS


@pytest.fixture()
def slidevqa_k8_records() -> list[dict]:
    return SLIDEVQA_K8_RECORDS

