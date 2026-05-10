"""
src/zeroshot_vdr/contracts.py 的测试（Step 2.1.1）

覆盖范围：
  - 数据类的实例化和字段访问：Page、Query、RetrievalResult、RelevanceJudgment
  - ID 辅助函数：normalize_doc_id()、build_page_id()、build_query_id()
"""

from __future__ import annotations

import pytest

from zeroshot_vdr.contracts import (
    Page,
    Query,
    RelevanceJudgment,
    RetrievalResult,
    build_page_id,
    build_query_id,
    normalize_doc_id,
)

# ===========================================================================
# Page 数据类
# ===========================================================================


class TestPage:
    """Page 应该是一个包含所有必需字段的普通数据类。"""

    def _make_page(self, **overrides) -> Page:
        defaults = dict(
            page_id="docqa/longdocurl_K4/4088173/p0",
            doc_id="4088173",
            raw_doc_name="4088173",
            task_family="docqa",
            subtask="longdocurl",
            length="K4",
            page_idx=0,
            image_path="mmlb_image/longdocurl/4088173/4088173_page101.jpg",
        )
        defaults.update(overrides)
        return Page(**defaults)

    def test_instantiation_succeeds(self):
        page = self._make_page()
        assert page is not None

    def test_page_id_field(self):
        page = self._make_page(page_id="docqa/longdocurl_K4/4088173/p0")
        assert page.page_id == "docqa/longdocurl_K4/4088173/p0"

    def test_doc_id_field(self):
        page = self._make_page(doc_id="4088173")
        assert page.doc_id == "4088173"

    def test_raw_doc_name_field(self):
        page = self._make_page(raw_doc_name="4088173")
        assert page.raw_doc_name == "4088173"

    def test_raw_doc_name_can_be_none(self):
        page = self._make_page(raw_doc_name=None)
        assert page.raw_doc_name is None

    def test_task_family_field(self):
        page = self._make_page(task_family="docqa")
        assert page.task_family == "docqa"

    def test_subtask_field(self):
        page = self._make_page(subtask="longdocurl")
        assert page.subtask == "longdocurl"

    def test_length_field(self):
        page = self._make_page(length="K4")
        assert page.length == "K4"

    def test_page_idx_field(self):
        page = self._make_page(page_idx=3)
        assert page.page_idx == 3

    def test_page_idx_zero_based(self):
        # page_idx 必须能够为 0
        page = self._make_page(page_idx=0)
        assert page.page_idx == 0

    def test_image_path_field(self):
        path = "mmlb_image/longdocurl/4088173/4088173_page101.jpg"
        page = self._make_page(image_path=path)
        assert page.image_path == path


# ===========================================================================
# Query 数据类
# ===========================================================================


class TestQuery:
    """Query 必须携带足够的上下文以支持文档级别的检索。"""

    def _make_query(self, **overrides) -> Query:
        defaults = dict(
            query_id="docqa/longdocurl_K4/q000",
            text="Which organization produced the Aponjon-MAMA project?",
            doc_id="4088173",
            raw_doc_name="4088173",
            task_family="docqa",
            subtask="longdocurl",
            length="K4",
        )
        defaults.update(overrides)
        return Query(**defaults)

    def test_instantiation_succeeds(self):
        assert self._make_query() is not None

    def test_query_id_field(self):
        q = self._make_query(query_id="docqa/longdocurl_K4/q000")
        assert q.query_id == "docqa/longdocurl_K4/q000"

    def test_text_field(self):
        q = self._make_query(text="Hello?")
        assert q.text == "Hello?"

    def test_doc_id_field(self):
        q = self._make_query(doc_id="4088173")
        assert q.doc_id == "4088173"

    def test_raw_doc_name_field(self):
        q = self._make_query(raw_doc_name="4088173")
        assert q.raw_doc_name == "4088173"

    def test_raw_doc_name_can_be_none(self):
        q = self._make_query(raw_doc_name=None)
        assert q.raw_doc_name is None

    def test_task_family_field(self):
        q = self._make_query(task_family="docqa")
        assert q.task_family == "docqa"

    def test_subtask_field(self):
        q = self._make_query(subtask="slidevqa")
        assert q.subtask == "slidevqa"

    def test_length_field(self):
        q = self._make_query(length="K128")
        assert q.length == "K128"


# ===========================================================================
# RetrievalResult 数据类
# ===========================================================================


class TestRetrievalResult:
    def test_instantiation_succeeds(self):
        r = RetrievalResult(
            query_id="docqa/longdocurl_K4/q000",
            page_id="docqa/longdocurl_K4/4088173/p1",
            score=0.85,
            rank=1,
        )
        assert r is not None

    def test_query_id_field(self):
        r = RetrievalResult(
            query_id="docqa/longdocurl_K4/q000",
            page_id="docqa/longdocurl_K4/4088173/p1",
            score=0.85,
            rank=1,
        )
        assert r.query_id == "docqa/longdocurl_K4/q000"

    def test_page_id_field(self):
        r = RetrievalResult(
            query_id="docqa/longdocurl_K4/q000",
            page_id="docqa/longdocurl_K4/4088173/p1",
            score=0.85,
            rank=1,
        )
        assert r.page_id == "docqa/longdocurl_K4/4088173/p1"

    def test_score_field(self):
        r = RetrievalResult(
            query_id="q", page_id="p", score=0.42, rank=3
        )
        assert r.score == pytest.approx(0.42)

    def test_rank_field(self):
        r = RetrievalResult(query_id="q", page_id="p", score=1.0, rank=1)
        assert r.rank == 1


# ===========================================================================
# RelevanceJudgment 数据类
# ===========================================================================


class TestRelevanceJudgment:
    def test_instantiation_succeeds(self):
        j = RelevanceJudgment(
            query_id="docqa/longdocurl_K4/q000",
            page_id="docqa/longdocurl_K4/4088173/p1",
            relevance=1,
        )
        assert j is not None

    def test_relevance_one(self):
        j = RelevanceJudgment(
            query_id="q", page_id="p", relevance=1
        )
        assert j.relevance == 1

    def test_relevance_zero(self):
        j = RelevanceJudgment(
            query_id="q", page_id="p", relevance=0
        )
        assert j.relevance == 0

    def test_query_id_field(self):
        j = RelevanceJudgment(query_id="my_qid", page_id="p", relevance=1)
        assert j.query_id == "my_qid"

    def test_page_id_field(self):
        j = RelevanceJudgment(query_id="q", page_id="my_pid", relevance=0)
        assert j.page_id == "my_pid"


# ===========================================================================
# normalize_doc_id() 函数
# ===========================================================================


class TestNormalizeDocId:
    """normalize_doc_id() 是创建 doc_id 的唯一合法入口。"""

    def test_returns_string(self):
        assert isinstance(normalize_doc_id("4088173"), str)

    def test_simple_numeric_string_unchanged(self):
        assert normalize_doc_id("4088173") == "4088173"

    def test_strips_leading_whitespace(self):
        result = normalize_doc_id("  4088173")
        assert result == "4088173"

    def test_strips_trailing_whitespace(self):
        result = normalize_doc_id("4088173  ")
        assert result == "4088173"

    def test_strips_both_sides_whitespace(self):
        result = normalize_doc_id("  4088173  ")
        assert result == "4088173"

    def test_replaces_forward_slash(self):
        result = normalize_doc_id("mmlongbench-doc/abc123")
        assert "/" not in result

    def test_replaces_backslash(self):
        result = normalize_doc_id("some\\path\\doc")
        assert "\\" not in result

    def test_forward_slash_becomes_underscore(self):
        result = normalize_doc_id("mmlongbench-doc/abc123")
        assert result == "mmlongbench-doc_abc123"

    def test_multiple_slashes_collapsed(self):
        # 两个连续斜杠不应产生双下划线
        result = normalize_doc_id("a//b")
        assert "__" not in result

    def test_collapses_multiple_spaces(self):
        result = normalize_doc_id("a  b")
        assert "  " not in result

    def test_hash_string_unchanged(self):
        hash_name = "0e94b4197b10096b1f4c699701570fbf"
        assert normalize_doc_id(hash_name) == hash_name

    def test_dashes_preserved(self):
        result = normalize_doc_id("agileftpbwm201008-1225530207520998-9_95")
        assert "-" in result

    def test_path_like_no_separators_in_result(self):
        result = normalize_doc_id("mmlongbench-doc/abc/def")
        assert "/" not in result
        assert "\\" not in result

    def test_idempotent_on_already_normalized(self):
        first = normalize_doc_id("4088173")
        second = normalize_doc_id(first)
        assert first == second


# ===========================================================================
# build_page_id() 函数
# ===========================================================================


class TestBuildPageId:
    """稳定的 page_id 格式：{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}"""

    def test_returns_string(self):
        assert isinstance(build_page_id("docqa", "longdocurl", "K4", "4088173", 0), str)

    def test_basic_format_page0(self):
        pid = build_page_id("docqa", "longdocurl", "K4", "4088173", 0)
        assert pid == "docqa/longdocurl_K4/4088173/p0"

    def test_basic_format_page5(self):
        pid = build_page_id("docqa", "longdocurl", "K4", "4088173", 5)
        assert pid == "docqa/longdocurl_K4/4088173/p5"

    def test_mmlongdoc_subtask(self):
        pid = build_page_id(
            "docqa", "mmlongdoc", "K8", "0e94b4197b10096b1f4c699701570fbf", 8
        )
        assert pid == "docqa/mmlongdoc_K8/0e94b4197b10096b1f4c699701570fbf/p8"

    def test_slidevqa_subtask(self):
        pid = build_page_id("docqa", "slidevqa", "K128", "slide_doc", 10)
        assert pid == "docqa/slidevqa_K128/slide_doc/p10"

    def test_large_page_idx(self):
        pid = build_page_id("docqa", "longdocurl", "K128", "doc123", 999)
        assert pid.endswith("/p999")

    def test_contains_all_components(self):
        pid = build_page_id("docqa", "longdocurl", "K32", "doc123", 2)
        assert "docqa" in pid
        assert "longdocurl_K32" in pid
        assert "doc123" in pid
        assert "/p2" in pid

    def test_different_lengths_produce_different_ids(self):
        pid_k4 = build_page_id("docqa", "longdocurl", "K4", "4088173", 0)
        pid_k8 = build_page_id("docqa", "longdocurl", "K8", "4088173", 0)
        assert pid_k4 != pid_k8

    def test_different_page_idx_produce_different_ids(self):
        pid0 = build_page_id("docqa", "longdocurl", "K4", "4088173", 0)
        pid1 = build_page_id("docqa", "longdocurl", "K4", "4088173", 1)
        assert pid0 != pid1

    def test_different_doc_ids_produce_different_ids(self):
        pid_a = build_page_id("docqa", "longdocurl", "K4", "4088173", 0)
        pid_b = build_page_id("docqa", "longdocurl", "K4", "4027862", 0)
        assert pid_a != pid_b


# ===========================================================================
# build_query_id() 函数
# ===========================================================================


class TestBuildQueryId:
    """稳定的 query_id 格式：{task_family}/{subtask}_{length}/q{query_index:0>3d}"""

    def test_returns_string(self):
        assert isinstance(build_query_id("docqa", "longdocurl", "K4", 0), str)

    def test_index_zero_zero_padded(self):
        qid = build_query_id("docqa", "longdocurl", "K4", 0)
        assert qid == "docqa/longdocurl_K4/q000"

    def test_index_one_zero_padded(self):
        qid = build_query_id("docqa", "longdocurl", "K4", 1)
        assert qid == "docqa/longdocurl_K4/q001"

    def test_index_nine_zero_padded(self):
        qid = build_query_id("docqa", "longdocurl", "K4", 9)
        assert qid == "docqa/longdocurl_K4/q009"

    def test_index_10_two_digit(self):
        qid = build_query_id("docqa", "longdocurl", "K4", 10)
        assert qid == "docqa/longdocurl_K4/q010"

    def test_index_100_three_digit(self):
        qid = build_query_id("docqa", "longdocurl", "K4", 100)
        assert qid == "docqa/longdocurl_K4/q100"

    def test_index_999_three_digit(self):
        qid = build_query_id("docqa", "longdocurl", "K4", 999)
        assert qid == "docqa/longdocurl_K4/q999"

    def test_contains_all_components(self):
        qid = build_query_id("docqa", "mmlongdoc", "K8", 5)
        assert "docqa" in qid
        assert "mmlongdoc_K8" in qid
        assert "q005" in qid

    def test_different_indices_produce_different_ids(self):
        qid0 = build_query_id("docqa", "longdocurl", "K4", 0)
        qid1 = build_query_id("docqa", "longdocurl", "K4", 1)
        assert qid0 != qid1

    def test_different_lengths_produce_different_ids(self):
        qid_k4 = build_query_id("docqa", "longdocurl", "K4", 0)
        qid_k8 = build_query_id("docqa", "longdocurl", "K8", 0)
        assert qid_k4 != qid_k8
