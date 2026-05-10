"""
src/zeroshot_vdr/data/adapters.py 的测试（Step 2.1.2）

测试 DocumentQAAdapter，它必须：
  - 接受 data_dir（MMLongBench 原始根目录）、subtasks 列表、lengths 列表
  - 从 data_dir/mmlb_data/documentQA/{subtask}_{length}.jsonl 加载 JSONL
  - 相对于 data_dir/mmlb_image/ 解析图片路径
  - 通过 iter_pages()、iter_queries()、iter_judgments() 生成 Page / Query / RelevanceJudgment 对象（均为生成器/可迭代对象）
  - 通过 normalize_doc_id() 标准化 doc_name -> doc_id
  - 通过规范的 build_* 辅助函数构造 page_id / query_id

所有测试均为黑盒测试（仅测试接口）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zeroshot_vdr.contracts import (
    Page,
    Query,
    RelevanceJudgment,
    build_page_id,
    build_query_id,
    normalize_doc_id,
)
from zeroshot_vdr.data.adapters import DocumentQAAdapter

# ---------------------------------------------------------------------------
# 辅助函数：从夹具构建适配器
# ---------------------------------------------------------------------------


def _longdocurl_adapter(data_dir: Path) -> DocumentQAAdapter:
    return DocumentQAAdapter(
        data_dir=str(data_dir),
        subtasks=["longdocurl"],
        lengths=["K4"],
    )


def _mmlongdoc_adapter(data_dir: Path) -> DocumentQAAdapter:
    return DocumentQAAdapter(
        data_dir=str(data_dir),
        subtasks=["mmlongdoc"],
        lengths=["K4"],
    )


# ===========================================================================
# 实例化
# ===========================================================================


class TestDocumentQAAdapterInstantiation:
    def test_can_instantiate_longdocurl(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        assert adapter is not None

    def test_can_instantiate_mmlongdoc(self, mmlongdoc_data_dir):
        adapter = _mmlongdoc_adapter(mmlongdoc_data_dir)
        assert adapter is not None

    def test_can_instantiate_with_defaults(self):
        """无参数构造函数不应抛出异常。"""
        adapter = DocumentQAAdapter()
        assert adapter is not None


# ===========================================================================
# iter_pages() 方法
# ===========================================================================


class TestIterPages:
    @pytest.fixture()
    def adapter(self, longdocurl_data_dir):
        return _longdocurl_adapter(longdocurl_data_dir)

    @pytest.fixture()
    def pages(self, adapter):
        return list(adapter.iter_pages())

    # --- Return type ---

    def test_returns_iterable(self, adapter):
        result = adapter.iter_pages()
        assert hasattr(result, "__iter__")

    def test_yields_page_objects(self, pages):
        assert all(isinstance(p, Page) for p in pages)

    def test_non_empty(self, pages):
        assert len(pages) > 0

    # --- 页面计数 ---

    def test_count_equals_unique_page_list_entries(
        self, pages, longdocurl_records
    ):
        """
        每个唯一的 page_id（从 page_list 条目派生）对应一个 Page。
        我们的夹具记录在不同文档之间的页面没有重叠。
        """
        expected = sum(len(r["page_list"]) for r in longdocurl_records)
        assert len(pages) == expected

    # --- 元数据字段 ---

    def test_task_family_is_docqa(self, pages):
        for page in pages:
            assert page.task_family == "docqa"

    def test_subtask_is_longdocurl(self, pages):
        for page in pages:
            assert page.subtask == "longdocurl"

    def test_length_is_K4(self, pages):
        for page in pages:
            assert page.length == "K4"

    # --- doc_id / raw_doc_name ---

    def test_raw_doc_name_comes_from_jsonl_doc_name(
        self, pages, longdocurl_records
    ):
        expected_names = {r["doc_name"] for r in longdocurl_records}
        page_raw_names = {p.raw_doc_name for p in pages}
        assert page_raw_names == expected_names

    def test_doc_id_is_normalized_from_raw_doc_name(self, pages):
        for page in pages:
            assert page.doc_id == normalize_doc_id(page.raw_doc_name)

    def test_doc_id_contains_no_path_separators(self, pages):
        for page in pages:
            assert "/" not in page.doc_id
            assert "\\" not in page.doc_id

    # --- page_idx ---

    def test_page_idx_non_negative(self, pages):
        for page in pages:
            assert page.page_idx >= 0

    def test_page_idx_unique_per_doc(self, pages):
        from collections import defaultdict

        groups: dict[tuple, list[int]] = defaultdict(list)
        for page in pages:
            key = (page.doc_id, page.subtask, page.length)
            groups[key].append(page.page_idx)

        for key, idxs in groups.items():
            assert len(idxs) == len(set(idxs)), (
                f"Duplicate page_idx for {key}: {idxs}"
            )

    def test_page_idx_starts_at_zero_per_doc(self, pages):
        from collections import defaultdict

        min_idx: dict[str, int] = defaultdict(lambda: 10**9)
        for page in pages:
            min_idx[page.doc_id] = min(min_idx[page.doc_id], page.page_idx)

        for doc_id, m in min_idx.items():
            assert m == 0, f"doc_id={doc_id!r} min page_idx={m}, expected 0"

    # --- page_id ---

    def test_page_id_is_non_empty_string(self, pages):
        for page in pages:
            assert isinstance(page.page_id, str) and page.page_id

    def test_page_id_consistent_with_build_page_id(self, pages):
        for page in pages:
            expected = build_page_id(
                page.task_family,
                page.subtask,
                page.length,
                page.doc_id,
                page.page_idx,
            )
            assert page.page_id == expected, (
                f"page.page_id={page.page_id!r} != {expected!r}"
            )

    def test_page_ids_are_globally_unique(self, pages):
        ids = [p.page_id for p in pages]
        assert len(ids) == len(set(ids)), "Duplicate page_ids detected"

    # --- image_path ---

    def test_image_path_is_non_empty_string(self, pages):
        for page in pages:
            assert isinstance(page.image_path, str) and page.image_path

    def test_image_path_contains_data_dir(self, pages, longdocurl_data_dir):
        """image_path 应该是绝对路径，并且位于 data_dir 下。"""
        data_dir_str = str(longdocurl_data_dir)
        for page in pages:
            assert page.image_path.startswith(data_dir_str), (
                f"image_path={page.image_path!r} not under data_dir"
            )


# ===========================================================================
# iter_queries() 方法
# ===========================================================================


class TestIterQueries:
    @pytest.fixture()
    def adapter(self, longdocurl_data_dir):
        return _longdocurl_adapter(longdocurl_data_dir)

    @pytest.fixture()
    def queries(self, adapter):
        return list(adapter.iter_queries())

    # --- 返回类型 ---

    def test_returns_iterable(self, adapter):
        assert hasattr(adapter.iter_queries(), "__iter__")

    def test_yields_query_objects(self, queries):
        assert all(isinstance(q, Query) for q in queries)

    def test_non_empty(self, queries):
        assert len(queries) > 0

    # --- 查询计数 ---

    def test_count_equals_record_count(self, queries, longdocurl_records):
        """每个 JSONL 记录恰好对应一个 Query。"""
        assert len(queries) == len(longdocurl_records)

    # --- 文本 ---

    def test_query_text_matches_jsonl_question(
        self, queries, longdocurl_records
    ):
        expected = {r["question"] for r in longdocurl_records}
        actual = {q.text for q in queries}
        assert actual == expected

    # --- 元数据 ---

    def test_task_family_is_docqa(self, queries):
        for q in queries:
            assert q.task_family == "docqa"

    def test_subtask_is_longdocurl(self, queries):
        for q in queries:
            assert q.subtask == "longdocurl"

    def test_length_is_K4(self, queries):
        for q in queries:
            assert q.length == "K4"

    # --- doc_id ---

    def test_raw_doc_name_from_jsonl(self, queries, longdocurl_records):
        expected = {r["doc_name"] for r in longdocurl_records}
        actual = {q.raw_doc_name for q in queries}
        assert actual == expected

    def test_doc_id_normalized(self, queries):
        for q in queries:
            assert q.doc_id == normalize_doc_id(q.raw_doc_name)

    # --- query_id ---

    def test_query_id_is_non_empty_string(self, queries):
        for q in queries:
            assert isinstance(q.query_id, str) and q.query_id

    def test_query_id_starts_with_docqa(self, queries):
        for q in queries:
            assert q.query_id.startswith("docqa/")

    def test_query_id_contains_subtask_and_length(self, queries):
        for q in queries:
            assert "longdocurl_K4" in q.query_id

    def test_query_id_has_q_prefix_in_last_segment(self, queries):
        for q in queries:
            last = q.query_id.split("/")[-1]
            assert last.startswith("q")

    def test_query_id_consistent_with_build_query_id(self, queries):
        for q in queries:
            last = q.query_id.split("/")[-1]   # e.g. "q016"
            index = int(last[1:])
            expected = build_query_id(
                q.task_family, q.subtask, q.length, index
            )
            assert q.query_id == expected

    def test_query_ids_are_unique(self, queries):
        ids = [q.query_id for q in queries]
        assert len(ids) == len(set(ids))


# ===========================================================================
# iter_judgments() 方法
# ===========================================================================


class TestIterJudgments:
    @pytest.fixture()
    def adapter(self, longdocurl_data_dir):
        return _longdocurl_adapter(longdocurl_data_dir)

    @pytest.fixture()
    def pages(self, adapter):
        return list(adapter.iter_pages())

    @pytest.fixture()
    def queries(self, adapter):
        return list(adapter.iter_queries())

    @pytest.fixture()
    def judgments(self, adapter):
        return list(adapter.iter_judgments())

    # --- 返回类型 ---

    def test_returns_iterable(self, adapter):
        assert hasattr(adapter.iter_judgments(), "__iter__")

    def test_yields_relevance_judgment_objects(self, judgments):
        assert all(isinstance(j, RelevanceJudgment) for j in judgments)

    def test_non_empty(self, judgments):
        assert len(judgments) > 0

    # --- 相关性值 ---

    def test_all_yielded_judgments_are_relevant(self, judgments):
        """
        iter_judgments() 仅发出正标签（relevance=1）。
        不在 ans_page_list 中的页面不会生成条目（relevance=0 是隐式的）。
        """
        for j in judgments:
            assert j.relevance == 1

    def test_relevance_is_int(self, judgments):
        for j in judgments:
            assert isinstance(j.relevance, int)

    # --- 引用完整性 ---

    def test_query_ids_reference_real_queries(self, queries, judgments):
        valid_qids = {q.query_id for q in queries}
        for j in judgments:
            assert j.query_id in valid_qids, (
                f"Unknown query_id={j.query_id!r}"
            )

    def test_page_ids_reference_real_pages(self, pages, judgments):
        valid_pids = {p.page_id for p in pages}
        for j in judgments:
            assert j.page_id in valid_pids, (
                f"Unknown page_id={j.page_id!r}"
            )

    # --- 每个查询的覆盖范围 ---

    def test_each_query_has_at_least_one_judgment(self, queries, judgments):
        judgment_qids = {j.query_id for j in judgments}
        for q in queries:
            assert q.query_id in judgment_qids, (
                f"No judgments for query_id={q.query_id!r}"
            )

    def test_no_duplicate_query_page_pair(self, judgments):
        seen: set[tuple[str, str]] = set()
        for j in judgments:
            pair = (j.query_id, j.page_id)
            assert pair not in seen, f"Duplicate judgment for {pair!r}"
            seen.add(pair)

    # --- 特定夹具计数 ---

    def test_record_one_relevant_page_count(
        self, queries, judgments, longdocurl_records
    ):
        """记录 0：ans_page_list=[102] -> 每个查询 1 个相关判断。"""
        rec = longdocurl_records[0]
        doc_id = normalize_doc_id(rec["doc_name"])
        doc_queries = [q for q in queries if q.doc_id == doc_id]
        assert len(doc_queries) >= 1

        for q in doc_queries:
            q_judgments = [j for j in judgments if j.query_id == q.query_id]
            assert len(q_judgments) == len(rec["ans_page_list"])

    def test_record_two_relevant_page_count(
        self, queries, judgments, longdocurl_records
    ):
        """记录 1：ans_page_list=[78] -> 每个查询 1 个相关判断。"""
        rec = longdocurl_records[1]
        doc_id = normalize_doc_id(rec["doc_name"])
        doc_queries = [q for q in queries if q.doc_id == doc_id]
        assert len(doc_queries) >= 1

        for q in doc_queries:
            q_judgments = [j for j in judgments if j.query_id == q.query_id]
            assert len(q_judgments) == len(rec["ans_page_list"])


# ===========================================================================
# 跨子任务：mmlongdoc
# ===========================================================================


class TestMmlongdocAdapter:
    """对 mmlongdoc 进行冒烟测试，验证子任务处理。"""

    @pytest.fixture()
    def adapter(self, mmlongdoc_data_dir):
        return _mmlongdoc_adapter(mmlongdoc_data_dir)

    def test_pages_have_mmlongdoc_subtask(self, adapter):
        pages = list(adapter.iter_pages())
        for p in pages:
            assert p.subtask == "mmlongdoc"

    def test_queries_have_mmlongdoc_subtask(self, adapter):
        queries = list(adapter.iter_queries())
        for q in queries:
            assert q.subtask == "mmlongdoc"

    def test_page_ids_use_mmlongdoc_subtask(self, adapter):
        pages = list(adapter.iter_pages())
        for p in pages:
            assert "mmlongdoc_K4" in p.page_id

    def test_hash_doc_name_normalized_correctly(
        self, adapter, mmlongdoc_records
    ):
        pages = list(adapter.iter_pages())
        raw_name = mmlongdoc_records[0]["doc_name"]
        expected_id = normalize_doc_id(raw_name)
        assert any(p.doc_id == expected_id for p in pages)

    def test_relevant_pages_match_ans_page_list(
        self, adapter, mmlongdoc_records
    ):
        queries = list(adapter.iter_queries())
        judgments = list(adapter.iter_judgments())

        for rec in mmlongdoc_records:
            doc_id = normalize_doc_id(rec["doc_name"])
            doc_queries = [q for q in queries if q.doc_id == doc_id]
            for q in doc_queries:
                q_judgments = [j for j in judgments if j.query_id == q.query_id]
                assert len(q_judgments) == len(rec["ans_page_list"])


# ===========================================================================
# build_ground_truth() 便捷方法
# ===========================================================================


class TestBuildGroundTruth:
    @pytest.fixture()
    def adapter(self, longdocurl_data_dir):
        return _longdocurl_adapter(longdocurl_data_dir)

    def test_returns_dict(self, adapter):
        gt = adapter.build_ground_truth()
        assert isinstance(gt, dict)

    def test_keys_are_query_ids(self, adapter):
        gt = adapter.build_ground_truth()
        queries = list(adapter.iter_queries())
        query_ids = {q.query_id for q in queries}
        for qid in gt:
            assert qid in query_ids

    def test_values_are_sets(self, adapter):
        gt = adapter.build_ground_truth()
        for v in gt.values():
            assert isinstance(v, set)

    def test_values_are_non_empty(self, adapter):
        gt = adapter.build_ground_truth()
        for v in gt.values():
            assert len(v) >= 1

    def test_page_ids_in_ground_truth_are_real(self, adapter):
        gt = adapter.build_ground_truth()
        pages = list(adapter.iter_pages())
        valid_pids = {p.page_id for p in pages}
        for page_ids in gt.values():
            for pid in page_ids:
                assert pid in valid_pids
