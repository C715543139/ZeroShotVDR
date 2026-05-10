"""
src/zeroshot_vdr/data/corpus.py 的测试（Step 2.1.3）

真实的 PageCorpus 接口：
  - PageCorpus(config: dict | None)
  - corpus.build(adapters: list[BaseAdapter]) -> list[Page]
  - corpus.build_from_adapter(adapter) -> list[Page]   （单适配器快捷方式）
  - corpus.save_metadata(path: str | None) -> str
  - PageCorpus.load_metadata(path: str) -> list[Page]  （类方法）
  - corpus.pages, corpus.num_pages, corpus.num_docs, corpus.doc_ids（属性）
  - corpus.get_page(page_id), corpus.get_doc_pages(doc_id)

JSON 磁盘格式：
  {"num_pages": N, "num_docs": M, "pages": [...], "doc_index": {...}}
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zeroshot_vdr.contracts import Page
from zeroshot_vdr.data.adapters import DocumentQAAdapter
from zeroshot_vdr.data.corpus import PageCorpus


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _longdocurl_adapter(data_dir: Path, *, length: str = "K4") -> DocumentQAAdapter:
    return DocumentQAAdapter(
        data_dir=str(data_dir),
        subtasks=["longdocurl"],
        lengths=[length],
    )


def _mmlongdoc_adapter(data_dir: Path) -> DocumentQAAdapter:
    return DocumentQAAdapter(
        data_dir=str(data_dir),
        subtasks=["mmlongdoc"],
        lengths=["K4"],
    )


# ===========================================================================
# build() 方法
# ===========================================================================


class TestBuild:
    # --- 返回类型 ---

    def test_build_returns_list(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        result = corpus.build([adapter])
        assert isinstance(result, list)

    def test_build_returns_page_objects(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        pages = corpus.build([adapter])
        assert all(isinstance(p, Page) for p in pages)

    def test_build_non_empty(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        pages = corpus.build([adapter])
        assert len(pages) > 0

    # --- 计数完整性 ---

    def test_count_matches_adapter_iter_pages(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        adapter_pages = list(adapter.iter_pages())
        corpus = PageCorpus(config={})
        corpus_pages = corpus.build([adapter])
        assert len(corpus_pages) == len(adapter_pages)

    # --- 唯一性 ---

    def test_page_ids_are_unique(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        pages = corpus.build([adapter])
        ids = [p.page_id for p in pages]
        assert len(ids) == len(set(ids))

    # --- 多个适配器（组合 data_dir） ---

    def test_two_subtasks_merged(self, combined_data_dir):
        adapter_ldu = DocumentQAAdapter(
            data_dir=str(combined_data_dir),
            subtasks=["longdocurl"],
            lengths=["K4"],
        )
        adapter_mld = DocumentQAAdapter(
            data_dir=str(combined_data_dir),
            subtasks=["mmlongdoc"],
            lengths=["K4"],
        )
        corpus = PageCorpus(config={})
        pages = corpus.build([adapter_ldu, adapter_mld])

        ldu_pages = [p for p in pages if p.subtask == "longdocurl"]
        mld_pages = [p for p in pages if p.subtask == "mmlongdoc"]
        assert len(ldu_pages) > 0
        assert len(mld_pages) > 0

    def test_two_subtasks_combined_unique_page_ids(self, combined_data_dir):
        adapter_ldu = DocumentQAAdapter(
            data_dir=str(combined_data_dir),
            subtasks=["longdocurl"],
            lengths=["K4"],
        )
        adapter_mld = DocumentQAAdapter(
            data_dir=str(combined_data_dir),
            subtasks=["mmlongdoc"],
            lengths=["K4"],
        )
        corpus = PageCorpus(config={})
        pages = corpus.build([adapter_ldu, adapter_mld])
        ids = [p.page_id for p in pages]
        assert len(ids) == len(set(ids))

    # --- 空适配器列表 ---

    def test_empty_adapters_returns_empty_list(self):
        corpus = PageCorpus(config={})
        pages = corpus.build([])
        assert isinstance(pages, list)
        assert len(pages) == 0

    # --- .pages 属性 ---

    def test_pages_property_matches_build_return(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        built = corpus.build([adapter])
        assert {p.page_id for p in corpus.pages} == {p.page_id for p in built}

    # --- num_pages / num_docs ---

    def test_num_pages_matches_build_count(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        pages = corpus.build([adapter])
        assert corpus.num_pages == len(pages)

    def test_num_docs_is_positive(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        corpus.build([adapter])
        assert corpus.num_docs >= 1

    # --- get_page() 方法 ---

    def test_get_page_returns_correct_page(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        pages = corpus.build([adapter])
        first = pages[0]
        result = corpus.get_page(first.page_id)
        assert result is not None
        assert result.page_id == first.page_id

    def test_get_page_unknown_returns_none(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        corpus.build([adapter])
        assert corpus.get_page("nonexistent/page_id") is None

    # --- get_doc_pages() 方法 ---

    def test_get_doc_pages_returns_list(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        corpus.build([adapter])
        doc_id = corpus.pages[0].doc_id
        result = corpus.get_doc_pages(doc_id)
        assert isinstance(result, list)

    def test_get_doc_pages_non_empty_for_valid_doc(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        corpus.build([adapter])
        doc_id = corpus.pages[0].doc_id
        assert len(corpus.get_doc_pages(doc_id)) >= 1

    def test_get_doc_pages_unknown_returns_empty(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        corpus.build([adapter])
        assert corpus.get_doc_pages("__nonexistent__") == []


# ===========================================================================
# build_from_adapter()  （单适配器便捷方法）
# ===========================================================================


class TestBuildFromAdapter:
    def test_returns_same_pages_as_build_single(self, longdocurl_data_dir):
        adapter1 = _longdocurl_adapter(longdocurl_data_dir)
        adapter2 = _longdocurl_adapter(longdocurl_data_dir)

        corpus1 = PageCorpus(config={})
        pages1 = corpus1.build([adapter1])

        corpus2 = PageCorpus(config={})
        pages2 = corpus2.build_from_adapter(adapter2)

        assert {p.page_id for p in pages1} == {p.page_id for p in pages2}


# ===========================================================================
# save_metadata() 和 load_metadata()
# ===========================================================================


class TestSaveAndLoadMetadata:
    @pytest.fixture()
    def built_corpus(self, longdocurl_data_dir):
        adapter = _longdocurl_adapter(longdocurl_data_dir)
        corpus = PageCorpus(config={})
        corpus.build([adapter])
        return corpus

    @pytest.fixture()
    def original_pages(self, built_corpus):
        return built_corpus.pages

    # --- save_metadata ---

    def test_save_creates_file(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        assert out.exists()

    def test_save_returns_string(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        returned = built_corpus.save_metadata(str(out))
        assert isinstance(returned, str)

    def test_save_returned_path_exists(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        returned = built_corpus.save_metadata(str(out))
        assert Path(returned).exists()

    def test_saved_file_is_valid_json(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data is not None

    def test_saved_json_has_pages_key(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "pages" in data

    def test_saved_json_pages_non_empty(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data["pages"]) > 0

    def test_saved_json_each_record_has_page_id(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        for rec in data["pages"]:
            assert "page_id" in rec

    def test_saved_json_has_num_pages(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "num_pages" in data
        assert data["num_pages"] == built_corpus.num_pages

    # --- load_metadata ---

    def test_load_returns_list(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        loaded = PageCorpus.load_metadata(str(out))
        assert isinstance(loaded, list)

    def test_load_returns_page_objects(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        loaded = PageCorpus.load_metadata(str(out))
        assert all(isinstance(p, Page) for p in loaded)

    # --- 往返保真度 ---

    def test_roundtrip_page_count(self, built_corpus, original_pages, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        loaded = PageCorpus.load_metadata(str(out))
        assert len(loaded) == len(original_pages)

    def test_roundtrip_page_ids(self, built_corpus, original_pages, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        loaded = PageCorpus.load_metadata(str(out))
        assert {p.page_id for p in loaded} == {p.page_id for p in original_pages}

    def test_roundtrip_doc_ids(self, built_corpus, original_pages, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        loaded = PageCorpus.load_metadata(str(out))
        assert {p.doc_id for p in loaded} == {p.doc_id for p in original_pages}

    def test_roundtrip_preserves_all_fields(
        self, built_corpus, original_pages, tmp_path
    ):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        loaded = PageCorpus.load_metadata(str(out))

        orig_by_id = {p.page_id: p for p in original_pages}
        loaded_by_id = {p.page_id: p for p in loaded}

        assert orig_by_id.keys() == loaded_by_id.keys()
        for pid, orig in orig_by_id.items():
            lo = loaded_by_id[pid]
            assert lo.doc_id == orig.doc_id
            assert lo.raw_doc_name == orig.raw_doc_name
            assert lo.task_family == orig.task_family
            assert lo.subtask == orig.subtask
            assert lo.length == orig.length
            assert lo.page_idx == orig.page_idx
            assert lo.image_path == orig.image_path

    def test_roundtrip_page_idx_is_int(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        built_corpus.save_metadata(str(out))
        loaded = PageCorpus.load_metadata(str(out))
        for p in loaded:
            assert isinstance(p.page_idx, int)

    # --- 覆盖行为 ---

    def test_save_overwrites_existing_file(self, built_corpus, tmp_path):
        out = tmp_path / "corpus_meta.json"
        out.write_text('{"sentinel": true}', encoding="utf-8")
        built_corpus.save_metadata(str(out))
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "pages" in data   # 真实内容，而非哨兵标记
