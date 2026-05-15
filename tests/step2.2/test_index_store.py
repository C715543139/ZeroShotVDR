"""
Tests for src/zeroshot_vdr/indexing/store.py  (Step 2.2.4 / 2.2.5)

IndexStore public interface (from API spec / Project_Plan.md):

    IndexStore(index_dir: str)

    # Core I/O
    .write_page(page_id: str, embedding: torch.Tensor) -> None
    .read_page(page_id: str) -> torch.Tensor

    # Bulk access
    .iter_pages(page_ids: list[str] | None = None) -> Iterator[(str, Tensor)]
    .list_page_ids(doc_id: str | None = None) -> list[str]
    .get_mean_pooled_view(page_ids: list[str] | None = None) -> Tensor | dict

    # Convenience (uniform patch count)
    .read_stacked(page_ids: list[str]) -> tuple[torch.Tensor, list[str]]

    # Metadata
    .stats -> dict

On-disk artefacts required by the spec:
    page_ids.json     – page_id → file-path mapping
    index_meta.json   – model, dims, timestamp, page count

Key invariants:
  * Each page is stored as an independent .pt file with shape [n_patches, dim].
  * A fresh IndexStore pointed at the same directory can read previously written
    pages (persistence / incremental-build guarantee).
  * list_page_ids(doc_id=X) returns only pages whose doc_id component == X.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from zeroshot_vdr.indexing.store import IndexStore


# ---------------------------------------------------------------------------
# Module-level constants (mirrors conftest._* private constants)
# ---------------------------------------------------------------------------

N_PATCHES = 16
DIM = 128

PAGE_ID_A0 = "docqa/longdocurl_K4/doc001/p0"
PAGE_ID_A1 = "docqa/longdocurl_K4/doc001/p1"
PAGE_ID_A0_K8 = "docqa/longdocurl_K8/doc001/p0"
PAGE_ID_B0 = "docqa/longdocurl_K4/doc002/p0"
PAGE_ID_C0 = "docqa/mmlongdoc_K4/doc003/p0"


def _emb(seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N_PATCHES, DIM)


# ===========================================================================
# Instantiation
# ===========================================================================


class TestInstantiation:
    def test_accepts_string_path(self, index_dir):
        store = IndexStore(str(index_dir))
        assert store is not None

    def test_accepts_existing_directory(self, index_dir):
        IndexStore(str(index_dir))  # must not raise

    def test_creates_directory_if_missing(self, tmp_path):
        new_dir = tmp_path / "brand_new_index"
        assert not new_dir.exists()
        store = IndexStore(str(new_dir))
        # Verify usable: write without raising
        store.write_page(PAGE_ID_A0, _emb())


# ===========================================================================
# write_page / read_page
# ===========================================================================


class TestWriteAndRead:
    def test_write_does_not_raise(self, store, emb_a0):
        store.write_page(PAGE_ID_A0, emb_a0)

    def test_read_returns_tensor(self, store, emb_a0):
        store.write_page(PAGE_ID_A0, emb_a0)
        assert isinstance(store.read_page(PAGE_ID_A0), torch.Tensor)

    def test_read_shape_preserved(self, store, emb_a0):
        store.write_page(PAGE_ID_A0, emb_a0)
        assert store.read_page(PAGE_ID_A0).shape == (N_PATCHES, DIM)

    def test_read_values_close(self, store, emb_a0):
        store.write_page(PAGE_ID_A0, emb_a0)
        result = store.read_page(PAGE_ID_A0)
        assert torch.allclose(result.float(), emb_a0.float(), atol=1e-4)

    def test_multiple_pages_independent(self, store, emb_a0, emb_b0):
        store.write_page(PAGE_ID_A0, emb_a0)
        store.write_page(PAGE_ID_B0, emb_b0)
        ra = store.read_page(PAGE_ID_A0)
        rb = store.read_page(PAGE_ID_B0)
        assert not torch.allclose(ra.float(), rb.float())

    def test_page_id_with_slashes_round_trips(self, store):
        """page_id contains '/' separators — store must handle this safely."""
        pid = "docqa/longdocurl_K4/mydoc/p99"
        emb = _emb(99)
        store.write_page(pid, emb)
        result = store.read_page(pid)
        assert result.shape == emb.shape

    def test_overwrite_same_page_id_updates_value(self, store):
        old = torch.zeros(N_PATCHES, DIM)
        new = torch.ones(N_PATCHES, DIM)
        store.write_page(PAGE_ID_A0, old)
        store.write_page(PAGE_ID_A0, new)
        result = store.read_page(PAGE_ID_A0)
        assert torch.allclose(result.float(), new.float(), atol=1e-4)

    def test_read_unknown_page_id_raises(self, store):
        with pytest.raises(Exception):
            store.read_page("docqa/longdocurl_K4/ghost/p0")


# ===========================================================================
# list_page_ids()
# ===========================================================================


class TestListPageIds:
    def test_empty_store_returns_empty_list(self, store):
        result = store.list_page_ids()
        assert isinstance(result, list) and len(result) == 0

    def test_returns_list(self, populated_store):
        assert isinstance(populated_store.list_page_ids(), list)

    def test_all_written_pages_listed(self, populated_store):
        ids = populated_store.list_page_ids()
        for pid in (PAGE_ID_A0, PAGE_ID_A1, PAGE_ID_B0, PAGE_ID_C0):
            assert pid in ids

    def test_count_matches_written_pages(self, populated_store):
        assert len(populated_store.list_page_ids()) == 4

    def test_filter_by_doc_id_doc001(self, populated_store):
        assert set(populated_store.list_page_ids(doc_id="doc001")) == {
            PAGE_ID_A0,
            PAGE_ID_A1,
        }

    def test_filter_by_doc_id_doc002(self, populated_store):
        assert set(populated_store.list_page_ids(doc_id="doc002")) == {PAGE_ID_B0}

    def test_filter_by_doc_id_doc003(self, populated_store):
        assert set(populated_store.list_page_ids(doc_id="doc003")) == {PAGE_ID_C0}

    def test_filter_by_unknown_doc_id_returns_empty(self, populated_store):
        assert populated_store.list_page_ids(doc_id="nonexistent_doc") == []

    def test_none_filter_same_as_no_filter(self, populated_store):
        assert set(populated_store.list_page_ids()) == set(
            populated_store.list_page_ids(doc_id=None)
        )

    def test_filter_by_length_excludes_same_doc_other_lengths(self, populated_store):
        populated_store.write_page(PAGE_ID_A0_K8, _emb(100))

        result = populated_store.list_page_ids(
            doc_id="doc001",
            task_family="docqa",
            subtask="longdocurl",
            length="K4",
        )

        assert set(result) == {PAGE_ID_A0, PAGE_ID_A1}

    def test_filter_by_subtask_and_length_without_doc_id(self, populated_store):
        populated_store.write_page(PAGE_ID_A0_K8, _emb(101))

        result = populated_store.list_page_ids(
            task_family="docqa",
            subtask="longdocurl",
            length="K8",
        )

        assert result == [PAGE_ID_A0_K8]


# ===========================================================================
# iter_pages()
# ===========================================================================


class TestIterPages:
    def test_returns_iterable(self, populated_store):
        assert hasattr(populated_store.iter_pages(), "__iter__")

    def test_yields_page_id_and_tensor_tuples(self, populated_store):
        for item in populated_store.iter_pages():
            assert len(item) == 2
            pid, emb = item
            assert isinstance(pid, str)
            assert isinstance(emb, torch.Tensor)

    def test_yields_all_pages_when_no_filter(self, populated_store):
        assert len(list(populated_store.iter_pages())) == 4

    def test_no_duplicate_page_ids_in_iteration(self, populated_store):
        pids = [pid for pid, _ in populated_store.iter_pages()]
        assert len(pids) == len(set(pids))

    def test_embedding_shape_per_page(self, populated_store):
        for _, emb in populated_store.iter_pages():
            assert emb.shape == (N_PATCHES, DIM)

    def test_filter_subset_of_page_ids(self, populated_store):
        target = [PAGE_ID_A0, PAGE_ID_B0]
        pids = {pid for pid, _ in populated_store.iter_pages(page_ids=target)}
        assert pids == set(target)

    def test_filter_single_page_id(self, populated_store):
        results = list(populated_store.iter_pages(page_ids=[PAGE_ID_A0]))
        assert len(results) == 1
        assert results[0][0] == PAGE_ID_A0

    def test_values_match_written_embedding(self, store, emb_a0):
        store.write_page(PAGE_ID_A0, emb_a0)
        results = list(store.iter_pages())
        assert len(results) == 1
        pid, emb = results[0]
        assert pid == PAGE_ID_A0
        assert torch.allclose(emb.float(), emb_a0.float(), atol=1e-4)

    def test_empty_store_yields_nothing(self, store):
        assert list(store.iter_pages()) == []


# ===========================================================================
# read_stacked()
# ===========================================================================


class TestReadStacked:
    def test_returns_tuple_of_length_two(self, populated_store):
        result = populated_store.read_stacked([PAGE_ID_A0, PAGE_ID_A1])
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_tensor(self, populated_store):
        tensor, _ = populated_store.read_stacked([PAGE_ID_A0, PAGE_ID_A1])
        assert isinstance(tensor, torch.Tensor)

    def test_second_element_is_list_of_strings(self, populated_store):
        _, ids = populated_store.read_stacked([PAGE_ID_A0, PAGE_ID_A1])
        assert isinstance(ids, list) and all(isinstance(p, str) for p in ids)

    def test_stacked_tensor_shape(self, populated_store):
        """Shape must be [n_pages, n_patches, dim] for uniform-size embeddings."""
        page_ids = [PAGE_ID_A0, PAGE_ID_A1]
        tensor, _ = populated_store.read_stacked(page_ids)
        assert tensor.ndim == 3
        assert tensor.shape == (len(page_ids), N_PATCHES, DIM)

    def test_output_page_ids_cover_requested(self, populated_store):
        requested = [PAGE_ID_A0, PAGE_ID_A1]
        _, returned = populated_store.read_stacked(requested)
        assert set(returned) == set(requested)

    def test_single_page_stacked(self, populated_store):
        tensor, ids = populated_store.read_stacked([PAGE_ID_B0])
        assert tensor.shape[0] == 1
        assert ids == [PAGE_ID_B0]

    def test_values_match_individual_reads(self, populated_store):
        page_ids = [PAGE_ID_A0, PAGE_ID_A1]
        tensor, returned_ids = populated_store.read_stacked(page_ids)
        for i, pid in enumerate(returned_ids):
            expected = populated_store.read_page(pid)
            assert torch.allclose(tensor[i].float(), expected.float(), atol=1e-4)

    def test_all_pages_stacked(self, populated_store):
        all_ids = populated_store.list_page_ids()
        tensor, returned_ids = populated_store.read_stacked(all_ids)
        assert tensor.shape[0] == len(all_ids)


# ===========================================================================
# get_mean_pooled_view()
# ===========================================================================


class TestGetMeanPooledView:
    def test_does_not_raise(self, populated_store):
        populated_store.get_mean_pooled_view()

    def test_returns_non_none(self, populated_store):
        assert populated_store.get_mean_pooled_view() is not None

    def test_tensor_result_has_one_vector_per_page(self, populated_store):
        """get_mean_pooled_view() returns (tensor, page_ids); tensor.shape[0] == n_pages."""
        tensor, pids = populated_store.get_mean_pooled_view()
        all_ids = populated_store.list_page_ids()
        assert tensor.shape[0] == len(all_ids)

    def test_tensor_result_has_correct_dim(self, populated_store):
        tensor, _ = populated_store.get_mean_pooled_view()
        assert tensor.shape[-1] == DIM

    def test_filter_by_page_ids(self, populated_store):
        target = [PAGE_ID_A0, PAGE_ID_B0]
        tensor, pids = populated_store.get_mean_pooled_view(page_ids=target)
        assert tensor.shape[0] == len(target)
        assert set(pids) == set(target)

    def test_pooling_produces_single_vector_per_page(self, store):
        """Mean-pooled output must collapse the patch dim: each row is shape [dim]."""
        emb = torch.ones(N_PATCHES, DIM)
        store.write_page(PAGE_ID_A0, emb)
        tensor, pids = store.get_mean_pooled_view()
        # tensor shape [1, dim]
        assert tensor.ndim == 2
        assert tensor.shape == (1, DIM)

    def test_pooled_value_is_mean_of_patches(self, store):
        """For an all-ones embedding, pooled vector should be all ones."""
        emb = torch.ones(N_PATCHES, DIM)
        store.write_page(PAGE_ID_A0, emb)
        tensor, _ = store.get_mean_pooled_view()
        assert torch.allclose(tensor[0].float(), torch.ones(DIM), atol=1e-4)


# ===========================================================================
# stats property
# ===========================================================================


class TestStats:
    def test_returns_dict(self, populated_store):
        assert isinstance(populated_store.stats, dict)

    def test_empty_store_stats_is_dict(self, store):
        assert isinstance(store.stats, dict)

    def test_stats_exposes_page_count(self, populated_store):
        s = populated_store.stats
        count = (
            s.get("num_pages")
            or s.get("page_count")
            or s.get("count")
            or s.get("n_pages")
        )
        assert count is not None
        assert int(count) == 4

    def test_stats_count_increases_after_write(self, store, emb_a0):
        before = store.stats
        store.write_page(PAGE_ID_A0, emb_a0)
        after = store.stats

        def _count(s):
            return int(
                s.get("num_pages")
                or s.get("page_count")
                or s.get("count")
                or s.get("n_pages")
                or 0
            )

        assert _count(after) > _count(before)


# ===========================================================================
# On-disk metadata files
# ===========================================================================


class TestMetadataFiles:
    def test_page_ids_json_created_after_write(self, store, emb_a0, index_dir):
        store.write_page(PAGE_ID_A0, emb_a0)
        assert (index_dir / "page_ids.json").exists()

    def test_page_ids_json_is_valid_json(self, store, emb_a0, index_dir):
        """page_ids.json stores a JSON array (list), not a dict."""
        store.write_page(PAGE_ID_A0, emb_a0)
        data = json.loads((index_dir / "page_ids.json").read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_page_ids_json_contains_written_page(self, store, emb_a0, index_dir):
        store.write_page(PAGE_ID_A0, emb_a0)
        data = json.loads((index_dir / "page_ids.json").read_text(encoding="utf-8"))
        assert PAGE_ID_A0 in data  # data is a list

    def test_index_meta_json_created_after_save_meta(self, store, emb_a0, index_dir):
        """index_meta.json is created by explicit save_meta(), not by write_page()."""
        store.write_page(PAGE_ID_A0, emb_a0)
        store.save_meta(model_name="test-model", dim=DIM)
        assert (index_dir / "index_meta.json").exists()

    def test_index_meta_json_is_valid_json(self, store, emb_a0, index_dir):
        store.write_page(PAGE_ID_A0, emb_a0)
        store.save_meta(model_name="test-model", dim=DIM)
        meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        assert isinstance(meta, dict)


# ===========================================================================
# Manifest recovery after interrupted indexing
# ===========================================================================


class TestManifestRecovery:
    def test_recover_page_ids_restores_existing_files(self, index_dir, emb_a0, emb_b0):
        store = IndexStore(str(index_dir))
        store.write_page(PAGE_ID_A0, emb_a0, update_manifest=False)
        store.write_page(PAGE_ID_B0, emb_b0, update_manifest=False)

        recovered = store.recover_page_ids([PAGE_ID_A0, PAGE_ID_B0, PAGE_ID_C0])

        assert recovered == 2
        assert set(store.list_page_ids()) == {PAGE_ID_A0, PAGE_ID_B0}

    def test_recover_page_ids_is_idempotent(self, index_dir, emb_a0):
        store = IndexStore(str(index_dir))
        store.write_page(PAGE_ID_A0, emb_a0, update_manifest=False)

        assert store.recover_page_ids([PAGE_ID_A0]) == 1
        assert store.recover_page_ids([PAGE_ID_A0]) == 0


# ===========================================================================
# Persistence — data survives a fresh IndexStore from the same directory
# ===========================================================================


class TestPersistence:
    def test_read_after_reopen(self, index_dir, emb_a0):
        store1 = IndexStore(str(index_dir))
        store1.write_page(PAGE_ID_A0, emb_a0)

        store2 = IndexStore(str(index_dir))
        result = store2.read_page(PAGE_ID_A0)
        assert torch.allclose(result.float(), emb_a0.float(), atol=1e-4)

    def test_list_page_ids_after_reopen(self, index_dir, emb_a0, emb_b0):
        store1 = IndexStore(str(index_dir))
        store1.write_page(PAGE_ID_A0, emb_a0)
        store1.write_page(PAGE_ID_B0, emb_b0)

        store2 = IndexStore(str(index_dir))
        ids = store2.list_page_ids()
        assert PAGE_ID_A0 in ids and PAGE_ID_B0 in ids

    def test_incremental_write_preserves_existing(self, index_dir, emb_a0, emb_b0):
        """Two separate IndexStore instances can append pages without data loss."""
        IndexStore(str(index_dir)).write_page(PAGE_ID_A0, emb_a0)
        IndexStore(str(index_dir)).write_page(PAGE_ID_B0, emb_b0)

        ids = IndexStore(str(index_dir)).list_page_ids()
        assert PAGE_ID_A0 in ids and PAGE_ID_B0 in ids
