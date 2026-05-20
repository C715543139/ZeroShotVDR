"""Stage 5 测试：MeanPoolCache。"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from zeroshot_vdr.advanced.mean_pool_cache import MeanPoolCache, build_mean_pool_cache


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tmp_cache_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp) / "cache"


@pytest.fixture
def sample_data():
    page_ids = [f"doc/p{i}" for i in range(100)]
    embeddings = torch.randn(100, 128)
    return page_ids, embeddings


# ===========================================================================
# Tests
# ===========================================================================


class TestMeanPoolCache:
    def test_save_and_exists(self, tmp_cache_dir, sample_data):
        page_ids, embeddings = sample_data
        cache = MeanPoolCache(tmp_cache_dir)

        assert not cache.exists()
        cache.save(page_ids, embeddings)
        assert cache.exists()

    def test_roundtrip(self, tmp_cache_dir, sample_data):
        page_ids, embeddings = sample_data
        cache = MeanPoolCache(tmp_cache_dir)
        cache.save(page_ids, embeddings)

        cache2 = MeanPoolCache(tmp_cache_dir)
        cache2.load()
        assert cache2.page_ids == page_ids

        retrieved = cache2.get(page_ids[:10])
        assert retrieved.shape == (10, 128)
        assert torch.allclose(retrieved, embeddings[:10])

    def test_get_preserves_input_order(self, tmp_cache_dir, sample_data):
        page_ids, embeddings = sample_data
        cache = MeanPoolCache(tmp_cache_dir)
        cache.save(page_ids, embeddings)
        cache.load()

        # 请求逆序
        reversed_ids = list(reversed(page_ids[:5]))
        result = cache.get(reversed_ids)
        for i, pid in enumerate(reversed_ids):
            orig_idx = page_ids.index(pid)
            assert torch.allclose(result[i], embeddings[orig_idx])

    def test_missing_page_id_raises(self, tmp_cache_dir, sample_data):
        page_ids, embeddings = sample_data
        cache = MeanPoolCache(tmp_cache_dir)
        cache.save(page_ids, embeddings)
        cache.load()

        with pytest.raises(KeyError):
            cache.get(["doc/not_exist"])

    def test_load_missing_raises(self, tmp_cache_dir):
        cache = MeanPoolCache(tmp_cache_dir)
        with pytest.raises(FileNotFoundError):
            cache.load()

    def test_meta_content(self, tmp_cache_dir, sample_data):
        page_ids, embeddings = sample_data
        cache = MeanPoolCache(tmp_cache_dir)
        custom_meta = {"index_dir": "/data/index", "version": "1.0"}
        cache.save(page_ids, embeddings, meta=custom_meta)

        meta = cache.load_meta()
        assert meta["num_pages"] == 100
        assert meta["embedding_dim"] == 128
        assert meta["index_dir"] == "/data/index"

    def test_contains(self, tmp_cache_dir, sample_data):
        page_ids, embeddings = sample_data
        cache = MeanPoolCache(tmp_cache_dir)
        cache.save(page_ids, embeddings)
        cache.load()

        assert "doc/p0" in cache
        assert "doc/not_exist" not in cache

    def test_save_creates_parent_dirs(self, tmp_cache_dir):
        deep_dir = tmp_cache_dir / "a" / "b" / "c"
        cache = MeanPoolCache(deep_dir)
        page_ids = ["p1", "p2"]
        embeddings = torch.randn(2, 64)
        cache.save(page_ids, embeddings)
        assert deep_dir.exists()

    def test_get_without_load_raises(self, tmp_cache_dir):
        cache = MeanPoolCache(tmp_cache_dir)
        with pytest.raises(RuntimeError, match="未加载"):
            cache.get(["p1"])

    def test_meta_json_valid(self, tmp_cache_dir, sample_data):
        page_ids, embeddings = sample_data
        cache = MeanPoolCache(tmp_cache_dir)
        cache.save(page_ids, embeddings)

        # 验证 meta.json 是合法 JSON
        with open(cache.meta_path, "r") as f:
            meta = json.load(f)
        assert "num_pages" in meta
        assert "embedding_dim" in meta


class FakeIndexStore:
    """fake IndexStore 用于测试 build_mean_pool_cache。"""

    def __init__(
        self,
        page_ids: list[str],
        dim: int = 128,
        max_batch_size: int | None = None,
    ):
        self._page_ids = page_ids
        self._dim = dim
        self._max_batch_size = max_batch_size
        self._requested_batch_sizes: list[int] = []

    def get_mean_pooled_view(self, page_ids):
        self._requested_batch_sizes.append(len(page_ids))
        if self._max_batch_size is not None and len(page_ids) > self._max_batch_size:
            raise AssertionError(f"batch 过大: {len(page_ids)} > {self._max_batch_size}")
        gen = torch.Generator()
        gen.manual_seed(42)
        embs = torch.randn(len(page_ids), self._dim, generator=gen)
        return embs, page_ids


class TestBuildMeanPoolCache:
    def test_build(self, tmp_cache_dir):
        store = FakeIndexStore([f"doc/p{i}" for i in range(50)])
        cache = MeanPoolCache(tmp_cache_dir)
        build_mean_pool_cache(store, store._page_ids, cache)

        assert cache.exists()
        cache.load()
        assert len(cache.page_ids) == 50
        assert cache.embeddings.shape == (50, 128)

    def test_build_batches_large_requests(self, tmp_cache_dir):
        store = FakeIndexStore(
            [f"doc/p{i}" for i in range(50)],
            max_batch_size=16,
        )
        cache = MeanPoolCache(tmp_cache_dir)

        build_mean_pool_cache(store, store._page_ids, cache, batch_size=16)

        assert cache.exists()
        assert store._requested_batch_sizes == [16, 16, 16, 2]
