"""
Tests for src/zeroshot_vdr/indexing/encoder.py  (Step 2.2.2 / 2.2.3)

PageEncoder public interface (from API spec / Project_Plan.md):

    PageEncoder(model, batch_size: int = 4, dtype=torch.float16)
    .encode_batch(images: list[Image.Image]) -> torch.Tensor
    .encode_corpus(pages: list[Page], store: IndexStore) -> None

Key invariants:
  * encode_batch takes N PIL Images → tensor shape [N, n_patches, dim].
  * encode_corpus writes exactly one entry to the IndexStore per Page;
    the page's image_path is used to load the image.
  * batch_size controls chunking; no crash when len(images) > batch_size or
    len(images) is not a multiple of batch_size.

Because PageEncoder wraps a real ML model, tests use a lightweight mock that
returns deterministic random tensors.  The mock is deliberately flexible so
that it captures any calling pattern the encoder implementation might use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from zeroshot_vdr.contracts import Page
from zeroshot_vdr.indexing.encoder import PageEncoder
from zeroshot_vdr.indexing.store import IndexStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PATCHES = 16
DIM = 128

PAGE_ID_A0 = "docqa/longdocurl_K4/doc001/p0"
PAGE_ID_A1 = "docqa/longdocurl_K4/doc001/p1"
PAGE_ID_B0 = "docqa/longdocurl_K4/doc002/p0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(w: int = 32, h: int = 32) -> Image.Image:
    return Image.new("RGB", (w, h), color=(100, 149, 237))


def _make_page(
    page_id: str,
    page_idx: int,
    image_path: str,
    doc_id: str = "doc001",
) -> Page:
    return Page(
        page_id=page_id,
        doc_id=doc_id,
        raw_doc_name=doc_id,
        task_family="docqa",
        subtask="longdocurl",
        length="K4",
        page_idx=page_idx,
        image_path=image_path,
    )


def _write_image_file(tmp_path: Path, filename: str) -> str:
    """Save a tiny JPEG image and return its absolute path string."""
    img_path = tmp_path / filename
    _make_image().save(str(img_path))
    return str(img_path)


def _write_pages(specs: list[tuple[str, int]], tmp_path: Path) -> list[Page]:
    """
    Given a list of (page_id, page_idx) tuples, write a real JPEG for each
    and return a list of Page objects whose image_path points to it.
    """
    pages = []
    for page_id, page_idx in specs:
        safe_name = page_id.replace("/", "_") + ".jpg"
        img_path = _write_image_file(tmp_path, safe_name)
        pages.append(
            _make_page(
                page_id=page_id,
                page_idx=page_idx,
                image_path=img_path,
            )
        )
    return pages


# ---------------------------------------------------------------------------
# Mock ColPali model & processor
# ---------------------------------------------------------------------------


class _MockBatch(dict):
    """A dict subclass that supports .to(device) so encoder can call batch.to(device)."""
    def to(self, *args, **kwargs):
        return self


class _MockColPaliProcessor:
    """
    Mock of ColPaliProcessor.
    process_images() must return an object with a .to(device) method
    (the actual encoder calls: batch = self._processor.process_images(images).to(self._device))
    """
    def process_images(self, images: list[Image.Image]):
        n = len(images)
        return _MockBatch(pixel_values=torch.zeros(n, 3, 32, 32))

    def process_queries(self, queries: list[str]):
        n = len(queries)
        return _MockBatch(input_ids=torch.zeros(n, 16, dtype=torch.long))


class _MockColPaliModel:
    """
    Lightweight mock of a ColPali model.
    Forward pass: model.forward(**batch) → Tensor[batch, n_patches, dim]
    """

    def forward(self, **kwargs):
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                break
        else:
            batch_size = 1
        return torch.randn(batch_size, N_PATCHES, DIM)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def parameters(self):
        return iter([])


@pytest.fixture()
def mock_model() -> _MockColPaliModel:
    return _MockColPaliModel()


@pytest.fixture()
def mock_processor() -> _MockColPaliProcessor:
    return _MockColPaliProcessor()


@pytest.fixture()
def encoder(mock_model: _MockColPaliModel, mock_processor: _MockColPaliProcessor) -> PageEncoder:
    return PageEncoder(mock_model, mock_processor, batch_size=2)


# ===========================================================================
# Instantiation
# ===========================================================================


class TestInstantiation:
    def test_default_parameters(self, mock_model, mock_processor):
        enc = PageEncoder(mock_model, mock_processor)
        assert enc is not None

    def test_custom_batch_size(self, mock_model, mock_processor):
        enc = PageEncoder(mock_model, mock_processor, batch_size=2)
        assert enc is not None

    def test_float16_dtype(self, mock_model, mock_processor):
        enc = PageEncoder(mock_model, mock_processor, dtype=torch.float16)
        assert enc is not None

    def test_float32_dtype(self, mock_model, mock_processor):
        enc = PageEncoder(mock_model, mock_processor, dtype=torch.float32)
        assert enc is not None


# ===========================================================================
# encode_batch()
# ===========================================================================


class TestEncodeBatch:
    def test_returns_tensor(self, encoder):
        result = encoder.encode_batch([_make_image()])
        assert isinstance(result, torch.Tensor)

    def test_single_image_batch_dim(self, encoder):
        result = encoder.encode_batch([_make_image()])
        assert result.ndim == 3
        assert result.shape[0] == 1

    def test_two_images_batch_dim(self, encoder):
        result = encoder.encode_batch([_make_image(), _make_image()])
        assert result.shape[0] == 2

    def test_n_images_batch_dim(self, encoder):
        n = 5
        result = encoder.encode_batch([_make_image() for _ in range(n)])
        assert result.shape[0] == n

    def test_patch_dimension(self, encoder):
        result = encoder.encode_batch([_make_image()])
        assert result.shape[1] == N_PATCHES

    def test_embedding_dim(self, encoder):
        result = encoder.encode_batch([_make_image()])
        assert result.shape[-1] == DIM

    def test_accepts_pil_rgb_images(self, encoder):
        pil_img = Image.new("RGB", (448, 448), color="white")
        result = encoder.encode_batch([pil_img])
        assert result is not None

    def test_larger_batch_than_batch_size_does_not_raise(self, mock_model, mock_processor):
        """
        When len(images) > batch_size, the encoder processes in chunks.
        The returned tensor must still contain one row per image.
        """
        enc = PageEncoder(mock_model, mock_processor, batch_size=2)
        imgs = [_make_image() for _ in range(5)]
        result = enc.encode_batch(imgs)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 5

    def test_batch_size_1_does_not_raise(self, mock_model, mock_processor):
        enc = PageEncoder(mock_model, mock_processor, batch_size=1)
        result = enc.encode_batch([_make_image(), _make_image()])
        assert result.shape[0] == 2

    def test_non_multiple_of_batch_size_correct_count(self, mock_model, mock_processor):
        """3 images with batch_size=2 must yield shape[0] == 3."""
        enc = PageEncoder(mock_model, mock_processor, batch_size=2)
        result = enc.encode_batch([_make_image() for _ in range(3)])
        assert result.shape[0] == 3


# ===========================================================================
# encode_corpus()
# ===========================================================================


class TestEncodeCorpus:
    def test_writes_one_entry_per_page(self, encoder, index_dir, tmp_path):
        pages = _write_pages(
            [(PAGE_ID_A0, 0), (PAGE_ID_A1, 1), (PAGE_ID_B0, 0)],
            tmp_path,
        )
        store = IndexStore(str(index_dir))
        encoder.encode_corpus(pages, store)

        ids = store.list_page_ids()
        for page in pages:
            assert page.page_id in ids, f"Missing page_id: {page.page_id!r}"

    def test_entry_count_matches_pages(self, encoder, index_dir, tmp_path):
        pages = _write_pages([(PAGE_ID_A0, 0), (PAGE_ID_A1, 1)], tmp_path)
        store = IndexStore(str(index_dir))
        encoder.encode_corpus(pages, store)
        assert len(store.list_page_ids()) == 2

    def test_stored_embedding_shape(self, encoder, index_dir, tmp_path):
        """Each stored embedding must have shape [n_patches, dim]."""
        pages = _write_pages([(PAGE_ID_A0, 0)], tmp_path)
        store = IndexStore(str(index_dir))
        encoder.encode_corpus(pages, store)
        emb = store.read_page(PAGE_ID_A0)
        assert emb.shape == (N_PATCHES, DIM)

    def test_stored_embedding_is_tensor(self, encoder, index_dir, tmp_path):
        pages = _write_pages([(PAGE_ID_A0, 0)], tmp_path)
        store = IndexStore(str(index_dir))
        encoder.encode_corpus(pages, store)
        assert isinstance(store.read_page(PAGE_ID_A0), torch.Tensor)

    def test_empty_pages_list_does_not_raise(self, encoder, index_dir):
        store = IndexStore(str(index_dir))
        encoder.encode_corpus([], store)
        assert store.list_page_ids() == []

    def test_multiple_batches_no_crash(self, mock_model, mock_processor, index_dir, tmp_path):
        """
        5 pages with batch_size=2 requires 3 forward passes (2+2+1).
        encode_corpus must handle the partial last batch without error.
        """
        enc = PageEncoder(mock_model, mock_processor, batch_size=2)
        specs = [
            ("docqa/longdocurl_K4/doc001/p0", 0),
            ("docqa/longdocurl_K4/doc001/p1", 1),
            ("docqa/longdocurl_K4/doc001/p2", 2),
            ("docqa/longdocurl_K4/doc001/p3", 3),
            ("docqa/longdocurl_K4/doc001/p4", 4),
        ]
        pages = _write_pages(specs, tmp_path)
        store = IndexStore(str(index_dir))
        enc.encode_corpus(pages, store)
        assert len(store.list_page_ids()) == 5

    def test_uses_page_image_path(self, encoder, index_dir, tmp_path):
        """
        encode_corpus must load the image from page.image_path.
        Providing a valid image path should not raise.
        """
        pages = _write_pages([(PAGE_ID_A0, 0)], tmp_path)
        store = IndexStore(str(index_dir))
        encoder.encode_corpus(pages, store)
        # If the image wasn't loaded, the store would be empty.
        assert PAGE_ID_A0 in store.list_page_ids()
