"""
Shared fixtures for Step 2.2 (indexing) tests.

Fixtures are auto-injected by pytest – do NOT import this file directly.
Helper functions and constants are defined in each test module independently.
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
from PIL import Image

from zeroshot_vdr.indexing.store import IndexStore


# ---------------------------------------------------------------------------
# Constants (duplicated intentionally in test modules so no import needed)
# ---------------------------------------------------------------------------
_N_PATCHES = 16
_DIM = 128

_PAGE_ID_A0 = "docqa/longdocurl_K4/doc001/p0"
_PAGE_ID_A1 = "docqa/longdocurl_K4/doc001/p1"
_PAGE_ID_B0 = "docqa/longdocurl_K4/doc002/p0"
_PAGE_ID_C0 = "docqa/mmlongdoc_K4/doc003/p0"


def _make_embedding(n_patches: int = _N_PATCHES, dim: int = _DIM) -> torch.Tensor:
    return torch.randn(n_patches, dim)


def _make_image(w: int = 32, h: int = 32) -> Image.Image:
    return Image.new("RGB", (w, h), color=(100, 149, 237))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def index_dir(tmp_path: Path) -> Path:
    d = tmp_path / "index"
    d.mkdir()
    return d


@pytest.fixture()
def store(index_dir: Path) -> IndexStore:
    return IndexStore(str(index_dir))


@pytest.fixture()
def emb_a0() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(_N_PATCHES, _DIM)


@pytest.fixture()
def emb_a1() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(_N_PATCHES, _DIM)


@pytest.fixture()
def emb_b0() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(_N_PATCHES, _DIM)


@pytest.fixture()
def emb_c0() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(_N_PATCHES, _DIM)


@pytest.fixture()
def populated_store(store: IndexStore, emb_a0, emb_a1, emb_b0, emb_c0) -> IndexStore:
    store.write_page(_PAGE_ID_A0, emb_a0)
    store.write_page(_PAGE_ID_A1, emb_a1)
    store.write_page(_PAGE_ID_B0, emb_b0)
    store.write_page(_PAGE_ID_C0, emb_c0)
    return store
