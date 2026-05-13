"""
Tests for src/zeroshot_vdr/retrieval/scoring.py  (Step 2.3.2 / 2.3.3)

Public interface (from Project_Plan.md):
    maxsim_score(query_emb: [n_tokens, dim], page_emb: [n_patches, dim]) -> scalar
    batched_maxsim(query_emb: [n_tokens, dim], pages_emb: [batch, n_patches, dim]) -> [batch]

Key invariants:
  * MaxSim = 对每个 query token，在页面 patches 中取最大相似度，再求和。
  * batched_maxsim 对每个页面返回一个分数。
  * batched_maxsim 的结果应与逐页循环调用 maxsim_score 一致。
"""

from __future__ import annotations

import pytest
import torch

from zeroshot_vdr.retrieval.scoring import batched_maxsim, maxsim_score


DIM = 4


def _unit(index: int) -> torch.Tensor:
    v = torch.zeros(DIM, dtype=torch.float32)
    v[index] = 1.0
    return v


def _query() -> torch.Tensor:
    return torch.stack([_unit(0), _unit(1)])


def _perfect_page() -> torch.Tensor:
    return torch.stack([_unit(0), _unit(1)])


def _partial_page() -> torch.Tensor:
    return torch.stack([_unit(0), _unit(2)])


def _irrelevant_page() -> torch.Tensor:
    return torch.stack([_unit(2), _unit(3)])


def _redundant_perfect_page() -> torch.Tensor:
    return torch.stack([_unit(2), _unit(0), _unit(3), _unit(1)])


def _duplicate_best_patches_page() -> torch.Tensor:
    return torch.stack([_unit(0), _unit(0), _unit(1), _unit(1)])


class TestMaxSimScore:
    def test_returns_scalar_tensor(self):
        score = maxsim_score(_query(), _perfect_page())
        assert isinstance(score, torch.Tensor)
        assert score.ndim == 0

    def test_perfect_match_equals_number_of_query_tokens(self):
        score = maxsim_score(_query(), _perfect_page())
        assert score.item() == pytest.approx(2.0, abs=1e-5)

    def test_sum_of_tokenwise_maxima(self):
        """
        query = [e0, e1]
        page  = [e0, e2]

        token e0 的最大相似度为 1；
        token e1 与任一 patch 都不匹配，最大相似度为 0；
        总分应为 1。
        """
        score = maxsim_score(_query(), _partial_page())
        assert score.item() == pytest.approx(1.0, abs=1e-5)

    def test_irrelevant_page_scores_zero(self):
        score = maxsim_score(_query(), _irrelevant_page())
        assert score.item() == pytest.approx(0.0, abs=1e-5)

    def test_extra_irrelevant_patches_do_not_change_best_match_score(self):
        score = maxsim_score(_query(), _redundant_perfect_page())
        assert score.item() == pytest.approx(2.0, abs=1e-5)

    def test_duplicate_best_patches_do_not_increase_score(self):
        score = maxsim_score(_query(), _duplicate_best_patches_page())
        assert score.item() == pytest.approx(2.0, abs=1e-5)


class TestBatchedMaxSim:
    def test_returns_one_score_per_page(self):
        pages = torch.stack([
            _perfect_page(),
            _partial_page(),
            _irrelevant_page(),
        ])
        scores = batched_maxsim(_query(), pages)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (3,)

    def test_agrees_with_loop_over_maxsim_score(self):
        pages = torch.stack([
            _perfect_page(),
            _partial_page(),
            _irrelevant_page(),
        ])

        batched_scores = batched_maxsim(_query(), pages)
        loop_scores = torch.tensor(
            [maxsim_score(_query(), page).item() for page in pages],
            dtype=batched_scores.dtype,
        )

        assert torch.allclose(batched_scores, loop_scores, atol=1e-5)

    def test_expected_score_ordering(self):
        pages = torch.stack([
            _perfect_page(),
            _partial_page(),
            _irrelevant_page(),
        ])
        scores = batched_maxsim(_query(), pages)

        assert scores[0].item() == pytest.approx(2.0, abs=1e-5)
        assert scores[1].item() == pytest.approx(1.0, abs=1e-5)
        assert scores[2].item() == pytest.approx(0.0, abs=1e-5)
        assert scores[0] > scores[1] > scores[2]

    def test_single_page_batch_matches_single_page_score(self):
        page = _partial_page()
        batched = batched_maxsim(_query(), page.unsqueeze(0))
        single = maxsim_score(_query(), page)
        assert batched.shape == (1,)
        assert torch.allclose(batched[0], single, atol=1e-5)

    def test_batch_output_preserves_input_order(self):
        pages = torch.stack([
            _partial_page(),
            _perfect_page(),
            _irrelevant_page(),
        ])
        scores = batched_maxsim(_query(), pages)
        assert scores[0].item() == pytest.approx(1.0, abs=1e-5)
        assert scores[1].item() == pytest.approx(2.0, abs=1e-5)
        assert scores[2].item() == pytest.approx(0.0, abs=1e-5)

    def test_mixed_precision_inputs_are_supported(self):
        pages = torch.stack([
            _perfect_page(),
            _partial_page(),
        ]).to(torch.float16)
        query = _query().to(torch.bfloat16)

        scores = batched_maxsim(query, pages)

        assert scores.shape == (2,)
        assert scores[0].item() == pytest.approx(2.0, abs=1e-5)
        assert scores[1].item() == pytest.approx(1.0, abs=1e-5)

