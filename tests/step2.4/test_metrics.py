"""
Tests for `src/zeroshot_vdr/evaluation/metrics.py` (Step 2.4.2 / 2.4.3)

Public interface (from `Project_Plan.md`):

	recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float
	precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float
	mrr(retrieved: list[str], relevant: set[str]) -> float
	ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float
	compute_all_metrics(
		retrieval_results: dict[str, list[str]],
		ground_truth: dict[str, set[str]],
		k_values: list[int] = [1, 3, 5, 10],
	) -> pandas.DataFrame

Key invariants:
  * 指标函数只依赖标准化输入 `(retrieved_page_ids, relevant_page_ids, k)`。
  * Recall@k / Precision@k / MRR / nDCG@k 与具体数据集解耦。
  * `compute_all_metrics()` 对多条查询进行批量聚合，并返回 DataFrame。
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from zeroshot_vdr.evaluation.metrics import (
	compute_all_metrics,
	mrr,
	ndcg_at_k,
	precision_at_k,
	recall_at_k,
)


class TestRecallAtK:
	def test_returns_float(self):
		result = recall_at_k(["p1", "p2"], {"p1"}, 1)
		assert isinstance(result, float)

	def test_perfect_recall_when_all_relevant_pages_appear_in_top_k(self):
		result = recall_at_k(["p2", "p1", "p3"], {"p1", "p2"}, 2)
		assert result == pytest.approx(1.0)

	def test_partial_recall_counts_only_hits_inside_top_k(self):
		result = recall_at_k(["p1", "p4", "p3"], {"p1", "p2", "p3"}, 2)
		assert result == pytest.approx(1.0 / 3.0)

	def test_top_k_larger_than_retrieved_list_uses_available_hits(self):
		result = recall_at_k(["p1"], {"p1", "p2"}, 10)
		assert result == pytest.approx(0.5)

	def test_empty_relevant_set_returns_one(self):
		result = recall_at_k(["p1", "p2"], set(), 3)
		assert result == pytest.approx(1.0)

	def test_non_positive_k_raises_value_error(self):
		with pytest.raises(ValueError):
			recall_at_k(["p1"], {"p1"}, 0)

	def test_duplicate_retrieved_hits_do_not_increase_recall(self):
		result = recall_at_k(["p1", "p1", "p9"], {"p1", "p2"}, 3)
		assert result == pytest.approx(0.5)


class TestPrecisionAtK:
	def test_returns_float(self):
		result = precision_at_k(["p1", "p2"], {"p1"}, 1)
		assert isinstance(result, float)

	def test_precision_is_hits_divided_by_k(self):
		result = precision_at_k(["p1", "p4", "p3"], {"p1", "p2", "p3"}, 2)
		assert result == pytest.approx(0.5)

	def test_ignores_relevant_pages_beyond_top_k(self):
		result = precision_at_k(["p4", "p1", "p2"], {"p1", "p2"}, 1)
		assert result == pytest.approx(0.0)

	def test_when_k_exceeds_list_length_denominator_is_still_k(self):
		result = precision_at_k(["p1"], {"p1"}, 5)
		assert result == pytest.approx(0.2)

	def test_empty_relevant_set_returns_zero(self):
		result = precision_at_k(["p1", "p2"], set(), 3)
		assert result == pytest.approx(0.0)

	def test_non_positive_k_raises_value_error(self):
		with pytest.raises(ValueError):
			precision_at_k(["p1"], {"p1"}, 0)

	def test_duplicate_retrieved_hits_do_not_increase_precision(self):
		result = precision_at_k(["p1", "p1", "p9"], {"p1"}, 3)
		assert result == pytest.approx(1.0 / 3.0)


class TestMRR:
	def test_returns_float(self):
		result = mrr(["p1", "p2"], {"p2"})
		assert isinstance(result, float)

	def test_first_relevant_at_rank_one_gives_one(self):
		result = mrr(["p2", "p1", "p3"], {"p2"})
		assert result == pytest.approx(1.0)

	def test_first_relevant_at_rank_three_gives_one_over_three(self):
		result = mrr(["p4", "p5", "p2"], {"p2", "p9"})
		assert result == pytest.approx(1.0 / 3.0)

	def test_only_first_relevant_rank_matters(self):
		result = mrr(["p4", "p2", "p3"], {"p2", "p3"})
		assert result == pytest.approx(0.5)

	def test_no_relevant_page_gives_zero(self):
		result = mrr(["p4", "p5", "p6"], {"p1", "p2"})
		assert result == pytest.approx(0.0)

	def test_empty_relevant_set_returns_one(self):
		result = mrr(["p4", "p5"], set())
		assert result == pytest.approx(1.0)


class TestNDCGAtK:
	def test_returns_float(self):
		result = ndcg_at_k(["p1", "p2"], {"p1"}, 1)
		assert isinstance(result, float)

	def test_perfect_ranking_has_ndcg_one(self):
		result = ndcg_at_k(["p1", "p2", "p3"], {"p1", "p2"}, 2)
		assert result == pytest.approx(1.0)

	def test_late_relevant_hit_has_lower_ndcg(self):
		result = ndcg_at_k(["p3", "p1", "p2"], {"p1"}, 2)
		assert 0.0 < result < 1.0
		assert result == pytest.approx(1.0 / math.log2(3), abs=1e-6)

	def test_no_relevant_hit_in_top_k_gives_zero(self):
		result = ndcg_at_k(["p3", "p4", "p1"], {"p1"}, 2)
		assert result == pytest.approx(0.0)

	def test_empty_relevant_set_returns_one(self):
		result = ndcg_at_k(["p1", "p2"], set(), 3)
		assert result == pytest.approx(1.0)

	def test_non_positive_k_raises_value_error(self):
		with pytest.raises(ValueError):
			ndcg_at_k(["p1"], {"p1"}, 0)

	def test_top_k_larger_than_retrieved_list_still_stays_within_unit_interval(self):
		result = ndcg_at_k(["p1", "p3"], {"p1", "p2"}, 10)
		assert 0.0 <= result <= 1.0


class TestComputeAllMetrics:
	def _retrieval_results(self) -> dict[str, list[str]]:
		return {
			"q1": ["p1", "p2", "p3"],
			"q2": ["p4", "p5"],
		}

	def _ground_truth(self) -> dict[str, set[str]]:
		return {
			"q1": {"p2", "p3"},
			"q2": {"p4"},
		}

	def test_returns_dataframe(self):
		df = compute_all_metrics(
			self._retrieval_results(),
			self._ground_truth(),
			k_values=[1, 3],
		)
		assert isinstance(df, pd.DataFrame)

	def test_has_one_row_per_requested_k(self):
		df = compute_all_metrics(
			self._retrieval_results(),
			self._ground_truth(),
			k_values=[1, 3],
		)
		assert list(df["k"]) == [1, 3]

	def test_exposes_required_metric_columns(self):
		df = compute_all_metrics(
			self._retrieval_results(),
			self._ground_truth(),
			k_values=[1, 3],
		)
		assert {"k", "Recall", "Precision", "MRR", "nDCG", "n_queries"}.issubset(df.columns)

	def test_aggregates_mean_metrics_correctly(self):
		df = compute_all_metrics(
			self._retrieval_results(),
			self._ground_truth(),
			k_values=[1, 3],
		)

		row_k1 = df[df["k"] == 1].iloc[0]
		row_k3 = df[df["k"] == 3].iloc[0]

		assert row_k1["Recall"] == pytest.approx(0.5, abs=1e-6)
		assert row_k1["Precision"] == pytest.approx(0.5, abs=1e-6)
		assert row_k1["MRR"] == pytest.approx(0.75, abs=1e-6)
		assert row_k1["nDCG"] == pytest.approx(0.5, abs=1e-6)
		assert row_k1["n_queries"] == 2

		assert row_k3["Recall"] == pytest.approx(1.0, abs=1e-6)
		assert row_k3["Precision"] == pytest.approx(0.5, abs=1e-6)
		assert row_k3["MRR"] == pytest.approx(0.75, abs=1e-6)
		assert row_k3["nDCG"] == pytest.approx(0.846713, abs=1e-6)
		assert row_k3["n_queries"] == 2

	def test_none_k_values_uses_default_cutoffs(self):
		df = compute_all_metrics(self._retrieval_results(), self._ground_truth())
		assert list(df["k"]) == [1, 3, 5, 10]

	def test_only_common_query_ids_contribute_to_aggregate(self):
		retrieval_results = {
			"q1": ["p1", "p2", "p3"],
			"q_extra": ["p9"],
		}
		ground_truth = {
			"q1": {"p2", "p3"},
			"q_missing": {"p7"},
		}
		df = compute_all_metrics(retrieval_results, ground_truth, k_values=[1])
		row = df.iloc[0]
		assert row["n_queries"] == 1
		assert row["Recall"] == pytest.approx(0.0)
		assert row["Precision"] == pytest.approx(0.0)
		assert row["MRR"] == pytest.approx(0.5)
		assert row["nDCG"] == pytest.approx(0.0)

	def test_no_common_query_ids_raises_value_error(self):
		with pytest.raises(ValueError):
			compute_all_metrics({"q1": ["p1"]}, {"q2": {"p1"}}, k_values=[1])

