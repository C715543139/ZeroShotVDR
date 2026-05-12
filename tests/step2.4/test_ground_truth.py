"""
Tests for `src/zeroshot_vdr/evaluation/ground_truth.py` (Step 2.4.1)

Current public interface (from the real implementation):

    class GroundTruthLoader:
        def __init__(self, config: dict | None = None): ...
        def load(
            subtasks: list[str] | None = None,
            lengths: list[str] | None = None,
            task_family: str = "docqa",
        ) -> dict[str, set[str]]: ...
        def load_by_subtask(subtask: str, lengths: list[str] | None = None) -> dict[str, set[str]]: ...
        def load_by_length(length: str, subtasks: list[str] | None = None) -> dict[str, set[str]]: ...
        @property
        def config(self) -> dict: ...

Key invariants:
  * `load()` 返回统一的 `{query_id: set[page_id]}`。
  * 输出的 page_id / query_id 必须与 `contracts.py` 的命名规则一致。
  * DocumentQA 的 `ans_page_list` 应映射到 0-based `page_idx` 对应的 page_id。
  * `load_by_subtask()` / `load_by_length()` 是 `load()` 的便利包装。
"""

from __future__ import annotations

from pathlib import Path

from zeroshot_vdr.contracts import build_page_id, build_query_id


def _make_config(data_dir: Path) -> dict:
    return {
        "data": {
            "root_dir": str(data_dir),
            "subtasks": ["longdocurl", "slidevqa"],
            "length": None,
        }
    }


def _ground_truth_loader_class():
    from zeroshot_vdr.evaluation.ground_truth import GroundTruthLoader

    return GroundTruthLoader


def _instantiate_loader(data_dir: Path):
    GroundTruthLoader = _ground_truth_loader_class()
    return GroundTruthLoader(_make_config(data_dir))


def _load_ground_truth(
    data_dir: Path,
    subtasks=None,
    lengths=None,
    task_family: str = "docqa",
) -> dict[str, set[str]]:
    loader = _instantiate_loader(data_dir)
    return loader.load(subtasks=subtasks, lengths=lengths, task_family=task_family)


class TestInstantiation:
    def test_can_instantiate_with_config_dict(self, docqa_ground_truth_data_dir):
        loader = _instantiate_loader(docqa_ground_truth_data_dir)
        assert loader is not None

    def test_config_property_returns_original_config(self, docqa_ground_truth_data_dir):
        config = _make_config(docqa_ground_truth_data_dir)
        GroundTruthLoader = _ground_truth_loader_class()
        loader = GroundTruthLoader(config)
        assert loader.config == config

    def test_can_instantiate_without_explicit_config(self):
        GroundTruthLoader = _ground_truth_loader_class()
        loader = GroundTruthLoader()
        assert loader is not None


class TestLoad:
    def test_returns_dict(self, docqa_ground_truth_data_dir):
        gt = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
        )
        assert isinstance(gt, dict)

    def test_values_are_sets_of_page_ids(self, docqa_ground_truth_data_dir):
        gt = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
        )
        assert all(isinstance(v, set) for v in gt.values())
        assert all(isinstance(pid, str) for values in gt.values() for pid in values)

    def test_loads_all_queries_from_available_docqa_files(self, docqa_ground_truth_data_dir):
        gt = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
        )
        expected_query_ids = {
            build_query_id("docqa", "longdocurl", "K4", 0),
            build_query_id("docqa", "longdocurl", "K4", 1),
            build_query_id("docqa", "slidevqa", "K8", 0),
        }
        assert set(gt.keys()) == expected_query_ids

    def test_longdocurl_multiple_answer_pages_map_to_page_id_set(self, docqa_ground_truth_data_dir):
        gt = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
        )
        query_id = build_query_id("docqa", "longdocurl", "K4", 0)
        assert gt[query_id] == {
            build_page_id("docqa", "longdocurl", "K4", "doc001", 0),
            build_page_id("docqa", "longdocurl", "K4", "doc001", 2),
        }

    def test_duplicate_answer_pages_are_collapsed_by_set_semantics(self, docqa_ground_truth_data_dir):
        gt = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
        )
        query_id = build_query_id("docqa", "longdocurl", "K4", 0)
        assert len(gt[query_id]) == 2

    def test_longdocurl_second_record_maps_to_single_relevant_page(self, docqa_ground_truth_data_dir):
        gt = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
        )
        query_id = build_query_id("docqa", "longdocurl", "K4", 1)
        assert gt[query_id] == {
            build_page_id("docqa", "longdocurl", "K4", "doc001", 1)
        }

    def test_slidevqa_filename_pattern_maps_slide_number_to_zero_based_page_idx(self, docqa_ground_truth_data_dir):
        gt = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
        )
        query_id = build_query_id("docqa", "slidevqa", "K8", 0)
        assert gt[query_id] == {
            build_page_id("docqa", "slidevqa", "K8", "deck_intro", 1)
        }

    def test_explicit_task_family_docqa_does_not_change_result(self, docqa_ground_truth_data_dir):
        implicit = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
        )
        explicit = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["longdocurl", "slidevqa"],
            lengths=["K4", "K8"],
            task_family="docqa",
        )
        assert explicit == implicit

    def test_default_subtasks_and_lengths_come_from_config(self, docqa_ground_truth_data_dir):
        loader = _instantiate_loader(docqa_ground_truth_data_dir)
        gt = loader.load()
        assert set(gt.keys()) == {
            build_query_id("docqa", "longdocurl", "K4", 0),
            build_query_id("docqa", "longdocurl", "K4", 1),
            build_query_id("docqa", "slidevqa", "K8", 0),
        }

    def test_config_single_length_string_restricts_default_load(self, docqa_ground_truth_data_dir):
        GroundTruthLoader = _ground_truth_loader_class()
        loader = GroundTruthLoader(
            {
                "data": {
                    "root_dir": str(docqa_ground_truth_data_dir),
                    "subtasks": ["longdocurl", "slidevqa"],
                    "length": "K4",
                }
            }
        )
        gt = loader.load()
        assert set(gt.keys()) == {
            build_query_id("docqa", "longdocurl", "K4", 0),
            build_query_id("docqa", "longdocurl", "K4", 1),
        }

    def test_missing_subtask_and_length_combination_returns_empty_dict(self, docqa_ground_truth_data_dir):
        gt = _load_ground_truth(
            docqa_ground_truth_data_dir,
            subtasks=["mmlongdoc"],
            lengths=["K128"],
        )
        assert gt == {}

    def test_repeated_load_with_same_arguments_uses_cached_result_object(self, docqa_ground_truth_data_dir):
        loader = _instantiate_loader(docqa_ground_truth_data_dir)
        first = loader.load(subtasks=["longdocurl"], lengths=["K4"])
        second = loader.load(subtasks=["longdocurl"], lengths=["K4"])
        assert first is second


class TestConvenienceMethods:
    def test_load_by_subtask_filters_to_single_subtask(self, docqa_ground_truth_data_dir):
        loader = _instantiate_loader(docqa_ground_truth_data_dir)
        gt = loader.load_by_subtask("longdocurl", lengths=["K4"])
        assert set(gt.keys()) == {
            build_query_id("docqa", "longdocurl", "K4", 0),
            build_query_id("docqa", "longdocurl", "K4", 1),
        }

    def test_load_by_length_filters_to_single_length(self, docqa_ground_truth_data_dir):
        loader = _instantiate_loader(docqa_ground_truth_data_dir)
        gt = loader.load_by_length("K8", subtasks=["slidevqa"])
        assert set(gt.keys()) == {build_query_id("docqa", "slidevqa", "K8", 0)}

    def test_load_by_subtask_unknown_subtask_returns_empty_dict(self, docqa_ground_truth_data_dir):
        loader = _instantiate_loader(docqa_ground_truth_data_dir)
        gt = loader.load_by_subtask("ghost_subtask", lengths=["K4"])
        assert gt == {}

    def test_load_by_length_unknown_length_returns_empty_dict(self, docqa_ground_truth_data_dir):
        loader = _instantiate_loader(docqa_ground_truth_data_dir)
        gt = loader.load_by_length("K128", subtasks=["longdocurl", "slidevqa"])
        assert gt == {}

