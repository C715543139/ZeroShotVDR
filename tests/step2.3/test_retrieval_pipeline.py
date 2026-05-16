"""
Tests for src/zeroshot_vdr/retrieval/pipeline.py  (Step 2.3.4 / 2.3.5 / 2.3.6)

RetrievalPipeline public interface (from Project_Plan.md):

    RetrievalPipeline(model, index_store, query_encoder=None, config=None)
    .encode_query(query_text: str) -> torch.Tensor
    .retrieve(query: Query, top_k: int = 10,
              candidate_ids: list[str] | None = None,
              score_batch_size: int = 64) -> list[RetrievalResult]
    .retrieve_text(text: str, candidate_ids: list[str], top_k: int = 10)
    .retrieve_batch(queries: list[Query], top_k: int = 10, **kwargs)
    .generate_candidates(query: Query, query_emb: torch.Tensor,
                         top_n: int | None = None) -> list[str]

Key invariants:
  * Baseline 默认候选范围为 query.doc_id 对应的文档内页面。
  * 返回结果按分数降序排列，并封装为 RetrievalResult。
  * rank 为 1-based 且与排序结果一致。
  * top_k 控制结果截断；score_batch_size 只影响分批方式，不影响语义。
"""

from __future__ import annotations

import torch

from zeroshot_vdr.contracts import RetrievalResult


PAGE_ID_A0 = "docqa/longdocurl_K4/doc001/p0"
PAGE_ID_A1 = "docqa/longdocurl_K4/doc001/p1"
PAGE_ID_A2 = "docqa/longdocurl_K4/doc001/p2"
PAGE_ID_A0_K8 = "docqa/longdocurl_K8/doc001/p0"
PAGE_ID_B0 = "docqa/longdocurl_K4/doc002/p0"


class TestEncodeQuery:
    def test_delegates_to_query_encoder(self, pipeline, fixed_query_encoder, query_embedding):
        result = pipeline.encode_query("find the answer")
        assert torch.allclose(result, query_embedding)
        assert fixed_query_encoder.calls[-1] == "find the answer"


class TestGenerateCandidates:
    def test_candidate_page_ids_override_doc_scope_when_present(self, pipeline, query_embedding):
        from zeroshot_vdr.contracts import Query

        query = Query(
            query_id="docqa/longdocurl_K4/q010",
            text="explicit candidate scope",
            doc_id="ghost_doc",
            raw_doc_name="ghost_doc",
            task_family="docqa",
            subtask="longdocurl",
            length="K4",
            candidate_page_ids=(PAGE_ID_A2, PAGE_ID_B0),
        )

        candidates = pipeline.generate_candidates(query, query_embedding)
        assert candidates == [PAGE_ID_A2, PAGE_ID_B0]

    def test_default_scope_is_pages_of_query_doc(self, pipeline, sample_query, query_embedding):
        candidates = pipeline.generate_candidates(sample_query, query_embedding)
        assert set(candidates) == {PAGE_ID_A0, PAGE_ID_A1, PAGE_ID_A2}

    def test_default_scope_contains_no_cross_document_pages(self, pipeline, sample_query, query_embedding):
        candidates = pipeline.generate_candidates(sample_query, query_embedding)
        assert PAGE_ID_B0 not in candidates

    def test_default_scope_excludes_same_doc_other_lengths(self, pipeline, sample_query, query_embedding):
        pipeline.index_store.write_page(PAGE_ID_A0_K8, torch.zeros(2, 4))

        candidates = pipeline.generate_candidates(sample_query, query_embedding)

        assert PAGE_ID_A0_K8 not in candidates

    def test_unknown_doc_id_produces_empty_candidates(self, pipeline, query_embedding):
        from zeroshot_vdr.contracts import Query

        query = Query(
            query_id="docqa/longdocurl_K4/q999",
            text="unknown document",
            doc_id="ghost_doc",
            raw_doc_name="ghost_doc",
            task_family="docqa",
            subtask="longdocurl",
            length="K4",
        )
        candidates = pipeline.generate_candidates(query, query_embedding)
        assert candidates == []


class TestRetrieve:
    def test_returns_retrieval_result_objects(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=3)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_default_retrieve_uses_doc_local_candidates(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=10)
        page_ids = [r.page_id for r in results]

        assert PAGE_ID_B0 not in page_ids
        assert set(page_ids).issubset({PAGE_ID_A0, PAGE_ID_A1, PAGE_ID_A2})

    def test_default_retrieve_returns_scores_in_descending_order(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_default_retrieve_assigns_consecutive_one_based_ranks(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    def test_default_retrieve_ranking_matches_expected_similarity_order(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=3)
        page_ids = [r.page_id for r in results]
        assert page_ids == [PAGE_ID_A1, PAGE_ID_A0, PAGE_ID_A2]

    def test_top_k_truncates_result_count(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=2)
        assert len(results) == 2
        assert [r.page_id for r in results] == [PAGE_ID_A1, PAGE_ID_A0]

    def test_top_k_larger_than_candidate_count_returns_all_available_results(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=10)
        assert len(results) == 3
        assert [r.page_id for r in results] == [PAGE_ID_A1, PAGE_ID_A0, PAGE_ID_A2]
        assert [r.rank for r in results] == [1, 2, 3]

    def test_explicit_candidate_ids_override_default_scope(self, pipeline, sample_query):
        candidate_ids = [PAGE_ID_B0, PAGE_ID_A2]
        results = pipeline.retrieve(sample_query, top_k=2, candidate_ids=candidate_ids)
        page_ids = [r.page_id for r in results]

        assert set(page_ids) == set(candidate_ids)
        assert page_ids[0] == PAGE_ID_B0

    def test_empty_candidate_ids_returns_empty_results(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=3, candidate_ids=[])
        assert results == []

    def test_score_batch_size_can_be_smaller_than_candidate_count(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=3, score_batch_size=1)
        assert len(results) == 3
        assert [r.page_id for r in results] == [PAGE_ID_A1, PAGE_ID_A0, PAGE_ID_A2]

    def test_results_keep_source_query_id(self, pipeline, sample_query):
        results = pipeline.retrieve(sample_query, top_k=3)
        assert all(r.query_id == sample_query.query_id for r in results)


class TestRetrieveText:
    def test_retrieve_text_uses_explicit_candidates(self, pipeline, fixed_query_encoder):
        candidate_ids = [PAGE_ID_A0, PAGE_ID_A1, PAGE_ID_A2]
        results = pipeline.retrieve_text(
            "find the answer",
            candidate_ids=candidate_ids,
            top_k=2,
        )

        assert len(results) == 2
        assert set(r.page_id for r in results).issubset(set(candidate_ids))
        assert [r.page_id for r in results] == [PAGE_ID_A1, PAGE_ID_A0]
        assert fixed_query_encoder.calls[-1] == "find the answer"

    def test_retrieve_text_returns_ranked_results(self, pipeline):
        results = pipeline.retrieve_text(
            "find the answer",
            candidate_ids=[PAGE_ID_B0, PAGE_ID_A2],
            top_k=2,
        )
        assert [r.rank for r in results] == [1, 2]
        assert results[0].score >= results[1].score

    def test_retrieve_text_top_k_larger_than_candidates_returns_all_candidates(self, pipeline):
        candidate_ids = [PAGE_ID_B0, PAGE_ID_A2]
        results = pipeline.retrieve_text(
            "find the answer",
            candidate_ids=candidate_ids,
            top_k=10,
        )
        assert len(results) == 2
        assert set(r.page_id for r in results) == set(candidate_ids)
        assert [r.rank for r in results] == [1, 2]


class TestRetrieveBatch:
    def test_returns_one_result_list_per_query(self, pipeline, sample_query, other_doc_query):
        results = pipeline.retrieve_batch([sample_query, other_doc_query], top_k=1)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(item, list) for item in results)

    def test_each_query_is_ranked_against_its_own_doc_scope(self, pipeline, sample_query, other_doc_query):
        batch_results = pipeline.retrieve_batch([sample_query, other_doc_query], top_k=1)
        assert batch_results[0][0].page_id == PAGE_ID_A1
        assert batch_results[1][0].page_id == PAGE_ID_B0
        assert batch_results[0][0].query_id == sample_query.query_id
        assert batch_results[1][0].query_id == other_doc_query.query_id

    def test_batch_retrieve_forwards_score_batch_size_without_changing_results(
        self,
        pipeline,
        sample_query,
        other_doc_query,
    ):
        batch_results = pipeline.retrieve_batch(
            [sample_query, other_doc_query],
            top_k=1,
            score_batch_size=1,
        )
        assert batch_results[0][0].page_id == PAGE_ID_A1
        assert batch_results[1][0].page_id == PAGE_ID_B0

