"""
Tests for src/zeroshot_vdr/retrieval/encoder.py  (Step 2.3.1)

QueryEncoder public interface (from Project_Plan.md):

    QueryEncoder(model)
    .encode(query: str) -> torch.Tensor

Key invariants:
  * encode() 接受单条文本查询并返回 torch.Tensor。
  * 输出表示单条查询的 token embeddings，形状为 [n_tokens, dim]。
  * 单条查询结果不应保留 batch 维度。
"""

from __future__ import annotations

import torch

from zeroshot_vdr.retrieval.encoder import QueryEncoder


# 与 conftest 中的 mock 设置保持一致（故意重复，避免跨测试模块导入）
N_TOKENS = 6
DIM = 8


class TestInstantiation:
    def test_can_instantiate(self, mock_query_model, mock_query_processor):
        encoder = QueryEncoder(mock_query_model, mock_query_processor)
        assert encoder is not None


class TestEncode:
    def test_returns_tensor(self, actual_query_encoder):
        result = actual_query_encoder.encode("Where is the answer?")
        assert isinstance(result, torch.Tensor)

    def test_returns_two_dim_tensor(self, actual_query_encoder):
        result = actual_query_encoder.encode("Where is the answer?")
        assert result.ndim == 2

    def test_token_dimension_matches_processed_query_length(self, actual_query_encoder):
        result = actual_query_encoder.encode("Where is the answer?")
        assert result.shape[0] == N_TOKENS

    def test_embedding_dimension_matches_model_output(self, actual_query_encoder):
        result = actual_query_encoder.encode("Where is the answer?")
        assert result.shape[1] == DIM

    def test_single_query_has_no_batch_dimension(self, actual_query_encoder):
        result = actual_query_encoder.encode("single query")
        assert result.shape == (N_TOKENS, DIM)

    def test_repeated_calls_keep_same_shape(self, actual_query_encoder):
        first = actual_query_encoder.encode("first question")
        second = actual_query_encoder.encode("second question")
        assert first.shape == second.shape == (N_TOKENS, DIM)

    def test_empty_string_still_returns_token_embeddings(self, actual_query_encoder):
        result = actual_query_encoder.encode("")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (N_TOKENS, DIM)

    def test_unicode_query_still_returns_token_embeddings(self, actual_query_encoder):
        result = actual_query_encoder.encode("请定位页面中的答案。")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (N_TOKENS, DIM)

