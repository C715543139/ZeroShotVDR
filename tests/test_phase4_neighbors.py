"""Stage 2 测试：page_id 解析与 neighbor expansion。"""

import pytest

from zeroshot_vdr.advanced.neighbors import (
    expand_neighbors,
    make_page_id,
    parse_page_id,
    ParsedPageId,
)


class TestParsePageId:
    def test_standard_format(self):
        parsed = parse_page_id("slidevqa/default_K128/doc_001/p15")
        assert parsed.prefix == "slidevqa/default_K128/doc_001"
        assert parsed.page_idx == 15

    def test_nested_prefix(self):
        parsed = parse_page_id("docqa/longdocurl_K64/some_doc_v2/p0")
        assert parsed.prefix == "docqa/longdocurl_K64/some_doc_v2"
        assert parsed.page_idx == 0

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid page_id format"):
            parse_page_id("no_page_suffix")

    def test_no_slash_p(self):
        with pytest.raises(ValueError, match="Invalid page_id format"):
            parse_page_id("prefix/p123_no_digit")

    def test_make_roundtrip(self):
        original = "a/b/c_K128/doc_x/p7"
        parsed = parse_page_id(original)
        rebuilt = make_page_id(parsed.prefix, parsed.page_idx)
        assert rebuilt == original

    def test_frozen_dataclass(self):
        parsed = parse_page_id("a/b/p3")
        with pytest.raises(Exception):
            parsed.prefix = "new"  # type: ignore[misc]


class TestExpandNeighbors:
    def test_window_zero_no_expand(self):
        coarse = ["a/b/p5"]
        universe = ["a/b/p4", "a/b/p5", "a/b/p6"]
        result = expand_neighbors(coarse, universe, window=0, seed_n=1)
        assert result == ["a/b/p5"]

    def test_seed_n_zero_no_expand(self):
        coarse = ["a/b/p5"]
        universe = ["a/b/p4", "a/b/p5", "a/b/p6"]
        result = expand_neighbors(coarse, universe, window=1, seed_n=0)
        assert result == ["a/b/p5"]

    def test_window_one_within_universe(self):
        coarse = ["a/b/p5"]
        universe = ["a/b/p4", "a/b/p5", "a/b/p6"]
        result = expand_neighbors(coarse, universe, window=1, seed_n=8)
        assert result == ["a/b/p5", "a/b/p4", "a/b/p6"]

    def test_does_not_leave_universe(self):
        coarse = ["a/b/p5"]
        universe = ["a/b/p5"]  # no neighbors in universe
        result = expand_neighbors(coarse, universe, window=1, seed_n=8)
        assert result == ["a/b/p5"]

    def test_no_negative_page_idx(self):
        coarse = ["a/b/p0"]
        universe = ["a/b/p0", "a/b/p1"]
        result = expand_neighbors(coarse, universe, window=1, seed_n=8)
        assert result == ["a/b/p0", "a/b/p1"]  # p-1 not added

    def test_stable_order_and_dedup(self):
        coarse = ["a/b/p5", "a/b/p5", "a/b/p6"]
        universe = ["a/b/p4", "a/b/p5", "a/b/p6", "a/b/p7"]
        result = expand_neighbors(coarse, universe, window=1, seed_n=2)
        # coarse order preserved (dedup applied): p5, p6
        assert result[0] == "a/b/p5"
        assert result[1] == "a/b/p6"
        # p5 gets neighbor p4; p6 is NOT in first seed_n=2 (original list
        # indices 0,1 are both p5), so p7 is not expanded
        assert "a/b/p4" in result

    def test_multi_doc_coarse(self):
        """不同文档的 coarse pages 各自扩展邻页。"""
        coarse = ["a/K128/doc1/p3", "a/K128/doc2/p7"]
        universe = [
            "a/K128/doc1/p2", "a/K128/doc1/p3", "a/K128/doc1/p4",
            "a/K128/doc2/p6", "a/K128/doc2/p7", "a/K128/doc2/p8",
        ]
        result = expand_neighbors(coarse, universe, window=1, seed_n=8)
        assert len(result) == 6  # 2 coarse + 4 neighbors
        assert result[0] == "a/K128/doc1/p3"
        assert result[1] == "a/K128/doc2/p7"

    def test_seed_n_limits_expansion(self):
        """只有前 seed_n 个 coarse page 扩展邻页。"""
        coarse = ["a/b/p5", "a/b/p10", "a/b/p20"]
        universe = [
            "a/b/p4", "a/b/p5", "a/b/p6",
            "a/b/p9", "a/b/p10", "a/b/p11",
            "a/b/p19", "a/b/p20", "a/b/p21",
        ]
        result = expand_neighbors(coarse, universe, window=1, seed_n=1)
        # only p5 gets neighbors
        assert "a/b/p4" in result
        assert "a/b/p6" in result
        assert "a/b/p9" not in result  # p10 not in seed_n=1
        assert "a/b/p11" not in result

    def test_invalid_page_id_skipped(self):
        """格式不正确的 page_id 不参与扩展。"""
        coarse = ["invalid", "a/b/p5"]
        universe = ["a/b/p4", "a/b/p5", "a/b/p6", "invalid"]
        result = expand_neighbors(coarse, universe, window=1, seed_n=8)
        assert result == ["invalid", "a/b/p5", "a/b/p4", "a/b/p6"]

    def test_window_two(self):
        coarse = ["a/b/p5"]
        universe = ["a/b/p3", "a/b/p4", "a/b/p5", "a/b/p6", "a/b/p7"]
        result = expand_neighbors(coarse, universe, window=2, seed_n=8)
        assert set(result) == {"a/b/p3", "a/b/p4", "a/b/p5", "a/b/p6", "a/b/p7"}
