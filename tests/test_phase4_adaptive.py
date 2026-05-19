"""Stage 3 测试：adaptive top-N 工具函数。"""

import torch

from zeroshot_vdr.advanced.two_stage import choose_adaptive_top_n


class TestChooseAdaptiveTopN:
    def test_empty_scores(self):
        scores = torch.tensor([])
        assert choose_adaptive_top_n(scores, universe_size=0) == 0

    def test_small_universe_below_min(self):
        """universe_size <= min_n → 返回 universe_size。"""
        scores = torch.rand(10)
        n = choose_adaptive_top_n(scores, universe_size=10, min_n=32)
        assert n == 10

    def test_universe_at_min_n(self):
        scores = torch.rand(32)
        n = choose_adaptive_top_n(scores, universe_size=32, min_n=32)
        assert n == 32

    def test_respects_min_bound(self):
        scores = torch.linspace(1.0, 0.0, steps=1000)
        n = choose_adaptive_top_n(
            scores,
            universe_size=1000,
            min_n=32,
            max_n=128,
            base_ratio=0.20,
        )
        assert n >= 32

    def test_respects_max_bound(self):
        scores = torch.linspace(1.0, 0.0, steps=1000)
        n = choose_adaptive_top_n(
            scores,
            universe_size=1000,
            min_n=32,
            max_n=128,
            base_ratio=0.20,
        )
        assert n <= 128

    def test_respects_universe_size_bound(self):
        """top-N 不能超过 universe_size。"""
        scores = torch.linspace(1.0, 0.0, steps=50)
        n = choose_adaptive_top_n(
            scores,
            universe_size=50,
            min_n=10,
            max_n=200,
            base_ratio=1.0,
        )
        assert n <= 50

    def test_sharp_distribution_no_expand(self):
        """分数分布尖锐（top-1 远大于其他）→ 不扩张。"""
        scores = torch.cat([
            torch.tensor([1.0]),
            torch.linspace(0.5, 0.0, steps=127),
        ])
        n = choose_adaptive_top_n(
            scores,
            universe_size=128,
            min_n=32,
            max_n=128,
            base_ratio=0.25,  # base_n = 32
            flat_margin=0.035,
        )
        assert n == 32  # 不扩张

    def test_flat_distribution_expand(self):
        """分数分布平坦（margin 小）→ 扩张。"""
        scores = torch.linspace(1.0, 0.98, steps=128)
        n = choose_adaptive_top_n(
            scores,
            universe_size=128,
            min_n=32,
            max_n=128,
            base_ratio=0.25,  # base_n = 32
            flat_margin=0.035,
        )
        assert n == 64  # 翻倍

    def test_expand_not_exceed_max(self):
        """扩张后不超过 max_n。"""
        scores = torch.linspace(1.0, 0.999, steps=128)
        n = choose_adaptive_top_n(
            scores,
            universe_size=128,
            min_n=32,
            max_n=48,
            base_ratio=0.25,  # base_n = 32
            flat_margin=0.035,
        )
        assert n <= 48

    def test_expand_not_exceed_universe(self):
        """扩张后不超过 universe_size。"""
        scores = torch.linspace(1.0, 0.999, steps=60)
        n = choose_adaptive_top_n(
            scores,
            universe_size=60,
            min_n=5,
            max_n=100,
            base_ratio=0.50,  # base_n = 30
            flat_margin=0.035,
        )
        assert n <= 60  # 翻倍是 60, 不超过 universe

    def test_single_element(self):
        scores = torch.tensor([0.5])
        n = choose_adaptive_top_n(scores, universe_size=1, min_n=32)
        assert n == 1
