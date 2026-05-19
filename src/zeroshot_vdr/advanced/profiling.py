"""Phase 4 Profiling: per-query trace 统计与 slice-level 分析工具。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_traces(trace_path: str | Path) -> list[dict[str, Any]]:
    """加载 phase4_trace.jsonl 为 dict 列表。"""
    records: list[dict[str, Any]] = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_slice_metrics(
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 subtask + length 分组统计 trace 指标。

    返回每组的聚合指标列表，字段包括:
    - slice_name, num_queries
    - Recall@1/5/10, MRR, nDCG@10（需 trace 中有 hit_at_k 字段）
    - avg_latency_ms, avg_coarse_ms, avg_rerank_ms
    - avg_universe_size, avg_coarse_top_n, avg_expanded_candidates
    - avg_neighbor_added

    Parameters
    ----------
    traces : list[dict]
        load_traces() 返回的 trace 记录列表

    Returns
    -------
    list[dict]
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for t in traces:
        key = f"{t.get('subtask', '?')}/{t.get('length', '?')}"
        groups[key].append(t)

    rows: list[dict[str, Any]] = []
    for slice_name, group in sorted(groups.items()):
        n = len(group)
        row: dict[str, Any] = {"slice_name": slice_name, "num_queries": n}

        # 命中率
        for k in [1, 5, 10]:
            key = f"hit_at_{k}"
            if all(key in t for t in group):
                row[f"Recall@{k}"] = mean(t[key] for t in group)

        # MRR（简化：用 hit_at_k 近似，k=10 时 1/rank；精确 MRR 需要 rank 信息）
        # 这里用 Recall@1 作为 MRR 的下界近似
        if all("hit_at_1" in t for t in group):
            # 简化 MRR 估算: 假设命中 rank 均匀分布
            hits = [t for t in group if t.get("hit_at_10")]
            row["hit_rate@10"] = len(hits) / n if n > 0 else 0.0

        # 延迟
        for field, label in [
            ("total_ms", "avg_latency_ms"),
            ("coarse_ms", "avg_coarse_ms"),
            ("rerank_ms", "avg_rerank_ms"),
        ]:
            if all(field in t for t in group):
                row[label] = mean(t[field] for t in group)

        # 候选统计
        for field, label in [
            ("universe_size", "avg_universe_size"),
            ("coarse_top_n", "avg_coarse_top_n"),
            ("expanded_candidate_count", "avg_expanded_candidates"),
            ("neighbor_added_count", "avg_neighbor_added"),
        ]:
            if all(field in t for t in group):
                row[label] = mean(t[field] for t in group)

        rows.append(row)

    return rows


def compute_universe_bucket_metrics(
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 universe_size 分桶统计指标。

    分桶: K8 (≤8), K16 (9-16), K32 (17-32), K64 (33-64),
          K128 (65-128), other (>128)
    """
    buckets = {
        "K8": (0, 8),
        "K16": (9, 16),
        "K32": (17, 32),
        "K64": (33, 64),
        "K128": (65, 128),
        "other": (129, 999999),
    }

    grouped: dict[str, list[dict]] = {k: [] for k in buckets}
    for t in traces:
        us = t.get("universe_size", 0)
        for bucket, (lo, hi) in buckets.items():
            if lo <= us <= hi:
                grouped[bucket].append(t)
                break

    rows: list[dict[str, Any]] = []
    for bucket, group in grouped.items():
        if not group:
            continue
        n = len(group)
        row: dict[str, Any] = {"bucket": bucket, "num_queries": n}
        for k in [1, 5, 10]:
            key = f"hit_at_{k}"
            if all(key in t for t in group):
                row[f"Recall@{k}"] = mean(t[key] for t in group)
        if all("total_ms" in t for t in group):
            row["avg_latency_ms"] = mean(t["total_ms"] for t in group)
        rows.append(row)

    return rows

