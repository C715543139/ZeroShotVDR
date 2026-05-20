"""Phase 4 Profiling: per-query trace 统计与 slice-level 分析工具。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _dcg(relevances: list[float]) -> float:
    total = 0.0
    for index, relevance in enumerate(relevances, start=1):
        total += relevance / math.log2(index + 1)
    return total


def _hit_at_k(trace: dict[str, Any], k: int) -> float:
    gt_ids = set(trace.get("gt_page_ids", []))
    pred_ids = trace.get("pred_page_ids", [])
    if not gt_ids:
        return 0.0
    return float(any(page_id in gt_ids for page_id in pred_ids[:k]))


def _mrr(trace: dict[str, Any]) -> float:
    gt_ids = set(trace.get("gt_page_ids", []))
    pred_ids = trace.get("pred_page_ids", [])
    if not gt_ids:
        return 0.0
    for rank, page_id in enumerate(pred_ids, start=1):
        if page_id in gt_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_10(trace: dict[str, Any]) -> float:
    gt_ids = set(trace.get("gt_page_ids", []))
    pred_ids = trace.get("pred_page_ids", [])
    if not gt_ids:
        return 0.0
    relevances = [1.0 if page_id in gt_ids else 0.0 for page_id in pred_ids[:10]]
    ideal = [1.0] * min(len(gt_ids), 10)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return _dcg(relevances) / ideal_dcg


def _method_label(group: list[dict[str, Any]]) -> str:
    methods = sorted({trace.get("method", "?") for trace in group})
    return methods[0] if len(methods) == 1 else ",".join(methods)


def _aggregate_group(
    group_type: str,
    slice_name: str,
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    total_ms = [float(trace.get("total_ms", 0.0)) for trace in group]
    row: dict[str, Any] = {
        "method": _method_label(group),
        "group_type": group_type,
        "slice_name": slice_name,
        "num_queries": len(group),
        "Recall@1": mean(_hit_at_k(trace, 1) for trace in group),
        "Recall@5": mean(_hit_at_k(trace, 5) for trace in group),
        "Recall@10": mean(_hit_at_k(trace, 10) for trace in group),
        "MRR": mean(_mrr(trace) for trace in group),
        "nDCG@10": mean(_ndcg_at_10(trace) for trace in group),
        "Avg latency": mean(total_ms),
        "P95 latency": _percentile(total_ms, 0.95),
        "Avg universe size": mean(float(trace.get("universe_size", 0.0)) for trace in group),
        "Avg rerank candidates": mean(
            float(trace.get("expanded_candidate_count", 0.0)) for trace in group
        ),
        "Avg neighbor added": mean(
            float(trace.get("neighbor_added_count", 0.0)) for trace in group
        ),
        "Avg coarse ms": mean(float(trace.get("coarse_ms", 0.0)) for trace in group),
        "Avg rerank ms": mean(float(trace.get("rerank_ms", 0.0)) for trace in group),
    }
    return row


def _bucket_name(universe_size: int) -> str:
    if universe_size <= 8:
        return "K8"
    if universe_size <= 16:
        return "K16"
    if universe_size <= 32:
        return "K32"
    if universe_size <= 64:
        return "K64"
    if universe_size <= 128:
        return "K128"
    return "other"


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
    """按文档要求的 slice 维度统计 trace 指标。

    分组包含：
    - task_family
    - task_family + length
    - subtask + length

    Parameters
    ----------
    traces : list[dict]
        load_traces() 返回的 trace 记录列表

    Returns
    -------
    list[dict]
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for t in traces:
        task_family = t.get("task_family", "?")
        subtask = t.get("subtask", "?")
        length = t.get("length", "?")
        groups[("task_family", str(task_family))].append(t)
        groups[("task_family_length", f"{task_family}/{length}")].append(t)
        groups[("subtask_length", f"{subtask}/{length}")].append(t)

    rows: list[dict[str, Any]] = []
    for (group_type, slice_name), group in sorted(groups.items()):
        rows.append(_aggregate_group(group_type, slice_name, group))

    return rows


def compute_universe_bucket_metrics(
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 universe_size bucket 输出与 slice_metrics 同 schema 的指标。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        grouped[_bucket_name(int(t.get("universe_size", 0)))].append(t)

    rows: list[dict[str, Any]] = []
    for bucket, group in sorted(grouped.items()):
        if not group:
            continue
        rows.append(_aggregate_group("universe_bucket", bucket, group))

    return rows

