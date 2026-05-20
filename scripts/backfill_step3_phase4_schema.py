"""为 Phase 3 stable run 回填与 Phase 4 兼容的 valid-only 产物。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 Phase 3 stable run 生成与 Phase 4 同 schema 的 valid-only 产物"
    )
    parser.add_argument(
        "--step3-run-dir",
        type=str,
        default="outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids",
        help="Phase 3 stable run 目录",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录；默认 <step3-run-dir>/analysis/phase4_schema_valid_only",
    )
    return parser.parse_args()


def dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def recall_at_k(pred_ids: list[str], gt_ids: set[str], k: int) -> float:
    if not gt_ids:
        return 1.0
    return len(set(pred_ids[:k]) & gt_ids) / len(gt_ids)


def precision_at_k(pred_ids: list[str], gt_ids: set[str], k: int) -> float:
    if not gt_ids:
        return 0.0
    return len(set(pred_ids[:k]) & gt_ids) / k


def mrr(pred_ids: list[str], gt_ids: set[str]) -> float:
    if not gt_ids:
        return 1.0
    for rank, page_id in enumerate(pred_ids, start=1):
        if page_id in gt_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pred_ids: list[str], gt_ids: set[str], k: int) -> float:
    if not gt_ids:
        return 1.0
    rels = [1.0 if pid in gt_ids else 0.0 for pid in pred_ids[:k]]
    ideal = [1.0] * min(len(gt_ids), k)
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return dcg(rels) / ideal_dcg


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def compute_slice_metrics(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        key = f"{trace.get('subtask', '?')}/{trace.get('length', '?')}"
        groups[key].append(trace)

    rows: list[dict[str, Any]] = []
    for slice_name, group in sorted(groups.items()):
        row: dict[str, Any] = {
            "slice_name": slice_name,
            "num_queries": len(group),
        }
        for k in [1, 5, 10]:
            key = f"hit_at_{k}"
            row[f"Recall@{k}"] = mean(bool(item[key]) for item in group)
        row["hit_rate@10"] = mean(bool(item["hit_at_10"]) for item in group)
        row["avg_latency_ms"] = mean(item["total_ms"] for item in group)
        row["avg_coarse_ms"] = mean(item["coarse_ms"] for item in group)
        row["avg_rerank_ms"] = mean(item["rerank_ms"] for item in group)
        row["avg_universe_size"] = mean(item["universe_size"] for item in group)
        row["avg_coarse_top_n"] = mean(item["coarse_top_n"] for item in group)
        row["avg_expanded_candidates"] = mean(
            item["expanded_candidate_count"] for item in group
        )
        row["avg_neighbor_added"] = mean(item["neighbor_added_count"] for item in group)
        rows.append(row)
    return rows


def compute_universe_bucket_metrics(
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets = {
        "K8": (0, 8),
        "K16": (9, 16),
        "K32": (17, 32),
        "K64": (33, 64),
        "K128": (65, 128),
        "other": (129, 999999),
    }
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in buckets}
    for trace in traces:
        universe_size = trace.get("universe_size", 0)
        for bucket, (lower, upper) in buckets.items():
            if lower <= universe_size <= upper:
                grouped[bucket].append(trace)
                break

    rows: list[dict[str, Any]] = []
    for bucket, group in grouped.items():
        if not group:
            continue
        row: dict[str, Any] = {
            "bucket": bucket,
            "num_queries": len(group),
            "avg_latency_ms": mean(item["total_ms"] for item in group),
        }
        for k in [1, 5, 10]:
            key = f"hit_at_{k}"
            row[f"Recall@{k}"] = mean(bool(item[key]) for item in group)
        rows.append(row)
    return rows


def load_records(step3_run_dir: Path) -> list[dict[str, Any]]:
    details_path = step3_run_dir / "retrieval_details.json"
    if not details_path.exists():
        raise FileNotFoundError(f"缺少 retrieval_details.json: {details_path}")
    return json.loads(details_path.read_text(encoding="utf-8"))


def filter_valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("relevant_page_ids")]


def build_trace_records(valid_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for record in valid_records:
        pred_page_ids = [item["page_id"] for item in record.get("results", [])]
        gt_page_ids = sorted(set(record.get("relevant_page_ids", [])))
        task_family = record.get("query_id", "").split("/", 1)[0] or "docqa"
        latency_ms = float(record.get("latency_s", 0.0)) * 1000.0
        candidate_pages = int(record.get("candidate_pages", 0))
        traces.append({
            "query_id": record["query_id"],
            "task_family": task_family,
            "subtask": record["subtask"],
            "length": record["length"],
            "method": "phase3_full_maxsim",
            "universe_size": candidate_pages,
            "coarse_top_n": candidate_pages,
            "expanded_candidate_count": candidate_pages,
            "neighbor_added_count": 0,
            "coarse_ms": 0.0,
            "rerank_ms": round(latency_ms, 3),
            "total_ms": round(latency_ms, 3),
            "top1_coarse_score": None,
            "topn_coarse_score": None,
            "coarse_margin": None,
            "adaptive_expanded": False,
            "hit_at_1": any(pid in gt_page_ids for pid in pred_page_ids[:1]),
            "hit_at_5": any(pid in gt_page_ids for pid in pred_page_ids[:5]),
            "hit_at_10": any(pid in gt_page_ids for pid in pred_page_ids[:10]),
            "gt_page_ids": gt_page_ids,
            "pred_page_ids": pred_page_ids[:10],
        })
    return traces


def _compute_rows(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    query_ids: list[str],
    k_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in k_values:
        recall_sum = 0.0
        precision_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        n = 0

        for qid in query_ids:
            pred_ids = retrieval_results[qid]
            gt_ids = ground_truth[qid]
            recall_sum += recall_at_k(pred_ids, gt_ids, k)
            precision_sum += precision_at_k(pred_ids, gt_ids, k)
            mrr_sum += mrr(pred_ids, gt_ids)
            ndcg_sum += ndcg_at_k(pred_ids, gt_ids, k)
            n += 1

        rows.append({
            "k": k,
            "Recall": recall_sum / n,
            "Precision": precision_sum / n,
            "MRR": mrr_sum / n,
            "nDCG": ndcg_sum / n,
            "n_queries": n,
        })
    return rows


def build_metrics_tables(valid_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    retrieval_results: dict[str, list[str]] = {}
    ground_truth: dict[str, set[str]] = {}
    query_lookup: dict[str, Any] = {}

    for record in valid_records:
        qid = record["query_id"]
        retrieval_results[qid] = [item["page_id"] for item in record.get("results", [])]
        ground_truth[qid] = set(record.get("relevant_page_ids", []))
        query_lookup[qid] = SimpleNamespace(
            subtask=record["subtask"],
            length=record["length"],
        )

    k_values = [1, 3, 5, 10]
    all_query_ids = sorted(retrieval_results)
    overall = _compute_rows(retrieval_results, ground_truth, all_query_ids, k_values)

    subtask_groups: dict[str, list[str]] = {}
    length_groups: dict[str, list[str]] = {}
    scope_groups: dict[str, list[str]] = {}
    for qid in all_query_ids:
        subtask_groups.setdefault(query_lookup[qid].subtask, []).append(qid)
        length_groups.setdefault(query_lookup[qid].length, []).append(qid)
        scope_groups.setdefault(
            f"{query_lookup[qid].subtask}/{query_lookup[qid].length}", []
        ).append(qid)

    by_subtask: list[dict[str, Any]] = []
    for group, query_ids in sorted(subtask_groups.items()):
        for row in _compute_rows(retrieval_results, ground_truth, query_ids, k_values):
            by_subtask.append({"group": group, **row})

    by_length: list[dict[str, Any]] = []
    for group, query_ids in sorted(length_groups.items()):
        for row in _compute_rows(retrieval_results, ground_truth, query_ids, k_values):
            by_length.append({"group": group, **row})

    by_scope: list[dict[str, Any]] = []
    for group, query_ids in sorted(scope_groups.items()):
        subtask, length = group.split("/", 1)
        for row in _compute_rows(retrieval_results, ground_truth, query_ids, k_values):
            by_scope.append({"group": group, "subtask": subtask, "length": length, **row})

    summary: list[dict[str, Any]] = []
    for row in overall:
        summary.append({
            "group_type": "overall",
            "group": None,
            "k": row["k"],
            "Recall": row["Recall"],
            "Precision": row["Precision"],
            "MRR": row["MRR"],
            "nDCG": row["nDCG"],
            "n_queries": row["n_queries"],
            "subtask": None,
            "length": None,
        })
    for row in by_subtask:
        summary.append({
            "group_type": "subtask",
            "group": row["group"],
            "k": row["k"],
            "Recall": row["Recall"],
            "Precision": row["Precision"],
            "MRR": row["MRR"],
            "nDCG": row["nDCG"],
            "n_queries": row["n_queries"],
            "subtask": None,
            "length": None,
        })
    for row in by_length:
        summary.append({
            "group_type": "length",
            "group": row["group"],
            "k": row["k"],
            "Recall": row["Recall"],
            "Precision": row["Precision"],
            "MRR": row["MRR"],
            "nDCG": row["nDCG"],
            "n_queries": row["n_queries"],
            "subtask": None,
            "length": None,
        })
    for row in by_scope:
        summary.append({
            "group_type": "subtask_length",
            "group": row["group"],
            "k": row["k"],
            "Recall": row["Recall"],
            "Precision": row["Precision"],
            "MRR": row["MRR"],
            "nDCG": row["nDCG"],
            "n_queries": row["n_queries"],
            "subtask": row["subtask"],
            "length": row["length"],
        })

    return {
        "overall": overall,
        "by_subtask": by_subtask,
        "by_length": by_length,
        "by_subtask_length": by_scope,
        "summary": summary,
    }


def save_rows_as_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_tables(out_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    for name, rows in tables.items():
        if not rows:
            continue
        save_rows_as_csv(out_dir / f"metrics_{name}.csv", rows)


def build_run_summary(
    step3_run_dir: Path,
    out_dir: Path,
    valid_records: list[dict[str, Any]],
    metrics_tables: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    latencies_s = [float(record.get("latency_s", 0.0)) for record in valid_records]
    candidate_counts = [int(record.get("candidate_pages", 0)) for record in valid_records]
    metrics_preview = metrics_tables["summary"][:24]

    return {
        "run_dir": str(out_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_run_dir": str(step3_run_dir),
        "phase4_schema_compatible": True,
        "phase4": False,
        "method": "phase3_full_maxsim",
        "scope": {
            "valid_only": True,
            "num_queries": len(valid_records),
            "k_values": [1, 3, 5, 10],
        },
        "candidate_stats": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "mean": mean(candidate_counts) if candidate_counts else 0.0,
        },
        "retrieval": {
            "avg_latency_s": mean(latencies_s) if latencies_s else 0.0,
            "p50_latency_s": percentile(latencies_s, 0.50),
            "p95_latency_s": percentile(latencies_s, 0.95),
            "max_latency_s": max(latencies_s) if latencies_s else 0.0,
        },
        "trace_enabled": True,
        "coarse_stats": {
            "avg_universe_size": mean(candidate_counts) if candidate_counts else 0.0,
            "avg_coarse_top_n": mean(candidate_counts) if candidate_counts else 0.0,
            "avg_expanded_candidates": mean(candidate_counts) if candidate_counts else 0.0,
            "avg_neighbor_added": 0.0,
            "avg_coarse_ms": 0.0,
            "avg_rerank_ms": mean(latencies_s) * 1000.0 if latencies_s else 0.0,
        },
        "metrics_preview": metrics_preview,
    }


def save_trace_analytics(out_dir: Path, traces: list[dict[str, Any]]) -> None:
    slice_metrics = compute_slice_metrics(traces)
    if slice_metrics:
        save_rows_as_csv(out_dir / "slice_metrics.csv", slice_metrics)

    bucket_metrics = compute_universe_bucket_metrics(traces)
    if bucket_metrics:
        save_rows_as_csv(out_dir / "bucket_metrics.csv", bucket_metrics)

    summary = {
        "num_traces": len(traces),
        "methods": ["phase3_full_maxsim"],
        "avg_total_ms": round(mean(t["total_ms"] for t in traces), 3) if traces else 0.0,
        "avg_coarse_ms": round(mean(t["coarse_ms"] for t in traces), 3) if traces else 0.0,
        "avg_rerank_ms": round(mean(t["rerank_ms"] for t in traces), 3) if traces else 0.0,
        "avg_universe_size": round(mean(t["universe_size"] for t in traces), 3) if traces else 0.0,
        "avg_coarse_top_n": round(mean(t["coarse_top_n"] for t in traces), 3) if traces else 0.0,
    }
    (out_dir / "trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    step3_run_dir = (PROJECT_ROOT / args.step3_run_dir).resolve()
    out_dir = (
        (PROJECT_ROOT / args.out_dir).resolve()
        if args.out_dir
        else step3_run_dir / "analysis" / "phase4_schema_valid_only"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(step3_run_dir)
    valid_records = filter_valid_records(records)
    if not valid_records:
        raise RuntimeError("没有可用的 valid-only 记录")

    traces = build_trace_records(valid_records)
    metrics_tables = build_metrics_tables(valid_records)
    save_metrics_tables(out_dir, metrics_tables)

    trace_path = out_dir / "phase4_trace.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(trace, ensure_ascii=False) for trace in traces) + "\n",
        encoding="utf-8",
    )

    save_trace_analytics(out_dir, traces)

    run_summary = build_run_summary(step3_run_dir, out_dir, valid_records, metrics_tables)
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已生成 valid-only phase4-compatible 产物: {out_dir}")
    print(f"  queries={len(valid_records)}")
    print(f"  trace={trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
