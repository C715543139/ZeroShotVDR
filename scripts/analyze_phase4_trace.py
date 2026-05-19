"""Phase 4 trace 分析脚本: 从 phase4_trace.jsonl 生成 slice-level 报告。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zeroshot_vdr.advanced.profiling import (
    compute_slice_metrics,
    compute_universe_bucket_metrics,
    load_traces,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 Phase 4 trace JSONL")
    parser.add_argument(
        "--trace", type=str, required=True,
        help="phase4_trace.jsonl 路径",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="输出目录；默认与 trace 同目录",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"错误: trace 文件不存在: {trace_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else trace_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = load_traces(trace_path)
    print(f"加载 {len(traces)} 条 trace")

    # slice 指标
    slice_metrics = compute_slice_metrics(traces)
    if slice_metrics:
        slice_path = out_dir / "slice_metrics.csv"
        keys = slice_metrics[0].keys()
        with open(slice_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            writer.writerows(slice_metrics)
        print(f"slice 指标已保存: {slice_path} ({len(slice_metrics)} 组)")

    # universe bucket 指标
    bucket_metrics = compute_universe_bucket_metrics(traces)
    if bucket_metrics:
        bucket_path = out_dir / "bucket_metrics.csv"
        keys = bucket_metrics[0].keys()
        with open(bucket_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            writer.writerows(bucket_metrics)
        print(f"bucket 指标已保存: {bucket_path} ({len(bucket_metrics)} 组)")

    # 汇总统计
    if traces:
        total = len(traces)
        methods = set(t.get("method", "?") for t in traces)
        avg_total_ms = sum(t.get("total_ms", 0) for t in traces) / total
        avg_coarse_ms = sum(t.get("coarse_ms", 0) for t in traces) / total
        avg_rerank_ms = sum(t.get("rerank_ms", 0) for t in traces) / total
        avg_universe = sum(t.get("universe_size", 0) for t in traces) / total
        avg_coarse_n = sum(t.get("coarse_top_n", 0) for t in traces) / total

        summary = {
            "num_traces": total,
            "methods": sorted(methods),
            "avg_total_ms": round(avg_total_ms, 3),
            "avg_coarse_ms": round(avg_coarse_ms, 3),
            "avg_rerank_ms": round(avg_rerank_ms, 3),
            "avg_universe_size": round(avg_universe, 1),
            "avg_coarse_top_n": round(avg_coarse_n, 1),
        }
        summary_path = out_dir / "trace_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"汇总已保存: {summary_path}")
        print(f"  平均 total_ms={avg_total_ms:.3f}, coarse_ms={avg_coarse_ms:.3f}, rerank_ms={avg_rerank_ms:.3f}")
        print(f"  平均 universe_size={avg_universe:.1f}, coarse_top_n={avg_coarse_n:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
