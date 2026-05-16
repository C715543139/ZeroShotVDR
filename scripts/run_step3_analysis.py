"""Step 3.2 结果分析脚本：曲线绘制、bad case 汇总与代表性样本筛选。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from zeroshot_vdr.contracts import build_page_id, normalize_doc_id
from zeroshot_vdr.data.adapters import DocumentQAAdapter
from zeroshot_vdr.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 Step 3.2 分析：绘图、bad case 汇总与代表性样本筛选。"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="outputs/eval_reports/step3_docqa_full_dual3090",
        help="Step 3.1 输出目录",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="按哪个 top-k 定义 bad case，默认 10",
    )
    parser.add_argument(
        "--representatives-per-subtask",
        type=int,
        default=3,
        help="每个子任务每类失败模式保留多少代表性样本",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args()


def _parse_page_idx(page_id: str) -> int | None:
    try:
        page_part = page_id.rsplit("/", 1)[-1]
        if page_part.startswith("p"):
            return int(page_part[1:])
    except Exception:
        return None
    return None


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _frame_to_text(df: pd.DataFrame, float_digits: int = 4) -> str:
    rendered = df.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: _format_float(value, float_digits))
    return rendered.to_string(index=False)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_page_lookup(run_summary: dict[str, Any]) -> dict[str, Any]:
    adapter = DocumentQAAdapter(
        data_dir=run_summary["data_dir"],
        subtasks=run_summary["scope"]["subtasks"],
        lengths=run_summary["scope"]["lengths"],
    )
    return {page.page_id: page for page in adapter.iter_pages()}


def _build_page_id_stability_summary(run_summary: dict[str, Any]) -> pd.DataFrame:
    data_dir = Path(run_summary["data_dir"])
    jsonl_dir = data_dir / "mmlb_data" / "documentQA"
    rows: list[dict[str, Any]] = []

    for subtask in run_summary["scope"]["subtasks"]:
        for length in run_summary["scope"]["lengths"]:
            jsonl_path = jsonl_dir / f"{subtask}_{length}.jsonl"
            page_map: dict[str, set[str]] = defaultdict(set)

            with open(jsonl_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    sample = json.loads(line)
                    doc_id = normalize_doc_id(sample.get("doc_name", ""))
                    for page_idx, rel_path in enumerate(sample.get("page_list", [])):
                        page_id = build_page_id("docqa", subtask, length, doc_id, page_idx)
                        page_map[page_id].add(rel_path)

            total_page_ids = len(page_map)
            unstable_page_ids = sum(1 for paths in page_map.values() if len(paths) > 1)
            rows.append(
                {
                    "subtask": subtask,
                    "length": length,
                    "page_ids": total_page_ids,
                    "unstable_page_ids": unstable_page_ids,
                    "unstable_rate": unstable_page_ids / total_page_ids if total_page_ids else 0.0,
                }
            )

    return pd.DataFrame(rows).sort_values(["unstable_rate", "unstable_page_ids"], ascending=[False, False])


def _build_case_frame(
    retrieval_details: list[dict[str, Any]],
    page_lookup: dict[str, Any],
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for detail in retrieval_details:
        results = detail.get("results", [])[:top_k]
        relevant_page_ids = detail.get("relevant_page_ids", [])
        relevant_set = set(relevant_page_ids)
        retrieved_page_ids = [item["page_id"] for item in results]

        hit_ranks = [
            index + 1
            for index, page_id in enumerate(retrieved_page_ids)
            if page_id in relevant_set
        ]
        recall_at_k = (
            len(hit_ranks) / len(relevant_set)
            if relevant_set
            else None
        )
        first_relevant_rank = min(hit_ranks) if hit_ranks else None

        top1 = results[0] if results else None
        best_relevant = next(
            (item for item in results if item["page_id"] in relevant_set),
            None,
        )

        top1_page_id = top1["page_id"] if top1 else None
        top1_score = float(top1["score"]) if top1 else None
        best_relevant_page_id = best_relevant["page_id"] if best_relevant else None
        best_relevant_score = (
            float(best_relevant["score"])
            if best_relevant
            else None
        )
        score_gap_to_best_relevant = (
            top1_score - best_relevant_score
            if top1_score is not None and best_relevant_score is not None
            else None
        )

        relevant_page_indices = [
            page_idx
            for page_idx in (_parse_page_idx(page_id) for page_id in relevant_page_ids)
            if page_idx is not None
        ]
        answer_span = (
            max(relevant_page_indices) - min(relevant_page_indices)
            if relevant_page_indices
            else None
        )

        if not relevant_set:
            failure_mode = "no_ground_truth"
        elif recall_at_k >= 1.0:
            failure_mode = "ok"
        elif len(relevant_set) > 1 and hit_ranks:
            failure_mode = "multi_page_partial"
        elif hit_ranks:
            failure_mode = "late_hit"
        else:
            failure_mode = "miss_top10"

        top1_page = page_lookup.get(top1_page_id) if top1_page_id else None
        relevant_image_paths = [
            page_lookup[page_id].image_path
            for page_id in relevant_page_ids
            if page_id in page_lookup
        ]

        rows.append(
            {
                "query_id": detail["query_id"],
                "subtask": detail["subtask"],
                "length": detail["length"],
                "doc_id": detail["doc_id"],
                "question": detail["question"],
                "candidate_pages": detail["candidate_pages"],
                "latency_s": detail["latency_s"],
                "num_relevant_pages": len(relevant_set),
                "recall_at_k": recall_at_k,
                "first_relevant_rank": first_relevant_rank,
                "failure_mode": failure_mode,
                "top1_page_id": top1_page_id,
                "top1_score": top1_score,
                "best_relevant_page_id": best_relevant_page_id,
                "best_relevant_score": best_relevant_score,
                "score_gap_to_best_relevant": score_gap_to_best_relevant,
                "answer_span": answer_span,
                "relevant_page_ids": " | ".join(relevant_page_ids),
                "top1_image_path": top1_page.image_path if top1_page else None,
                "relevant_image_paths": " | ".join(relevant_image_paths),
            }
        )

    return pd.DataFrame(rows)


def _build_bad_case_summary(case_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        case_frame.assign(
            has_ground_truth=lambda df: df["num_relevant_pages"] > 0,
            is_bad=lambda df: (df["num_relevant_pages"] > 0) & (df["recall_at_k"] < 1.0),
        )
        .groupby(["subtask", "length"], as_index=False)
        .agg(
            queries=("query_id", "count"),
            queries_with_ground_truth=("has_ground_truth", "sum"),
            no_ground_truth=("failure_mode", lambda values: int((values == "no_ground_truth").sum())),
            bad_cases=("is_bad", "sum"),
            miss_top10=("failure_mode", lambda values: int((values == "miss_top10").sum())),
            late_hit=("failure_mode", lambda values: int((values == "late_hit").sum())),
            multi_page_partial=("failure_mode", lambda values: int((values == "multi_page_partial").sum())),
            mean_candidate_pages=("candidate_pages", "mean"),
            mean_latency_s=("latency_s", "mean"),
        )
    )
    grouped["bad_case_rate"] = grouped.apply(
        lambda row: row["bad_cases"] / row["queries_with_ground_truth"]
        if row["queries_with_ground_truth"]
        else 0.0,
        axis=1,
    )
    return grouped.sort_values(["bad_case_rate", "mean_candidate_pages"], ascending=[False, False])


def _select_representatives(
    bad_cases: pd.DataFrame,
    representatives_per_subtask: int,
) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    candidate_groups = bad_cases[bad_cases["failure_mode"] != "multi_page_partial"]

    for failure_mode in ["miss_top10", "late_hit", "multi_page_partial"]:
        current = bad_cases[bad_cases["failure_mode"] == failure_mode]
        if current.empty:
            continue

        for subtask in sorted(current["subtask"].unique()):
            subset = current[current["subtask"] == subtask].copy()
            subset = subset.sort_values(
                by=["candidate_pages", "score_gap_to_best_relevant", "latency_s"],
                ascending=[False, False, False],
                na_position="last",
            )
            selected_frames.append(subset.head(representatives_per_subtask))

    if not selected_frames:
        return bad_cases.head(0).copy()

    representatives = pd.concat(selected_frames, ignore_index=True)
    representatives = representatives.drop_duplicates(subset=["query_id"])
    return representatives.sort_values(["failure_mode", "subtask", "length", "candidate_pages"], ascending=[True, True, True, False])


def _plot_group_curves(
    metrics_summary: pd.DataFrame,
    group_type: str,
    output_path: Path,
    title: str,
) -> None:
    metrics = ["Recall", "Precision", "nDCG"]
    subset = metrics_summary[metrics_summary["group_type"] == group_type].copy()
    if subset.empty:
        return

    groups = list(dict.fromkeys(subset["group"].tolist()))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for axis, metric in zip(axes, metrics):
        for group in groups:
            group_frame = subset[subset["group"] == group].sort_values("k")
            axis.plot(group_frame["k"], group_frame[metric], marker="o", linewidth=2, label=group)
        axis.set_title(metric)
        axis.set_xlabel("k")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_curves(metrics_summary: pd.DataFrame, output_path: Path) -> None:
    subset = metrics_summary[metrics_summary["group_type"] == "overall"].sort_values("k")
    if subset.empty:
        return

    fig, axis = plt.subplots(figsize=(8, 5))
    for metric in ["Recall", "Precision", "nDCG"]:
        axis.plot(subset["k"], subset[metric], marker="o", linewidth=2, label=metric)
    axis.set_title("Overall k-Metric Curves")
    axis.set_xlabel("k")
    axis.set_ylabel("Score")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    mrr = float(subset.iloc[0]["MRR"])
    axis.text(
        0.98,
        0.04,
        f"MRR = {mrr:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_k32_subtask(metrics_summary: pd.DataFrame, output_path: Path) -> None:
    subset = metrics_summary[
        (metrics_summary["group_type"] == "subtask_length")
        & (metrics_summary["length"] == "K32")
    ].copy()
    if subset.empty:
        return

    subset = subset.sort_values(["subtask", "k"])
    pivots = {
        metric: subset.pivot(index="subtask", columns="k", values=metric)
        for metric in ["Recall", "nDCG"]
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (metric, pivot) in zip(axes, pivots.items()):
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(f"K32 {metric} by Subtask")
        axis.set_xlabel("subtask")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="k", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary_markdown(
    output_path: Path,
    run_summary: dict[str, Any],
    metrics_summary: pd.DataFrame,
    bad_case_summary: pd.DataFrame,
    representatives: pd.DataFrame,
    page_id_stability: pd.DataFrame,
    top_k: int,
) -> None:
    overall = metrics_summary[metrics_summary["group_type"] == "overall"].copy()
    k32_subtask = metrics_summary[
        (metrics_summary["group_type"] == "subtask_length")
        & (metrics_summary["length"] == "K32")
    ][["subtask", "k", "Recall", "Precision", "MRR", "nDCG"]].copy()

    total_queries = int(run_summary["scope_stats"]["selected_queries"])
    queries_with_ground_truth = int(bad_case_summary["queries_with_ground_truth"].sum())
    no_ground_truth = int(bad_case_summary["no_ground_truth"].sum())
    bad_cases = int(bad_case_summary["bad_cases"].sum())
    bad_case_rate = bad_cases / queries_with_ground_truth if queries_with_ground_truth else 0.0

    hotspot_summary = bad_case_summary.head(8)[
        [
            "subtask",
            "length",
            "bad_cases",
            "queries",
            "queries_with_ground_truth",
            "no_ground_truth",
            "bad_case_rate",
            "miss_top10",
            "late_hit",
            "multi_page_partial",
        ]
    ]
    representative_view = representatives[
        [
            "query_id",
            "subtask",
            "length",
            "failure_mode",
            "candidate_pages",
            "first_relevant_rank",
            "score_gap_to_best_relevant",
            "question",
            "top1_image_path",
            "relevant_image_paths",
        ]
    ].copy()
    stability_view = page_id_stability[
        ["subtask", "length", "unstable_page_ids", "page_ids", "unstable_rate"]
    ].copy()

    content = [
        "# Step 3.2 Analysis Summary",
        "",
        "## Run Snapshot",
        f"- Run: {run_summary['run_name']}",
        f"- Queries: {run_summary['scope_stats']['selected_queries']}",
        f"- Docs: {run_summary['scope_stats']['selected_docs']}",
        f"- Pages: {run_summary['scope_stats']['required_pages']}",
        f"- Index build time: {run_summary['index']['index_build_time_s']:.1f}s",
        f"- Retrieval time: {run_summary['retrieval']['total_time_s']:.1f}s",
        f"- Avg latency: {run_summary['retrieval']['avg_latency_s']:.3f}s/query",
        f"- P95 latency: {run_summary['retrieval']['p95_latency_s']:.3f}s/query",
        "",
        "## Overall Metrics",
        "```text",
        _frame_to_text(overall[["k", "Recall", "Precision", "MRR", "nDCG", "n_queries"]]),
        "```",
        "",
        "## K32 Subtask Snapshot",
        "```text",
        _frame_to_text(k32_subtask),
        "```",
        "",
        "## Historical Page ID Stability Baseline",
        "- This intentionally recomputes the old doc-local page_id semantics to show why the redesign was necessary.",
        "- High instability here is historical evidence for the previous `doc_id/page_idx` contract, not a defect in the current stable-page-id run.",
        "```text",
        _frame_to_text(stability_view),
        "```",
        "",
        f"## Bad Cases at Recall@{top_k} < 1.0",
        f"- Queries with usable ground truth: {queries_with_ground_truth}/{total_queries}",
        f"- Queries with missing/invalid ground truth: {no_ground_truth}/{total_queries}",
        f"- True bad cases among queries with usable ground truth: {bad_cases}/{queries_with_ground_truth} ({bad_case_rate:.2%})",
        "- Hotspots below are sorted by bad case rate over queries with usable ground truth.",
        "```text",
        _frame_to_text(hotspot_summary),
        "```",
        "",
        "## Representative Cases for Manual Review",
        "```text",
        _frame_to_text(representative_view, float_digits=3),
        "```",
        "",
        "## Generated Artifacts",
        "- plots/overall_k_curves.png",
        "- plots/subtask_k_curves.png",
        "- plots/length_k_curves.png",
        "- plots/k32_subtask_comparison.png",
        "- bad_cases_all.csv",
        "- no_ground_truth_cases.csv",
        "- bad_case_summary.csv",
        "- representative_bad_cases.csv",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    log_level = getattr(__import__("logging"), args.log_level)
    logger = setup_logging("step3_analysis", level=log_level)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"结果目录不存在: {run_dir}")

    analysis_dir = run_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_summary = _load_json(run_dir / "run_summary.json")
    retrieval_details = _load_json(run_dir / "retrieval_details.json")
    metrics_summary = pd.read_csv(run_dir / "metrics_summary.csv")

    logger.info("加载页面映射，用于 bad case 样本追踪")
    page_lookup = _load_page_lookup(run_summary)

    logger.info("构建 bad case 明细与汇总")
    case_frame = _build_case_frame(retrieval_details, page_lookup, top_k=args.top_k)
    no_ground_truth_cases = case_frame[case_frame["failure_mode"] == "no_ground_truth"].copy()
    bad_cases = case_frame[
        (case_frame["num_relevant_pages"] > 0)
        & (case_frame["recall_at_k"] < 1.0)
    ].copy()
    bad_case_summary = _build_bad_case_summary(case_frame)
    representatives = _select_representatives(
        bad_cases,
        representatives_per_subtask=args.representatives_per_subtask,
    )
    page_id_stability = _build_page_id_stability_summary(run_summary)

    case_frame.to_csv(analysis_dir / "all_cases.csv", index=False, encoding="utf-8-sig")
    bad_cases.to_csv(analysis_dir / "bad_cases_all.csv", index=False, encoding="utf-8-sig")
    no_ground_truth_cases.to_csv(analysis_dir / "no_ground_truth_cases.csv", index=False, encoding="utf-8-sig")
    bad_case_summary.to_csv(analysis_dir / "bad_case_summary.csv", index=False, encoding="utf-8-sig")
    representatives.to_csv(analysis_dir / "representative_bad_cases.csv", index=False, encoding="utf-8-sig")
    page_id_stability.to_csv(analysis_dir / "page_id_stability_summary.csv", index=False, encoding="utf-8-sig")

    logger.info("绘制 Step 3.2 曲线图")
    _plot_overall_curves(metrics_summary, plots_dir / "overall_k_curves.png")
    _plot_group_curves(metrics_summary, "subtask", plots_dir / "subtask_k_curves.png", "k-Metric Curves by Subtask")
    _plot_group_curves(metrics_summary, "length", plots_dir / "length_k_curves.png", "k-Metric Curves by Length")
    _plot_k32_subtask(metrics_summary, plots_dir / "k32_subtask_comparison.png")

    _write_summary_markdown(
        analysis_dir / "step3_2_analysis_summary.md",
        run_summary,
        metrics_summary,
        bad_case_summary,
        representatives,
        page_id_stability,
        top_k=args.top_k,
    )

    logger.info("Step 3.2 分析完成，产物写入: %s", analysis_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())