"""Phase 4 评测脚本: two-stage coarse-to-fine retrieval 评测。

基于 run_step3_eval.py 的数据加载和指标计算逻辑，
使用 TwoStageRetriever 替代 RetrievalPipeline 进行检索。
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"),
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sitecustomize  # noqa: F401
import pandas as pd

from zeroshot_vdr.advanced.two_stage import TwoStageRetriever, TwoStageOutput
from zeroshot_vdr.config import (
    get_evaluation_config,
    get_index_config,
    get_model_config,
    get_retrieval_config,
    load_config,
)
from zeroshot_vdr.data.adapters import DocumentQAAdapter
from zeroshot_vdr.evaluation.metrics import compute_all_metrics, compute_metrics_by_group
from zeroshot_vdr.indexing.encoder import PageEncoder
from zeroshot_vdr.indexing.store import IndexStore
from zeroshot_vdr.retrieval.pipeline import RetrievalPipeline
from zeroshot_vdr.utils import format_duration, resolve_path, setup_logging


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 two-stage coarse-to-fine retrieval 评测"
    )
    # 基础参数
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument(
        "--subtasks", nargs="+", default=None,
        help="评测的子任务列表",
    )
    parser.add_argument(
        "--lengths", nargs="+", default=None,
        help="评测的长度档位列表",
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="仅评测前 N 条查询（smoke test）",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=None,
        help="评测的 k 值列表",
    )
    parser.add_argument(
        "--index-dir", type=str, default=None,
        help="索引目录",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出根目录",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="输出子目录名；默认自动生成",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="覆盖模型设备",
    )
    parser.add_argument(
        "--skip-index-build", action="store_true",
        help="若索引缺页则直接报错",
    )

    # Phase 4 特有参数
    parser.add_argument(
        "--method", type=str, default="fixed_topn",
        choices=["fixed_topn", "adaptive", "adaptive_neighbors"],
        help="Phase 4 检索方法",
    )
    parser.add_argument(
        "--coarse-top-n", type=int, default=64,
        help="固定 coarse top-N（method=fixed_topn 时使用）",
    )
    parser.add_argument(
        "--min-candidates", type=int, default=32,
        help="adaptive 下限",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=128,
        help="adaptive 上限",
    )
    parser.add_argument(
        "--base-ratio", type=float, default=0.20,
        help="adaptive 基础比例",
    )
    parser.add_argument(
        "--flat-margin", type=float, default=0.035,
        help="adaptive 平坦阈值",
    )
    parser.add_argument(
        "--neighbor-window", type=int, default=0,
        help="邻页窗口大小",
    )
    parser.add_argument(
        "--neighbor-seed-n", type=int, default=8,
        help="邻页扩展的 seed 数量",
    )

    # 过滤与 trace
    parser.add_argument(
        "--valid-only", action="store_true",
        help="仅评测有有效 ground truth 的 query（14,385 条）",
    )
    parser.add_argument(
        "--trace-enabled", action="store_true",
        help="输出 per-query trace (phase4_trace.jsonl)",
    )

    # 日志
    parser.add_argument(
        "--resume", action="store_true",
        help="从上次中断处续跑（自动检测 _checkpoint.json）",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="忽略已有 checkpoint，从头重新跑",
    )

    # 日志
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ===========================================================================
# 辅助函数（复用自 run_step3_eval.py）
# ===========================================================================


def _percentile(values: list[float], q: float) -> float:
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


def _resolve_subtasks(args: argparse.Namespace, config: dict) -> list[str]:
    if args.subtasks:
        return args.subtasks
    return config.get("data", {}).get("subtasks", ["longdocurl", "mmlongdoc", "slidevqa"])


def _resolve_lengths(args: argparse.Namespace, config: dict) -> list[str]:
    if args.lengths:
        return args.lengths
    length_cfg = config.get("data", {}).get("length")
    if isinstance(length_cfg, str):
        return [length_cfg]
    if isinstance(length_cfg, list) and length_cfg:
        return length_cfg
    return ["K4", "K8", "K16", "K32", "K64", "K128"]


def _resolve_k_values(args: argparse.Namespace, evaluation_cfg: dict) -> list[int]:
    if args.k_values:
        return sorted(set(args.k_values))
    return evaluation_cfg.get("k_values", [1, 3, 5, 10])


def _parse_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    import torch
    mapping = {
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    normalized = dtype_name.lower()
    if normalized not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype_name}")
    return mapping[normalized]


def _make_run_name(
    subtasks: list[str],
    lengths: list[str],
    method: str,
    coarse_top_n: int,
    max_queries: int | None,
) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    subtask_label = "-".join(subtasks[:3]) if len(subtasks) <= 3 else f"{subtasks[0]}-plus"
    length_label = "-".join(lengths[:3]) if len(lengths) <= 3 else f"{lengths[0]}-plus"
    query_label = f"q{max_queries}" if max_queries is not None else "qall"
    topn_label = f"top{coarse_top_n}" if method == "fixed_topn" else method
    return f"phase4_{topn_label}_{subtask_label}_{length_label}_{query_label}_{timestamp}"


# ===========================================================================
# 指标计算
# ===========================================================================


def _build_metrics_tables(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    query_lookup: dict[str, Any],
    k_values: list[int],
) -> dict[str, pd.DataFrame]:
    overall = compute_all_metrics(retrieval_results, ground_truth, k_values=k_values)

    by_subtask = compute_metrics_by_group(
        retrieval_results, ground_truth,
        group_fn=lambda qid: query_lookup[qid].subtask,
        k_values=k_values,
    )
    by_length = compute_metrics_by_group(
        retrieval_results, ground_truth,
        group_fn=lambda qid: query_lookup[qid].length,
        k_values=k_values,
    )
    by_scope = compute_metrics_by_group(
        retrieval_results, ground_truth,
        group_fn=lambda qid: f"{query_lookup[qid].subtask}/{query_lookup[qid].length}",
        k_values=k_values,
    )
    if not by_scope.empty:
        split = by_scope["group"].str.split("/", n=1, expand=True)
        by_scope.insert(1, "subtask", split[0])
        by_scope.insert(2, "length", split[1])

    frames = []
    for name, df, gtype in [
        ("overall", overall, "overall"),
        ("by_subtask", by_subtask, "subtask"),
        ("by_length", by_length, "length"),
        ("by_subtask_length", by_scope, "subtask_length"),
    ]:
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "group_type", gtype)
        frames.append(df)

    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {
        "overall": overall,
        "by_subtask": by_subtask,
        "by_length": by_length,
        "by_subtask_length": by_scope,
        "summary": summary,
    }


def _save_metrics_tables(run_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        if df.empty:
            continue
        df.to_csv(run_dir / f"metrics_{name}.csv", index=False, encoding="utf-8-sig")


# ===========================================================================
# Valid-only 过滤
# ===========================================================================


def filter_valid_queries(
    queries: list,
    ground_truth: dict[str, set[str]],
    logger,
) -> tuple[list, dict[str, set[str]]]:
    """过滤掉没有有效 ground truth 的 query。"""
    valid_qids = {qid for qid, pages in ground_truth.items() if pages}
    valid_queries = [q for q in queries if q.query_id in valid_qids]
    valid_gt = {qid: ground_truth[qid] for qid in valid_qids if qid in valid_qids}

    removed = len(queries) - len(valid_queries)
    if removed > 0:
        logger.info(
            "valid-only 过滤: %d → %d queries（移除 %d 条无 ground truth）",
            len(queries), len(valid_queries), removed,
        )

    return valid_queries, valid_gt


# ===========================================================================
# Model / Index 加载（复用 run_step3_eval.py 逻辑）
# ===========================================================================


def _load_page_encoder(
    model_cfg: dict,
    index_cfg: dict,
    args: argparse.Namespace,
    *,
    device: str | None = None,
    batch_size: int | None = None,
):
    model_repo = model_cfg.get("repo", "vidore/colpali-v1.3")
    base_repo = model_cfg.get("base_repo", "vidore/colpaligemma-3b-pt-448-base")
    device = device or args.device or model_cfg.get("device", "cuda:0")
    dtype = _parse_torch_dtype(model_cfg.get("dtype"))
    batch_size = batch_size or index_cfg.get("batch_size", 4)
    storage_dtype = _parse_torch_dtype(index_cfg.get("storage_dtype", "float16"))

    try:
        return PageEncoder.from_pretrained(
            model_repo=model_repo,
            base_repo=base_repo,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            storage_dtype=storage_dtype,
        )
    except Exception as exc:
        raise RuntimeError(
            "加载 ColPali 模型失败。请先确认项目内缓存完整，并可先运行 scripts/test_model_load.py 验证。"
        ) from exc


def _get_device_memory_gb(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    try:
        import torch
    except Exception:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    if ":" in device:
        device_index = int(device.split(":", 1)[1])
    else:
        device_index = 0
    props = torch.cuda.get_device_properties(device_index)
    return props.total_memory / (1024**3)


def _recommend_score_batch_size(explicit_value: int | None, retrieval_device: str) -> int:
    if explicit_value is not None:
        return explicit_value
    memory_gb = _get_device_memory_gb(retrieval_device)
    if memory_gb >= 22:
        return 512
    if memory_gb >= 14:
        return 256
    return 64


def _split_pages_for_devices(pages: list, devices: list[str]) -> list[tuple[str, list]]:
    if not devices:
        return []
    shards: list[list] = [[] for _ in devices]
    for index, page in enumerate(pages):
        shards[index % len(devices)].append(page)
    return [(device, shard) for device, shard in zip(devices, shards) if shard]


def _encode_pages_worker(payload: dict) -> dict:
    device = payload["device"]
    pages = payload["pages"]
    if not pages:
        return {"device": device, "encoded_pages": 0}

    model_cfg = payload["model_cfg"]
    dtype = _parse_torch_dtype(model_cfg.get("dtype"))
    storage_dtype = _parse_torch_dtype(payload["storage_dtype"])

    encoder = PageEncoder.from_pretrained(
        model_repo=model_cfg.get("repo", "vidore/colpali-v1.3"),
        base_repo=model_cfg.get("base_repo", "vidore/colpaligemma-3b-pt-448-base"),
        device=device,
        dtype=dtype,
        batch_size=payload["page_batch_size"],
        storage_dtype=storage_dtype,
    )
    store_obj = IndexStore(payload["index_dir"])
    encoder.encode_corpus(pages, store_obj, show_progress=False, resume=False, update_manifest=False)
    return {"device": device, "encoded_pages": len(pages)}


# ===========================================================================
# Trace 保存
# ===========================================================================


# ===========================================================================
# Checkpoint / Resume
# ===========================================================================


def _checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "_checkpoint.json"


def _load_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    """加载断点文件，不存在则返回 None。"""
    cp_path = _checkpoint_path(run_dir)
    if not cp_path.exists():
        return None
    try:
        return json.loads(cp_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_checkpoint(
    run_dir: Path,
    completed_ids: list[str],
    retrieval_results: dict[str, list[str]],
    latencies: list[float],
    trace_lines: list[str],
    queries_done: int,
    queries_total: int,
    elapsed: float = 0.0,
) -> None:
    """保存断点信息。"""
    cp = {
        "completed_query_ids": completed_ids,
        "retrieval_results": retrieval_results,
        "latencies": latencies,
        "trace_lines": trace_lines,
        "queries_done": queries_done,
        "queries_total": queries_total,
        "elapsed_so_far": elapsed,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _checkpoint_path(run_dir).write_text(
        json.dumps(cp, ensure_ascii=False), encoding="utf-8"
    )


def _clear_checkpoint(run_dir: Path) -> None:
    """完成后清除断点文件。"""
    cp_path = _checkpoint_path(run_dir)
    if cp_path.exists():
        cp_path.unlink()


# ===========================================================================
# Main
# ===========================================================================


def main() -> int:
    args = parse_args()
    log_level = getattr(__import__("logging"), args.log_level)
    logger = setup_logging("phase4_eval", level=log_level)

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = get_model_config(config)
    index_cfg = get_index_config(config)
    retrieval_cfg = get_retrieval_config(config).copy()
    evaluation_cfg = get_evaluation_config(config)

    subtasks = _resolve_subtasks(args, config)
    lengths = _resolve_lengths(args, config)
    k_values = _resolve_k_values(args, evaluation_cfg)
    max_k = max(k_values)

    retrieval_device = args.device or model_cfg.get("device", "cuda:0")
    retrieval_cfg["score_batch_size"] = _recommend_score_batch_size(
        None, retrieval_device
    )

    data_dir = resolve_path(data_cfg.get("root_dir", "data/MMLongBench/raw"))
    index_dir = resolve_path(
        args.index_dir or index_cfg.get("dir", "data/processed/index")
    )
    output_root = resolve_path(
        args.output_dir or evaluation_cfg.get("output_dir", "outputs/eval_reports")
    )
    run_name = args.run_name or _make_run_name(
        subtasks, lengths, args.method, args.coarse_top_n, args.max_queries,
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 4 评测开始: method=%s subtasks=%s lengths=%s", args.method, subtasks, lengths)
    logger.info("输出目录: %s", run_dir)
    logger.info(
        "参数: coarse_top_n=%d min=%d max=%d ratio=%.2f margin=%.3f window=%d seed=%d",
        args.coarse_top_n, args.min_candidates, args.max_candidates,
        args.base_ratio, args.flat_margin,
        args.neighbor_window, args.neighbor_seed_n,
    )

    # ---- 加载数据 ----
    adapter = DocumentQAAdapter(
        data_dir=str(data_dir),
        subtasks=subtasks,
        lengths=lengths,
    )

    all_queries = list(adapter.iter_queries())
    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max-queries 必须为正整数")
        selected_queries = all_queries[:args.max_queries]
    else:
        selected_queries = all_queries

    if not selected_queries:
        raise RuntimeError("当前筛选条件下没有可评测的查询")

    selected_query_ids = {q.query_id for q in selected_queries}
    query_lookup = {q.query_id: q for q in selected_queries}

    # ---- Ground truth ----
    ground_truth_all = adapter.build_ground_truth()
    ground_truth = {qid: ground_truth_all.get(qid, set()) for qid in selected_query_ids}

    # ---- Valid-only 过滤 ----
    if args.valid_only:
        selected_queries, ground_truth = filter_valid_queries(
            selected_queries, ground_truth, logger,
        )
        selected_query_ids = {q.query_id for q in selected_queries}
        query_lookup = {q.query_id: q for q in selected_queries}
        if not selected_queries:
            raise RuntimeError("valid-only 过滤后没有可评测的查询")

    logger.info("评测范围: %d queries", len(selected_queries))

    # ---- 计算所需页面 ----
    all_pages = list(adapter.iter_pages())
    page_lookup = {page.page_id: page for page in all_pages}
    required_page_ids = {
        page_id
        for query in selected_queries
        for page_id in query.candidate_page_ids
    }
    required_pages = [
        page_lookup[page_id]
        for page_id in sorted(required_page_ids)
        if page_id in page_lookup
    ]

    # ---- 索引检查与补建 ----
    store = IndexStore(str(index_dir))
    store.recover_page_ids(sorted(required_page_ids))

    existing_page_ids = set(store.list_page_ids())
    missing_pages = [p for p in required_pages if p.page_id not in existing_page_ids]

    if missing_pages and args.skip_index_build:
        raise RuntimeError(
            f"索引中缺少当前评测范围的 {len(missing_pages)} 页，且指定了 --skip-index-build。"
        )

    if missing_pages:
        logger.info("开始补建索引: %d 页", len(missing_pages))
        page_encoder = _load_page_encoder(
            model_cfg, index_cfg, args, device=retrieval_device,
        )
        page_encoder.encode_corpus(missing_pages, store, show_progress=True, resume=True)
        scope_page_ids = store.list_page_ids()
        if scope_page_ids:
            dim = int(store.read_page(scope_page_ids[0]).shape[-1])
            store.save_meta(model_cfg.get("repo", "vidore/colpali-v1.3"), dim=dim)
    else:
        page_encoder = _load_page_encoder(
            model_cfg, index_cfg, args, device=retrieval_device,
        )

    # ---- 构建 Pipeline + TwoStageRetriever ----
    pipeline = RetrievalPipeline(
        model=page_encoder,
        index_store=store,
        config=retrieval_cfg,
    )

    retriever = TwoStageRetriever(
        base_pipeline=pipeline,
        index_store=store,
        method=args.method,
        coarse_top_n=args.coarse_top_n,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        base_ratio=args.base_ratio,
        flat_margin=args.flat_margin,
        neighbor_window=args.neighbor_window,
        neighbor_seed_n=args.neighbor_seed_n,
    )

    # ---- 检索循环（支持断点续跑） ----
    retrieval_results: dict[str, list[str]] = {}
    query_latencies: list[float] = []
    all_outputs: list[TwoStageOutput] = []
    trace_lines: list[str] = []

    # 尝试加载 checkpoint
    checkpoint = None
    prev_time = 0.0
    if not args.no_resume:
        checkpoint = _load_checkpoint(run_dir)
    if checkpoint is not None:
        retrieval_results = checkpoint.get("retrieval_results", {})
        query_latencies = checkpoint.get("latencies", [])
        trace_lines = checkpoint.get("trace_lines", [])
        completed_ids = set(checkpoint.get("completed_query_ids", []))
        done_count = checkpoint.get("queries_done", 0)
        total_count = checkpoint.get("queries_total", len(selected_queries))
        prev_time = checkpoint.get("elapsed_so_far", 0.0)
        logger.info(
            "从断点续跑: %d/%d queries 已完成", done_count, total_count,
        )
    else:
        completed_ids: set[str] = set()
        done_count = 0
        total_count = len(selected_queries)

    total_start = time.perf_counter()
    save_interval = max(1, min(50, len(selected_queries) // 20))  # 每 ~5% 存一次

    for i, query in enumerate(selected_queries):
        if query.query_id in completed_ids:
            continue  # 跳过已完成的 query

        output = retriever.retrieve(query, top_k=max_k)
        latency = output.trace.total_ms / 1000.0
        query_latencies.append(latency)
        retrieval_results[query.query_id] = [r.page_id for r in output.results]
        all_outputs.append(output)
        completed_ids.add(query.query_id)
        done_count += 1

        # 增量保存 trace 行
        if args.trace_enabled:
            gt_pages = ground_truth.get(query.query_id, set())
            pred_pages = [r.page_id for r in output.results]
            trace = output.trace
            record = {
                "query_id": query.query_id,
                "task_family": query.task_family,
                "subtask": query.subtask,
                "length": query.length,
                "method": trace.method,
                "universe_size": trace.universe_size,
                "coarse_top_n": trace.coarse_top_n,
                "expanded_candidate_count": trace.expanded_candidate_count,
                "neighbor_added_count": trace.neighbor_added_count,
                "coarse_ms": round(trace.coarse_ms, 3),
                "rerank_ms": round(trace.rerank_ms, 3),
                "total_ms": round(trace.total_ms, 3),
                "top1_coarse_score": (
                    round(trace.top1_coarse_score, 6)
                    if trace.top1_coarse_score is not None else None
                ),
                "topn_coarse_score": (
                    round(trace.topn_coarse_score, 6)
                    if trace.topn_coarse_score is not None else None
                ),
                "coarse_margin": (
                    round(trace.coarse_margin, 6)
                    if trace.coarse_margin is not None else None
                ),
                "adaptive_expanded": trace.adaptive_expanded,
                "hit_at_1": any(p in gt_pages for p in pred_pages[:1]),
                "hit_at_5": any(p in gt_pages for p in pred_pages[:5]),
                "hit_at_10": any(p in gt_pages for p in pred_pages[:10]),
                "gt_page_ids": sorted(gt_pages),
                "pred_page_ids": pred_pages[:10],
            }
            trace_lines.append(json.dumps(record, ensure_ascii=False))

        # 定期保存 checkpoint
        if done_count % save_interval == 0:
            _save_checkpoint(
                run_dir,
                completed_ids=sorted(completed_ids),
                retrieval_results=retrieval_results,
                latencies=query_latencies,
                trace_lines=trace_lines,
                queries_done=done_count,
                queries_total=len(selected_queries),
                elapsed=prev_time + (time.perf_counter() - total_start),
            )
            logger.debug(
                "checkpoint: %d/%d queries", done_count, len(selected_queries),
            )

    # 最终保存一次 checkpoint
    _save_checkpoint(
        run_dir,
        completed_ids=sorted(completed_ids),
        retrieval_results=retrieval_results,
        latencies=query_latencies,
        trace_lines=trace_lines,
        queries_done=done_count,
        queries_total=len(selected_queries),
        elapsed=prev_time + (time.perf_counter() - total_start),
    )

    total_time = time.perf_counter() - total_start

    # ---- 指标计算 ----
    tables = _build_metrics_tables(retrieval_results, ground_truth, query_lookup, k_values)
    _save_metrics_tables(run_dir, tables)

    # ---- Trace 输出 ----
    if args.trace_enabled and trace_lines:
        trace_path = run_dir / "phase4_trace.jsonl"
        trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
        logger.info("trace 已保存: %s (%d 行)", trace_path, len(trace_lines))

    # ---- 清除 checkpoint（正常完成） ----
    _clear_checkpoint(run_dir)

    # ---- 汇总 ----
    query_candidate_counts = {
        q.query_id: len(q.candidate_page_ids) for q in selected_queries
    }
    candidate_counts = list(query_candidate_counts.values())

    # 合并 checkpoint 中之前累积的时间
    prev_time_val = checkpoint.get("elapsed_so_far", 0.0) if checkpoint else 0.0

    run_summary: dict[str, Any] = {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase4": True,
        "method": args.method,
        "resumed": checkpoint is not None,
        "parameters": {
            "coarse_top_n": args.coarse_top_n,
            "min_candidates": args.min_candidates,
            "max_candidates": args.max_candidates,
            "base_ratio": args.base_ratio,
            "flat_margin": args.flat_margin,
            "neighbor_window": args.neighbor_window,
            "neighbor_seed_n": args.neighbor_seed_n,
        },
        "scope": {
            "subtasks": subtasks,
            "lengths": lengths,
            "max_queries": args.max_queries,
            "valid_only": args.valid_only,
            "k_values": k_values,
            "num_queries": len(selected_queries),
        },
        "candidate_stats": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "mean": mean(candidate_counts) if candidate_counts else 0.0,
        },
        "retrieval": {
            "total_time_s": total_time + prev_time_val,
            "session_time_s": total_time,
            "prev_time_s": prev_time_val,
            "avg_latency_s": mean(query_latencies) if query_latencies else 0.0,
            "p50_latency_s": _percentile(query_latencies, 0.50),
            "p95_latency_s": _percentile(query_latencies, 0.95),
            "max_latency_s": max(query_latencies) if query_latencies else 0.0,
        },
        "trace_enabled": args.trace_enabled,
    }

    # 聚合 trace 统计
    if all_outputs:
        traces = [o.trace for o in all_outputs]
        run_summary["coarse_stats"] = {
            "avg_universe_size": mean(t.universe_size for t in traces),
            "avg_coarse_top_n": mean(t.coarse_top_n for t in traces),
            "avg_expanded_candidates": mean(t.expanded_candidate_count for t in traces),
            "avg_neighbor_added": mean(t.neighbor_added_count for t in traces),
            "avg_coarse_ms": mean(t.coarse_ms for t in traces),
            "avg_rerank_ms": mean(t.rerank_ms for t in traces),
        }

    if not tables["summary"].empty:
        metrics_preview = tables["summary"].where(pd.notna(tables["summary"]), None)
        run_summary["metrics_preview"] = metrics_preview.to_dict(orient="records")

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # ---- 日志输出 ----
    logger.info("检索耗时: %s", format_duration(total_time))
    logger.info("平均延迟: %.3fs/query", run_summary["retrieval"]["avg_latency_s"])
    logger.info("P95 延迟: %.3fs", run_summary["retrieval"]["p95_latency_s"])

    overall = tables.get("overall")
    if overall is not None and not overall.empty:
        recall10_row = overall[overall["k"] == 10]
        if not recall10_row.empty:
            logger.info("Recall@10: %.4f", recall10_row["Recall"].values[0])
            logger.info("nDCG@10: %.4f", recall10_row["nDCG"].values[0])

    logger.info("结果已写入: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
