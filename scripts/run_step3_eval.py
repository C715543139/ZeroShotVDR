"""Step 3.1 评测脚本：DocumentQA 文档内页级检索。"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "运行 Step 3.1 DocumentQA 页级检索评测。"
            "默认强制 HuggingFace 离线，并在索引缺失时自动补建当前评测范围需要的页面。"
        )
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument(
        "--subtasks",
        nargs="+",
        default=None,
        help="评测的子任务列表，如 longdocurl mmlongdoc slidevqa",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=None,
        help="评测的长度档位列表，如 K4 K32 K128",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="仅评测前 N 条查询，用于 smoke test",
    )
    parser.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help="从第几条查询开始截取，默认 0",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help="评测的 k 值列表，默认读取 config/default.yaml",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="索引目录；默认读取配置中的 index.dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出根目录；默认读取配置中的 evaluation.output_dir",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="输出子目录名；默认自动生成",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="覆盖模型设备，如 cuda:0 或 cpu",
    )
    parser.add_argument(
        "--index-devices",
        nargs="+",
        default=None,
        help="索引构建使用的设备列表；默认自动使用全部可见 CUDA 设备",
    )
    parser.add_argument(
        "--page-batch-size",
        type=int,
        default=None,
        help="页面编码 batch size；未显式指定时按设备显存自动调优",
    )
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=None,
        help="MaxSim 评分 batch size；未显式指定时按检索设备自动调优",
    )
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        help="若索引缺页则直接报错，不自动补建",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="仅收集当前评测范围统计，不加载模型也不执行检索",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args()


def _resolve_subtasks(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if args.subtasks:
        return args.subtasks
    return config.get("data", {}).get(
        "subtasks", ["longdocurl", "mmlongdoc", "slidevqa"]
    )


def _resolve_lengths(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if args.lengths:
        return args.lengths

    length_cfg = config.get("data", {}).get("length")
    if isinstance(length_cfg, str):
        return [length_cfg]
    if isinstance(length_cfg, list) and length_cfg:
        return length_cfg
    return ["K4", "K8", "K16", "K32", "K64", "K128"]


def _resolve_k_values(args: argparse.Namespace, evaluation_cfg: dict[str, Any]) -> list[int]:
    if args.k_values:
        return sorted(set(args.k_values))
    return evaluation_cfg.get("k_values", [1, 3, 5, 10])


def _parse_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None

    import torch

    normalized = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype_name}")
    return mapping[normalized]


def _make_run_name(subtasks: list[str], lengths: list[str], max_queries: int | None) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    subtask_label = "-".join(subtasks[:3]) if len(subtasks) <= 3 else f"{subtasks[0]}-plus{len(subtasks)-1}"
    length_label = "-".join(lengths[:3]) if len(lengths) <= 3 else f"{lengths[0]}-plus{len(lengths)-1}"
    query_label = f"q{max_queries}" if max_queries is not None else "qall"
    return f"step3_1_{subtask_label}_{length_label}_{query_label}_{timestamp}"


def _page_scope_key(page) -> tuple[str, str, str, str]:
    return (page.task_family, page.subtask, page.length, page.doc_id)


def _query_scope_key(query) -> tuple[str, str, str, str]:
    return (query.task_family, query.subtask, query.length, query.doc_id)


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


def _build_scope_stats(
    all_queries,
    selected_queries,
    required_pages,
    pages_by_scope,
) -> dict[str, Any]:
    all_query_counter = Counter((q.subtask, q.length) for q in all_queries)
    selected_query_counter = Counter((q.subtask, q.length) for q in selected_queries)
    selected_page_counter = Counter((p.subtask, p.length) for p in required_pages)
    candidate_counts = [len(pages_by_scope[_query_scope_key(q)]) for q in selected_queries]

    return {
        "available_queries": len(all_queries),
        "selected_queries": len(selected_queries),
        "selected_docs": len({_query_scope_key(q) for q in selected_queries}),
        "required_pages": len(required_pages),
        "available_queries_by_scope": {
            f"{subtask}/{length}": count
            for (subtask, length), count in sorted(all_query_counter.items())
        },
        "selected_queries_by_scope": {
            f"{subtask}/{length}": count
            for (subtask, length), count in sorted(selected_query_counter.items())
        },
        "required_pages_by_scope": {
            f"{subtask}/{length}": count
            for (subtask, length), count in sorted(selected_page_counter.items())
        },
        "candidate_pages_per_query": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "mean": mean(candidate_counts) if candidate_counts else 0.0,
        },
    }


def _build_metrics_tables(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    query_lookup: dict[str, Any],
    k_values: list[int],
) -> dict[str, pd.DataFrame]:
    overall = compute_all_metrics(retrieval_results, ground_truth, k_values=k_values)

    by_subtask = compute_metrics_by_group(
        retrieval_results,
        ground_truth,
        group_fn=lambda qid: query_lookup[qid].subtask,
        k_values=k_values,
    )

    by_length = compute_metrics_by_group(
        retrieval_results,
        ground_truth,
        group_fn=lambda qid: query_lookup[qid].length,
        k_values=k_values,
    )

    by_scope = compute_metrics_by_group(
        retrieval_results,
        ground_truth,
        group_fn=lambda qid: f"{query_lookup[qid].subtask}/{query_lookup[qid].length}",
        k_values=k_values,
    )

    if not by_scope.empty:
        split = by_scope["group"].str.split("/", n=1, expand=True)
        by_scope.insert(1, "subtask", split[0])
        by_scope.insert(2, "length", split[1])

    overall_with_group = overall.copy()
    overall_with_group.insert(0, "group", "all")
    overall_with_group.insert(0, "group_type", "overall")

    by_subtask_summary = by_subtask.copy()
    if not by_subtask_summary.empty:
        by_subtask_summary.insert(0, "group_type", "subtask")

    by_length_summary = by_length.copy()
    if not by_length_summary.empty:
        by_length_summary.insert(0, "group_type", "length")

    by_scope_summary = by_scope.copy()
    if not by_scope_summary.empty:
        by_scope_summary.insert(0, "group_type", "subtask_length")

    frames = [overall_with_group]
    if not by_subtask_summary.empty:
        frames.append(by_subtask_summary)
    if not by_length_summary.empty:
        frames.append(by_length_summary)
    if not by_scope_summary.empty:
        frames.append(by_scope_summary)

    summary = pd.concat(frames, ignore_index=True)

    return {
        "overall": overall,
        "by_subtask": by_subtask,
        "by_length": by_length,
        "by_subtask_length": by_scope,
        "summary": summary,
    }


def _save_metrics_tables(run_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(run_dir / f"metrics_{name}.csv", index=False, encoding="utf-8-sig")


def _load_page_encoder(
    model_cfg: dict[str, Any],
    index_cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    device: str | None = None,
    batch_size: int | None = None,
):
    model_repo = model_cfg.get("repo", "vidore/colpali-v1.3")
    base_repo = model_cfg.get("base_repo", "vidore/colpaligemma-3b-pt-448-base")
    device = device or args.device or model_cfg.get("device", "cuda:0")
    dtype = _parse_torch_dtype(model_cfg.get("dtype"))
    batch_size = batch_size or args.page_batch_size or index_cfg.get("batch_size", 4)
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


def _list_visible_cuda_devices() -> list[str]:
    try:
        import torch
    except Exception:
        return []

    if not torch.cuda.is_available():
        return []
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def _resolve_index_devices(args: argparse.Namespace, default_device: str) -> list[str]:
    if args.index_devices:
        return args.index_devices

    if not default_device.startswith("cuda"):
        return [default_device]

    visible_devices = _list_visible_cuda_devices()
    return visible_devices or [default_device]


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
        _, raw_index = device.split(":", 1)
        device_index = int(raw_index)
    else:
        device_index = 0

    props = torch.cuda.get_device_properties(device_index)
    return props.total_memory / (1024**3)


def _recommend_page_batch_size(explicit_value: int | None, index_devices: list[str]) -> int:
    if explicit_value is not None:
        return explicit_value

    if not index_devices:
        return 4

    min_memory_gb = min(_get_device_memory_gb(device) for device in index_devices)
    if min_memory_gb >= 22:
        return 4
    if min_memory_gb >= 14:
        return 3
    return 4


def _recommend_score_batch_size(explicit_value: int | None, retrieval_device: str) -> int:
    if explicit_value is not None:
        return explicit_value

    memory_gb = _get_device_memory_gb(retrieval_device)
    if memory_gb >= 22:
        return 512
    if memory_gb >= 14:
        return 256
    return 64


def _split_pages_for_devices(pages: list[Any], devices: list[str]) -> list[tuple[str, list[Any]]]:
    if not devices:
        return []

    shards: list[list[Any]] = [[] for _ in devices]
    for index, page in enumerate(pages):
        shards[index % len(devices)].append(page)

    return [
        (device, shard)
        for device, shard in zip(devices, shards)
        if shard
    ]


def _encode_pages_worker(payload: dict[str, Any]) -> dict[str, Any]:
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
    store = IndexStore(payload["index_dir"])
    encoder.encode_corpus(
        pages,
        store,
        show_progress=False,
        resume=False,
        update_manifest=False,
    )
    return {
        "device": device,
        "encoded_pages": len(pages),
    }


def main() -> int:
    args = parse_args()
    log_level = getattr(__import__("logging"), args.log_level)
    logger = setup_logging("step3_eval", level=log_level)

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
    index_devices = _resolve_index_devices(args, retrieval_device)
    page_batch_size = _recommend_page_batch_size(args.page_batch_size, index_devices)
    retrieval_cfg["score_batch_size"] = _recommend_score_batch_size(
        args.score_batch_size,
        retrieval_device,
    )

    data_dir = resolve_path(data_cfg.get("root_dir", "data/MMLongBench/raw"))
    index_dir = resolve_path(args.index_dir or index_cfg.get("dir", "data/processed/index"))
    output_root = resolve_path(
        args.output_dir or evaluation_cfg.get("output_dir", "outputs/eval_reports")
    )
    run_name = args.run_name or _make_run_name(subtasks, lengths, args.max_queries)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 3.1 评测开始: subtasks=%s lengths=%s", subtasks, lengths)
    logger.info("HuggingFace 离线模式: HF_HUB_OFFLINE=%s", os.environ.get("HF_HUB_OFFLINE"))
    logger.info("输出目录: %s", run_dir)
    logger.info(
        "运行参数: retrieval_device=%s index_devices=%s page_batch_size=%d score_batch_size=%d",
        retrieval_device,
        index_devices,
        page_batch_size,
        retrieval_cfg["score_batch_size"],
    )

    adapter = DocumentQAAdapter(
        data_dir=str(data_dir),
        subtasks=subtasks,
        lengths=lengths,
    )

    all_queries = list(adapter.iter_queries())
    if args.query_offset < 0:
        raise ValueError("--query-offset 不能为负数")

    selected_queries = all_queries[args.query_offset :]
    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max-queries 必须为正整数")
        selected_queries = selected_queries[: args.max_queries]

    if not selected_queries:
        raise RuntimeError("当前筛选条件下没有可评测的查询")

    selected_query_ids = {q.query_id for q in selected_queries}
    query_lookup = {q.query_id: q for q in selected_queries}

    selected_scope_keys = {_query_scope_key(q) for q in selected_queries}
    all_pages = list(adapter.iter_pages())
    required_pages = [p for p in all_pages if _page_scope_key(p) in selected_scope_keys]

    pages_by_scope: dict[tuple[str, str, str, str], list[Any]] = {}
    for page in required_pages:
        pages_by_scope.setdefault(_page_scope_key(page), []).append(page)

    ground_truth_all = adapter.build_ground_truth()
    ground_truth = {
        qid: ground_truth_all.get(qid, set())
        for qid in selected_query_ids
    }

    scope_stats = _build_scope_stats(all_queries, selected_queries, required_pages, pages_by_scope)
    logger.info(
        "评测范围: %d queries, %d docs, %d pages",
        scope_stats["selected_queries"],
        scope_stats["selected_docs"],
        scope_stats["required_pages"],
    )

    store = IndexStore(str(index_dir))
    existing_page_ids = set(store.list_page_ids())
    existing_total_pages_before = len(existing_page_ids)
    missing_pages = [page for page in required_pages if page.page_id not in existing_page_ids]

    run_summary: dict[str, Any] = {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(data_dir),
        "index_dir": str(index_dir),
        "output_dir": str(run_dir),
        "offline": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "scope": {
            "subtasks": subtasks,
            "lengths": lengths,
            "query_offset": args.query_offset,
            "max_queries": args.max_queries,
            "k_values": k_values,
        },
        "scope_stats": scope_stats,
        "index": {
            "existing_total_pages_before": existing_total_pages_before,
            "missing_scope_pages": len(missing_pages),
            "skip_index_build": args.skip_index_build,
        },
        "model": {
            "repo": model_cfg.get("repo"),
            "base_repo": model_cfg.get("base_repo"),
            "device": retrieval_device,
            "dtype": model_cfg.get("dtype"),
        },
        "runtime": {
            "retrieval_device": retrieval_device,
            "index_devices": index_devices,
            "page_batch_size": page_batch_size,
            "score_batch_size": retrieval_cfg["score_batch_size"],
            "multi_gpu_index": len(index_devices) > 1,
        },
    }

    if args.stats_only:
        with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
        logger.info("已输出统计信息（未加载模型）: %s", run_dir / "run_summary.json")
        return 0

    if missing_pages and args.skip_index_build:
        raise RuntimeError(
            f"索引中缺少当前评测范围的 {len(missing_pages)} 页，且指定了 --skip-index-build。"
        )

    page_encoder = None

    index_build_start = time.perf_counter()
    if missing_pages:
        logger.info("开始补建索引: %d 页", len(missing_pages))
        if len(index_devices) == 1:
            page_encoder = _load_page_encoder(
                model_cfg,
                index_cfg,
                args,
                device=index_devices[0],
                batch_size=page_batch_size,
            )
            page_encoder.encode_corpus(missing_pages, store, show_progress=True, resume=True)
        else:
            jobs = _split_pages_for_devices(missing_pages, index_devices)
            logger.info(
                "启用多 GPU 索引构建: %s",
                ", ".join(f"{device}={len(pages)}页" for device, pages in jobs),
            )
            worker_payloads = [
                {
                    "device": device,
                    "pages": pages,
                    "model_cfg": {
                        "repo": model_cfg.get("repo"),
                        "base_repo": model_cfg.get("base_repo"),
                        "dtype": model_cfg.get("dtype"),
                    },
                    "page_batch_size": page_batch_size,
                    "storage_dtype": index_cfg.get("storage_dtype", "float16"),
                    "index_dir": str(index_dir),
                }
                for device, pages in jobs
            ]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(worker_payloads)) as pool:
                worker_results = pool.map(_encode_pages_worker, worker_payloads)
            store.register_page_ids([page.page_id for page in missing_pages])
            run_summary["index"]["worker_results"] = worker_results

        scope_page_ids = store.list_page_ids()
        if scope_page_ids:
            dim = int(store.read_page(scope_page_ids[0]).shape[-1])
            store.save_meta(model_cfg.get("repo", "vidore/colpali-v1.3"), dim=dim)
    index_build_time = time.perf_counter() - index_build_start

    if page_encoder is None:
        page_encoder = _load_page_encoder(
            model_cfg,
            index_cfg,
            args,
            device=retrieval_device,
            batch_size=page_batch_size,
        )

    pipeline = RetrievalPipeline(
        model=page_encoder,
        index_store=store,
        config=retrieval_cfg,
    )

    retrieval_results: dict[str, list[str]] = {}
    retrieval_details: list[dict[str, Any]] = []
    query_latencies: list[float] = []

    total_retrieval_start = time.perf_counter()
    for query in selected_queries:
        query_start = time.perf_counter()
        results = pipeline.retrieve(query, top_k=max_k)
        latency = time.perf_counter() - query_start
        query_latencies.append(latency)

        relevant = ground_truth.get(query.query_id, set())
        retrieval_results[query.query_id] = [item.page_id for item in results]

        retrieval_details.append(
            {
                "query_id": query.query_id,
                "subtask": query.subtask,
                "length": query.length,
                "doc_id": query.doc_id,
                "question": query.text,
                "candidate_pages": len(pages_by_scope[_query_scope_key(query)]),
                "latency_s": latency,
                "relevant_page_ids": sorted(relevant),
                "results": [
                    {
                        "page_id": item.page_id,
                        "rank": item.rank,
                        "score": item.score,
                        "is_relevant": item.page_id in relevant,
                    }
                    for item in results
                ],
            }
        )

    total_retrieval_time = time.perf_counter() - total_retrieval_start

    tables = _build_metrics_tables(retrieval_results, ground_truth, query_lookup, k_values)
    _save_metrics_tables(run_dir, tables)

    with open(run_dir / "retrieval_details.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_details, f, ensure_ascii=False, indent=2)

    run_summary["index"].update(
        {
            "index_build_time_s": index_build_time,
            "encoded_pages": len(missing_pages),
            "store_stats": store.stats,
        }
    )
    run_summary["retrieval"] = {
        "total_time_s": total_retrieval_time,
        "avg_latency_s": mean(query_latencies) if query_latencies else 0.0,
        "p50_latency_s": _percentile(query_latencies, 0.50),
        "p95_latency_s": _percentile(query_latencies, 0.95),
        "max_latency_s": max(query_latencies) if query_latencies else 0.0,
    }
    metrics_preview = tables["summary"].where(pd.notna(tables["summary"]), None)
    run_summary["metrics_preview"] = metrics_preview.to_dict(orient="records")

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    logger.info("索引补建耗时: %s", format_duration(index_build_time))
    logger.info("检索耗时: %s", format_duration(total_retrieval_time))
    logger.info("平均单查询延迟: %.3fs", run_summary["retrieval"]["avg_latency_s"])
    logger.info("结果已写入: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())