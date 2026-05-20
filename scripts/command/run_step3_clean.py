"""Step 3 清理脚本：清理评测输出目录和/或匹配范围的索引页面。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any


def _detect_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"未找到项目根目录: {current}")


PROJECT_ROOT = _detect_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zeroshot_vdr.config import get_evaluation_config, get_index_config, load_config
from zeroshot_vdr.indexing.store import IndexStore
from zeroshot_vdr.utils import resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "清理 Step 3 生成的评测输出目录和/或索引页面。"
            "默认仅预览删除计划；传入 --yes 后才会真的删除。"
        )
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument(
        "--subtasks",
        nargs="+",
        default=None,
        help="清理的子任务列表，如 longdocurl mmlongdoc slidevqa",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=None,
        help="清理的长度档位列表，如 K4 K32 K128",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        help="精确指定要清理的评测输出目录名（位于 outputs/eval_reports 下）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="评测输出根目录；默认读取配置中的 evaluation.output_dir",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="索引目录；默认读取配置中的 index.dir",
    )
    parser.add_argument(
        "--clean-outputs",
        action="store_true",
        help="清理 outputs/eval_reports 下匹配的 run 目录",
    )
    parser.add_argument(
        "--clean-index",
        action="store_true",
        help="清理 index 中匹配子任务/长度范围的页面 embedding",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="清理全部 Step 3 输出目录，并删除索引中的全部页面",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印清理计划，不实际删除",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认执行实际删除；未提供时默认仅预览",
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


def _build_scope_filters(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> set[tuple[str, str]]:
    if not (args.subtasks or args.lengths):
        return set()

    subtasks = _resolve_subtasks(args, config)
    lengths = _resolve_lengths(args, config)
    return {(subtask, length) for subtask, length in product(subtasks, lengths)}


def _load_run_summaries(output_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if not output_dir.exists():
        return summaries

    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        summary_path = child / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summaries[child.name] = json.load(f)
        except Exception:
            continue

    return summaries


def _summary_scope_pairs(summary: dict[str, Any]) -> set[tuple[str, str]]:
    scope = summary.get("scope", {})
    subtasks = scope.get("subtasks") or []
    lengths = scope.get("lengths") or []
    return {(subtask, length) for subtask, length in product(subtasks, lengths)}


def _select_output_runs(
    args: argparse.Namespace,
    output_dir: Path,
    scope_filters: set[tuple[str, str]],
    run_summaries: dict[str, dict[str, Any]],
) -> list[str]:
    selected: list[str] = []

    if args.all:
        selected = sorted(child.name for child in output_dir.iterdir() if child.is_dir())
        return selected

    selected_set: set[str] = set()

    if args.run_names:
        for run_name in args.run_names:
            run_path = output_dir / run_name
            if run_path.exists() and run_path.is_dir():
                selected_set.add(run_name)
            else:
                raise FileNotFoundError(f"评测输出目录不存在: {run_path}")

    if scope_filters:
        for run_name, summary in run_summaries.items():
            pairs = _summary_scope_pairs(summary)
            if pairs and pairs.issubset(scope_filters):
                selected_set.add(run_name)

    return sorted(selected_set)


def _derive_scope_filters_from_runs(
    run_names: list[str],
    run_summaries: dict[str, dict[str, Any]],
) -> set[tuple[str, str]]:
    scope_filters: set[tuple[str, str]] = set()
    for run_name in run_names:
        summary = run_summaries.get(run_name)
        if summary is None:
            continue
        scope_filters.update(_summary_scope_pairs(summary))
    return scope_filters


def _page_file_path(index_dir: Path, page_id: str) -> Path:
    safe_id = page_id.replace("/", "_").replace("\\", "_")
    return index_dir / "pages" / f"{safe_id}.pt"


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _select_index_pages(
    args: argparse.Namespace,
    store: IndexStore,
    config: dict[str, Any],
    run_names: list[str],
    run_summaries: dict[str, dict[str, Any]],
) -> list[str]:
    if args.all:
        return store.list_page_ids(task_family="docqa")

    scope_filters = _build_scope_filters(args, config)
    scope_filters.update(_derive_scope_filters_from_runs(run_names, run_summaries))

    if not scope_filters:
        return []

    selected: list[str] = []
    seen: set[str] = set()
    for subtask, length in sorted(scope_filters):
        for page_id in store.list_page_ids(
            task_family="docqa",
            subtask=subtask,
            length=length,
        ):
            if page_id in seen:
                continue
            seen.add(page_id)
            selected.append(page_id)

    return selected


def _format_size_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _rewrite_index_metadata(index_dir: Path, remaining_page_ids: list[str]) -> None:
    page_ids_path = index_dir / "page_ids.json"
    with open(page_ids_path, "w", encoding="utf-8") as f:
        json.dump(remaining_page_ids, f, ensure_ascii=False)

    meta_path = index_dir / "index_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["num_pages"] = len(remaining_page_ids)
        meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    log_level = getattr(__import__("logging"), args.log_level)
    logger = setup_logging("step3_clean", level=log_level)

    if not args.clean_outputs and not args.clean_index:
        args.clean_outputs = True
        args.clean_index = True

    config = load_config(args.config)
    output_dir = resolve_path(
        args.output_dir or get_evaluation_config(config).get("output_dir", "outputs/eval_reports")
    )
    index_dir = resolve_path(
        args.index_dir or get_index_config(config).get("dir", "data/processed/index")
    )

    explicit_scope_filters = _build_scope_filters(args, config)
    if not args.all and not args.run_names and not explicit_scope_filters:
        raise ValueError(
            "请至少提供一种清理范围：--all，或 --run-names，或 --subtasks/--lengths。"
        )

    run_summaries = _load_run_summaries(output_dir)
    selected_runs: list[str] = []
    if args.clean_outputs:
        if not output_dir.exists():
            logger.warning("评测输出目录不存在: %s", output_dir)
        else:
            selected_runs = _select_output_runs(
                args=args,
                output_dir=output_dir,
                scope_filters=explicit_scope_filters,
                run_summaries=run_summaries,
            )

    selected_pages: list[str] = []
    all_page_ids: list[str] = []
    if args.clean_index:
        store = IndexStore(str(index_dir))
        all_page_ids = store.list_page_ids()
        selected_pages = _select_index_pages(
            args=args,
            store=store,
            config=config,
            run_names=selected_runs if selected_runs else (args.run_names or []),
            run_summaries=run_summaries,
        )

    output_bytes = sum(_path_size_bytes(output_dir / run_name) for run_name in selected_runs)
    index_bytes = sum(_path_size_bytes(_page_file_path(index_dir, page_id)) for page_id in selected_pages)

    logger.info("清理目标目录: %s", output_dir)
    logger.info("清理目标索引: %s", index_dir)
    logger.info(
        "匹配到 %d 个输出目录（约 %s），%d 个索引页面文件（约 %s）",
        len(selected_runs),
        _format_size_mb(output_bytes),
        len(selected_pages),
        _format_size_mb(index_bytes),
    )

    if selected_runs:
        logger.info("将处理的输出目录: %s", ", ".join(selected_runs))
    if selected_pages:
        sample = ", ".join(selected_pages[:5])
        logger.info("将处理的索引页面示例: %s", sample)

    execute = args.yes and not args.dry_run
    if not execute:
        logger.info("当前为预览模式；传入 --yes 才会实际删除。")
        return 0

    if args.clean_outputs:
        for run_name in selected_runs:
            shutil.rmtree(output_dir / run_name, ignore_errors=False)

    if args.clean_index and selected_pages:
        to_remove = set(selected_pages)
        for page_id in selected_pages:
            page_path = _page_file_path(index_dir, page_id)
            if page_path.exists():
                page_path.unlink()

        remaining_page_ids = [page_id for page_id in all_page_ids if page_id not in to_remove]
        _rewrite_index_metadata(index_dir, remaining_page_ids)

    logger.info("清理完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())