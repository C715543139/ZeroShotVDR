from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CommandSpec:
    kind: str
    target: str
    description: str


COMMANDS: dict[str, dict[str, CommandSpec]] = {
    "phase4": {
        "eval": CommandSpec("python", "scripts/run/run_phase4_eval.py", "运行单次 Phase 4 评测"),
        "full": CommandSpec("bash", "scripts/run/run_phase4_full.sh", "顺序执行 Phase 4 三方法全量评测"),
    },
    "step3": {
        "eval": CommandSpec("python", "scripts/run/run_step3_eval.py", "运行 Step 3.1 页级检索评测"),
        "analysis": CommandSpec("python", "scripts/run/run_step3_analysis.py", "运行 Step 3.2 结果分析"),
    },
    "command": {
        "check-env": CommandSpec("python", "scripts/command/check_env.py", "检查运行环境"),
        "phase4-progress": CommandSpec("bash", "scripts/command/check_phase4_progress.sh", "查看 Phase 4 全量任务进度"),
        "step3-clean": CommandSpec("python", "scripts/command/run_step3_clean.py", "清理 Step 3 评测输出和索引页面"),
        "trace-analyze": CommandSpec("python", "scripts/analyze_phase4_trace.py", "分析 Phase 4 trace 产物"),
        "backfill-step3-phase4-schema": CommandSpec("python", "scripts/backfill_step3_phase4_schema.py", "为 Phase 3 stable run 回填 Phase 4 兼容产物"),
        "test-model-load": CommandSpec("python", "scripts/test_model_load.py", "验证 ColPali 模型加载"),
    },
}


def _project_python() -> str:
    candidates = [
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _print_main_help() -> None:
    print("ZeroShotVDR 主入口")
    print("")
    print("用法:")
    print("  python main.py <group> <command> [args...]")
    print("")
    print("可用分组:")
    for group, actions in COMMANDS.items():
        print(f"  {group}")
        for action, spec in actions.items():
            print(f"    {action:<28} {spec.description}")
    print("")
    print("示例:")
    print("  python main.py phase4 eval --run-name smoke_fixed64 --method fixed_topn --coarse-top-n 64 --max-queries 50 --valid-only")
    print("  python main.py phase4 full")
    print("  python main.py command phase4-progress --watch")
    print("  python main.py step3 analysis --run-dir outputs/eval_reports/step3_docqa_full_dual3090")


def _print_group_help(group: str) -> None:
    actions = COMMANDS[group]
    print(f"ZeroShotVDR 主入口: {group}")
    print("")
    print("可用命令:")
    for action, spec in actions.items():
        print(f"  {action:<28} {spec.description}")
    print("")
    print(f"用法:")
    print(f"  python main.py {group} <command> [args...]")


def _build_command(spec: CommandSpec, forwarded_args: list[str]) -> list[str]:
    target = PROJECT_ROOT / spec.target
    if spec.kind == "python":
        return [_project_python(), str(target), *forwarded_args]
    if spec.kind == "bash":
        return ["bash", str(target), *forwarded_args]
    raise ValueError(f"未知命令类型: {spec.kind}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_main_help()
        return 0

    group = args.pop(0)
    if group not in COMMANDS:
        print(f"错误: 未知分组 {group}\n", file=sys.stderr)
        _print_main_help()
        return 2

    if not args or args[0] in {"-h", "--help"}:
        _print_group_help(group)
        return 0

    command = args.pop(0)
    spec = COMMANDS[group].get(command)
    if spec is None:
        print(f"错误: {group} 下不存在命令 {command}\n", file=sys.stderr)
        _print_group_help(group)
        return 2

    forwarded_args = args
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    cmd = _build_command(spec, forwarded_args)
    print("执行:", " ".join(shlex.quote(part) for part in cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())