#!/usr/bin/env bash
# ===========================================================================
# Phase 4 进度查看工具
#
# 用法:
#   bash scripts/check_phase4_progress.sh           # 查看所有任务进度
#   bash scripts/check_phase4_progress.sh --watch   # 每 30 秒刷新
#   bash scripts/check_phase4_progress.sh --tail    # 查看当前任务实时日志
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="outputs/eval_reports"
MAIN_LOG="$LOG_DIR/phase4_full_run.log"

MODE="${1:-}"

# ---- 实时日志 ----
if [ "$MODE" == "--tail" ]; then
    for TASK in fixed_topn adaptive adaptive_neighbors; do
        TASK_LOG="$LOG_DIR/phase4_${TASK}.log"
        if [ -f "$TASK_LOG" ]; then
            echo "=== $TASK (最后 5 行) ==="
            tail -5 "$TASK_LOG"
            echo ""
        fi
    done
    exit 0
fi

# ---- 单次 / 循环刷新 ----
show_progress() {
    clear 2>/dev/null || true
    echo "========== Phase 4 全量评测进度 [$(date '+%H:%M:%S')] =========="
    echo ""

    # 主日志状态
    if [ -f "$MAIN_LOG" ]; then
        echo "--- 主日志 ---"
        grep -E "(开始|完成|失败|结束)" "$MAIN_LOG" | tail -8
        echo ""
    else
        echo "主日志尚未创建，评测可能还未启动。"
        echo ""
    fi

    # 各任务 checkpoint
    for TASK in fixed_topn adaptive adaptive_neighbors; do
        CP="outputs/eval_reports/phase4_${TASK}/_checkpoint.json"
        SUMMARY="outputs/eval_reports/phase4_${TASK}/run_summary.json"

        if [ -f "$CP" ]; then
            # 从 checkpoint 读取进度
            PROGRESS=$($VENV_PYTHON -c "import json
d=json.load(open('$CP'))
print(f\"{d['queries_done']}/{d['queries_total']} ({d['queries_done']/max(1,d['queries_total'])*100:.0f}%)\")
" 2>/dev/null || echo "读取失败")
            echo "  [$TASK] 运行中: $PROGRESS"
        elif [ -f "$SUMMARY" ]; then
            # 已完成
            INFO=$($VENV_PYTHON -c "
import json
d=json.load(open('$SUMMARY'))
n=d.get('scope',{}).get('num_queries','?')
r=d.get('retrieval',{})
lat=r.get('avg_latency_s',0)
t=r.get('total_time_s',0)
print(f'完成: {n} queries, avg {lat:.3f}s, total {t/60:.0f}min')
" 2>/dev/null || echo "完成(读取失败)")
            echo "  [$TASK] $INFO"
        else
            echo "  [$TASK] 等待中..."
        fi
    done

    echo ""
    echo "--- 系统资源 ---"
    # GPU 使用
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>/dev/null | while read line; do
        echo "  GPU$line"
    done || echo "  (nvidia-smi 不可用)"

    # 最近的主日志
    if [ -f "$MAIN_LOG" ]; then
        echo ""
        echo "--- 最近输出 ---"
        tail -3 "$MAIN_LOG"
    fi
}

if [ "$MODE" == "--watch" ]; then
    echo "每 30 秒刷新，Ctrl+C 退出..."
    while true; do
        show_progress
        sleep 30
    done
else
    show_progress
fi
