#!/usr/bin/env bash
# ===========================================================================
# Phase 4 全量评测（后台运行）
#
# 用法:
#   bash scripts/run_phase4_full.sh              # 启动后台评测
#   bash scripts/run_phase4_full.sh --resume     # 续跑中断的任务
#   bash scripts/check_phase4_progress.sh        # 查看进度
#
# 输出:
#   outputs/eval_reports/phase4_fixed_topn/
#   outputs/eval_reports/phase4_adaptive/
#   outputs/eval_reports/phase4_adaptive_neighbors/
# ===========================================================================

set -euo pipefail
# 但不要让 grep 无匹配时导致脚本退出
FAIL_ON_NO_MATCH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---- 配置 ----
INDEX_DIR="data/processed/index_stable_page_ids"
LOG_DIR="outputs/eval_reports"
mkdir -p "$LOG_DIR"

RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "[$(date '+%H:%M:%S')] 续跑模式"
else
    echo "[$(date '+%H:%M:%S')] 全新运行模式"
fi

# ---- 激活环境 ----
# 使用 venv 绝对路径，避免 nohup 下 conda/source 失败
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "错误: 找不到 $VENV_PYTHON，请先创建虚拟环境"
    exit 1
fi

# 设置必要的环境变量（HF 缓存路径等）
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT_ROOT/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "[$(date '+%H:%M:%S')] Python: $VENV_PYTHON ($($VENV_PYTHON --version))"

# ---- 定义三个任务 ----
declare -A TASKS
TASK_ORDER=("fixed_topn" "adaptive" "adaptive_neighbors")

TASKS["fixed_topn"]="--method fixed_topn --coarse-top-n 64 --valid-only --trace-enabled"
TASKS["adaptive"]="--method adaptive --valid-only --trace-enabled"
TASKS["adaptive_neighbors"]="--method adaptive_neighbors --neighbor-window 1 --neighbor-seed-n 8 --valid-only --trace-enabled"

# ---- 顺序执行 ----
MAIN_LOG="$LOG_DIR/phase4_full_run.log"
echo "==========================================" | tee -a "$MAIN_LOG"
echo " Phase 4 全量评测开始: $(date)" | tee -a "$MAIN_LOG"
echo " 日志: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"

TOTAL=${#TASK_ORDER[@]}
CURRENT=0

for TASK in "${TASK_ORDER[@]}"; do
    CURRENT=$((CURRENT + 1))
    TASK_LOG="$LOG_DIR/phase4_${TASK}.log"
    CMD="$VENV_PYTHON scripts/run_phase4_eval.py ${TASKS[$TASK]} --index-dir $INDEX_DIR $RESUME_FLAG"

    echo "" | tee -a "$MAIN_LOG"
    echo "---- [$CURRENT/$TOTAL] $TASK ----" | tee -a "$MAIN_LOG"
    echo "  开始: $(date '+%H:%M:%S')" | tee -a "$MAIN_LOG"
    echo "  命令: $CMD" | tee -a "$MAIN_LOG"
    echo "  日志: $TASK_LOG" | tee -a "$MAIN_LOG"

    START_TS=$(date +%s)
    $CMD > "$TASK_LOG" 2>&1
    RC=$?
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    if [ $RC -eq 0 ]; then
        echo "  完成: $(date '+%H:%M:%S') (耗时 ${ELAPSED}s)" | tee -a "$MAIN_LOG"
        # 提取关键指标
        grep -E "(Recall@10|nDCG@10|平均延迟|评测范围)" "$TASK_LOG" | tail -5 | tee -a "$MAIN_LOG" || true
    else
        echo "  失败: exit code=$RC (耗时 ${ELAPSED}s)" | tee -a "$MAIN_LOG"
        echo "  请查看日志: $TASK_LOG" | tee -a "$MAIN_LOG"
        tail -20 "$TASK_LOG" | tee -a "$MAIN_LOG"
    fi
done

echo "" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"
echo " Phase 4 全量评测结束: $(date)" | tee -a "$MAIN_LOG"
echo " 结果目录:" | tee -a "$MAIN_LOG"
for TASK in "${TASK_ORDER[@]}"; do
    echo "   outputs/eval_reports/phase4_${TASK}/" | tee -a "$MAIN_LOG"
done
echo "==========================================" | tee -a "$MAIN_LOG"
