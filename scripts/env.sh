#!/usr/bin/env bash
# ZeroShotVDR environment activation
# Usage: source scripts/env.sh

# Resolve script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Initialize conda and activate environment
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate zeroshotvdr 2>/dev/null || echo "conda activate zeroshotvdr failed"

# Activate project virtual environment
source "$SCRIPT_DIR/../.venv/bin/activate" 2>/dev/null || echo ".venv not found"
