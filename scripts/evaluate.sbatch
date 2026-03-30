#!/bin/bash
#SBATCH --job-name=spa-eval
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=spa-eval-%j.out
#SBATCH --error=spa-eval-%j.err

set -euo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

WORK_DIR="$HOME/markov-chain-sentence-generator"
SCRIPT="$WORK_DIR/evaluate_generator.py"
GENERATED="$WORK_DIR/outputs/hybrid/hybrid_batch_2026462.csv"
OUT_DIR="$WORK_DIR/eval_2026462_9000"

mkdir -p "$OUT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentgen

export PYTHONUNBUFFERED=1

echo "Python path:   $(which python3)"
python3 --version

python3 - <<'PY'
mods = [
    "pickle",
    "csv",
    "json",
    "re",
    "math",
    "statistics",
    "sklearn",
]
for m in mods:
    try:
        __import__(m)
        print(f"OK import {m}", flush=True)
    except Exception as e:
        print(f"FAIL import {m}: {e}", flush=True)
        raise
PY

for f in \
    "$SCRIPT" \
    "$GENERATED"
do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

echo "Script:        $SCRIPT"
echo "Generated CSV: $GENERATED ($(wc -l < "$GENERATED") lines)"
echo "Output dir:    $OUT_DIR"

python3 "$SCRIPT" \
    --generated "$GENERATED" \
    --out-dir "$OUT_DIR"

echo "============================================"
echo "Finished: $(date)"
ls -lh "$OUT_DIR" || true
echo "============================================"