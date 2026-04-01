#!/bin/bash
#SBATCH --job-name=spa-translate
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=spa-translate-%j.out
#SBATCH --error=spa-translate-%j.err

set -euo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

WORK_DIR="$HOME/markov-chain-sentence-generator"
SCRIPT="/mnt/data/translate.py"

INPUT_CSV="$WORK_DIR/outputs/sentances.csv"
OUT_DIR="$WORK_DIR/outputs/translated"
mkdir -p "$OUT_DIR"
OUTPUT_CSV="$OUT_DIR/final_sentences_en_${SLURM_JOB_ID}.csv"

BATCH_SIZE=16
HEARTBEAT_SECS=60

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentgen

export PYTHONUNBUFFERED=1

echo "Python path:   $(which python)"
python --version

python - <<'PY'
mods = [
    "argparse",
    "pandas",
    "torch",
    "tqdm",
    "transformers",
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
    "$INPUT_CSV"
do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

echo "Script:        $SCRIPT"
echo "Input CSV:     $INPUT_CSV ($(wc -l < "$INPUT_CSV") lines)"
echo "Output CSV:    $OUTPUT_CSV"
echo "Batch size:    $BATCH_SIZE"

START_TS=$(date +%s)

progress_heartbeat() {
    while true; do
        sleep "$HEARTBEAT_SECS"

        NOW_TS=$(date +%s)
        ELAPSED=$((NOW_TS - START_TS))
        H=$((ELAPSED / 3600))
        M=$(((ELAPSED % 3600) / 60))
        S=$((ELAPSED % 60))

        OUT_ROWS=0
        if [ -f "$OUTPUT_CSV" ]; then
            OUT_ROWS=$(( $(wc -l < "$OUTPUT_CSV") - 1 ))
            [ "$OUT_ROWS" -lt 0 ] && OUT_ROWS=0
        fi

        echo "[heartbeat] elapsed=${H}h${M}m${S}s translated_rows=${OUT_ROWS}"

        if command -v sstat >/dev/null 2>&1; then
            sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU --noheader 2>/dev/null \
                | sed 's/^/[sstat] /' || true
        fi
    done
}

progress_heartbeat &
HEARTBEAT_PID=$!

cleanup() {
    if [ -n "${HEARTBEAT_PID:-}" ] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
        kill "$HEARTBEAT_PID" || true
    fi
}
trap cleanup EXIT

python -u "$SCRIPT" \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_CSV" \
    --batch-size "$BATCH_SIZE"

cleanup

echo "============================================"
echo "Finished: $(date)"
ls -lh "$OUTPUT_CSV" || true
echo "============================================"