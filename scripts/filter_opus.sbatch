#!/bin/bash
#SBATCH --job-name=spa-opus-filter
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=spa-opus-filter-%j.out
#SBATCH --error=spa-opus-filter-%j.err

set -euo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

WORK_DIR="$HOME/markov-chain-sentence-generator"
SCRIPT="$WORK_DIR/scripts/filter_opus.py"

# Tighten the already es-filtered len4-25 corpus down to the 4-22 window.
INPUT="/projects/b35cg/corpora/opus_filtered_len4-25_es.txt"
MODEL="/projects/b35cg/envs/lid.176.bin"

MIN_TOKENS=4
MAX_TOKENS=22
LANG=es
THRESHOLD=0.70
HEARTBEAT_SECS=60

# Output to scratch (large; not project/home). Name reflects the 4-22 window.
OUT="$SCRATCHDIR/opus_filtered_len4-22_es.txt"
mkdir -p "$(dirname "$OUT")"

# Env: fastText + lid.176.bin live in this venv (not conda sentgen, not module
# cray-python). Activate the venv only; the import check below fails loudly if
# fasttext is missing.
source /projects/b35cg/envs/seg_env/bin/activate

export PYTHONUNBUFFERED=1

echo "Python path:   $(which python)"
python --version

python - <<'PY'
mods = ["argparse", "fasttext"]
failed = False
for m in mods:
    try:
        __import__(m)
        print(f"OK import {m}", flush=True)
    except Exception as e:
        print(f"FAIL import {m}: {e}", flush=True)
        failed = True
if failed:
    raise SystemExit(1)
PY

for f in "$SCRIPT" "$INPUT" "$MODEL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

echo "Script:        $SCRIPT"
echo "Input:         $INPUT ($(ls -lh "$INPUT" | awk '{print $5}'))"
echo "Model:         $MODEL"
echo "Output:        $OUT"
echo "Window:        [$MIN_TOKENS, $MAX_TOKENS] tokens; lang=$LANG; threshold=$THRESHOLD"

# filter_opus.py prints its own per-million-line progress heartbeat (read/kept/
# survival%/lines-per-sec). This background loop adds only sstat memory telemetry,
# matching the house style without duplicating the progress lines.
START_TS=$(date +%s)
progress_heartbeat() {
    while true; do
        sleep "$HEARTBEAT_SECS"
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
    --input "$INPUT" \
    --output "$OUT" \
    --min-tokens "$MIN_TOKENS" \
    --max-tokens "$MAX_TOKENS" \
    --lang "$LANG" \
    --threshold "$THRESHOLD" \
    --model "$MODEL"

cleanup

echo "============================================"
echo "Finished: $(date)"
echo "FINAL OUTPUT PATH: $OUT"
echo -n "FINAL LINE COUNT: "
wc -l "$OUT" || true
ls -lh "$OUT" || true
echo "============================================"
