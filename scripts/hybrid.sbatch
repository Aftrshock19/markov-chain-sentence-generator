#!/bin/bash
#SBATCH --job-name=spa-hybrid-batch
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=spa-hybrid-batch-%j.out
#SBATCH --error=spa-hybrid-batch-%j.err

set -euo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

WORK_DIR="$HOME/markov-chain-sentence-generator"
LEXICON="$WORK_DIR/stg_words_spa.csv"
MODELS_DIR="$WORK_DIR/models"
SCRIPT="$WORK_DIR/hybrid_generator.py"

OUT_DIR="$WORK_DIR/outputs/hybrid"
mkdir -p "$OUT_DIR"

OUTFILE="$OUT_DIR/hybrid_batch.csv"
REVIEW_OUT="$OUT_DIR/hybrid_batch_review.csv"
CANDIDATES_OUT="$OUT_DIR/hybrid_batch_candidates.csv"

OVERRIDE_1="$WORK_DIR/starter_overrides.csv"
OVERRIDE_2="$WORK_DIR/starter_exclusions.csv"

TARGET_LIMIT=15000
HEARTBEAT_SECS=60

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentgen

export PYTHONUNBUFFERED=1

echo "Python path:   $(which python)"
python --version

python - <<'PY'
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
    "$LEXICON" \
    "$MODELS_DIR/lemma_forms.pkl" \
    "$MODELS_DIR/lemma_contexts.pkl" \
    "$MODELS_DIR/bigrams.pkl" \
    "$MODELS_DIR/trigrams.pkl" \
    "$MODELS_DIR/pos_transitions.pkl" \
    "$MODELS_DIR/enriched_lexicon.pkl" \
    "$OVERRIDE_1" \
    "$OVERRIDE_2"
do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

echo "Script:        $SCRIPT"
echo "Lexicon:       $LEXICON ($(wc -l < "$LEXICON") lines)"
echo "Models dir:    $MODELS_DIR"
echo "Output dir:    $OUT_DIR"

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
        REVIEW_ROWS=0
        CAND_ROWS=0

        if [ -f "$OUTFILE" ]; then
            OUT_ROWS=$(( $(wc -l < "$OUTFILE") - 1 ))
            [ "$OUT_ROWS" -lt 0 ] && OUT_ROWS=0
        fi

        if [ -f "$REVIEW_OUT" ]; then
            REVIEW_ROWS=$(( $(wc -l < "$REVIEW_OUT") - 1 ))
            [ "$REVIEW_ROWS" -lt 0 ] && REVIEW_ROWS=0
        fi

        if [ -f "$CANDIDATES_OUT" ]; then
            CAND_ROWS=$(( $(wc -l < "$CANDIDATES_OUT") - 1 ))
            [ "$CAND_ROWS" -lt 0 ] && CAND_ROWS=0
        fi

        PCT="0.00"
        ETA_STR="unknown"
        if [ "$OUT_ROWS" -gt 0 ]; then
            PCT=$(python - <<PY
done = $OUT_ROWS
total = $TARGET_LIMIT
print(f"{100.0 * done / total:.2f}")
PY
)
            ETA_STR=$(python - <<PY
done = $OUT_ROWS
total = $TARGET_LIMIT
elapsed = $ELAPSED
remaining = max(total - done, 0)
if done > 0 and remaining > 0:
    eta = int(elapsed * remaining / done)
    h = eta // 3600
    m = (eta % 3600) // 60
    s = eta % 60
    print(f"{h:02d}:{m:02d}:{s:02d}")
elif remaining == 0:
    print("00:00:00")
else:
    print("unknown")
PY
)
        fi

        echo "[heartbeat] elapsed=${H}h${M}m${S}s out_rows=${OUT_ROWS}/${TARGET_LIMIT} (${PCT}%) review_rows=${REVIEW_ROWS} candidate_rows=${CAND_ROWS} eta=${ETA_STR}"

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
    --lexicon "$LEXICON" \
    --models-dir "$MODELS_DIR" \
    --out "$OUTFILE" \
    --review-out "$REVIEW_OUT" \
    --candidates-out "$CANDIDATES_OUT" \
    --lexicon-overrides "$OVERRIDE_1" \
    --lexicon-overrides "$OVERRIDE_2" \
    --min-rank 1 \
    --max-rank 15000 \
    --limit 15000 \
    --seed 42 \
    --max-total-attempts 350 \
    --max-candidates-to-keep 25 \
    --max-candidates-per-lemma 20

cleanup

echo "============================================"
echo "Finished: $(date)"
ls -lh "$OUTFILE" "$REVIEW_OUT" "$CANDIDATES_OUT" || true
echo "============================================"