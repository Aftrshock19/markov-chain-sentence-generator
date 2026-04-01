#!/bin/bash
#SBATCH --job-name=spa-rebuild-artifacts
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=spa-rebuild-artifacts-%j.out
#SBATCH --error=spa-rebuild-artifacts-%j.err
#SBATCH --exclusive 

set -euo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"


PROJECT_ROOT="/projects/b35cg/samtoughan/sentence_generator"
CORPUS="$PROJECT_ROOT/data/sbwce.clean.quarter.filtered.txt"
LEXICON="$HOME/markov-chain-sentence-generator/stg_words_spa.csv"
OUTDIR="$PROJECT_ROOT/models"
SCRIPT="$HOME/markov-chain-sentence-generator/rebuild_artifacts_compressed.py"

HEARTBEAT_SECS=30
EXTRA_ARGS="--skip-spacy"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentgen

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=5

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
    "tqdm",
    "spacy",
]
for m in mods:
    try:
        __import__(m)
        print(f"OK import {m}", flush=True)
    except Exception as e:
        print(f"FAIL import {m}: {e}", flush=True)
        raise

try:
    import spacy
    spacy.load("es_core_news_lg", disable=["ner"])
    print("OK load es_core_news_lg", flush=True)
except Exception as e:
    print(f"FAIL load es_core_news_lg: {e}", flush=True)
    raise
PY

for f in "$SCRIPT" "$LEXICON" "$CORPUS"
do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

mkdir -p "$OUTDIR"

echo "Script:        $SCRIPT"
echo "Corpus:        $CORPUS"
echo "Lexicon:       $LEXICON ($(wc -l < "$LEXICON") lines)"
echo "Out dir:       $OUTDIR"

START_TS=$(date +%s)
LOGFILE="$HOME/markov-chain-sentence-generator/spa-rebuild-artifacts-${SLURM_JOB_ID}.live.log"

progress_heartbeat() {
    while true; do
        sleep "$HEARTBEAT_SECS"

        NOW_TS=$(date +%s)
        ELAPSED=$((NOW_TS - START_TS))
        H=$((ELAPSED / 3600))
        M=$(((ELAPSED % 3600) / 60))
        S=$((ELAPSED % 60))

        echo "[heartbeat] elapsed=${H}h${M}m${S}s"

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

stdbuf -oL -eL python -u "$SCRIPT" \
    --corpus "$CORPUS" \
    --lexicon "$LEXICON" \
    --outdir "$OUTDIR" \
    --context-reservoir-size 50 \
    --spacy-sample-per-lemma 100 \
    --min-words 3 \
    --max-words 19 \
    --seed 42 \
    $EXTRA_ARGS 2>&1 | tee -a "$LOGFILE"

cleanup

echo "============================================"
echo "Finished: $(date)"
ls -lh \
    "$OUTDIR/lemma_forms.pkl" \
    "$OUTDIR/lemma_contexts.pkl" \
    "$OUTDIR/bigrams.pkl" \
    "$OUTDIR/trigrams.pkl" \
    "$OUTDIR/pos_transitions.pkl" \
    "$OUTDIR/enriched_lexicon.pkl" || true
echo "============================================"