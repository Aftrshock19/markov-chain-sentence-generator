#!/bin/bash
#SBATCH --job-name=spa-starter-seed
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=spa-starter-seed-%j.out
#SBATCH --error=spa-starter-seed-%j.err

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
SCRIPT="$WORK_DIR/generate.py"

OUT_DIR="$WORK_DIR/outputs"
OUTFILE="$OUT_DIR/starter.csv"
REVIEW_OUT="$OUT_DIR/app_seed_a1_a2_review.csv"
QUARANTINE_OUT="$OUT_DIR/app_seed_a1_a2_quarantine.csv"
CANDIDATES_OUT="$OUT_DIR/app_seed_a1_a2_candidates.csv"

OVERRIDE_1="$WORK_DIR/starter_overrides.csv"
OVERRIDE_2="$WORK_DIR/starter_exclusions.csv"

mkdir -p "$OUT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentgen

echo "Python path:   $(which python)"
python --version

python - <<'PY'
mods = [
    "pickle",
    "csv",
    "re",
    "math",
    "statistics",
    "sklearn",
]
for m in mods:
    try:
        __import__(m)
        print(f"OK import {m}")
    except Exception as e:
        print(f"FAIL import {m}: {e}")
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

python "$SCRIPT" \
    --lexicon "$LEXICON" \
    --models-dir "$MODELS_DIR" \
    --starter-max-attempts 5000 \
    --limit 1000 \
    --out "$OUTFILE" \
    --review-out "$REVIEW_OUT" \
    --quarantine-out "$QUARANTINE_OUT" \
    --candidates-out "$CANDIDATES_OUT" \
    --lexicon-overrides "$OVERRIDE_1" \
    --lexicon-overrides "$OVERRIDE_2"

echo "============================================"
echo "Finished: $(date)"
ls -lh "$OUTFILE" "$REVIEW_OUT" "$CANDIDATES_OUT" || true
[ -f "$QUARANTINE_OUT" ] && ls -lh "$QUARANTINE_OUT" || true
echo "============================================"