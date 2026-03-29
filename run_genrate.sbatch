#!/bin/bash
#SBATCH --job-name=spa-generate
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=spa-generate-%j.out
#SBATCH --error=spa-generate-%j.err

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
OUTFILE="$WORK_DIR/generated_sentences_${SLURM_JOB_ID}.csv"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentgen

python -c "import pickle, csv, re, math" >/dev/null 2>&1 || {
    echo "ERROR: Missing basic Python dependencies in conda env"
    exit 1
}

for f in \
    "$SCRIPT" \
    "$LEXICON" \
    "$MODELS_DIR/lemma_forms.pkl" \
    "$MODELS_DIR/lemma_contexts.pkl" \
    "$MODELS_DIR/bigrams.pkl" \
    "$MODELS_DIR/trigrams.pkl" \
    "$MODELS_DIR/pos_transitions.pkl" \
    "$MODELS_DIR/enriched_lexicon.pkl"
do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

echo "Script:   $SCRIPT"
echo "Lexicon:  $LEXICON ($(wc -l < "$LEXICON") lines)"
echo "Models:   $MODELS_DIR"
echo "Output:   $OUTFILE"

python -u "$SCRIPT" \
    --lexicon "$LEXICON" \
    --models-dir "$MODELS_DIR" \
    --out "$OUTFILE" \
    --limit 100

echo "============================================"
echo "Finished: $(date)"
ls -lh "$OUTFILE"
echo "============================================"
