#!/bin/bash
#SBATCH --job-name=spa-train
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=spa-train-%j.out
#SBATCH --error=spa-train-%j.err

set -euo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

WORK_DIR="$HOME/markov-chain-sentence-generator"
TATOEBA="$WORK_DIR/spa_sentences.tsv"
LEXICON="$WORK_DIR/stg_words_spa.csv"
OUTDIR="$WORK_DIR/models"
SCRIPT="$WORK_DIR/train.py"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentgen

python -c "import spacy; import tqdm" >/dev/null 2>&1 || {
    echo "ERROR: Missing dependencies in conda env"
    exit 1
}

for f in "$SCRIPT" "$TATOEBA" "$LEXICON"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

mkdir -p "$OUTDIR"

echo "Tatoeba: $TATOEBA ($(wc -l < "$TATOEBA") lines)"
echo "Lexicon: $LEXICON ($(wc -l < "$LEXICON") lines)"
echo "Output:  $OUTDIR"

python "$SCRIPT" \
    --tatoeba "$TATOEBA" \
    --lexicon "$LEXICON" \
    --outdir "$OUTDIR" \
    --seed 42

echo "============================================"
echo "Finished: $(date)"
ls -lh "$OUTDIR"
echo "============================================"