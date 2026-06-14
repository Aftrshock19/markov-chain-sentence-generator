#!/bin/bash
#SBATCH --job-name=sbwce-segaudit
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=sbwce-segaudit-%j.out
#SBATCH --error=sbwce-segaudit-%j.err

set -euo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "============================================"

WORK_DIR="$HOME/markov-chain-sentence-generator"
SCRIPT="$WORK_DIR/scripts/corpus_audit_segmented.py"
INPUT="$WORK_DIR/sbwce.sample1000.txt"

cd "$WORK_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentgen

for f in "$SCRIPT" "$INPUT"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

# Report segmenter availability (do NOT install anything).
python - <<'PY'
import importlib.util as u
print("spacy installed:", bool(u.find_spec("spacy")))
print("es_core_news_sm:", bool(u.find_spec("es_core_news_sm")))
print("stanza installed:", bool(u.find_spec("stanza")))
PY

echo "Input:  $INPUT ($(wc -l < "$INPUT") lines)"
echo "Output: $WORK_DIR/outputs/audit_sbwce_segmented.md"

python -u "$SCRIPT"

echo "============================================"
echo "Finished: $(date)"
ls -lh "$WORK_DIR/outputs/audit_sbwce_segmented.md"
echo "============================================"
