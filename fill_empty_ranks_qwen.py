import csv
import argparse

RANK_CANDIDATES = ["rank", "rango", "id", "idx", "index"]
WORD_CANDIDATES = ["word", "lemma", "palabra", "spanish", "target", "lexeme", "lemma_lower"]

def normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def find_column(fieldnames, candidates):
    if not fieldnames:
        return None
    normalized = {normalize_header(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for original in fieldnames:
        n = normalize_header(original)
        for candidate in candidates:
            if candidate in n:
                return original
    return None

def detect_rank_and_word_columns(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {csv_path}")
        rank_col = find_column(reader.fieldnames, RANK_CANDIDATES)
        word_col = find_column(reader.fieldnames, WORD_CANDIDATES)
        if rank_col is None:
            raise ValueError(
                f"Could not detect rank column in {csv_path}. Available columns: {reader.fieldnames}"
            )
        if word_col is None:
            raise ValueError(
                f"Could not detect word column in {csv_path}. Available columns: {reader.fieldnames}"
            )
        return rank_col, word_col

def load_empty_ranks(csv_path):
    ranks = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {csv_path}")
        rank_col = find_column(reader.fieldnames, RANK_CANDIDATES)
        if rank_col is None:
            raise ValueError(
                f"Could not detect rank column in {csv_path}. Available columns: {reader.fieldnames}"
            )
        for row in reader:
            rank = (row.get(rank_col) or "").strip()
            if rank:
                ranks.append(rank)
    return ranks

def load_lexicon_map(csv_path):
    rank_col, word_col = detect_rank_and_word_columns(csv_path)
    lexicon = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = (row.get(rank_col) or "").strip()
            word = (row.get(word_col) or "").strip()
            if rank and word and rank not in lexicon:
                lexicon[rank] = word
    return lexicon

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty-ranks", required=True)
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    empty_ranks = load_empty_ranks(args.empty_ranks)
    lexicon = load_lexicon_map(args.lexicon)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word"])
        for rank in empty_ranks:
            writer.writerow([rank, lexicon.get(rank, "")])

    print(f"Saved {len(empty_ranks)} rows to {args.out}")

if __name__ == "__main__":
    main()