import csv

INPUT_CSV = "outputs/sentences.csv"
OUTPUT_CSV = "empty_ranks.csv"

empty_ranks = []

with open(INPUT_CSV, "r", encoding="utf-8", newline="") as infile:
    reader = csv.reader(infile)
    
    for row in reader:
        if not row:
            continue
        
        rank = row[0].strip()
        sentence = row[1].strip() if len(row) > 1 else ""
        
        if rank and sentence == "":
            empty_ranks.append([rank])

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["rank"])
    writer.writerows(empty_ranks)

print(f"saved {len(empty_ranks)} empty ranks to {OUTPUT_CSV}")