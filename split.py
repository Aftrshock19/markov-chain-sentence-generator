import csv

input_file = "stg_words_spa.csv"
output_file = "lemma.csv"

with open(input_file, "r", encoding="utf-8", newline="") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:
    
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=["lemma", "rank"])
    
    writer.writeheader()
    for row in reader:
        writer.writerow({
            "lemma": row["lemma"],
            "rank": row["rank"]
        })