import pandas as pd

sent_df = pd.read_csv("final_sentences_en.csv")
lemma_df = pd.read_csv("stg_words_spa.csv")

sent_df["rank"] = pd.to_numeric(sent_df["rank"], errors="coerce")
lemma_df["rank"] = pd.to_numeric(lemma_df["rank"], errors="coerce")

merged = lemma_df.merge(
    sent_df[["rank", "sentence", "english_sentence"]],
    on="rank",
    how="inner"
)

merged = merged[["rank", "lemma", "sentence", "english_sentence"]]
merged = merged.sort_values("rank")

merged.to_csv("merged.csv", index=False)