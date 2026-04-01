import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "sentence" not in df.columns:
        raise ValueError("CSV must contain a 'sentence' column")

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True).to(device)

    sentences = df["sentence"].fillna("").astype(str).tolist()
    translations = []

    for i in tqdm(range(0, len(sentences), args.batch_size)):
        batch = sentences[i:i + args.batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

    df["english_sentence"] = translations
    df.to_csv(args.output, index=False)
    print(f"saved to {args.output}")

if __name__ == "__main__":
    main()