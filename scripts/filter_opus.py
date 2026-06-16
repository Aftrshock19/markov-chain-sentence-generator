#!/usr/bin/env python3
"""Streaming length + fastText language filter for the OPUS subtitle corpus.

Keeps a line iff:
  - whitespace token count in [min_tokens, max_tokens] inclusive, AND
  - fastText lid.176 top label == __label__<lang> with confidence >= threshold.

Length is checked FIRST and short-circuits: a line failing length is counted as
dropped_length even if it would also have failed the language test (it is never
sent to fastText). Memory: input is streamed line by line; the model is loaded
once before the loop; language prediction is batched for throughput.
"""
import argparse
import sys
import time

import fasttext


def parse_args():
    ap = argparse.ArgumentParser(description="Length + fastText language filter (streaming).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-tokens", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=25)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--model", default="/projects/b35cg/envs/lid.176.bin")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--heartbeat", type=int, default=1_000_000)
    return ap.parse_args()


def main():
    args = parse_args()
    target_label = "__label__" + args.lang

    # Load the fastText model ONCE, before the loop. Never reload per line.
    model = fasttext.load_model(args.model)

    total_in = 0
    kept = 0
    dropped_length = 0
    dropped_lang = 0

    # Batch buffers: only length-passing lines reach the language test.
    batch_text = []   # newline-free text fed to model.predict
    batch_raw = []    # text to write out if the line is kept

    t0 = time.time()

    def heartbeat(final=False):
        elapsed = time.time() - t0
        rate = total_in / elapsed if elapsed > 0 else 0.0
        surv = (100.0 * kept / total_in) if total_in else 0.0
        tag = "[final]   " if final else "[heartbeat]"
        print(
            f"{tag} read={total_in:,} kept={kept:,} drop_len={dropped_length:,} "
            f"drop_lang={dropped_lang:,} survival={surv:.2f}% "
            f"rate={rate:,.0f} lines/s elapsed={elapsed:,.0f}s",
            flush=True,
        )

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        def flush_batch():
            # Run one batched language prediction over accumulated length-passing lines.
            nonlocal kept, dropped_lang
            if not batch_text:
                return
            labels, probs = model.predict(batch_text, k=1)
            for raw, lab, prob in zip(batch_raw, labels, probs):
                if lab[0] == target_label and prob[0] >= args.threshold:
                    kept += 1
                    fout.write(raw + "\n")
                else:
                    dropped_lang += 1
            batch_text.clear()
            batch_raw.clear()

        for line in fin:
            total_in += 1
            text = line.rstrip("\n")
            # fastText cannot accept newlines; strip any stray CR/internal newlines too.
            clean = text.replace("\r", " ").replace("\n", " ")

            # 1) length check first (short-circuits the language test)
            ntok = len(text.split())
            if ntok < args.min_tokens or ntok > args.max_tokens:
                dropped_length += 1
            else:
                batch_text.append(clean)
                batch_raw.append(text)
                if len(batch_text) >= args.batch_size:
                    flush_batch()

            # 2) progress heartbeat on million-line boundaries (flush batch first
            #    so kept/drop_lang are exact at the moment we print)
            if total_in % args.heartbeat == 0:
                flush_batch()
                heartbeat()

        # drain any remaining length-passing lines
        flush_batch()

    heartbeat(final=True)
    print(
        f"SUMMARY input={args.input} output={args.output} "
        f"min_tokens={args.min_tokens} max_tokens={args.max_tokens} "
        f"lang={args.lang} threshold={args.threshold} "
        f"total_in={total_in:,} kept={kept:,} "
        f"dropped_length={dropped_length:,} dropped_lang={dropped_lang:,}",
        flush=True,
    )
    # sanity: kept + dropped_length + dropped_lang should equal total_in
    assert kept + dropped_length + dropped_lang == total_in, "counter mismatch"


if __name__ == "__main__":
    main()
