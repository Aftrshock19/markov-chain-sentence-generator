#!/usr/bin/env python3
"""Evaluate the coherence discriminator against the PRE-REGISTERED frozen gate.

Pass condition (data_clean/disc_eval_protocol.md): pairwise accuracy >= 0.70 on
the held-out good-vs-nonsense eval set, i.e. over all (good, nonsense) cross
pairs, the fraction where score(good) > score(nonsense).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disc_model import load_discriminator  # noqa: E402

GATE = 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disc", default="data_clean/disc.pt")
    ap.add_argument("--frozen", default="data_clean/disc_eval_frozen.json")
    args = ap.parse_args()

    disc = load_discriminator(Path(args.disc))
    if disc is None:
        raise SystemExit(f"discriminator not found: {args.disc}")
    fr = json.loads(Path(args.frozen).read_text())
    good, non = fr["good"], fr["nonsense"]
    sg = [disc.score(s.split()) for s in good]
    sn = [disc.score(s.split()) for s in non]

    # pairwise accuracy (ties count as 0.5)
    wins = ties = 0
    for a in sg:
        for b in sn:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    total = len(sg) * len(sn)
    acc = (wins + 0.5 * ties) / total

    import statistics as st
    print(f"[eval] good n={len(sg)} mean={st.mean(sg):.3f}  nonsense n={len(sn)} mean={st.mean(sn):.3f}", file=sys.stderr)
    print(f"[eval] pairwise accuracy = {acc:.4f}  (gate {GATE})", file=sys.stderr)
    verdict = "PASS" if acc >= GATE else "FAIL"
    print(f"[eval] === {verdict} ===", file=sys.stderr)
    # show worst-ranked good and best-ranked nonsense (failure modes)
    gi = sorted(range(len(sg)), key=lambda i: sg[i])[:6]
    ni = sorted(range(len(sn)), key=lambda i: sn[i], reverse=True)[:6]
    print("\nlowest-scored GOOD (false rejects):", file=sys.stderr)
    for i in gi:
        print(f"   {sg[i]:.3f}  {good[i]}", file=sys.stderr)
    print("highest-scored NONSENSE (false accepts):", file=sys.stderr)
    for i in ni:
        print(f"   {sn[i]:.3f}  {non[i]}", file=sys.stderr)
    print(json.dumps({"pairwise_acc": acc, "gate": GATE, "verdict": verdict,
                      "good_mean": st.mean(sg), "nonsense_mean": st.mean(sn)}))


if __name__ == "__main__":
    main()
