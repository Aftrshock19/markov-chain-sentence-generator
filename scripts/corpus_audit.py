#!/usr/bin/env python3
"""Read-only corpus audit. Streams input, never writes/modifies corpora.
Emits outputs/audit_opus.md and outputs/audit_sbwce.md."""
import re, sys, unicodedata, random, os

ZW = dict.fromkeys(map(ord, "­​‌‍﻿"), None)

def norm(line):
    s = line.rstrip("\n")
    s = unicodedata.normalize("NFC", s)
    s = s.translate(ZW)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^-\s?", "", s, count=1)  # strip single leading dialogue dash
    return s

# --- predicates ---
RE_SPACE = re.compile(r"\bSPACE\b")
LEGAL = ["Nº","N°","ARTÍCULO","ARTICULO","VISTO","CONSIDERANDO","Expediente","CCT",
         "Decreto","Resolución","RESUELVE","período de sesiones","Señor Presidente"]
RE_BRACKET = re.compile(r"\([^)]*\)|\[[^\]]*\]")
RE_OCR_I = re.compile(r"[a-záéíóúñ]I[a-záéíóúñ]")
RE_DIGIT_IN_ALPHA = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]+\d|\d[A-Za-zÁÉÍÓÚÑáéíóúñ]+")
EN = ["the","and","with","whole","miss","Hail","you","for","this","that","of","your"]
RE_EN = re.compile(r"\b(" + "|".join(EN) + r")\b")
RE_TOKEN = re.compile(r"\S+")
RE_ALPHA = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]")

def tokens(s):
    return RE_TOKEN.findall(s)

def is_allcaps(s):
    toks = tokens(s)
    if len(toks) <= 3:
        return False
    has_alpha = any(RE_ALPHA.search(t) for t in toks)
    if not has_alpha:
        return False
    return s == s.upper()

def evaluate(s):
    n = len(s)
    digits = sum(c.isdigit() for c in s)
    toks = tokens(s)
    tc = len(toks)
    p = {}
    p["has_space_token"] = bool(RE_SPACE.search(s))
    p["legal_marker"] = any(m in s for m in LEGAL)
    p["subtitle_furniture"] = ("♪" in s or "*" in s or bool(RE_BRACKET.search(s))
                               or s.endswith(":") or is_allcaps(s))
    p["ocr_signature"] = bool(RE_OCR_I.search(s)) or bool(RE_DIGIT_IN_ALPHA.search(s))
    p["high_digit_ratio"] = (digits / n) > 0.10 if n else False
    p["length_out_of_window"] = tc < 4 or tc > 30
    en_fired = RE_EN.findall(s)
    p["english_content_token"] = bool(en_fired)
    return p, tc, en_fired

PREDS = ["has_space_token","legal_marker","subtitle_furniture","ocr_signature",
         "high_digit_ratio","length_out_of_window","english_content_token"]
TBUCKETS = [("1",1,1),("2",2,2),("3",3,3),("4-7",4,7),("8-15",8,15),("16-30",16,30),("31+",31,10**9)]

def bucket(tc):
    for name,lo,hi in TBUCKETS:
        if lo <= tc <= hi:
            return name
    return "31+"

def audit(path, k=15, seed=1234):
    rnd = random.Random(seed)
    total = 0
    pcount = {p:0 for p in PREDS}
    psamp = {p:[] for p in PREDS}   # reservoir of (idx, line)
    none_samp = []
    atleast1 = 0; atleast2 = 0
    tbuck = {name:0 for name,_,_ in TBUCKETS}
    en_tok_counts = {}
    linelen_hist = {}  # char-length buckets for non-line-delimited detection
    LL = [(0,40),(41,80),(81,160),(161,320),(321,640),(641,1280),(1281,5000),(5001,10**9)]
    llh = {f"{lo}-{hi if hi<10**9 else 'inf'}":0 for lo,hi in LL}
    def llbucket(L):
        for lo,hi in LL:
            if lo<=L<=hi: return f"{lo}-{hi if hi<10**9 else 'inf'}"
    def reservoir(res, item, seen):
        if len(res) < k: res.append(item)
        else:
            j = rnd.randint(0, seen-1)
            if j < k: res[j] = item

    seen_none = 0
    seen_pred = {p:0 for p in PREDS}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            total += 1
            llh[llbucket(len(raw.rstrip("\n")))] += 1
            s = norm(raw)
            p, tc, en_fired = evaluate(s)
            tbuck[bucket(tc)] += 1
            for tk in en_fired:
                en_tok_counts[tk] = en_tok_counts.get(tk,0)+1
            nfired = 0
            for pred in PREDS:
                if p[pred]:
                    pcount[pred] += 1
                    nfired += 1
                    seen_pred[pred] += 1
                    reservoir(psamp[pred], (total, s), seen_pred[pred])
            if nfired >= 1: atleast1 += 1
            if nfired >= 2: atleast2 += 1
            if nfired == 0:
                seen_none += 1
                reservoir(none_samp, (total, s), seen_none)
    return dict(total=total, pcount=pcount, psamp=psamp, none_samp=none_samp,
                atleast1=atleast1, atleast2=atleast2, tbuck=tbuck,
                en_tok_counts=en_tok_counts, llh=llh)

def pct(n, d): return f"{(100.0*n/d):.2f}%" if d else "0%"

def report(name, path, r):
    t = r["total"]
    L = []
    L.append(f"# Corpus audit — {name}")
    L.append("")
    L.append(f"Source: `{path}`")
    L.append(f"Read-only. NFC + soft-hyphen/zero-width strip + whitespace collapse + single leading dash strip applied for evaluation only.")
    L.append("")
    L.append(f"**Total lines:** {t:,}")
    L.append("")
    L.append("## Line char-length distribution (line-delimitation check)")
    L.append("")
    L.append("| char-length bucket | lines | % |")
    L.append("|---|---|---|")
    for kk,v in r["llh"].items():
        L.append(f"| {kk} | {v:,} | {pct(v,t)} |")
    L.append("")
    L.append("## Per-predicate firing rates")
    L.append("")
    L.append("| predicate | lines firing | % |")
    L.append("|---|---|---|")
    for p in PREDS:
        L.append(f"| {p} | {r['pcount'][p]:,} | {pct(r['pcount'][p],t)} |")
    L.append("")
    L.append("## Co-occurrence")
    L.append("")
    L.append(f"- firing ≥1 predicate (would-drop rate): {r['atleast1']:,} ({pct(r['atleast1'],t)})")
    L.append(f"- firing ≥2 predicates: {r['atleast2']:,} ({pct(r['atleast2'],t)})")
    L.append(f"- firing 0 predicates (estimated survival): {t-r['atleast1']:,} ({pct(t-r['atleast1'],t)})")
    L.append("")
    L.append("## Token-count histogram")
    L.append("")
    L.append("| tokens | lines | % |")
    L.append("|---|---|---|")
    for nm,_,_ in TBUCKETS:
        L.append(f"| {nm} | {r['tbuck'][nm]:,} | {pct(r['tbuck'][nm],t)} |")
    L.append("")
    L.append("## english_content_token — which tokens fired")
    L.append("")
    if r["en_tok_counts"]:
        L.append("| token | line hits |")
        L.append("|---|---|")
        for tk,c in sorted(r["en_tok_counts"].items(), key=lambda x:-x[1]):
            L.append(f"| {tk} | {c:,} |")
    else:
        L.append("_none fired_")
    L.append("")
    L.append("## Sampled firing lines per predicate (verbatim, judge false positives)")
    for p in PREDS:
        L.append("")
        L.append(f"### {p} ({r['pcount'][p]:,} firing; up to 15 random)")
        L.append("")
        if not r["psamp"][p]:
            L.append("_no firing lines_")
        for idx, s in r["psamp"][p]:
            L.append(f"- L{idx}: `{s}`")
    L.append("")
    L.append("## Sampled lines firing NO predicate (would-keep set, judge false negatives)")
    L.append("")
    if not r["none_samp"]:
        L.append("_none_")
    for idx, s in r["none_samp"]:
        L.append(f"- L{idx}: `{s}`")
    L.append("")
    L.append("## Combined note")
    L.append("")
    L.append(f"Estimated survival (lines firing zero predicates): **{pct(t-r['atleast1'],t)}** "
             f"({t-r['atleast1']:,} of {t:,}).")
    L.append("")
    return "\n".join(L)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    # NOTE: filenames are inverted vs. corpus identity. Verified by content:
    #   sbwce.sample1000.txt = SBWCE (encyclopedic + Argentine legal prose, SPACE tokens)
    #   es (1).txt           = OPUS OpenSubtitles (one-utterance-per-line dialogue)
    jobs = [
        ("SBWCE", "sbwce.sample1000.txt", "outputs/audit_sbwce.md"),
        ("OPUS OpenSubtitles", "es (1).txt", "outputs/audit_opus.md"),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for nm, path, out in jobs:
        if only and only not in path:
            continue
        sys.stderr.write(f"auditing {path}...\n"); sys.stderr.flush()
        r = audit(path)
        with open(out, "w", encoding="utf-8") as fo:
            fo.write(report(nm, path, r))
        sys.stderr.write(f"wrote {out} (total={r['total']:,})\n"); sys.stderr.flush()
