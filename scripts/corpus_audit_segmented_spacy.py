#!/usr/bin/env python3
"""Read-only SBWCE re-audit WITH stage-0 spaCy sentence segmentation.

Read-only: does NOT modify the input and does NOT write a cleaned corpus.
The only file written is the report at OUTPUT. Run on the login node
(1000-line sample, senter-only pipe = lightweight). No pip, no sbatch.

Stage 0: spaCy es_core_news_sm, senter only (tagger/parser/ner/lemmatizer/
attribute_ruler/morphologizer disabled), nlp.pipe(batch_size=200).
Predicates: identical to the prior unsegmented audit, with two corrections —
subtitle_furniture also fires on {...} (ASS/SSA), and ocr_signature is split
into two reported sub-predicates ocr_iforl / digit_in_alpha (NOT merged).
"""
import re, unicodedata, random
import spacy

INPUT  = "/projects/b35cg/corpora/sbwce_sample1000.txt"
OUTPUT = "/lfs1i3/home/b35cg/samtoughan.b35cg/markov-chain-sentence-generator/outputs/audit_sbwce_segmented.md"
SEED, K = 1234, 15

# ---------- normalisation (evaluation only; input never rewritten) ----------
ZW = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)  # soft hyphen + zero-width

def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = s.translate(ZW)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^-\s?", "", s, count=1)   # strip a single leading dialogue dash
    return s

# ---------- Stage 0: spaCy senter ----------
def load_senter():
    nlp = spacy.load("es_core_news_sm",
                     disable=["tagger","parser","ner","lemmatizer","attribute_ruler","morphologizer"])
    nlp.enable_pipe("senter")
    nlp.max_length = 2_000_000
    method = (f"spaCy es_core_news_sm v{spacy.__version__}, senter only "
              f"(disabled: tagger/parser/ner/lemmatizer/attribute_ruler/morphologizer); "
              f"pipes={nlp.pipe_names}; nlp.pipe(batch_size=200)")
    return nlp, method

# ---------- predicates (same as prior audit; ocr split; {} added) ----------
RE_SPACE = re.compile(r"\bSPACE\b")
LEGAL = ["Nº","N°","ARTÍCULO","ARTICULO","VISTO","CONSIDERANDO","Expediente","CCT",
         "Decreto","Resolución","RESUELVE","período de sesiones","Señor Presidente"]
RE_BRACKET = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")      # () [] {}  (ASS/SSA = {})
RE_OCR_IFORL = re.compile(r"[a-záéíóúñ]I[a-záéíóúñ]")          # uppercase I flanked by lowercase
RE_DIGIT_IN_ALPHA = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]\d|\d[A-Za-zÁÉÍÓÚÑáéíóúñ]")  # digit adjacent to letter
EN = ["the","and","with","whole","miss","Hail","you","for","this","that","of","your"]
RE_EN = re.compile(r"\b(" + "|".join(EN) + r")\b")             # case-sensitive, matches prior audit
RE_TOKEN = re.compile(r"\S+")
RE_ALPHA = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]")

def tokens(s): return RE_TOKEN.findall(s)

def is_allcaps(s):
    toks = tokens(s)
    if len(toks) <= 3: return False
    if not any(RE_ALPHA.search(t) for t in toks): return False
    return s == s.upper()

def evaluate(s):
    n = len(s); digits = sum(c.isdigit() for c in s); tc = len(tokens(s))
    p = {}
    p["has_space_token"]       = bool(RE_SPACE.search(s))
    p["legal_marker"]          = any(m in s for m in LEGAL)
    p["subtitle_furniture"]    = ("♪" in s or "*" in s or bool(RE_BRACKET.search(s))
                                  or s.endswith(":") or is_allcaps(s))
    p["ocr_iforl"]             = bool(RE_OCR_IFORL.search(s))
    p["digit_in_alpha"]        = bool(RE_DIGIT_IN_ALPHA.search(s))
    p["high_digit_ratio"]      = (digits / n) > 0.10 if n else False
    p["length_out_of_window"]  = tc < 4 or tc > 30
    en = RE_EN.findall(s)
    p["english_content_token"] = bool(en)
    return p, tc, en

PREDS = ["has_space_token","legal_marker","subtitle_furniture","ocr_iforl",
         "digit_in_alpha","high_digit_ratio","length_out_of_window","english_content_token"]
TBUCKETS = [("1",1,1),("2",2,2),("3",3,3),("4-7",4,7),("8-15",8,15),("16-30",16,30),("31+",31,10**9)]
def bucket(tc):
    for nm,lo,hi in TBUCKETS:
        if lo <= tc <= hi: return nm
    return "31+"
def pct(n,d): return f"{(100.0*n/d):.2f}%" if d else "0%"

# ---------- run ----------
def main():
    nlp, method = load_senter()
    with open(INPUT, encoding="utf-8", errors="replace") as f:
        raw = [l.rstrip("\n") for l in f]
    norm_lines = [norm(l) for l in raw]

    n_lines = len(raw)
    sents = []                      # (src_line_1based, sentence_text)
    for i, doc in enumerate(nlp.pipe(norm_lines, batch_size=200)):
        for sp in doc.sents:
            t = re.sub(r"^-\s?", "", sp.text.strip(), count=1)
            if t:
                sents.append((i + 1, t))
    n_sents = len(sents)

    pcount = {p: 0 for p in PREDS}
    pfire  = {p: [] for p in PREDS}       # (line, sentence) lists, full (sample later)
    none_list = []
    atleast1 = atleast2 = 0
    tbuck = {nm: 0 for nm, _, _ in TBUCKETS}
    en_counts = {}

    for ln, sent in sents:
        p, tc, en = evaluate(sent)
        tbuck[bucket(tc)] += 1
        for tk in en: en_counts[tk] = en_counts.get(tk, 0) + 1
        nf = 0
        for pred in PREDS:
            if p[pred]:
                pcount[pred] += 1; nf += 1; pfire[pred].append((ln, sent))
        if nf >= 1: atleast1 += 1
        if nf >= 2: atleast2 += 1
        if nf == 0: none_list.append((ln, sent))

    rnd = random.Random(SEED)
    def sample(lst): return sorted(rnd.sample(lst, min(K, len(lst))), key=lambda x: x[0]) if lst else []

    # ---------- report ----------
    L = []
    L.append("# Corpus audit — SBWCE (segmented, stage-0 sentence split)")
    L.append("")
    L.append(f"Source: `{INPUT}`  —  read-only; input unmodified, no cleaned corpus written.")
    L.append("Normalisation (evaluation only): NFC + soft-hyphen/zero-width strip + whitespace "
             "collapse + single leading dash strip. Case preserved.")
    L.append("")
    L.append("## Stage 0 — sentence segmentation")
    L.append("")
    L.append(f"- **Method:** {method}")
    L.append(f"- **Input lines:** {n_lines:,}")
    L.append(f"- **Output sentences:** {n_sents:,}")
    L.append(f"- **Sentences per line (weld factor):** {n_sents / n_lines:.2f}")
    L.append("")
    L.append("## Per-predicate firing rates (over SENTENCES)")
    L.append("")
    L.append("| predicate | sentences firing | % |")
    L.append("|---|---|---|")
    for p in PREDS:
        L.append(f"| {p} | {pcount[p]:,} | {pct(pcount[p], n_sents)} |")
    L.append("")
    L.append("_`ocr_signature` is reported split into `ocr_iforl` and `digit_in_alpha` (not merged), per request._")
    L.append("")
    L.append("## Co-occurrence")
    L.append("")
    L.append(f"- firing ≥1 predicate (would-drop): {atleast1:,} ({pct(atleast1, n_sents)})")
    L.append(f"- firing ≥2 predicates: {atleast2:,} ({pct(atleast2, n_sents)})")
    L.append(f"- firing 0 predicates (estimated survival): {n_sents - atleast1:,} ({pct(n_sents - atleast1, n_sents)})")
    L.append("")
    L.append("## Token-count histogram (over sentences)")
    L.append("")
    L.append("| tokens | sentences | % |")
    L.append("|---|---|---|")
    for nm, _, _ in TBUCKETS:
        L.append(f"| {nm} | {tbuck[nm]:,} | {pct(tbuck[nm], n_sents)} |")
    L.append("")
    L.append("## english_content_token — which tokens fired")
    L.append("")
    if en_counts:
        L.append("| token | sentence hits |"); L.append("|---|---|")
        for tk, c in sorted(en_counts.items(), key=lambda x: -x[1]):
            L.append(f"| {tk} | {c:,} |")
    else:
        L.append("_none fired_")
    L.append("")
    L.append("## Sampled firing sentences per predicate (verbatim, judge false positives)")
    for p in PREDS:
        L.append(""); L.append(f"### {p} ({pcount[p]:,} firing; up to {K} random)"); L.append("")
        s = sample(pfire[p])
        if not s: L.append("_none_")
        for ln, sent in s: L.append(f"- L{ln}: `{sent}`")
    L.append(""); L.append(f"## Sampled zero-predicate sentences (would-keep set; up to {K} random)"); L.append("")
    s = sample(none_list)
    if not s: L.append("_none_")
    for ln, sent in s: L.append(f"- L{ln}: `{sent}`")
    L.append(""); L.append("## Combined note"); L.append("")
    L.append(f"Estimated post-segmentation survival (sentences firing zero predicates): "
             f"**{pct(n_sents - atleast1, n_sents)}** ({n_sents - atleast1:,} of {n_sents:,} sentences).")
    L.append("")

    with open(OUTPUT, "w", encoding="utf-8") as fo:
        fo.write("\n".join(L))
    print("method:", method)
    print(f"lines={n_lines:,} sents={n_sents:,} ratio={n_sents/n_lines:.2f}")
    print(f"survival(0-pred)={pct(n_sents - atleast1, n_sents)}  >=1={pct(atleast1, n_sents)}  >=2={pct(atleast2, n_sents)}")
    print("wrote", OUTPUT)

if __name__ == "__main__":
    main()
