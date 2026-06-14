#!/usr/bin/env python3
"""Read-only SBWCE re-audit WITH stage-0 sentence segmentation.
Does not write a cleaned corpus or modify input. Emits one report."""
import re, unicodedata, random, os

ZW = dict.fromkeys(map(ord, "­​‌‍﻿"), None)

def norm(line):
    s = unicodedata.normalize("NFC", line.rstrip("\n"))
    s = s.translate(ZW)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^-\s?", "", s, count=1)
    return s

# ---------- Stage 0: sentence segmentation ----------
# Prefer a real Spanish segmenter if present (spaCy es_core_news_sm senter, then
# stanza es). Never pip-install. Fall back to the protected punctuation splitter.
_SEG = None
SEG_METHOD = None

def _try_spacy():
    global _SEG, SEG_METHOD
    import spacy
    nlp = spacy.load("es_core_news_sm", disable=["ner","tagger","lemmatizer","attribute_ruler"])
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.enable_pipe("senter") if "senter" in nlp.disabled else nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000
    def seg(text):
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    _SEG = seg
    SEG_METHOD = f"spaCy es_core_news_sm ({spacy.__version__}), sentence boundaries via senter/parser"

def _try_stanza():
    global _SEG, SEG_METHOD
    import stanza
    nlp = stanza.Pipeline(lang="es", processors="tokenize", tokenize_no_ssplit=False, verbose=False)
    def seg(text):
        return [s.text.strip() for s in nlp(text).sentences if s.text.strip()]
    _SEG = seg
    SEG_METHOD = f"stanza es ({stanza.__version__}) tokenize+ssplit"

def init_segmenter():
    global _SEG, SEG_METHOD
    for fn in (_try_spacy, _try_stanza):
        try:
            fn(); return
        except Exception:
            continue
    _SEG = segment  # fallback defined below
    SEG_METHOD = ("Fallback protected punctuation splitter (spaCy & stanza unavailable; "
                  "no pip per instructions). Splits on [.!?]+ + optional close-quote/paren + "
                  "whitespace before a sentence-starter; guards: abbreviation list "
                  "(Sr., Sra., Dr., Nº, art., etc.) + single initials + decimals.")

# Fallback punctuation splitter (used if no spaCy/stanza; no pip per instructions).
# Split on [.!?] (+ optional closing quote/paren) followed by whitespace and a
# sentence-starter (uppercase letter, ¿, ¡, or opening quote). Protect abbreviations
# and decimals by NOT splitting when the token before the period is a known abbrev
# or when the period sits between two digits.
ABBREV = {
    "sr","sra","srta","dr","dra","d","da","ing","lic","gral","tte","cnel","cap",
    "av","avda","ej","etc","art","arts","núm","nro","nº","no","c","cía","cia","ud","uds",
    "vol","pág","págs","ss","fig","op","cit","ph","ee","uu","aa","ss","s","p","pp","n",
    "ltda","s.a","ej","aprox","máx","mín","ref","exp","expte","res","dto","mr","mrs","ms",
    "st","jr","prof","tel","depto","dpto","ada","atte","caps","cap","i.e","e.g",
}
# match a candidate boundary: terminator, optional close-quote/paren, whitespace, starter
BOUNDARY = re.compile(r'([.!?]+)(["»”’\')\]]?)(\s+)(?=[«"“¿¡(\[A-ZÁÉÍÓÚÑ])')
WORD_BEFORE = re.compile(r'([A-Za-zÁÉÍÓÚÑáéíóúñ\.]+)\.?$')

def looks_abbrev(left_text):
    # token immediately before the period
    m = re.search(r'([A-Za-zÁÉÍÓÚÑáéíóúñ\.]+)$', left_text)
    if not m:
        return False
    tok = m.group(1).strip(".").lower()
    if tok in ABBREV:
        return True
    # single initial like "J" in "J. D. Salinger"
    if len(tok) == 1 and tok.isalpha():
        return True
    return False

def is_decimal(text, dotpos):
    # period flanked by digits e.g. 3.14  or  1.000
    before = text[dotpos-1] if dotpos-1 >= 0 else ""
    after = text[dotpos+1] if dotpos+1 < len(text) else ""
    return before.isdigit() and after.isdigit()

def segment(line):
    s = line
    if not s:
        return []
    out = []
    start = 0
    for m in BOUNDARY.finditer(s):
        term_start = m.start(1)
        # left text up to (and incl) terminator's first char position
        left = s[start:term_start]
        # decimal guard: terminator is single '.' between digits
        if m.group(1) == "." and is_decimal(s, term_start):
            continue
        # abbreviation guard
        if looks_abbrev(left):
            continue
        end = m.end(2)  # include terminator + optional closing quote/paren
        sent = s[start:end].strip()
        if sent:
            out.append(sent)
        start = m.end(3)  # resume after the whitespace
    tail = s[start:].strip()
    if tail:
        out.append(tail)
    return out

# ---------- predicates (same as prior audit; ocr split; { } added) ----------
RE_SPACE = re.compile(r"\bSPACE\b")
LEGAL = ["Nº","N°","ARTÍCULO","ARTICULO","VISTO","CONSIDERANDO","Expediente","CCT",
         "Decreto","Resolución","RESUELVE","período de sesiones","Señor Presidente"]
RE_BRACKET = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")   # () [] {} ASS/SSA
RE_OCR_IFORL = re.compile(r"[a-záéíóúñ]I[a-záéíóúñ]")
RE_DIGIT_IN_ALPHA = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]+\d|\d[A-Za-zÁÉÍÓÚÑáéíóúñ]+")
EN = ["the","and","with","whole","miss","Hail","you","for","this","that","of","your"]
RE_EN = re.compile(r"\b(" + "|".join(EN) + r")\b")
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
    p["has_space_token"] = bool(RE_SPACE.search(s))
    p["legal_marker"] = any(m in s for m in LEGAL)
    p["subtitle_furniture"] = ("♪" in s or "*" in s or bool(RE_BRACKET.search(s))
                               or s.endswith(":") or is_allcaps(s))
    p["ocr_iforl"] = bool(RE_OCR_IFORL.search(s))
    p["digit_in_alpha"] = bool(RE_DIGIT_IN_ALPHA.search(s))
    p["high_digit_ratio"] = (digits / n) > 0.10 if n else False
    p["length_out_of_window"] = tc < 4 or tc > 30
    en = RE_EN.findall(s)
    p["english_content_token"] = bool(en)
    return p, tc, en

PREDS = ["has_space_token","legal_marker","subtitle_furniture","ocr_iforl",
         "digit_in_alpha","high_digit_ratio","length_out_of_window","english_content_token"]
TBUCKETS = [("1",1,1),("2",2,2),("3",3,3),("4-7",4,7),("8-15",8,15),("16-30",16,30),("31+",31,10**9)]
def bucket(tc):
    for nm,lo,hi in TBUCKETS:
        if lo<=tc<=hi: return nm
    return "31+"

def pct(n,d): return f"{(100.0*n/d):.2f}%" if d else "0%"

def audit(path, k=15, seed=1234):
    rnd = random.Random(seed)
    lines = 0; sents = 0
    pcount = {p:0 for p in PREDS}; psamp = {p:[] for p in PREDS}
    none_samp = []; atleast1=0; atleast2=0
    tbuck = {nm:0 for nm,_,_ in TBUCKETS}; en_counts={}
    seen_pred = {p:0 for p in PREDS}; seen_none=0
    def reservoir(res,item,seen):
        if len(res)<k: res.append(item)
        else:
            j=rnd.randint(0,seen-1)
            if j<k: res[j]=item
    with open(path,encoding="utf-8",errors="replace") as f:
        for raw in f:
            lines += 1
            base = norm(raw)
            for sent in _SEG(base):
                sents += 1
                p,tc,en = evaluate(sent)
                tbuck[bucket(tc)] += 1
                for tk in en: en_counts[tk]=en_counts.get(tk,0)+1
                nf=0
                for pred in PREDS:
                    if p[pred]:
                        pcount[pred]+=1; nf+=1; seen_pred[pred]+=1
                        reservoir(psamp[pred],(lines,sent),seen_pred[pred])
                if nf>=1: atleast1+=1
                if nf>=2: atleast2+=1
                if nf==0:
                    seen_none+=1; reservoir(none_samp,(lines,sent),seen_none)
    return dict(lines=lines,sents=sents,pcount=pcount,psamp=psamp,none_samp=none_samp,
                atleast1=atleast1,atleast2=atleast2,tbuck=tbuck,en_counts=en_counts)

def report(path,r,method):
    s=r["sents"]; L=[]
    L.append("# Corpus audit — SBWCE (segmented, stage-0 sentence split)")
    L.append("")
    L.append(f"Source: `{path}`  —  read-only; input unmodified.")
    L.append(f"Normalisation (eval only): NFC + soft-hyphen/zero-width strip + whitespace collapse + single leading dash strip.")
    L.append("")
    L.append("## Stage 0 — segmentation")
    L.append("")
    L.append(f"- **Method:** {method}")
    L.append(f"- **Input lines:** {r['lines']:,}")
    L.append(f"- **Output sentences:** {s:,}")
    L.append(f"- **Sentences per line (weld factor):** {s/r['lines']:.2f}")
    L.append("")
    L.append("## Per-predicate firing rates (over SENTENCES)")
    L.append("")
    L.append("| predicate | sentences firing | % |")
    L.append("|---|---|---|")
    for p in PREDS:
        L.append(f"| {p} | {r['pcount'][p]:,} | {pct(r['pcount'][p],s)} |")
    L.append("")
    L.append("_ocr_signature is reported split into `ocr_iforl` and `digit_in_alpha` (not merged), per request._")
    L.append("")
    L.append("## Co-occurrence")
    L.append("")
    L.append(f"- firing ≥1 predicate (would-drop): {r['atleast1']:,} ({pct(r['atleast1'],s)})")
    L.append(f"- firing ≥2 predicates: {r['atleast2']:,} ({pct(r['atleast2'],s)})")
    L.append(f"- firing 0 predicates (estimated survival): {s-r['atleast1']:,} ({pct(s-r['atleast1'],s)})")
    L.append("")
    L.append("## Token-count histogram (over sentences)")
    L.append("")
    L.append("| tokens | sentences | % |")
    L.append("|---|---|---|")
    for nm,_,_ in TBUCKETS:
        L.append(f"| {nm} | {r['tbuck'][nm]:,} | {pct(r['tbuck'][nm],s)} |")
    L.append("")
    L.append("## english_content_token — which tokens fired")
    L.append("")
    if r["en_counts"]:
        L.append("| token | sentence hits |"); L.append("|---|---|")
        for tk,c in sorted(r["en_counts"].items(),key=lambda x:-x[1]):
            L.append(f"| {tk} | {c:,} |")
    else: L.append("_none_")
    L.append("")
    L.append("## Sampled firing sentences per predicate (verbatim)")
    for p in PREDS:
        L.append(""); L.append(f"### {p} ({r['pcount'][p]:,} firing; up to 15 random)"); L.append("")
        if not r["psamp"][p]: L.append("_none_")
        for idx,sent in r["psamp"][p]: L.append(f"- L{idx}: `{sent}`")
    L.append(""); L.append("## Sampled zero-predicate sentences (would-keep set)"); L.append("")
    if not r["none_samp"]: L.append("_none_")
    for idx,sent in r["none_samp"]: L.append(f"- L{idx}: `{sent}`")
    L.append(""); L.append("## Combined note"); L.append("")
    L.append(f"Estimated post-segmentation survival (sentences firing zero predicates): "
             f"**{pct(s-r['atleast1'],s)}** ({s-r['atleast1']:,} of {s:,} sentences).")
    L.append("")
    return "\n".join(L)

if __name__ == "__main__":
    os.makedirs("outputs",exist_ok=True)
    path="sbwce.sample1000.txt"
    init_segmenter()
    r=audit(path)
    with open("outputs/audit_sbwce_segmented.md","w",encoding="utf-8") as fo:
        fo.write(report(path,r,SEG_METHOD))
    print("method:", SEG_METHOD)
    print(f"lines={r['lines']:,} sents={r['sents']:,} ratio={r['sents']/r['lines']:.2f}")
