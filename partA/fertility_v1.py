#!/usr/bin/env python3
"""
fertility_v1.py -- corrected replacement for the starter kit's fertility.py.

What changed vs v0, and why (each is measured in ../partA/A2_AUDIT.md):

  * no .lower()           -- casing is a property of the script, not of the
                             tokenizer's efficiency. Lowercasing only touches
                             cased languages, so it silently rescales the
                             English baseline and nothing else.  [F2]
  * str.split()           -- v0's split(" ") emits empty strings for runs of
                             whitespace, inflating the word count.            [F1]
  * corpus-level ratios   -- v0 averages per-line ratios, which weights a
                             4-word sentence the same as a 40-word one.       [F3]
  * four denominators     -- word / grapheme cluster / UTF-8 byte / parallel
                             sentence, reported side by side, because the
                             denominator IS the analysis.                     [A3]
  * parallel-sentence     -- the only denominator that holds *meaning*
    ratios                  constant across languages. Requires a line-aligned
                             corpus, which is asserted, not assumed.

Kept from v0 deliberately:
  * unicodedata NFC normalisation -- correct, see D1.

Usage:
    python fertility_v1.py --corpus-dir corpus \
        --tokenizer gpt2 --tokenizer hf:sarvamai/sarvam-1 \
        --out results/a3

    python fertility_v1.py --corpus-dir corpus --preset all --out results/a3
"""

import argparse
import csv
import os
import sys
import unicodedata

import regex

HERE = os.path.dirname(os.path.abspath(__file__))

# label -> spec.  gpt2 is the tokenizer REPORT_v0 used.
PRESET = [
    ("gpt2", "gpt2"),
    ("qwen2.5", "hf:Qwen/Qwen2.5-0.5B"),
    ("xlm-r", "hf:xlm-roberta-base"),
    ("bloom", "hf:bigscience/bloom-560m"),
    ("sarvam-1", "hf:sarvamai/sarvam-1"),
]

LANG_ORDER = ["eng", "hin", "ben", "mar", "kan", "tam", "tel"]
FAMILY = {"eng": "Germanic", "hin": "Indo-Aryan", "ben": "Indo-Aryan",
          "mar": "Indo-Aryan", "kan": "Dravidian", "tam": "Dravidian",
          "tel": "Dravidian"}


def load_tokenizer(spec):
    if spec.startswith("hf:"):
        import warnings
        warnings.filterwarnings("ignore")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    import tiktoken
    return tiktoken.get_encoding(spec).encode


def read_corpus(corpus_dir, langs):
    corp = {}
    for lang in langs:
        path = os.path.join(corpus_dir, lang + ".txt")
        lines = []
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                lines.append(unicodedata.normalize("NFC", line))
        corp[lang] = lines
    n = {l: len(v) for l, v in corp.items()}
    if len(set(n.values())) != 1:
        raise SystemExit(
            "corpus is not line-aligned: %s\n"
            "The per-parallel-sentence denominator requires alignment." % n)
    return corp


def totals(lines, encode):
    """Corpus-level counts. No per-line averaging -- see F3."""
    t = w = g = b = 0
    for line in lines:
        t += len(encode(line))
        w += len(line.split())
        g += len(regex.findall(r"\X", line))
        b += len(line.encode("utf-8"))
    return dict(tokens=t, words=w, graphemes=g, bytes=b, sents=len(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=os.path.join(HERE, "corpus"))
    ap.add_argument("--tokenizer", action="append", default=None,
                    help="repeatable. 'gpt2' or 'hf:<repo>'. Label = spec.")
    ap.add_argument("--preset", choices=["all"], default=None)
    ap.add_argument("--langs", default=",".join(LANG_ORDER),
                    help="comma list; the FIRST is the pivot language")
    ap.add_argument("--out", default=None, help="path prefix for .csv output")
    args = ap.parse_args()

    if args.preset == "all":
        toks = PRESET
    elif args.tokenizer:
        toks = [(s.replace("hf:", ""), s) for s in args.tokenizer]
    else:
        toks = [("gpt2", "gpt2")]

    langs = args.langs.split(",")
    pivot = langs[0]
    corp = read_corpus(args.corpus_dir, langs)
    n_sent = len(corp[pivot])

    print("corpus     : %s  (%d parallel sentences x %d languages)"
          % (os.path.normpath(args.corpus_dir), n_sent, len(langs)))
    print("pivot      : %s" % pivot)
    print("aggregation: corpus-level totals (micro), no lowercasing, NFC on\n")

    rows = []
    for label, spec in toks:
        try:
            encode = load_tokenizer(spec)
        except Exception as e:
            print("SKIP %s (%s: %s)" % (label, type(e).__name__, str(e)[:70]),
                  file=sys.stderr)
            continue

        tot = {l: totals(corp[l], encode) for l in langs}
        p = tot[pivot]

        print("=" * 100)
        print("tokenizer: %s   (%s)" % (label, spec))
        print("=" * 100)
        hdr = ("%-6s %-11s %9s %8s %8s %8s %9s %9s"
               % ("lang", "family", "tok/word", "tok/graph", "tok/byte",
                  "tok/sent", "x" + pivot + "(sent)", "tokens"))
        print(hdr)
        print("-" * len(hdr))
        for l in langs:
            d = tot[l]
            tpw = d["tokens"] / d["words"]
            tpg = d["tokens"] / d["graphemes"]
            tpb = d["tokens"] / d["bytes"]
            tps = d["tokens"] / d["sents"]
            # parallel-sentence ratio: same sentence count, so this is simply
            # the ratio of total tokens for identical semantic content.
            rel = d["tokens"] / p["tokens"]
            print("%-6s %-11s %9.3f %8.3f %8.3f %8.2f %9.2fx %9d"
                  % (l, FAMILY.get(l, "?"), tpw, tpg, tpb, tps, rel, d["tokens"]))
            rows.append(dict(tokenizer=label, spec=spec, lang=l,
                             family=FAMILY.get(l, "?"),
                             tokens=d["tokens"], words=d["words"],
                             graphemes=d["graphemes"], bytes=d["bytes"],
                             sents=d["sents"],
                             tok_per_word=round(tpw, 4),
                             tok_per_grapheme=round(tpg, 4),
                             tok_per_byte=round(tpb, 4),
                             tok_per_sent=round(tps, 3),
                             ratio_vs_pivot_per_sent=round(rel, 4),
                             ratio_vs_pivot_per_word=round(tpw / (p["tokens"] / p["words"]), 4)))
        print()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        csv_path = args.out + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print("wrote %s (%d rows)" % (csv_path, len(rows)))

    # ---- the headline comparison: same content, different tokenizers --------
    if len(rows) > len(langs):
        print("=" * 100)
        print("COST MULTIPLIER vs %s for IDENTICAL CONTENT (tokens_lang / tokens_%s)"
              % (pivot, pivot))
        print("=" * 100)
        labels = []
        for r in rows:
            if r["tokenizer"] not in labels:
                labels.append(r["tokenizer"])
        hdr = "%-6s " % "lang" + "".join("%12s" % l for l in labels)
        print(hdr)
        print("-" * len(hdr))
        for l in langs:
            cells = ""
            for lab in labels:
                m = [r for r in rows if r["tokenizer"] == lab and r["lang"] == l]
                cells += "%11.2fx" % m[0]["ratio_vs_pivot_per_sent"] if m else "%12s" % "-"
            print("%-6s " % l + cells)
        print()
        print("Read this table as: 'one %s sentence of content costs Nx as many"
              % pivot)
        print("tokens when written in this language and encoded by this tokenizer.'")


if __name__ == "__main__":
    main()
