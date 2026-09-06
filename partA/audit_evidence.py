#!/usr/bin/env python3
"""
audit_evidence.py -- the A2 evidence harness.

Every claim in A2_AUDIT.md is produced by this file. The method is ablation:
start from a faithful re-implementation of fertility.py's `analyze()`, flip
exactly ONE behaviour, and report the change in the headline numbers.

Anything that changes the numbers by 0.00 is reported as harmless, not as a bug.

Usage:
    python audit_evidence.py                       # sample corpus (reproduces REPORT_v0)
    python audit_evidence.py --corpus-dir corpus   # the real A1 corpus
    python audit_evidence.py --tokenizer hf:xlm-roberta-base
"""

import argparse
import os
import sys
import unicodedata

import regex  # for grapheme clusters (\X)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- tokenizers
def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        import warnings
        warnings.filterwarnings("ignore")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    import tiktoken
    return tiktoken.get_encoding(spec).encode


# ---------------------------------------------------------------- corpus
def read_lines(path, nfc=True):
    """v0's reader. `nfc` toggles the normalisation step so we can price it."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if nfc:
                line = unicodedata.normalize("NFC", line)
            out.append(line)
    return out


# ---------------------------------------------------------------- the metric
def analyze(lines, encode, *, lower, split_naive, macro, nfc_already=True,
            char_unit="codepoint"):
    """
    v0 defaults are: lower=True, split_naive=True, macro=True, char_unit='codepoint'.

    macro=True  -> mean of per-line (tokens/words)      <- what v0 does
    macro=False -> (sum tokens) / (sum words)           <- corpus-level ratio
    """
    per_f, per_c = [], []
    tot_tok = tot_word = tot_char = 0
    for line in lines:
        if lower:
            line = line.lower()
        tokens = encode(line)
        words = line.split(" ") if split_naive else line.split()
        if char_unit == "codepoint":
            chars = len(line)
        elif char_unit == "grapheme":
            chars = len(regex.findall(r"\X", line))
        elif char_unit == "byte":
            chars = len(line.encode("utf-8"))
        else:
            raise ValueError(char_unit)
        tot_tok += len(tokens)
        tot_word += len(words)
        tot_char += chars
        if words:
            per_f.append(len(tokens) / len(words))
        if chars:
            per_c.append(len(tokens) / chars)
    if macro:
        return sum(per_f) / len(per_f), sum(per_c) / len(per_c)
    return tot_tok / tot_word, tot_tok / tot_char


# ---------------------------------------------------------------- driver
def run(corpora, encode, **kw):
    """corpora: dict lang -> list[str]. Returns dict lang -> (fertility, tok_per_char)."""
    return {lang: analyze(lines, encode, **kw) for lang, lines in corpora.items()}


def fmt_row(label, res, base_lang, cmp_lang, ref=None):
    f_b, c_b = res[base_lang]
    f_c, c_c = res[cmp_lang]
    ratio = f_c / f_b
    s = (f"{label:<44}{f_b:>8.3f}{f_c:>9.3f}{ratio:>9.2f}x{c_b:>9.3f}{c_c:>9.3f}")
    if ref is not None:
        s += f"{ratio - ref:>+9.2f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=None,
                    help="directory of <lang>.txt files. Default: the v0 sample corpus.")
    ap.add_argument("--langs", default=None,
                    help="comma list, first is the baseline. Default eng,hin")
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    if args.corpus_dir:
        cdir = args.corpus_dir
        langs = (args.langs or "eng,hin").split(",")
        paths = {l: os.path.join(cdir, f"{l}.txt") for l in langs}
    else:
        cdir = os.path.join(HERE, "..", "starter_kit_original", "corpus_sample")
        langs = (args.langs or "eng,hin").split(",")
        paths = {l: os.path.join(cdir, f"{l}_sample.txt") for l in langs}

    base, cmp_ = langs[0], langs[1]
    encode = load_tokenizer(args.tokenizer)

    corp_nfc = {l: read_lines(p, nfc=True) for l, p in paths.items()}
    corp_raw = {l: read_lines(p, nfc=False) for l, p in paths.items()}

    print(f"tokenizer : {args.tokenizer}")
    print(f"corpus    : {os.path.normpath(cdir)}  "
          f"({', '.join(f'{l}={len(v)} lines' for l, v in corp_nfc.items())})")
    print()
    hdr = (f"{'configuration':<44}{base+' fert':>8}{cmp_+' fert':>9}{'ratio':>10}"
           f"{base+' t/c':>9}{cmp_+' t/c':>9}{'d(ratio)':>9}")
    print(hdr)
    print("-" * len(hdr))

    V0 = dict(lower=True, split_naive=True, macro=True, char_unit="codepoint")
    base_res = run(corp_nfc, encode, **V0)
    ref = base_res[cmp_][0] / base_res[base][0]
    print(fmt_row("V0 (as shipped, reproduces REPORT_v0)", base_res, base, cmp_))
    print()

    def ablate(label, **over):
        cfg = dict(V0); cfg.update(over)
        res = run(corp_nfc, encode, **cfg)
        print(fmt_row(label, res, base, cmp_, ref))
        return res

    print("--- single-flag ablations (one change from V0) " + "-" * 30)
    ablate("F1  split() instead of split(' ')", split_naive=False)
    ablate("F2  no .lower()", lower=False)
    ablate("F3  corpus-level ratio (micro avg)", macro=False)
    ablate("F4a chars = grapheme clusters", char_unit="grapheme")
    ablate("F4b chars = UTF-8 bytes", char_unit="byte")
    print()

    print("--- decoys: things that LOOK wrong " + "-" * 42)
    res_raw = run(corp_raw, encode, **V0)
    print(fmt_row("D1  drop NFC normalisation", res_raw, base, cmp_, ref))
    n_changed = {l: sum(1 for a, b in zip(corp_raw[l], corp_nfc[l]) if a != b)
                 for l in langs}
    print(f"    -> lines altered by NFC: {n_changed}")
    print()

    print("--- all fixes together " + "-" * 54)
    fixed = ablate("F1+F2+F3 (corrected tok/word)",
                   split_naive=False, lower=False, macro=False)
    print()
    print(f"V0 headline ratio       : {ref:.2f}x")
    print(f"Corrected tok/word ratio: {fixed[cmp_][0] / fixed[base][0]:.2f}x")


if __name__ == "__main__":
    main()
