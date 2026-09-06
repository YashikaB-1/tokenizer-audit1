#!/usr/bin/env python3
"""
probe.py -- tokenize arbitrary text across every tokenizer in the study.

Built for the live defense: someone pastes a sentence and asks what happens to
it. This shows the counts, the per-denominator rates, and (with -v) the actual
token pieces, so "GPT-2 degrades to near-byte-level on Devanagari" stops being
an assertion and becomes something visible on screen.

Usage:
    python probe.py "मुझे सुबह की चाय बहुत पसंद है।"
    python probe.py -v "ಬೆಂಗಳೂರು"                    # show the token pieces
    python probe.py --file some.txt                   # whole file, aggregated
    echo "..." | python probe.py                      # stdin
    python probe.py --pair "I want tea" "मुझे चाय चाहिए"   # aligned pair -> multiplier
"""

import argparse
import sys
import unicodedata

import regex

# Windows consoles default to cp1252, which cannot encode Devanagari/Kannada/Tamil
# and would crash this script mid-demo. Force UTF-8 on stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TOKS = [("gpt2", "gpt2"),
        ("qwen2.5", "hf:Qwen/Qwen2.5-0.5B"),
        ("xlm-r", "hf:xlm-roberta-base"),
        ("bloom", "hf:bigscience/bloom-560m"),
        ("sarvam-1", "hf:sarvamai/sarvam-1")]

_cache = {}


def load(spec):
    if spec in _cache:
        return _cache[spec]
    if spec.startswith("hf:"):
        import warnings
        warnings.filterwarnings("ignore")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        fns = (lambda s: tok.encode(s, add_special_tokens=False),
               lambda ids: [tok.decode([i]) for i in ids])
    else:
        import tiktoken
        enc = tiktoken.get_encoding(spec)
        fns = (enc.encode,
               lambda ids: [enc.decode([i]) for i in ids])
    _cache[spec] = fns
    return fns


def counts(text):
    return (len(text.split()),
            len(regex.findall(r"\X", text)),
            len(text.encode("utf-8")))


def report(text, verbose, label=None):
    text = unicodedata.normalize("NFC", text)
    w, g, b = counts(text)
    if label:
        print("\n### %s" % label)
    preview = text if len(text) <= 70 else text[:67] + "..."
    print('text      : %s' % preview)
    print("units     : %d words | %d grapheme clusters | %d UTF-8 bytes" % (w, g, b))
    print()
    hdr = "  %-10s%9s%11s%12s%11s" % ("tokenizer", "tokens", "tok/word", "tok/graph", "tok/byte")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    out = {}
    for name, spec in TOKS:
        try:
            enc, dec = load(spec)
        except Exception as e:
            print("  %-10s SKIP (%s)" % (name, type(e).__name__))
            continue
        ids = enc(text)
        out[name] = len(ids)
        tpb = len(ids) / b if b else 0
        flag = "   <- at byte level" if tpb >= 0.98 else ""
        print("  %-10s%9d%11.3f%11.3f%12.3f%s"
              % (name, len(ids), len(ids) / w if w else 0,
                 len(ids) / g if g else 0, tpb, flag))
        if verbose:
            pieces = dec(ids)
            shown = pieces[:40]
            print("             %s%s"
                  % (" ".join("[%s]" % p.replace("\n", "\\n") for p in shown),
                     " ... (+%d)" % (len(pieces) - 40) if len(pieces) > 40 else ""))
    if out:
        best = min(out, key=out.get)
        worst = max(out, key=out.get)
        if out[worst] == out[best]:
            print("\n  all tokenizers agree at %d tokens (no spread on this input)"
                  % out[best])
        else:
            print("\n  best %s (%d tokens) vs worst %s (%d) = %.2fx spread on identical text"
                  % (best, out[best], worst, out[worst], out[worst] / out[best]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="text to tokenize (or use --file / stdin)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show token pieces")
    ap.add_argument("--file", help="read text from a file instead")
    ap.add_argument("--pair", nargs=2, metavar=("ENG", "OTHER"),
                    help="two translations of the same thing -> cost multiplier")
    args = ap.parse_args()

    if args.pair:
        a = report(args.pair[0], args.verbose, label="A (pivot)")
        b = report(args.pair[1], args.verbose, label="B")
        print("\n" + "=" * 62)
        print("COST MULTIPLIER for identical content (B tokens / A tokens)")
        print("=" * 62)
        print("  %-12s%10s%10s%12s" % ("tokenizer", "A", "B", "B/A"))
        print("  " + "-" * 42)
        for name, _ in TOKS:
            if name in a and name in b and a[name]:
                print("  %-12s%10d%10d%11.2fx" % (name, a[name], b[name], b[name] / a[name]))
        print("\n  This is the A3 metric on a single sentence pair. It is the same")
        print("  computation fertility_v1.py does over 997 of them.")
        return

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = " ".join(args.text)
    else:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        sys.exit("no input")
    report(text, args.verbose)


if __name__ == "__main__":
    main()
