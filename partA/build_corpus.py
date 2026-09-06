#!/usr/bin/env python3
"""
build_corpus.py -- assemble the A1 evaluation corpus from FLORES-200.

Downloads the official FLORES-200 release, verifies its SHA-256, and writes
one plain-text file per language into partA/corpus/, plus a manifest.

The FLORES-200 `dev` split is 997 sentences that are *parallel*: line i of
every language file is a translation of the same source sentence. That
alignment is the whole point -- it is the only thing that lets us build a
denominator which holds meaning constant across languages (see A3).

Usage:
    python build_corpus.py                 # full 997-sentence dev split
    python build_corpus.py --n 200         # deterministic 200-sentence subset
    python build_corpus.py --cache-dir DIR # reuse an already-downloaded tarball
"""

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"

# code -> FLORES-200 file stem.  Script tags are part of the FLORES naming.
LANGS = {
    "eng": "eng_Latn",   # English            (Germanic, Latin script)   -- pivot
    "hin": "hin_Deva",   # Hindi              (Indo-Aryan, Devanagari)
    "ben": "ben_Beng",   # Bengali            (Indo-Aryan, Bengali)
    "mar": "mar_Deva",   # Marathi            (Indo-Aryan, Devanagari)
    "kan": "kan_Knda",   # Kannada            (DRAVIDIAN, Kannada)
    "tam": "tam_Taml",   # Tamil              (DRAVIDIAN, Tamil)
    "tel": "tel_Telu",   # Telugu             (DRAVIDIAN, Telugu)
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "corpus")


def fetch(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    tgz = os.path.join(cache_dir, "flores200.tar.gz")
    if not os.path.exists(tgz):
        print(f"downloading {FLORES_URL} ...", file=sys.stderr)
        urllib.request.urlretrieve(FLORES_URL, tgz)
    digest = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
    if digest != FLORES_SHA256:
        raise SystemExit(f"SHA-256 mismatch!\n  expected {FLORES_SHA256}\n  got      {digest}")
    print(f"sha256 ok: {digest}", file=sys.stderr)
    root = os.path.join(cache_dir, "flores200_dataset")
    if not os.path.isdir(root):
        with tarfile.open(tgz) as tf:
            tf.extractall(cache_dir)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "devtest"])
    ap.add_argument("--n", type=int, default=0,
                    help="keep only the first N sentences (0 = all). Deterministic: "
                         "a prefix, not a random sample, so the set stays parallel and "
                         "re-derivable without a seed.")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, ".flores_cache"))
    ap.add_argument("--out-dir", default=None,
                    help="where to write <lang>.txt. Default: partA/corpus. Point this "
                         "at a separate directory for the held-out devtest split so the "
                         "primary corpus is never clobbered.")
    args = ap.parse_args()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = args.out_dir
    root = fetch(args.cache_dir)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Read every language first so we can assert alignment before writing anything.
    texts = {}
    for code, stem in LANGS.items():
        path = os.path.join(root, args.split, f"{stem}.{args.split}")
        with open(path, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        while lines and not lines[-1].strip():
            lines.pop()
        texts[code] = lines

    counts = {c: len(v) for c, v in texts.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"FLORES files are not line-aligned: {counts}")
    n_total = next(iter(counts.values()))
    n_keep = n_total if args.n <= 0 else min(args.n, n_total)

    # A blank line in any language would silently break alignment downstream,
    # because the v0 reader drops blank lines per-file. Drop such rows from
    # *every* language together so line i keeps meaning the same thing.
    keep_idx = [i for i in range(n_keep)
                if all(texts[c][i].strip() for c in LANGS)]
    dropped = n_keep - len(keep_idx)

    for code in LANGS:
        out = os.path.join(OUT_DIR, f"{code}.txt")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for i in keep_idx:
                f.write(texts[code][i] + "\n")

    # Carry the per-sentence metadata across so domain claims in CORPUS.md are checkable.
    meta_src = os.path.join(root, f"metadata_{args.split}.tsv")
    with open(meta_src, encoding="utf-8") as f:
        meta = [ln.rstrip("\n") for ln in f]
    with open(os.path.join(OUT_DIR, "metadata.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write(meta[0] + "\n")
        for i in keep_idx:
            f.write(meta[1 + i] + "\n")

    with open(os.path.join(OUT_DIR, "MANIFEST.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source           FLORES-200 ({args.split} split)\n")
        f.write(f"url              {FLORES_URL}\n")
        f.write(f"sha256           {FLORES_SHA256}\n")
        f.write(f"sentences        {len(keep_idx)} (of {n_total} in split; "
                f"{dropped} dropped as blank in >=1 language)\n")
        f.write(f"languages        {', '.join(LANGS)}\n")
        f.write("alignment        line i is the same sentence in every file\n")
        f.write("preprocessing    none beyond blank-line removal; NO lowercasing, "
                "NO unicode normalisation, NO punctuation stripping\n")

    print(f"wrote {len(keep_idx)} parallel sentences x {len(LANGS)} languages -> {OUT_DIR}")
    if dropped:
        print(f"  ({dropped} rows dropped: blank in at least one language)")


if __name__ == "__main__":
    main()
