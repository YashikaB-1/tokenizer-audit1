#!/usr/bin/env python3
"""
b1_kv_capacity.py -- B1: KV-cache bytes per token from the model spec alone,
the implied concurrency limit, and reconciliation against bench_log.csv.

Everything here is derived from bench/model_spec.md. No fitted constants.

Usage:  python b1_kv_capacity.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.normpath(os.path.join(HERE, "..", "starter_kit_original", "bench", "bench_log.csv"))

# ---- straight from bench/model_spec.md -------------------------------------
LAYERS = 28
KV_HEADS = 8            # GQA: KV heads, NOT the 24 query heads
HEAD_DIM = 128
KV_DTYPE_BYTES = 2      # fp16
PARAMS = 4.2e9
W_DTYPE_BYTES = 2       # fp16
GPU_GB = 24             # 1x L4
UTIL = 0.92             # gpu_memory_utilization
OVERHEAD_BYTES = 1.6e9  # "non-KV runtime overhead ... assume ~1.6 GB"
MAX_MODEL_LEN = 4096


def rule(c="-", n=78):
    print(c * n)


def main():
    print("=" * 78)
    print("B1(a)  KV-cache bytes per token")
    print("=" * 78)
    print("""One cached token needs a K vector and a V vector in every layer.
With grouped-query attention the cache is sized by the KV head count (8),
not the query head count (24) -- that is the whole point of GQA.

  bytes/token = 2 (K and V)
              x layers        (%d)
              x kv_heads      (%d)     <- GQA: 8, not 24
              x head_dim      (%d)
              x bytes/elem    (%d)     <- fp16 KV cache
""" % (LAYERS, KV_HEADS, HEAD_DIM, KV_DTYPE_BYTES))

    per_tok = 2 * LAYERS * KV_HEADS * HEAD_DIM * KV_DTYPE_BYTES
    print("  = 2 x %d x %d x %d x %d" % (LAYERS, KV_HEADS, HEAD_DIM, KV_DTYPE_BYTES))
    print("  = %s bytes/token  =  %.0f KiB/token" % (f"{per_tok:,}", per_tok / 1024))
    print()
    print("  Sanity check -- if someone wrongly used the 24 query heads:")
    print("    2 x %d x 24 x %d x %d = %s bytes/token (3x too big)"
          % (LAYERS, HEAD_DIM, KV_DTYPE_BYTES,
             f"{2 * LAYERS * 24 * HEAD_DIM * KV_DTYPE_BYTES:,}"))

    per_seq = per_tok * MAX_MODEL_LEN
    print()
    print("  One full %d-token sequence: %s x %d = %s bytes = %.1f MiB = %.3f GB"
          % (MAX_MODEL_LEN, f"{per_tok:,}", MAX_MODEL_LEN, f"{per_seq:,}",
             per_seq / 2**20, per_seq / 1e9))

    print()
    print("=" * 78)
    print("B1(b)  How many concurrent %d-token sequences fit" % MAX_MODEL_LEN)
    print("=" * 78)
    weights = PARAMS * W_DTYPE_BYTES
    print("  weights        = %.1e params x %d B = %.2f GB"
          % (PARAMS, W_DTYPE_BYTES, weights / 1e9))
    print("  overhead       = %.2f GB (given)" % (OVERHEAD_BYTES / 1e9))
    print()
    print("  The spec says '24 GB' and vLLM's gpu_memory_utilization applies to")
    print("  total device memory, so the GB-vs-GiB reading of '24' matters. Both:")
    print()

    results = {}
    hdr = "  %-26s%14s%14s%12s%12s" % ("interpretation", "usable", "KV pool", "max tokens", "max seqs")
    print(hdr)
    rule(" ", 0)
    print("  " + "-" * (len(hdr) - 2))
    for name, total in [("24 GB  (decimal, 24e9)", 24e9),
                        ("24 GiB (binary, 2^30)", 24 * 2**30)]:
        usable = total * UTIL
        pool = usable - weights - OVERHEAD_BYTES
        toks = pool / per_tok
        seqs = toks / MAX_MODEL_LEN
        results[name] = (pool, toks, seqs)
        print("  %-26s%11.2f GB%11.2f GB%12s%12.2f"
              % (name, usable / 1e9, pool / 1e9, f"{toks:,.0f}", seqs))

    print()
    print("  Arithmetic for the decimal reading, in full:")
    usable = 24e9 * UTIL
    pool = usable - weights - OVERHEAD_BYTES
    print("    24.00 GB x 0.92                 = %.2f GB usable" % (usable / 1e9))
    print("    %.2f - %.2f (weights) - %.2f (overhead) = %.2f GB for KV"
          % (usable / 1e9, weights / 1e9, OVERHEAD_BYTES / 1e9, pool / 1e9))
    print("    %.2f GB / %s B per token       = %s tokens"
          % (pool / 1e9, f"{per_tok:,}", f"{pool/per_tok:,.0f}"))
    print("    %s tokens / %d                = %.1f sequences  -> floor %d"
          % (f"{pool/per_tok:,.0f}", MAX_MODEL_LEN, pool / per_tok / MAX_MODEL_LEN,
             int(pool / per_tok / MAX_MODEL_LEN)))

    # ---- reconcile against the log -----------------------------------------
    print()
    print("=" * 78)
    print("B1(c)  Check the prediction against bench_log.csv")
    print("=" * 78)
    print("kv_cache_util is the fraction of the KV pool in use. Inverting it on")
    print("each row gives an OBSERVED pool size, in tokens and in 4096-tok seqs.")
    print("kv_cache_util is logged to 2 decimals, so rows below ~0.05 are dominated")
    print("by rounding (at util=0.01 one ulp is +-50%) and are excluded from the mean.\n")

    rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
    hdr = ("  %-6s%7s%6s%9s%9s%13s%15s%14s"
           % ("batch", "plen", "glen", "tok/seq", "kv_util", "live tokens",
              "implied pool", "implied seqs"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    implied = []
    for r in rows:
        b = int(r["batch_size"]); pl = int(r["prompt_len"]); gl = int(r["gen_len"])
        u = float(r["kv_cache_util"]); pre = int(r["preempted_seqs"])
        live = b * (pl + gl)
        if u <= 0:
            continue
        pool_tok = live / u
        note = ""
        if pre > 0 or u >= 0.97:
            note = "  <- saturated / preempting, excluded"
        elif u < 0.05:
            note = "  <- util quantisation, excluded"
        else:
            implied.append(pool_tok)
        print("  %-6d%7d%6d%9d%9.2f%13s%15s%14.1f%s"
              % (b, pl, gl, pl + gl, u, f"{live:,}", f"{pool_tok:,.0f}",
                 pool_tok / MAX_MODEL_LEN, note))

    obs = sum(implied) / len(implied)
    pred_tok = pool / per_tok
    print()
    print("  observed pool (mean of %d usable rows)     : %s tokens = %.1f seqs of %d"
          % (len(implied), f"{obs:,.0f}", obs / MAX_MODEL_LEN, MAX_MODEL_LEN))
    print("  predicted pool (decimal 24 GB)           : %s tokens = %.1f seqs"
          % (f"{pred_tok:,.0f}", pred_tok / MAX_MODEL_LEN))
    print("  error                                    : %+.2f%%"
          % ((pred_tok - obs) / obs * 100))
    print()
    pb = results["24 GiB (binary, 2^30)"][1]
    print("  (the 24 GiB reading predicts %s tokens = %.1f seqs, %+.1f%% -- ruled out)"
          % (f"{pb:,.0f}", pb / MAX_MODEL_LEN, (pb - obs) / obs * 100))
    print()
    print("  CONCLUSION: ~%d concurrent 4096-token sequences. The log agrees:"
          % int(obs / MAX_MODEL_LEN))
    print("  batch 24 sits at 0.93 util with 0 preemptions; batch 32 (which would")
    print("  need %d tokens, %.0f%% of the pool) pins at 0.97 and preempts 7."
          % (32 * MAX_MODEL_LEN, 32 * MAX_MODEL_LEN / obs * 100))


if __name__ == "__main__":
    main()
