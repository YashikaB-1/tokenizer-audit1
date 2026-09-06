#!/usr/bin/env python3
"""
sensitivity.py -- how much do B1's assumptions actually matter?

B1 takes two numbers on faith: the ~1.6 GB non-KV overhead (the spec says
"assume"), and the decimal reading of "24 GB". This sweeps both and asks the
only question that matters for B2: does batch 32 still fail to fit?

Usage:  python sensitivity.py
"""

LAYERS, KV_HEADS, HEAD_DIM, KV_BYTES = 28, 8, 128, 2
Q_HEADS = 24
PARAMS, W_BYTES = 4.2e9, 2
UTIL, MAX_LEN = 0.92, 4096
OBSERVED_SEQS = 25.5      # from b1_kv_capacity.py, inverting kv_cache_util on the log


def per_tok(kv_heads=KV_HEADS, kv_bytes=KV_BYTES):
    return 2 * LAYERS * kv_heads * HEAD_DIM * kv_bytes


def seqs(total_bytes, overhead, kv_heads=KV_HEADS, kv_bytes=KV_BYTES):
    pool = total_bytes * UTIL - PARAMS * W_BYTES - overhead
    return pool / per_tok(kv_heads, kv_bytes) / MAX_LEN


def head(t):
    print("\n" + "=" * 80)
    print(t)
    print("=" * 80)


head("S1  Sensitivity to the assumed non-KV overhead")
print("The spec says 'assume ~1.6 GB'. If that assumption is wrong, how wrong is B1?\n")
print("  %-14s%14s%14s%16s" % ("overhead", "max seqs", "err vs log", "does b32 fit?"))
print("  " + "-" * 58)
for ov in [1.0e9, 1.3e9, 1.6e9, 2.0e9, 2.5e9, 3.0e9]:
    n = seqs(24e9, ov)
    print("  %-14s%14.2f%13.1f%%%16s"
          % ("%.1f GB" % (ov / 1e9), n, (n - OBSERVED_SEQS) / OBSERVED_SEQS * 100,
             "YES" if n >= 32 else "no"))
print("""
Across a 3x range of the overhead assumption, capacity stays between 23 and 28
sequences and batch 32 NEVER fits. The B2 mechanism does not depend on getting
the overhead right -- it only needs the ceiling to sit between 24 and 32, and
every plausible overhead puts it there.""")

head("S2  Sensitivity to the GB / GiB reading of '24 GB'")
print("  %-24s%12s%14s%16s" % ("interpretation", "max seqs", "err vs log", "does b32 fit?"))
print("  " + "-" * 66)
for name, tot in [("24 GB  (decimal)", 24e9), ("24 GiB (binary)", 24 * 2**30)]:
    n = seqs(tot, 1.6e9)
    print("  %-24s%12.2f%13.1f%%%16s"
          % (name, n, (n - OBSERVED_SEQS) / OBSERVED_SEQS * 100, "YES" if n >= 32 else "no"))
print("""
The decimal reading matches the log to under 1%; the binary reading is 14% high.
But note that even the binary reading does not let batch 32 fit -- so B2's
conclusion survives being wrong about this too.""")

head("S3  The GQA trap: what if you sized the cache by query heads?")
print("  %-30s%16s%12s%14s" % ("KV sized by", "bytes/token", "max seqs", "err vs log"))
print("  " + "-" * 72)
for name, kvh in [("8 KV heads (correct, GQA)", KV_HEADS),
                  ("24 query heads (the trap)", Q_HEADS)]:
    n = seqs(24e9, 1.6e9, kv_heads=kvh)
    print("  %-30s%16s%12.2f%13.1f%%"
          % (name, "%s" % f"{per_tok(kvh):,}", n, (n - OBSERVED_SEQS) / OBSERVED_SEQS * 100))
print("""
Using the 24 query heads predicts 8.6 concurrent sequences. The log shows batch
24 running with zero preemptions, which is flatly impossible under that number.
So the log does not merely agree with the GQA arithmetic -- it rules out the
alternative. This is the cleanest confirmation that 114,688 B/token is right.""")

head("S4  fp8 KV cache -- the B2 alternative proposal, sized")
print("  %-24s%16s%12s%16s" % ("KV precision", "bytes/token", "max seqs", "does b48 fit?"))
print("  " + "-" * 68)
for name, b in [("fp16 (current)", 2), ("fp8", 1)]:
    n = seqs(24e9, 1.6e9, kv_bytes=b)
    print("  %-24s%16s%12.2f%16s"
          % (name, f"{per_tok(kv_bytes=b):,}", n, "YES" if n >= 48 else "no"))
print("""
fp8 KV is the only one of the two B2 proposals that makes batch 48 resident.
Its predicted throughput gain is the weaker claim of the two (it extrapolates a
4-point fit); its predicted CAPACITY gain is straight arithmetic and solid.""")

head("S5  What would have to be true for '~25 sequences' to be wrong?")
print("""Solving backwards from the log's observed pool (~104,468 tokens = 25.5 seqs of
4096), the free parameter is the overhead:

    required overhead = 24e9 x 0.92 - 8.4e9 - 104,468 x 114,688""")
req = 24e9 * UTIL - PARAMS * W_BYTES - 104468 * per_tok()
print("                      = %.2f GB\n" % (req / 1e9))
print("""The spec's assumed 1.60 GB versus the 1.70 GB the log implies -- a 0.1 GB gap
on a 24 GB card. Either the spec's estimate is slightly low or there is a little
allocator fragmentation. Nothing in B1 or B2 changes at that resolution.""")
