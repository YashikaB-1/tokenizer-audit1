#!/usr/bin/env python3
"""
b2b3_goodput.py -- B2 (throughput anomaly + mechanism) and B3 (the misread
column, and the honest goodput of the batch-24 long-prompt row).

Usage:  python b2b3_goodput.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.normpath(os.path.join(HERE, "..", "starter_kit_original", "bench", "bench_log.csv"))
KV_POOL_TOKENS = 105329      # from b1_kv_capacity.py, model-spec arithmetic
MAX_MODEL_LEN = 4096


def load():
    return [{k: (float(v) if "." in v else int(v)) for k, v in r.items()}
            for r in csv.DictReader(open(LOG, encoding="utf-8"))]


def head(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


# --------------------------------------------------------------------- B3(i)
def what_is_reported_tok_s(rows):
    head("B3(i)  What `reported_tok_s` actually counts")
    print("Hypothesis: the harness counter divides ALL tokens it touched -- prompt")
    print("tokens included -- by wall clock. Prompt tokens are processed in a")
    print("parallel prefill and are nearly free per token; generated tokens are not.")
    print("Test both candidate formulas against every row.\n")

    hdr = ("  %-6s%7s%6s%12s%13s%9s%13s%9s"
           % ("batch", "plen", "glen", "reported", "(p+g)*n/wall", "err%",
              "g*n/wall", "err%"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    e_tot, e_gen = [], []
    for r in rows:
        n, pl, gl, w = r["num_requests"], r["prompt_len"], r["gen_len"], r["wall_clock_s"]
        tot = (pl + gl) * n / w
        gen = gl * n / w
        a = (tot - r["reported_tok_s"]) / r["reported_tok_s"] * 100
        b = (gen - r["reported_tok_s"]) / r["reported_tok_s"] * 100
        e_tot.append(abs(a)); e_gen.append(abs(b))
        print("  %-6d%7d%6d%12.1f%13.1f%+9.2f%13.1f%+9.1f"
              % (r["batch_size"], pl, gl, r["reported_tok_s"], tot, a, gen, b))
    print()
    print("  mean |error| vs (prompt+gen)/wall : %.2f%%   <-- this is the formula"
          % (sum(e_tot) / len(e_tot)))
    print("  mean |error| vs (gen only)/wall   : %.1f%%" % (sum(e_gen) / len(e_gen)))
    print()
    print("  CONFIRMED: reported_tok_s = (prompt_len + gen_len) * num_requests / wall.")
    print("  It is a *total tokens processed* counter, not a generation-rate counter.")
    print("  On the 3584/512 rows, %.0f%% of that number is prompt tokens."
          % (3584 / 4096 * 100))


# -------------------------------------------------------------------- B3(ii)
def goodput_table(rows):
    head("B3(ii)  Goodput -- generated tokens per second -- for every row")
    print("Two independent derivations of the batch-24 long-prompt row are shown")
    print("below the table; they must agree, and they do.\n")

    hdr = ("  %-6s%7s%6s%11s%11s%10s%13s%9s"
           % ("batch", "plen", "glen", "reported", "goodput", "prompt%",
              "itl-implied", "preempt"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        n, pl, gl, w = r["num_requests"], r["prompt_len"], r["gen_len"], r["wall_clock_s"]
        good = gl * n / w
        itl_imp = n / (r["itl_ms_p50"] / 1000.0)
        mark = "  *" if (r["batch_size"] == 24 and pl == 3584) else ""
        print("  %-6d%7d%6d%11.1f%11.1f%9.0f%%%13.1f%9d%s"
              % (r["batch_size"], pl, gl, r["reported_tok_s"], good,
                 pl / (pl + gl) * 100, itl_imp, r["preempted_seqs"], mark))

    r24 = [r for r in rows if r["batch_size"] == 24 and r["prompt_len"] == 3584][0]
    n, pl, gl, w = r24["num_requests"], r24["prompt_len"], r24["gen_len"], r24["wall_clock_s"]
    w1 = gl * n / w
    w2 = r24["reported_tok_s"] * gl / (pl + gl)
    w3 = n / (r24["itl_ms_p50"] / 1000.0)

    print("\n  Batch-24 long-prompt row (*), honest goodput -- two independent routes:")
    print("    route 1, from the raw counts:")
    print("       %d requests x %d generated tokens / %.2f s = %.1f tok/s"
          % (n, gl, w, w1))
    print("    route 2, strip the prompt fraction out of the harness counter:")
    print("       %.1f tok/s x %d/(%d+%d) = %.1f tok/s"
          % (r24["reported_tok_s"], gl, pl, gl, w2))
    print("    agreement: %.4f%% apart\n" % (abs(w1 - w2) / w1 * 100))
    print("    cross-check from the decode-rate column (not independent of wall clock,")
    print("    but a useful upper bound):")
    print("       %d concurrent / %.2f ms per token = %.1f tok/s steady-state decode"
          % (n, r24["itl_ms_p50"], w3))
    print("       %.1f / %.1f = %.2f  ->  only %.0f%% of wall clock is spent decoding;"
          % (w1, w3, w1 / w3, w1 / w3 * 100))
    print("       the other %.0f%% is prefill (ttft %.0f ms) and scheduling."
          % (100 - w1 / w3 * 100, r24["ttft_ms_p50"]))

    short16 = [r for r in rows if r["batch_size"] == 16 and r["prompt_len"] == 512][0]
    long16 = [r for r in rows if r["batch_size"] == 16 and r["prompt_len"] == 3584][0]
    g_s = short16["gen_len"] * short16["num_requests"] / short16["wall_clock_s"]
    g_l = long16["gen_len"] * long16["num_requests"] / long16["wall_clock_s"]
    print("\n  The comparison REPORT_v0 actually made, redone on goodput (batch 16):")
    print("    short prompts (512) : reported %.1f -> goodput %.1f tok/s"
          % (short16["reported_tok_s"], g_s))
    print("    long  prompts (3584): reported %.1f -> goodput %.1f tok/s"
          % (long16["reported_tok_s"], g_l))
    print("    reported_tok_s says long is %.2fx BETTER."
          % (long16["reported_tok_s"] / short16["reported_tok_s"]))
    print("    goodput says long is %.2fx WORSE (%.0f%% less useful output)."
          % (g_s / g_l, (1 - g_l / g_s) * 100))
    print("    The sign of the conclusion flips.")

    best_rep = max(rows, key=lambda r: r["reported_tok_s"])
    best_good = max(rows, key=lambda r: r["gen_len"] * r["num_requests"] / r["wall_clock_s"])
    print("\n  'best observed ~1600 tok/s' in REPORT_v0 is also not the max of any column:")
    print("    max reported_tok_s : %.1f (batch %d, prompt %d)"
          % (best_rep["reported_tok_s"], best_rep["batch_size"], best_rep["prompt_len"]))
    print("    max goodput        : %.1f (batch %d, prompt %d)"
          % (best_good["gen_len"] * best_good["num_requests"] / best_good["wall_clock_s"],
             best_good["batch_size"], best_good["prompt_len"]))

    r48 = [r for r in rows if r["batch_size"] == 48][0]
    print("\n  The batch-48 prediction, checked against the row that already exists:")
    print("    REPORT_v0 predicts   ~3200 tok/s at batch 48")
    print("    log row batch 48     %.1f tok/s reported (%.0f%% of the prediction)"
          % (r48["reported_tok_s"], r48["reported_tok_s"] / 3200 * 100))
    print("    log row batch 48     %.1f tok/s goodput  (%.0f%% of the prediction)"
          % (r48["gen_len"] * r48["num_requests"] / r48["wall_clock_s"],
             (r48["gen_len"] * r48["num_requests"] / r48["wall_clock_s"]) / 3200 * 100))
    print("    The batch-48 run was already IN the log. It did not need extrapolating.")


# ----------------------------------------------------------------------- B2
def anomaly(rows):
    head("B2  The long-context (prompt 3584) throughput anomaly")
    lc = [r for r in rows if r["prompt_len"] == 3584]

    print("Naive expectation: throughput rises monotonically with batch until the")
    print("GPU is compute-bound. Observed:\n")
    hdr = ("  %-6s%11s%10s%11s%10s%10s%10s%9s"
           % ("batch", "reported", "goodput", "wall_s", "kv_util", "preempt",
              "ttft_ms", "itl_ms"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    prev = None
    for r in lc:
        good = r["gen_len"] * r["num_requests"] / r["wall_clock_s"]
        d = "" if prev is None else "  %+.1f%%" % ((r["reported_tok_s"] - prev) / prev * 100)
        print("  %-6d%11.1f%10.1f%11.2f%10.2f%10d%10.1f%9.2f%s"
              % (r["batch_size"], r["reported_tok_s"], good, r["wall_clock_s"],
                 r["kv_cache_util"], r["preempted_seqs"], r["ttft_ms_p50"],
                 r["itl_ms_p50"], d))
        prev = r["reported_tok_s"]

    peak = max(lc, key=lambda r: r["reported_tok_s"])
    b32 = [r for r in lc if r["batch_size"] == 32][0]
    b48 = [r for r in lc if r["batch_size"] == 48][0]
    b24 = [r for r in lc if r["batch_size"] == 24][0]

    print("\nTHE ANOMALY: throughput peaks at batch %d and then FALLS. Going 24 -> 32"
          % peak["batch_size"])
    print("adds 33%% more concurrent work and loses %.1f%% of throughput; 24 -> 48"
          % ((b24["reported_tok_s"] - b32["reported_tok_s"]) / b24["reported_tok_s"] * 100))
    print("doubles the work and loses %.1f%%. Even the inflated counter goes DOWN."
          % ((b24["reported_tok_s"] - b48["reported_tok_s"]) / b24["reported_tok_s"] * 100))

    print("\nMECHANISM -- KV-cache exhaustion forcing preemption + prefill recompute.")
    print("Named rows and columns:\n")
    print("  1. Capacity. B1 gives a KV pool of %s tokens = %.1f sequences of %d."
          % (f"{KV_POOL_TOKENS:,}", KV_POOL_TOKENS / MAX_MODEL_LEN, MAX_MODEL_LEN))
    for r in lc:
        need = r["batch_size"] * MAX_MODEL_LEN
        print("     batch %-2d needs %7s tokens = %5.0f%% of the pool%s"
              % (r["batch_size"], f"{need:,}", need / KV_POOL_TOKENS * 100,
                 "   <-- does not fit" if need > KV_POOL_TOKENS else ""))
    print()
    print("  2. `kv_cache_util` pins: %.2f (b24) -> %.2f (b32) -> %.2f (b48)."
          % (b24["kv_cache_util"], b32["kv_cache_util"], b48["kv_cache_util"]))
    print("     It cannot exceed ~0.97, so the extra sequences are not resident.")
    print()
    print("  3. `preempted_seqs` turns on exactly there: %d -> %d -> %d."
          % (b24["preempted_seqs"], b32["preempted_seqs"], b48["preempted_seqs"]))
    print("     At b48, %d of %d requests (%.0f%%) were evicted at least once."
          % (b48["preempted_seqs"], b48["num_requests"],
             b48["preempted_seqs"] / b48["num_requests"] * 100))
    print()
    print("  4. `ttft_ms_p50` jumps %.1f -> %.1f -> %.1f ms (%+.0f%% at b48)."
          % (b24["ttft_ms_p50"], b32["ttft_ms_p50"], b48["ttft_ms_p50"],
             (b48["ttft_ms_p50"] - b24["ttft_ms_p50"]) / b24["ttft_ms_p50"] * 100))
    print("     Preempted sequences must RE-PREFILL, which is a second first-token.")
    print()
    print("  5. The decisive column pair. `itl_ms_p50` barely moves:")
    print("     %.2f -> %.2f -> %.2f ms (%+.1f%%). Decode speed per resident token"
          % (b24["itl_ms_p50"], b32["itl_ms_p50"], b48["itl_ms_p50"],
             (b48["itl_ms_p50"] - b24["itl_ms_p50"]) / b24["itl_ms_p50"] * 100))
    print("     is essentially UNCHANGED -- the GPU is not saturated on compute.")
    print("     But `wall_clock_s` explodes %.2f -> %.2f -> %.2f s."
          % (b24["wall_clock_s"], b32["wall_clock_s"], b48["wall_clock_s"]))
    dec24 = b24["gen_len"] * b24["itl_ms_p50"] / 1000
    dec48 = b48["gen_len"] * b48["itl_ms_p50"] / 1000
    print("     Time that SHOULD be decoding, gen_len x itl: %.1f s (b24), %.1f s (b48)."
          % (dec24, dec48))
    print("     Fraction of wall clock actually decoding: %.0f%% (b24) -> %.0f%% (b48)."
          % (dec24 / b24["wall_clock_s"] * 100, dec48 / b48["wall_clock_s"] * 100))
    print("     So the loss is not slower decoding. It is time spent NOT decoding:")
    print("     evicting, re-queuing and re-prefilling the same sequences.")

    # ---- proposed change, with a prediction --------------------------------
    print("\n" + "-" * 88)
    print("PROPOSED CHANGE (primary): cap --max-num-seqs at 24.")
    print("-" * 88)
    unpre = [r for r in lc if r["preempted_seqs"] == 0]
    xs = [r["batch_size"] for r in unpre]
    ys = [r["wall_clock_s"] for r in unpre]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    icept = my - slope * mx
    ss_res = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    print("Fit wall_clock vs batch on the %d preemption-free long-context rows"
          % n)
    print("(batches %s):  wall = %.2f + %.3f * batch   (R^2 = %.4f)"
          % (xs, icept, slope, 1 - ss_res / ss_tot))
    print()
    waves = 2
    pred_wall = waves * b24["wall_clock_s"]
    cur_good = b48["gen_len"] * b48["num_requests"] / b48["wall_clock_s"]
    pred_good = b48["gen_len"] * b48["num_requests"] / pred_wall
    print("With max_num_seqs=24, the 48-request load runs as 2 sequential waves of 24,")
    print("each of which the log already measured at %.2f s with 0 preemptions."
          % b24["wall_clock_s"])
    print()
    print("  PREDICTION for the batch-48 workload:")
    print("    wall clock : %.2f s -> %.2f s        (%+.1f%%)"
          % (b48["wall_clock_s"], pred_wall,
             (pred_wall - b48["wall_clock_s"]) / b48["wall_clock_s"] * 100))
    print("    goodput    : %.1f -> %.1f tok/s      (%+.1f%%)"
          % (cur_good, pred_good, (pred_good - cur_good) / cur_good * 100))
    print("    preemptions: %d -> 0" % b48["preempted_seqs"])
    print("    COST: wave-2 requests queue behind wave 1, so ttft p95 goes from")
    print("          %.0f ms to roughly %.1f s. This buys throughput with tail latency"
          % (b48["ttft_ms_p50"], b24["wall_clock_s"] + b24["ttft_ms_p50"] / 1000))
    print("          and is the right trade only if the SLO is throughput-shaped.")

    print("\n" + "-" * 88)
    print("PROPOSED CHANGE (alternative): fp8 KV cache.")
    print("-" * 88)
    print("Halving KV bytes/token 114,688 -> 57,344 doubles the pool to %s tokens"
          % f"{KV_POOL_TOKENS*2:,}")
    print("= %.1f concurrent 4096-token sequences, so batch 48 becomes resident."
          % (KV_POOL_TOKENS * 2 / MAX_MODEL_LEN))
    fit48 = icept + slope * 48
    good48 = b48["gen_len"] * b48["num_requests"] / fit48
    print("  Extrapolating the preemption-free fit to batch 48: wall = %.1f s" % fit48)
    print("  PREDICTION: wall %.2f -> %.1f s (%+.0f%%), goodput %.1f -> %.1f tok/s (%+.0f%%)"
          % (b48["wall_clock_s"], fit48, (fit48 - b48["wall_clock_s"]) / b48["wall_clock_s"] * 100,
             cur_good, good48, (good48 - cur_good) / cur_good * 100))
    print("  Weaker claim than the first: it extrapolates a 4-point fit beyond its")
    print("  range and assumes fp8 KV costs no accuracy. Verify before trusting.")


# ----------------------------------------------------------------------- B4
def b4():
    head("B4  Which counter confirms the B2 mechanism")
    print("""Pull `vllm:num_preemptions_total` (the scheduler's cumulative preemption
counter), scraped from /metrics across one batch-48 run.

  Expected value: >= 23 for the batch-48 row and >= 7 for batch-32, matching
  the `preempted_seqs` column -- and STRICTLY greater if any sequence was
  preempted more than once, which the column ("preempted at least once")
  cannot show. At batch 24 and below it must stay at exactly 0.

If I could pull a second, `vllm:prompt_tokens_total` settles it quantitatively.
A preempted sequence in the default RECOMPUTE policy re-runs its whole prefill,
so prompt tokens get counted twice for it:

  no preemption : 48 x 3584                    = 172,032 prompt tokens
  observed      : (48 + 23) x 3584             = 254,464   (+47.9%)

Seeing ~254k rather than ~172k proves the lost time is recomputed prefill and
not, say, memory-bandwidth saturation -- a bandwidth wall would inflate
`itl_ms_p50`, and the log shows it flat at ~100 ms from batch 24 to 48.""")


if __name__ == "__main__":
    rows = load()
    what_is_reported_tok_s(rows)
    goodput_table(rows)
    anomaly(rows)
    b4()
