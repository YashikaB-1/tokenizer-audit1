# Part B — Capacity reconciliation

```bash
python partB/b1_kv_capacity.py    # B1
python partB/b2b3_goodput.py      # B2, B3, B4
```

Saved output: [`results/b1_output.txt`](results/b1_output.txt), [`results/b2b3_output.txt`](results/b2b3_output.txt).

---

## B1 — KV-cache size and concurrency

### (a) KV-cache bytes per token, exactly

One cached token stores a K vector and a V vector in every layer. With grouped-query attention the cache is sized by the **8 KV heads**, not the 24 query heads — that is what GQA is for.

```
bytes/token = 2 (K and V) × 28 layers × 8 kv_heads × 128 head_dim × 2 bytes (fp16)
            = 114,688 bytes/token
            = 112 KiB/token
```

Using the 24 query heads instead would give 344,064 B/token — 3× too big, and it is the mistake this spec is shaped to catch.

One full 4096-token sequence: `114,688 × 4096 = 469,762,048 bytes = 448 MiB = 0.470 GB`.

### (b) Maximum concurrent 4096-token sequences

```
weights        = 4.2e9 params × 2 bytes (fp16)  = 8.40 GB
overhead       = 1.60 GB                          (given in the spec)

usable         = 24.00 GB × 0.92                = 22.08 GB
KV pool        = 22.08 − 8.40 − 1.60            = 12.08 GB
max tokens     = 12.08e9 ÷ 114,688              = 105,329 tokens
max sequences  = 105,329 ÷ 4096                 = 25.7  →  25 concurrent
```

The GB-vs-GiB reading of "24 GB" matters here, so I computed both and let the log decide:

| reading | KV pool | max tokens | max 4096-tok seqs |
|---|---|---|---|
| **24 GB (decimal, 24e9)** | 12.08 GB | **105,329** | **25.7** |
| 24 GiB (binary, 2³⁰) | 13.71 GB | 119,526 | 29.2 |

### (c) Checking the prediction against the log

`kv_cache_util` is the fraction of the KV pool in use, so inverting it on each row recovers an **observed** pool size. `kv_cache_util` is logged to 2 decimals, so rows below ~0.05 are rounding-dominated (at 0.01, one ulp is ±50%) and are excluded.

| batch | prompt | gen | tok/seq | kv_util | live tokens | implied pool | implied seqs |
|---|---|---|---|---|---|---|---|
| 8 | 512 | 256 | 768 | 0.06 | 6,144 | 102,400 | 25.0 |
| 16 | 512 | 256 | 768 | 0.12 | 12,288 | 102,400 | 25.0 |
| 32 | 512 | 256 | 768 | 0.23 | 24,576 | 106,852 | 26.1 |
| 64 | 512 | 256 | 768 | 0.47 | 49,152 | 104,579 | 25.5 |
| 4 | 3584 | 512 | 4096 | 0.16 | 16,384 | 102,400 | 25.0 |
| 8 | 3584 | 512 | 4096 | 0.31 | 32,768 | 105,703 | 25.8 |
| 16 | 3584 | 512 | 4096 | 0.62 | 65,536 | 105,703 | 25.8 |
| 24 | 3584 | 512 | 4096 | 0.93 | 98,304 | 105,703 | 25.8 |
| 32 | 3584 | 512 | 4096 | 0.97 | 131,072 | *saturated — excluded* | |
| 48 | 3584 | 512 | 4096 | 0.97 | 196,608 | *saturated — excluded* | |

**Observed pool: 104,468 tokens (25.5 seqs). Predicted: 105,329 tokens (25.7 seqs). Error +0.82%.**

The 24 GiB reading predicts 29.2 sequences — 14% high, and inconsistent with the log. **The decimal reading is correct**; `gpu_memory_utilization` is being applied to 24×10⁹ bytes.

Three independent confirmations that ~25–26 is the real ceiling:

- The three unsaturated long-context rows (8/16/24) all imply **exactly 105,703 tokens** — `kv_cache_util` is linear in batch at 0.03875/seq, i.e. 1/0.03875 = 25.8 sequences.
- Batch 24 sits at 0.93 util with **0 preemptions**; batch 32 would need 131,072 tokens (**125%** of the pool) and pins at 0.97 with **7** preemptions.
- The short-prompt rows, which have a completely different tokens-per-sequence, imply the same pool (102,400–106,852).

**Answer: ~25 concurrent 4096-token sequences, and the log agrees to within 1%.**

---

## B2 — The long-context throughput anomaly

### The anomaly

| batch | reported_tok_s | goodput | wall_s | kv_util | preempted | ttft_ms | itl_ms |
|---|---|---|---|---|---|---|---|
| 4 | 565.4 | 70.7 | 28.98 | 0.16 | 0 | 483.2 | 51.33 |
| 8 | 902.6 | 112.8 | 36.30 | 0.31 | 0 | 519.0 | 62.26 |
| 16 | 1311.4 | 163.9 | 49.97 | 0.62 | 0 | 498.3 | 77.20 |
| **24** | **1607.4** | **200.9** | 61.16 | 0.93 | 0 | 500.5 | 96.07 |
| 32 | 1384.0 | 173.0 | 94.71 | 0.97 | **7** | 636.9 | 101.79 |
| 48 | 1298.5 | 162.3 | 151.41 | 0.97 | **23** | 955.4 | 100.00 |

Throughput **peaks at batch 24 and then falls**. Batch 24→32 adds 33% more concurrent work and *loses* 13.9%; 24→48 doubles the work and loses 19.2%. Batch 48 is worse than batch 16. Even the inflated `reported_tok_s` counter goes down.

### Mechanism: KV-cache exhaustion → preemption → prefill recompute

**1. The capacity wall is exactly where B1 predicted.** The pool holds 105,329 tokens = 25.7 sequences of 4096:

| batch | KV needed | % of pool | |
|---|---|---|---|
| 16 | 65,536 | 62% | fits |
| 24 | 98,304 | 93% | fits, barely |
| 32 | 131,072 | **125%** | **does not fit** |
| 48 | 196,608 | **187%** | **does not fit** |

The throughput peak is at batch 24 because batch 24 is the largest batch that fits in the KV cache. This is not a coincidence — it is B1's number, derived from the model spec alone, showing up in the log.

**2. `kv_cache_util` pins at 0.97** (b32 and b48) and cannot rise further. The additional sequences are not resident; they are admitted and then evicted.

**3. `preempted_seqs` turns on precisely at the wall: 0 → 7 → 23.** At batch 48, 23 of 48 requests (48%) were evicted at least once.

**4. `ttft_ms_p50` jumps 500.5 → 636.9 → 955.4 ms (+91%).** A preempted sequence must re-prefill, which registers as a second time-to-first-token.

**5. The decisive column pair — `itl_ms_p50` vs `wall_clock_s`.** This is what rules out the alternative explanations:

- `itl_ms_p50` moves 96.07 → 101.79 → 100.00 ms — **+4.1% across a 2× batch increase**. Per-token decode speed is essentially unchanged, so the GPU is *not* compute-saturated and *not* bandwidth-saturated. Either of those would show up here first.
- `wall_clock_s` explodes 61.16 → 94.71 → **151.41 s**.
- Time that *should* be spent decoding is `gen_len × itl` = 49.2 s at b24 and 51.2 s at b48 — barely different. As a fraction of wall clock that is **80% at b24 and 34% at b48**.

**Two thirds of the batch-48 run is spent not decoding.** The loss is not slower work; it is repeated work — evicting, re-queuing and re-prefilling the same sequences. That is the mechanism.

### Proposed change (primary): cap `--max-num-seqs` at 24

Admit at most 24 sequences so the scheduler never oversubscribes the KV pool. The 48-request load then runs as two sequential waves of 24, and the log already measured a preemption-free wave of 24 at 61.16 s.

**Predicted effect on the batch-48 workload:**

| | now | predicted | |
|---|---|---|---|
| wall clock | 151.41 s | **122.32 s** | **−19.2%** |
| goodput | 162.3 tok/s | **200.9 tok/s** | **+23.8%** |
| preemptions | 23 | **0** | |

**Cost, stated honestly:** wave-2 requests queue behind wave 1, so ttft p95 goes from 955 ms to roughly 61.7 s. This buys throughput with tail latency and is the right trade only if the SLO is throughput-shaped. If p95 latency is the SLO, the answer is to add a second replica, not to raise the batch.

### Alternative: fp8 KV cache

Halving KV bytes/token (114,688 → 57,344) doubles the pool to 210,658 tokens = **51.4** concurrent 4096-token sequences, so batch 48 becomes resident. Fitting `wall_clock` against batch on the four preemption-free long-context rows gives `wall = 23.15 + 1.612 × batch` (R² = 0.9968); at batch 48 that extrapolates to 100.5 s.

Predicted: wall 151.41 → 100.5 s (−34%), goodput 162.3 → 244.5 tok/s (+51%).

**This claim is weaker than the first** and I flag it as such: it extrapolates a 4-point fit beyond its fitted range and assumes fp8 KV costs no output quality. The `max_num_seqs` prediction interpolates a row that was actually measured; this one does not.

---

## B3 — The misread column

### What the column is

**`reported_tok_s` counts prompt tokens as throughput.** It is total tokens processed ÷ wall clock, not generation rate. Tested against every row in the log:

```
reported_tok_s = (prompt_len + gen_len) × num_requests / wall_clock_s
```

| | mean absolute error |
|---|---|
| vs `(prompt+gen)×n / wall` | **0.01%** |
| vs `gen×n / wall` | 76.3% |

Thirteen of thirteen rows match the first formula to within 0.02%. On the 3584/512 rows, **88% of `reported_tok_s` is prompt tokens.** Prompt tokens are processed in one parallel prefill and are cheap per token; generated tokens are produced one step at a time and are what a capacity plan is actually buying.

**This single misreading produces both of §2's conclusions:**

- *"Longer prompts clearly give better GPU utilization"* — longer prompts mechanically inflate a counter whose numerator is dominated by prompt length. The comparison is between a number that is 67% prompt and a number that is 88% prompt.
- *"batch 48 should give us ~3200 tok/s"* — scaling that inflated counter linearly.

### Honest goodput of the batch-24 long-prompt row

**200.9 generated tokens/s.** Two independent derivations:

**Route 1 — from the raw counts:**
```
24 requests × 512 generated tokens ÷ 61.16 s = 200.92 tok/s
```

**Route 2 — strip the prompt fraction out of the harness counter:**
```
1607.4 tok/s × 512/(3584+512) = 1607.4 / 8 = 200.93 tok/s
```

The two agree to **0.005%**. (They are independent in the sense that route 2 never touches `num_requests` or the raw token counts — it only needs the reported figure and the prompt:gen split. If the harness counter were computed some other way, route 2 would disagree with route 1; it does not.)

**Third cross-check, from a different column entirely:** `itl_ms_p50 = 96.07 ms` gives 24 concurrent ÷ 0.09607 s = 249.8 tok/s of steady-state decode. Goodput is 200.9/249.8 = 80% of that, i.e. 80% of wall clock is spent decoding and 20% is prefill (ttft 500 ms) and scheduling. Consistent, and it also bounds the answer from above.

So `reported_tok_s` overstates the useful output of that row by **exactly 8×**.

### What the report should have said

Redoing §2's own comparison on goodput, at batch 16:

| | reported_tok_s | goodput |
|---|---|---|
| short prompts (512/256) | 883.2 | **294.5 tok/s** |
| long prompts (3584/512) | 1311.4 | **163.9 tok/s** |

**`reported_tok_s` says long prompts are 1.48× better. Goodput says they are 1.80× worse — 44% less useful output for the same GPU-second.** The sign of the conclusion flips.

Two further errors in the same paragraph:

- *"assume ~1600 tok/s per L4 (best observed)"* — 1607.4 is not the best observed anything. Max `reported_tok_s` in the log is **2267.3** (batch 64, short prompts); max goodput is **755.7 tok/s** (batch 64, short prompts). 1600 is the max of one sweep, quoted as a global best.
- *"batch 48 should give us ~3200 tok/s"* — **the batch-48 run is already in the log.** It did not need extrapolating. It measured 1298.5 tok/s reported (41% of the prediction) and 162.3 tok/s goodput (5% of it).

**§2 should have said:** *"Decode goodput peaks at 755.7 tok/s with short prompts at batch 64, and at 200.9 tok/s with 3584-token prompts at batch 24. Long prompts reduce useful throughput by ~44% at equal batch, because prefill consumes wall clock that produces no output tokens. Batch 24 is the ceiling for 4096-token sequences: KV cache holds ~25 of them, and batch 32 and 48 were measured at 14% and 19% below batch 24 because of preemption. Plan capacity at ~200 tok/s per L4 for long-context traffic, and do not encourage clients to pack more context per request — it costs goodput and it costs concurrency."*

---

## B4 — The counter that would confirm the mechanism

I would pull **`vllm:num_preemptions_total`**, the scheduler's cumulative preemption counter, scraped from `/metrics` across one batch-48 run.

**Expected value: ≥ 23 for batch 48 and ≥ 7 for batch 32**, matching the `preempted_seqs` column — and strictly greater if any sequence was preempted more than once, which that column ("preempted at least once") cannot distinguish. At batch 24 and below it must stay at **exactly 0**. That last part is what makes it a real test: the mechanism predicts a sharp threshold at the KV capacity boundary from B1, not a gradual rise, so a nonzero reading at batch 16 or 24 would falsify it.

If I could pull a second counter, **`vllm:prompt_tokens_total`** settles it quantitatively. Under vLLM's default RECOMPUTE preemption policy an evicted sequence re-runs its entire prefill, so its prompt tokens are counted twice:

```
no preemption : 48 × 3584              = 172,032 prompt tokens
predicted     : (48 + 23) × 3584       = 254,464 prompt tokens   (+47.9%)
```

Seeing ~254k rather than ~172k proves the lost wall clock is recomputed prefill specifically, and rules out the main alternative — memory-bandwidth saturation — which would instead show up as a rising `itl_ms_p50`, and the log has that flat at ~100 ms from batch 24 to batch 48.
