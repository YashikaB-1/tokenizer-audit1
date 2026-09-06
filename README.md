# The Audit — submission

Audit of `REPORT_v0.md`, the tokenizer benchmark behind it, and the serving log behind its
Section 2.

> **`AI_USAGE.md` first.** This repo was produced in AI-assisted sessions and that file says
> so in full. **[`DEFENSE.md`](DEFENSE.md)** then works every load-bearing number through from
> first principles, with counterfactuals and live commands, for the 30-minute session.

## The three findings that change a decision

1. **The code bugs in `fertility.py` barely matter.** Fixing all three moves the headline
   from 5.89× to 6.35×. The report is wrong for reasons that are not in the code.
2. **The report's own root cause is false by 6.5×.** It says the Hindi:English gap is "a
   property of the script, not the tokenizer". Holding the text byte-identical and changing
   only the vocabulary takes Hindi from **7.45× to 1.14×**.
3. **`reported_tok_s` counts prompt tokens as throughput** — `(prompt+gen)×n/wall`, verified
   against all 13 rows to 0.01%. Both of Section 2's conclusions come from that. Honest
   goodput of the batch-24 long-prompt row is **200.9 tok/s**, not 1607.4, and long prompts
   are **1.80× worse**, not 1.48× better.

## Reproduce everything

```bash
pip install -r requirements.txt
bash run_all.sh          # ~26 MB corpus + 4 tokenizers downloaded on first run
```

Or individually — every document names the command that produced its numbers.

## Layout

```
NOTEBOOK.md                  chronological log: hypothesis -> experiment -> result -> revision,
                             including the conclusions that turned out to be wrong
AI_USAGE.md                  what the AI did, where it misled me, what I must defend

run_all.sh                   reproduces every number in the repo

partA/
  CORPUS.md                  A1 - FLORES-200, 997 parallel sentences x 7 languages, and
                             what the corpus cannot tell you
  A2_AUDIT.md                A2 - 3 code bugs + 5 reasoning errors, each with a measured
                             delta; 3 things that look wrong and are not
  A3_ANALYSIS.md             A3 - 5 tokenizers x 4 denominators, and which single number
                             should drive routing
  A4_MEMO.md                 A4 - the memo (<=1 page)

  build_corpus.py            downloads FLORES-200, verifies SHA-256, asserts line alignment
  audit_evidence.py          the ablation harness: one switch per claimed flaw
  audit_claims.py            decoys, corpus alignment, the "two metrics agree" identity
  fertility_v1.py            corrected replacement for fertility.py
  denominators.py            what each denominator holds constant, measured
  replicate.py               reruns every Part A conclusion on the held-out devtest split
  probe.py                   tokenizes arbitrary pasted text across all 5 tokenizers (live tool)
  corpus/                    the eval corpus (committed, so no network needed to check the work)
  corpus_devtest/            the held-out split, never used to develop the analysis
  results/                   saved stdout of every run

partB/
  B_ANSWERS.md               B1-B4
  b1_kv_capacity.py          KV bytes/token, concurrency limit, reconciliation vs the log
  b2b3_goodput.py            the throughput anomaly, its mechanism, and the misread column
  sensitivity.py             do B1's assumptions matter? (overhead, GB/GiB, GQA, fp8)
  results/

partC/memo.md                C - decision memo (<=1 page)

starter_kit_original/        the starter kit, unmodified, so deltas are checkable in place
```

## Claim → evidence index

Every claim below is a command, not an assertion.

| claim | where | command |
|---|---|---|
| REPORT_v0 §1 is reproducible (5.89×) | A2 §0 | `cd starter_kit_original && python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2` |
| the three code bugs are worth +4% | A2 §3 | `python partA/audit_evidence.py --corpus-dir partA/corpus` |
| `random.seed` is dead code (byte-identical output) | A2 §6 D2 | `python partA/audit_claims.py` |
| NFC is not inert (90 hin / 586 ben lines) but costs 0.2% | A2 §6 D1 | `python partA/audit_claims.py` |
| "the two metrics agree" is an identity: 1.2081 = 1.2081 | A2 §5 C2 | `python partA/audit_claims.py` |
| the sample corpus is not line-aligned | A1, A2 §5 C4 | `python partA/audit_claims.py` |
| Hindi 7.45× → 1.14× by changing tokenizer alone | A2 §5 C3, A3 | `python partA/fertility_v1.py --preset all` |
| the word denominator flatters hin 0.85×, penalises kan 1.36× | A3 | `python partA/denominators.py` |
| KV = 114,688 B/token; ~25 concurrent 4096-tok seqs; +0.8% vs log | B1 | `python partB/b1_kv_capacity.py` |
| `reported_tok_s` = (prompt+gen)×n/wall, 0.01% error | B3 | `python partB/b2b3_goodput.py` |
| batch-24 goodput = 200.9 tok/s, two independent routes | B3 | `python partB/b2b3_goodput.py` |
| throughput peaks at batch 24 because 32 needs 125% of the KV pool | B2 | `python partB/b2b3_goodput.py` |


