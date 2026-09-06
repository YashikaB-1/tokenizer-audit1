#!/usr/bin/env bash
# Reproduce every number in this submission from scratch.
#   bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "### 0. reproduce REPORT_v0's table"
( cd starter_kit_original && python fertility.py \
    --corpus eng=corpus_sample/eng_sample.txt \
    --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2 )

echo; echo "### A1. build the eval corpus (downloads ~26 MB on first run)"
python partA/build_corpus.py

echo; echo "### A2. ablations + claim-level evidence"
python partA/audit_evidence.py                 | tee partA/results/a2_ablation_sample.txt
python partA/audit_evidence.py --corpus-dir partA/corpus \
                                               | tee partA/results/a2_ablation_flores.txt
python partA/audit_claims.py                   | tee partA/results/a2_claims.txt

echo; echo "### A3. corrected analysis (downloads 4 tokenizers on first run)"
python partA/denominators.py                   | tee partA/results/a3_denominators.txt
python partA/fertility_v1.py --preset all --out partA/results/a3_full \
                                               | tee partA/results/a3_full.txt

echo; echo "### A. held-out replication (builds the devtest corpus if absent)"
[ -d partA/corpus_devtest ] || python partA/build_corpus.py --split devtest --out-dir partA/corpus_devtest
python partA/replicate.py                      | tee partA/results/replication.txt

echo; echo "### B. capacity reconciliation"
python partB/b1_kv_capacity.py                 | tee partB/results/b1_output.txt
python partB/b2b3_goodput.py                   | tee partB/results/b2b3_output.txt
python partB/sensitivity.py                    | tee partB/results/sensitivity.txt

echo; echo "### live-demo tool smoke test"
{
  echo "### probe: Kannada 'Bengaluru' -- one word, gpt2 falls to byte level"
  python partA/probe.py -v "ಬೆಂಗಳೂರು"
  echo
  echo "### probe: a parallel pair -- the A3 metric on one sentence"
  python partA/probe.py --pair "I need a little water." "मुझे थोड़ा पानी चाहिए।"
} 2>&1 | grep -v "^Warning" | tee partA/results/probe_examples.txt

echo; echo "done."
