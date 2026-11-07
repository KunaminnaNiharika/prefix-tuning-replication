"""
Score saved E2E generations with the official E2E NLG Challenge scorer (tuetschek/e2e-metrics),
so they can be compared with the NLTK / rouge_score numbers from src/train_prefix.py.

Writes the two files measure_scores.py expects (references grouped per MR with a blank line between
groups; one system output per line), runs it, and saves the scores as JSON.

Setup (once):
  git clone https://github.com/tuetschek/e2e-metrics
  pip install -r e2e-metrics/requirements.txt
  # the BLEU/NIST step calls mteval-v13a-sig.pl, which needs the Perl modules XML::Twig and Sort::Naturally
  # (Ubuntu: apt-get install libxml-twig-perl libsort-naturally-perl); METEOR needs Java.

Usage:
  python src/official_e2e_scores.py results/e2e_full/generations.csv data/e2e/testset_w_refs.csv \
         e2e-metrics results/e2e_full/official_e2e_metrics.json
"""
import json
import os
import re
import subprocess
import sys

import pandas as pd

gen_csv, test_csv, metrics_dir, out_json = sys.argv[1:5]
gen = pd.read_csv(gen_csv)
refs = pd.read_csv(test_csv).groupby("mr", sort=False)["ref"].apply(list)

work = os.path.dirname(os.path.abspath(out_json))
ref_txt, sys_txt = os.path.join(work, "official_ref.txt"), os.path.join(work, "official_sys.txt")
with open(ref_txt, "w", encoding="utf-8") as f:
    for mr in gen["mr"]:
        f.write("\n".join(r.strip() for r in refs[mr]) + "\n\n")
with open(sys_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(str(h).strip() for h in gen["generated"]) + "\n")

out = subprocess.run([sys.executable, os.path.join(metrics_dir, "measure_scores.py"), ref_txt, sys_txt],
                     capture_output=True, text=True, check=True).stdout
scores = {k: float(v) for k, v in re.findall(r"^(BLEU|NIST|METEOR|ROUGE_L|CIDEr): ([\d.]+)", out, re.M)}
json.dump({"scorer": "tuetschek/e2e-metrics measure_scores.py", "n": len(gen), "source": gen_csv,
           "official": scores}, open(out_json, "w"), indent=2)
print(json.dumps(scores, indent=2))
