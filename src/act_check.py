"""
ViGGO error checks, per run:
  1) question vs statement, by dialogue act (does the model ask when the human asks?)
  2) invented ESRB ratings: outputs that mention a rating although the MR has no esrb slot
Usage:  python src/act_check.py results/viggo_warm results/viggo_cold
"""
import os, re, sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ESRB = re.compile(r"\(for\b|\bESRB\b|\brated [MET]\b", re.I)
for d in sys.argv[1:]:
    g = pd.read_csv(os.path.join(d.rstrip("\\/"), "generations.csv"), encoding="utf-8-sig").fillna("")
    g["act"] = g["mr"].str.extract(r"^\s*([a-z_]+)\(")[0].fillna("(none)")
    g["human_q"] = g["reference_1"].str.contains(r"\?")
    g["model_q"] = g["generated"].str.contains(r"\?")
    print("=" * 70); print(os.path.basename(d.rstrip("\\/")), f"({len(g)} outputs)")
    t = g.groupby("act").agg(n=("act", "size"), human_asks=("human_q", "mean"), model_asks=("model_q", "mean"))
    print(f"{'dialogue act':20s} {'n':>4s} {'human asks':>11s} {'model asks':>11s}")
    for act, r in t.sort_values("n", ascending=False).iterrows():
        print(f"{act:20s} {int(r.n):4d} {100*r.human_asks:10.0f}% {100*r.model_asks:10.0f}%")
    q = g[g["human_q"]]
    if len(q):
        print(f"\nwhere the human sentence is a question ({len(q)}): the model also asks in {100*q['model_q'].mean():.0f}%")
    st = g[~g["human_q"]]
    if len(st):
        print(f"where the human sentence is a statement ({len(st)}): the model asks a question in {100*st['model_q'].mean():.0f}%")
    no_esrb = g[~g["mr"].str.contains(r"esrb\[", case=False)]
    inv_m = int(no_esrb["generated"].map(lambda x: bool(ESRB.search(x))).sum())
    inv_h = int(no_esrb["reference_1"].map(lambda x: bool(ESRB.search(x))).sum())
    print(f"\ninputs with NO esrb slot: {len(no_esrb)} | outputs that mention a rating anyway: MODEL {inv_m}, HUMAN {inv_h}")
