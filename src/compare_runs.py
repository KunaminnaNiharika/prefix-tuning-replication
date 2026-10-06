"""
Compare two or more runs: a metrics table + side-by-side generated sentences.
Usage:  python src/compare_runs.py results/newdata_warm results/newdata_cold
"""
import json, os, re, sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

dirs = [d.rstrip("\\/") for d in sys.argv[1:]]
if len(dirs) < 1:
    sys.exit("give at least one results folder")
names = [os.path.basename(d) for d in dirs]
res = {d: json.load(open(os.path.join(d, "results.json"), encoding="utf-8")) for d in dirs}


def fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


rows = []
def add(label, fn):
    rows.append([label] + [fn(res[d]) for d in dirs])

add("loss, first step", lambda r: fmt(r["loss_history"][0], 2) if r["loss_history"] else "-")
add("loss, last 10 steps (avg)", lambda r: fmt(sum(r["loss_history"][-10:]) / len(r["loss_history"][-10:]), 2) if r["loss_history"] else "-")
add("BLEU (smoothed)", lambda r: fmt(r["metrics"].get("BLEU_smoothed", r["metrics"].get("BLEU"))))
add("NIST", lambda r: fmt(r["metrics"].get("NIST"), 2))
add("ROUGE-L", lambda r: fmt(r["metrics"].get("ROUGE_L")))
add("METEOR", lambda r: fmt(r["metrics"].get("METEOR")))
add("slot value coverage %", lambda r: fmt(r["metrics"].get("slot_value_coverage_%")))
w0 = max(len(r[0]) for r in rows) + 2
print(" " * w0 + "".join(f"{n:>22s}" for n in names))
for r in rows:
    print(f"{r[0]:{w0}s}" + "".join(f"{v:>22s}" for v in r[1:]))
print("\ncoverage by slot:")
for d, n in zip(dirs, names):
    print(f"  {n}: {res[d].get('coverage_by_slot')}")

gens = [pd.read_csv(os.path.join(d, "generations.csv"), encoding="utf-8-sig").fillna("") for d in dirs]
n_show = min(len(g) for g in gens)
print(f"\nsentences side by side ({n_show} test inputs):")
for i in range(n_show):
    mr = gens[0]["mr"][i]
    nm = re.search(r"name\[([^\]]*)\]", mr)
    print("\n" + "-" * 78)
    print("INPUT    :", mr[:230])
    print("REFERENCE:", gens[0]["reference_1"][i])
    for g, n in zip(gens, names):
        print(f"{n[:9]:9s}:", g["generated"][i])
