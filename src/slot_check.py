"""
Compare how often the MODEL vs the HUMAN reference mentions each slot value (verbatim), plus a looser check for 'area'.
Usage:  python src/slot_check.py results/e2e_eval/generations.csv
        python src/slot_check.py results/newdata_warm/generations.csv name,brand,color,fabric,material,pattern,type
"""
import re, sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

path = sys.argv[1] if len(sys.argv) > 1 else "results/e2e_eval/generations.csv"
g = pd.read_csv(path, encoding="utf-8-sig").fillna("")
SLOTS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["name", "eattype", "area", "near", "food", "pricerange"])
SLOTS = [x.strip().lower() for x in SLOTS]


def slots(mr):
    return {k.strip().lower(): v.strip() for k, v in re.findall(r"([A-Za-z_ ]+?)\[([^\]]*)\]", mr)}


stat = {s: [0, 0, 0] for s in SLOTS}       # [total, model_found, human_ref_found]
area_rows = []
for mr, gen, ref in zip(g["mr"], g["generated"], g["reference_1"]):
    sl = slots(mr)
    for s in SLOTS:
        if s in sl:
            v = sl[s].lower()
            stat[s][0] += 1
            stat[s][1] += v in gen.lower()
            stat[s][2] += v in ref.lower()
    if "area" in sl and "area" in SLOTS:
        area_rows.append((sl["area"], gen, ref))

print(f"{'slot':12s} {'n':>5s} {'MODEL':>8s} {'HUMAN ref':>10s}   (verbatim value found in the sentence)")
for s, (n, m, h) in stat.items():
    if n:
        print(f"{s:12s} {n:5d} {100*m/n:7.0f}% {100*h/n:9.0f}%")

# 'area' values are 'riverside' or 'city centre'; a model may write 'city center' or 'centre of the city'
loose = lambda v, t: bool(re.search("riverside" if v.lower() == "riverside" else r"cent(re|er)", t.lower()))
if area_rows:
    n = len(area_rows)
    print(f"\narea, looser check ('riverside' / 'centre|center' anywhere): MODEL {100*sum(loose(v,g_) for v,g_,_ in area_rows)/n:.0f}% | "
          f"HUMAN {100*sum(loose(v,r_) for v,_,r_ in area_rows)/n:.0f}%   (n={n})")
    print("\nexamples where the model's sentence has no area at all:")
    shown = 0
    for v, gen, ref in area_rows:
        if not loose(v, gen):
            print(f"  area[{v}]\n    MODEL: {gen}\n    HUMAN: {ref}")
            shown += 1
            if shown == 4:
                break
