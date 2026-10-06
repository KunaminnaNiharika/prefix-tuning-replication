"""
Quantify degenerate repetition and cut-off outputs in one or more generations.csv files.
Usage:  python src/repetition_check.py results/newdata_warm results/newdata_cold results/viggo_cold
"""
import os, re, sys
from collections import Counter
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def toks(s):
    return re.findall(r"[a-z0-9']+", str(s).lower())


def distinct2(s):
    t = toks(s)
    bg = list(zip(t, t[1:]))
    return len(set(bg)) / len(bg) if bg else 1.0


def looping(s, k=4):
    t = toks(s)
    bg = Counter(zip(t, t[1:]))
    return bool(bg) and max(bg.values()) >= k          # same word pair 4+ times = a loop


print(f"{'run':16s} {'n':>4s} {'loops':>6s} {'no end punct':>13s} {'distinct-2 model':>17s} {'distinct-2 human':>17s}")
for d in sys.argv[1:]:
    g = pd.read_csv(os.path.join(d.rstrip("\\/"), "generations.csv"), encoding="utf-8-sig").fillna("")
    gen = g["generated"].astype(str)
    n = len(g)
    loops = sum(looping(x) for x in gen)
    noend = int((~gen.str.strip().str.endswith((".", "!", "?"))).sum())
    d2m = sum(distinct2(x) for x in gen) / n
    d2h = sum(distinct2(x) for x in g["reference_1"]) / n
    print(f"{os.path.basename(d.rstrip(chr(92)+'/')):16s} {n:4d} {loops:6d} {noend:13d} {d2m:17.2f} {d2h:17.2f}")
print("\nloops = outputs where one word pair repeats 4+ times.  distinct-2 = share of unique word pairs (1.0 = no repetition).")
