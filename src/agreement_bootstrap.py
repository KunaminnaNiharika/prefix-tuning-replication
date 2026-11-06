"""
Cohen's kappa with a bootstrap 95% CI for each batch and for all batches pooled.
Usage:
  python src/agreement_bootstrap.py data/newdata/annotation_sheet.csv data/newdata/rater2_batch1.csv data/newdata/rater2_batch2.csv
"""
import sys
import numpy as np
import pandas as pd

L = ["yes", "partial", "no"]
IX = {l: i for i, l in enumerate(L)}


def kappa(x, y):
    po = (x == y).mean()
    pe = sum((x == k).mean() * (y == k).mean() for k in range(3))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def boot_ci(x, y, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    ks = [kappa(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(ks, [2.5, 97.5])


a = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
pairs = []
for p in sys.argv[2:]:
    b = pd.read_csv(p, encoding="utf-8-sig")
    m = a.merge(b[["mr", "ref", "supported_by_MR"]], on=["mr", "ref"], suffixes=("_1", "_2"))
    x = m["supported_by_MR_1"].astype(str).str.strip().str.lower().map(IX)
    y = m["supported_by_MR_2"].astype(str).str.strip().str.lower().map(IX)
    ok = x.notna() & y.notna()
    pairs.append((p, x[ok].astype(int).values, y[ok].astype(int).values))
pairs.append(("ALL", np.concatenate([q[1] for q in pairs]), np.concatenate([q[2] for q in pairs])))
print(f"{'sheet':40s} {'n':>4s} {'raw':>5s} {'kappa':>6s}   95% bootstrap CI (10,000 resamples)")
for name, x, y in pairs:
    lo, hi = boot_ci(x, y)
    print(f"{name[-40:]:40s} {len(x):4d} {100*(x==y).mean():4.0f}% {kappa(x,y):6.2f}   [{lo:.2f}, {hi:.2f}]")
