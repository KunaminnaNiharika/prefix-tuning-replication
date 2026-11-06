"""
Inter-annotator agreement (Cohen's kappa) between two annotation sheets.
Usage: python src/agreement.py data/newdata/annotation_sheet.csv data/newdata/rater2_batch1.csv [data/newdata/rater2_batch2.csv ...]
All sheets after the first are rater 2 and are combined; give one of them to score a single batch.
"""
import sys
import pandas as pd

COL = "supported_by_MR"
LABELS = ["yes", "partial", "no"]
a = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
b = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in sys.argv[2:]], ignore_index=True)
m = a.merge(b[["mr", "ref", COL]], on=["mr", "ref"], suffixes=("_1", "_2"))
m["r1"] = m[COL + "_1"].astype(str).str.strip().str.lower()
m["r2"] = m[COL + "_2"].astype(str).str.strip().str.lower()
m = m[m["r1"].isin(LABELS) & m["r2"].isin(LABELS)]
n = len(m)
if n == 0:
    sys.exit("No rows matched. Check both files have labels and the same mr/ref text (re-save as CSV UTF-8).")
po = (m["r1"] == m["r2"]).mean()
pe = sum((m["r1"] == l).mean() * (m["r2"] == l).mean() for l in LABELS)
kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
print(f"items compared: {n}")
print(f"raw agreement : {100*po:.0f}%")
def band(k):
    # Landis & Koch (1977) interpretation scale
    for lim, name in [(0, "poor"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if k <= lim:
            return name
    return "almost perfect"
print(f"Cohen's kappa : {kappa:.2f}   ({band(kappa)} agreement, Landis & Koch 1977)")
print("\nrows = rater 1, columns = rater 2")
print(pd.crosstab(m["r1"], m["r2"]).reindex(index=LABELS, columns=LABELS, fill_value=0))
