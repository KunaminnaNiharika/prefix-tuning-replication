"""
Remove rows you annotated as 'no' from train/dev/test and report how the annotation maps onto each split.
Usage:
  python src/filter_by_annotation.py --dir data/newdata --out data/newdata_clean
"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/newdata")
ap.add_argument("--out", default="data/newdata_clean")
a = ap.parse_args()

ann = pd.read_csv(os.path.join(a.dir, "annotation_sheet.csv"))
ann["label"] = ann["supported_by_MR"].astype(str).str.strip().str.lower()
print("annotation labels:", ann["label"].value_counts().to_dict(), "| rows:", len(ann))
bad_labels = set(ann["label"]) - {"yes", "partial", "no"}
if bad_labels:
    print("WARNING: unexpected labels (fix the sheet):", bad_labels)

label_of = {(m, r): l for m, r, l in zip(ann["mr"], ann["ref"], ann["label"])}
os.makedirs(a.out, exist_ok=True)
matched_no = 0
print(f"\n{'split':6s} {'before':>7s} {'annotated':>10s} {'yes':>5s} {'partial':>8s} {'no':>4s} {'after':>6s}")
for split in ["train", "dev", "test"]:
    d = pd.read_csv(os.path.join(a.dir, f"{split}.csv"))
    labs = [label_of.get((m, r)) for m, r in zip(d["mr"], d["ref"])]
    keep = [l != "no" for l in labs]
    cnt = pd.Series([l for l in labs if l]).value_counts()
    matched_no += int(cnt.get("no", 0))
    d[keep].to_csv(os.path.join(a.out, f"{split}.csv"), index=False)
    print(f"{split:6s} {len(d):7d} {sum(l is not None for l in labs):10d} {cnt.get('yes',0):5d} "
          f"{cnt.get('partial',0):8d} {cnt.get('no',0):4d} {sum(keep):6d}")
n_no = int((ann["label"] == "no").sum())
print(f"\nremoved {matched_no} of {n_no} rows labelled 'no'")
if matched_no != n_no:
    print("Some 'no' rows did not match any split row. Excel may have changed characters when saving: "
          "re-save the sheet as 'CSV UTF-8' and run again.")
print("clean files written to", a.out)
