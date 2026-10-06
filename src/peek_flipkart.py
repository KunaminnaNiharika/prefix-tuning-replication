"""Print real row examples + stats for the Flipkart CSV, so the build script can be written against the true formats.
Usage:  python src/peek_flipkart.py data/raw/flipkart.csv
"""
import sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # avoid Windows console encoding errors
except Exception:
    pass

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/raw/flipkart.csv")
print("rows:", len(d))
for c in ["product_name", "description", "product_specifications", "retail_price", "brand", "product_category_tree"]:
    print(f"{c:25s} non-null: {d[c].notna().sum()}")
desc = d["description"].fillna("").astype(str)
print("description length in characters (10/25/50/75/90 percentile):",
      desc.str.len().quantile([.1, .25, .5, .75, .9]).round(0).tolist())
print("\nmost common description openings (first 3 words):")
print(desc.str.split().str[:3].str.join(" ").value_counts().head(8).to_string())
for i in range(5):
    r = d.iloc[i]
    print("\n" + "=" * 78)
    print("NAME     :", r["product_name"])
    print("CATEGORY :", str(r["product_category_tree"])[:200])
    print("PRICE    :", r["retail_price"], "| BRAND:", r["brand"])
    print("DESCRIPTION:", repr(str(r["description"])[:800]))
    print("SPECS    :", repr(str(r["product_specifications"])[:500]))
