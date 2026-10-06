"""
Build the 'product-spec-to-blurb' dataset (E2E-style MR -> sentence).

Usage:
  python src/build_dataset.py --csv data/raw/Products.csv --out data/newdata --n_annotate 100

!! FIRST run with --inspect and check the real column names, then edit COLS below.
"""
import argparse, html, json, os, re
import pandas as pd

# ---- edit these to match the real column names printed by --inspect ----
COLS = {"name": "product_name", "category": "product_category",
        "desc": "product_description", "price": "product_price"}
# -------------------------------------------------------------------------

COLORS = ["black", "white", "red", "blue", "green", "pink", "silver", "gold", "grey", "gray",
          "brown", "purple", "yellow", "orange"]
MATERIALS = ["leather", "cotton", "wood", "wooden", "steel", "stainless steel", "plastic",
             "glass", "metal", "ceramic", "silicone", "wool", "polyester", "aluminum", "bamboo"]


def clean_text(t):
    t = html.unescape(str(t))
    t = re.sub(r"<[^>]+>", " ", t)          # strip HTML tags
    t = re.sub(r"\s+", " ", t).strip()
    return t


def first_sentence(t):
    m = re.match(r"(.+?[.!?])(\s|$)", t)
    return (m.group(1) if m else t).strip()


def parse_price(p):
    m = re.search(r"\d+(?:[.,]\d+)?", str(p).replace(",", ""))
    return float(m.group(0)) if m else None


def find_first(words, text):
    low = text.lower()
    for w in words:
        if re.search(rf"\b{re.escape(w)}\b", low):
            return w
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/newdata")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--min_words", type=int, default=8)
    ap.add_argument("--max_words", type=int, default=40)
    ap.add_argument("--n_annotate", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if a.inspect:
        print("columns:", list(df.columns)); print(df.head(3).T); return
    os.makedirs(a.out, exist_ok=True)
    log = {"raw_rows": len(df)}

    df = df.rename(columns={v: k for k, v in COLS.items()})[list(COLS)]
    df = df.dropna(); log["after_dropna"] = len(df)
    df["desc"] = df["desc"].map(clean_text)
    df["name"] = df["name"].map(clean_text)
    df = df.drop_duplicates(subset="desc").drop_duplicates(subset="name")
    log["after_dedupe"] = len(df)

    df["ref"] = df["desc"].map(first_sentence)
    nw = df["ref"].str.split().str.len()
    df = df[(nw >= a.min_words) & (nw <= a.max_words)]
    log["after_length_filter"] = len(df)

    df["price_num"] = df["price"].map(parse_price)
    df = df.dropna(subset=["price_num"])
    q1, q2 = df["price_num"].quantile([1 / 3, 2 / 3])
    df["priceRange"] = df["price_num"].map(lambda p: "cheap" if p <= q1 else ("moderate" if p <= q2 else "expensive"))

    def make_mr(r):
        slots = [("name", r["name"][:60]), ("category", r["category"]), ("priceRange", r["priceRange"])]
        for slot, vocab in (("color", COLORS), ("material", MATERIALS)):  # extracted attributes
            v = find_first(vocab, r["ref"])
            if v:
                slots.append((slot, v))
        return ", ".join(f"{k}[{v}]" for k, v in slots)

    df["mr"] = df.apply(make_mr, axis=1)
    df = df.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    n = len(df); n_dev = n_test = max(1, n // 10)
    test, dev, train = df[:n_test], df[n_test:n_test + n_dev], df[n_test + n_dev:]
    for nm, part in (("train", train), ("dev", dev), ("test", test)):
        part[["mr", "ref"]].to_csv(os.path.join(a.out, f"{nm}.csv"), index=False)

    ann = df.sample(n=min(a.n_annotate, n), random_state=a.seed)[["mr", "ref"]].copy()
    ann["supported_by_MR"] = ""   # fill in: yes / partial / no
    ann["notes"] = ""
    ann.to_csv(os.path.join(a.out, "annotation_sheet.csv"), index=False)

    log.update({"final_rows": n, "train": len(train), "dev": len(dev), "test": len(test),
                "avg_ref_words": round(float(df["ref"].str.split().str.len().mean()), 1),
                "avg_slots_per_MR": round(float(df["mr"].str.count(r"\[").mean()), 2),
                "pct_with_color": round(100 * df["mr"].str.contains(r"color\[").mean(), 1),
                "pct_with_material": round(100 * df["mr"].str.contains(r"material\[").mean(), 1),
                "price_cutoffs": [round(float(q1), 2), round(float(q2), 2)]})
    json.dump(log, open(os.path.join(a.out, "build_log.json"), "w"), indent=2)
    print(json.dumps(log, indent=2))
    print("\nexample:\n", df.iloc[0]["mr"], "\n ->", df.iloc[0]["ref"])


if __name__ == "__main__":
    main()
