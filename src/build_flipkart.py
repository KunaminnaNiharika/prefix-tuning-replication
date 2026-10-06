"""
Build the 'product-spec-to-blurb' dataset from the Flipkart sample CSV.
Output format matches E2E: columns `mr` ("slot[value], slot[value], ...") and `ref` (one sentence).

Usage:
  python src/build_flipkart.py --csv data/raw/flipkart.csv --out data/newdata --n_annotate 100
"""
import argparse, html, json, os, re
from collections import Counter

import pandas as pd

# lines in a description that are sales boilerplate, not content
BOILER = ["flipkart", "genuine product", "free shipping", "cash on delivery",
          "replacement guarantee", "specifications of", "key features of"]
# spec keys that are identifiers / logistics, not describable attributes
EXCLUDE_KEY = re.compile(r"(model|code|sku|\bean\b|\bupc\b|\bid\b|number of|sales package|contents|"
                         r"warranty|care|certif|manufactur|country|pack of|in the box|net quantity)", re.I)
SPEC_RE = re.compile(r'\{"key"=>"((?:[^"\\]|\\.)*)", "value"=>"((?:[^"\\]|\\.)*)"\}')


def parse_specs(s):
    """'{"product_specification"=>[{"key"=>"Fabric", "value"=>"cotton"}, {"value"=>"x"}]}' -> [(key, value)]"""
    out = []
    for k, v in SPEC_RE.findall(str(s)):
        k = k.replace('\\"', '"').strip()
        v = re.sub(r"\s+", " ", v.replace('\\"', '"')).strip()
        out.append((k, v))
    return out


def parse_category(s):
    try:
        import ast
        t = ast.literal_eval(str(s))
        t = t[0] if isinstance(t, list) else str(t)
    except Exception:
        t = str(s).strip('[]"')
    return [p.strip() for p in t.split(">>") if p.strip()]


def clean_description(desc, name):
    """Strip the product-name line, 'Price: Rs.' line, tabs, boilerplate, and repeated lines."""
    t = html.unescape(str(desc))
    lines = [re.sub(r"\s+", " ", l).strip() for l in re.split(r"[\r\n]+", t)]
    out = []
    for l in lines:
        low = l.lower()
        if not l:
            continue
        if re.match(r"price\s*:", low):
            continue
        if low.startswith("buy "):
            continue
        if any(b in low for b in BOILER):
            continue
        if name.lower() in low and len(l) <= len(name) + 25:   # the title line (maybe with "(Pack of 2)")
            continue
        if l in out:                                           # descriptions repeat each line twice
            continue
        out.append(l)
    return " ".join(out).strip()


def first_sentences(text, max_sent=2):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:max_sent]).strip()


def slot_name(k):
    return re.sub(r"[^A-Za-z]+", " ", k).strip().lower()


def clean_val(v, maxlen=40):
    v = re.sub(r"[\[\]]", "", v).strip()
    return v if 0 < len(v) <= maxlen and v.upper() not in ("NA", "N/A") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/newdata")
    ap.add_argument("--min_words", type=int, default=8)
    ap.add_argument("--max_words", type=int, default=45)
    ap.add_argument("--max_specs", type=int, default=4)
    ap.add_argument("--n_annotate", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    log = {"raw_rows": len(df)}
    drop = Counter()

    # global frequency of spec keys -> we prefer commonly shared slots (like E2E's 8 shared slots)
    keyfreq = Counter()
    parsed_specs = []
    for s in df["product_specifications"]:
        sp = [(slot_name(k), clean_val(v)) for k, v in parse_specs(s)
              if not EXCLUDE_KEY.search(k) and slot_name(k)]
        sp = [(k, v) for k, v in sp if v]
        parsed_specs.append(sp)
        keyfreq.update({k for k, _ in sp})

    # price terciles
    prices = pd.to_numeric(df["retail_price"], errors="coerce")
    q1, q2 = prices.quantile([1 / 3, 2 / 3])

    rows = []
    for i, r in df.iterrows():
        name = re.sub(r"\s+", " ", str(r["product_name"])).strip()
        raw_desc = str(r["description"]).lstrip()
        if raw_desc.lower().startswith("key features of"):
            drop["description is a spec dump ('Key Features of ...')"] += 1; continue
        ref = first_sentences(clean_description(raw_desc, name))
        if not ref:
            drop["description was only sales boilerplate"] += 1; continue
        nw = len(ref.split())
        if nw < a.min_words:
            drop[f"cleaned description < {a.min_words} words"] += 1; continue
        if nw > a.max_words:
            drop[f"cleaned description > {a.max_words} words"] += 1; continue
        if pd.isna(prices[i]):
            drop["no price"] += 1; continue
        cat = parse_category(r["product_category_tree"])
        specs = parsed_specs[i]
        brand = r["brand"] if isinstance(r["brand"], str) and r["brand"].strip() else None
        if brand is None:
            brand = next((v for k, v in specs if k == "brand"), None)
        specs = [(k, v) for k, v in specs if k != "brand"]
        seen, uniq = set(), []
        for k, v in sorted(specs, key=lambda kv: -keyfreq[kv[0]]):
            if k not in seen:
                seen.add(k); uniq.append((k, v))
        specs = uniq[: a.max_specs]
        if not specs:
            drop["no usable specs"] += 1; continue
        price_bucket = "cheap" if prices[i] <= q1 else ("moderate" if prices[i] <= q2 else "expensive")
        slots = [("name", clean_val(name, 80) or name)]
        if brand:
            slots.append(("brand", clean_val(brand) or brand[:40]))
        if cat:
            slots.append(("category", clean_val(cat[0]) or cat[0][:40]))
        if len(cat) > 1:
            slots.append(("subcategory", clean_val(cat[1]) or cat[1][:40]))
        slots.append(("priceRange", price_bucket))
        slots += specs
        mr = ", ".join(f"{k}[{v}]" for k, v in slots)
        mentions = any(len(v) >= 3 and v.lower() in ref.lower() for k, v in slots if k != "priceRange")
        rows.append({"pid": r.get("pid", i), "mr": mr, "ref": ref, "n_words": nw,
                     "n_slots": len(slots), "ref_mentions_a_slot_value": mentions})

    out = pd.DataFrame(rows)
    before = len(out)
    out = out.drop_duplicates(subset="ref")
    drop["duplicate target sentence"] += before - len(out)
    out = out.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    n = len(out)
    n_dev = n_test = max(1, n // 10)
    parts = {"test": out[:n_test], "dev": out[n_test:n_test + n_dev], "train": out[n_test + n_dev:]}
    for nm, p in parts.items():
        p[["mr", "ref"]].to_csv(os.path.join(a.out, f"{nm}.csv"), index=False)
    out.to_csv(os.path.join(a.out, "all_with_meta.csv"), index=False)

    ann = out.sample(n=min(a.n_annotate, n), random_state=a.seed)[["mr", "ref"]].copy()
    ann["supported_by_MR"] = ""   # yes = every fact in the sentence can be read off the MR; partial; no
    ann["notes"] = ""
    ann.to_csv(os.path.join(a.out, "annotation_sheet.csv"), index=False)

    log.update({"dropped": dict(drop), "final_rows": n,
                "train": len(parts["train"]), "dev": len(parts["dev"]), "test": len(parts["test"]),
                "avg_target_words": round(float(out["n_words"].mean()), 1),
                "avg_slots_per_MR": round(float(out["n_slots"].mean()), 2),
                "pct_targets_mentioning_an_MR_value": round(100 * float(out["ref_mentions_a_slot_value"].mean()), 1),
                "price_cutoffs": [round(float(q1), 1), round(float(q2), 1)],
                "most_common_spec_slots": keyfreq.most_common(10)})
    json.dump(log, open(os.path.join(a.out, "build_log.json"), "w"), indent=2)
    print(json.dumps(log, indent=2))
    if n:
        print("\nexample:\n ", out.iloc[0]["mr"], "\n  ->", out.iloc[0]["ref"])


if __name__ == "__main__":
    main()
