"""
Download ViGGO from the Hugging Face auto-converted (parquet) branch and save data/viggo/{train,dev,test}.csv
with columns mr, ref  (the format train_prefix.py expects).
Usage:  python src/get_viggo.py
"""
import os, sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO, REV, OUT = "GEM/viggo", "refs/convert/parquet", "data/viggo"
WANT = {"train": "train", "validation": "dev", "test": "test"}


def convert(local_files):
    """local_files: {split_name: [parquet paths]} -> writes CSVs, returns {name: dataframe}"""
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for split, paths in local_files.items():
        df = pd.concat([pd.read_parquet(p) for p in sorted(paths)], ignore_index=True)
        print(f"  {split:11s} columns: {list(df.columns)}")
        df = df.rename(columns={"meaning_representation": "mr", "target": "ref"})[["mr", "ref"]].dropna()
        df.to_csv(os.path.join(OUT, f"{WANT[split]}.csv"), index=False, encoding="utf-8")
        out[WANT[split]] = df
    return out


def main():
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
    parquet = [f for f in files if f.endswith(".parquet")]
    print(f"{len(parquet)} parquet files on branch {REV}; first few: {parquet[:6]}")
    by_split = {}
    for f in parquet:
        parts = f.split("/")
        split = parts[-2] if len(parts) >= 2 else ""
        if split in WANT:
            by_split.setdefault(split, []).append(hf_hub_download(REPO, f, repo_type="dataset", revision=REV))
    if not by_split:
        sys.exit("No train/validation/test parquet files found. Paste me the 'first few' list above, "
                 "or use the form on https://nlds.soe.ucsc.edu/viggo")
    data = convert(by_split)
    print("\nrows:", {k: len(v) for k, v in data.items()})
    print("unique MRs:", {k: int(v["mr"].nunique()) for k, v in data.items()})
    tr = data["train"]
    print("\nexample:\n  MR :", tr.iloc[0]["mr"], "\n  REF:", tr.iloc[0]["ref"])
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
