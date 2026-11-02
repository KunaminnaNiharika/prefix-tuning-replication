"""
Score the E2E output of the authors' code (run_generation.py) with the SAME scorer as src/train_prefix.py,
so the authors'-code run and my peft run can be compared directly.

The authors' code writes two files:
  <name>_test_gold  : the human references, one per line, a blank line after each MR's group
  <name>_test_beam  : one generated sentence per MR, in the same order

Usage:
  python src/score_authors_output.py /content/gen_out/e2e_results/short_run_test_gold \
                                     /content/gen_out/e2e_results/short_run_test_beam
"""
import json
import re
import sys


def read_gold(path):
    groups, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cur.append(line)
            elif cur:
                groups.append(cur)
                cur = []
    if cur:
        groups.append(cur)
    return groups


def tok(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())


def main(gold_path, out_path):
    refs_list = read_gold(gold_path)
    with open(out_path, encoding="utf-8") as f:
        hyps = [l.strip() for l in f]
    hyps = [h for h in hyps if h != ""][: len(refs_list)]
    assert len(hyps) == len(refs_list), f"{len(hyps)} outputs vs {len(refs_list)} reference groups"

    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.nist_score import corpus_nist
    from rouge_score import rouge_scorer
    import nltk
    nltk.download("wordnet", quiet=True)
    from nltk.translate.meteor_score import meteor_score

    H = [tok(h) for h in hyps]
    R = [[tok(r) for r in refs] for refs in refs_list]
    sc = rouge_scorer.RougeScorer(["rougeL"])
    res = {
        "n_MRs": len(hyps),
        "BLEU": 100 * corpus_bleu(R, H),
        "NIST": corpus_nist(R, H, n=5),
        "METEOR": 100 * sum(meteor_score(rt, ht) for ht, rt in zip(H, R)) / len(H),
        "ROUGE_L": 100 * sum(max(sc.score(r, h)["rougeL"].fmeasure for r in refs)
                             for h, refs in zip(hyps, refs_list)) / len(hyps),
        "n_Error_outputs": sum(h == "Error" for h in hyps),
    }
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
