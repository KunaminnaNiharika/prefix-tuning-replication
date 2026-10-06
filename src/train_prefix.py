"""
Prefix-tuning GPT-2 on meaning-representation (MR) -> text data.
Works on any CSV with columns `mr` and `ref` (E2E, ViGGO, or your new dataset).

Example (CPU-friendly subset run):
  python src/train_prefix.py --train_csv data/e2e/trainset.csv \
      --test_csv data/e2e/testset_w_refs.csv --n_train 2000 --epochs 2 \
      --n_eval 100 --prefix_len 10 --out_dir results/e2e_run1
"""
import argparse, json, os, random, re, time
from collections import OrderedDict

import pandas as pd
import torch

SEP = " ==>"


KEEP_ACT = False   # set by --keep_act: put the dialogue act (inform, confirm, ...) into the input


def linearize(mr):
    """'name[The Eagle], food[Chinese]' -> 'name : The Eagle | food : Chinese'
    With KEEP_ACT, 'inform(name[X], ...)' -> 'act : inform | name : X | ...'  (MRs without an act are unchanged)"""
    pairs = re.findall(r"([A-Za-z_ ]+?)\[([^\]]*)\]", mr)
    out = " | ".join(f"{k.strip()} : {v.strip()}" for k, v in pairs)
    if KEEP_ACT:
        m = re.match(r"\s*([a-z_]+)\(", mr)
        if m:
            out = f"act : {m.group(1)}" + (f" | {out}" if out else "")
    return out


def parse_slots(mr):
    return [(k.strip(), v.strip()) for k, v in re.findall(r"([A-Za-z_ ]+?)\[([^\]]*)\]", mr)]


def encode(tok, text):
    return tok(text)["input_ids"]


def build_examples(df, tok, max_len):
    examples = []
    for mr, ref in zip(df["mr"], df["ref"]):
        src = encode(tok, linearize(mr) + SEP)
        tgt = encode(tok, " " + str(ref).strip()) + [tok.eos_token_id]
        ids = (src + tgt)[:max_len]
        labels = ([-100] * len(src) + tgt)[:max_len]  # loss only on the target sentence
        examples.append((ids, labels))
    return examples


def collate(batch, pad_id):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for r, (i, l) in enumerate(batch):
        ids[r, : len(i)] = torch.tensor(i)
        lab[r, : len(l)] = torch.tensor(l)
        att[r, : len(i)] = 1
    return ids, att, lab


def train(model, examples, pad_id, epochs, bs, lr, log_every=20, log=print, device="cpu"):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    total_steps = epochs * ((len(examples) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, total_steps)))
    model.train()
    step, history = 0, []
    t0 = time.time()
    for ep in range(epochs):
        random.shuffle(examples)
        for b in range(0, len(examples), bs):
            ids, att, lab = collate(examples[b : b + bs], pad_id)
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            sched.step()
            step += 1
            history.append(out.loss.item())
            if step % log_every == 0 or step == 1:
                log(f"epoch {ep+1} step {step}/{total_steps} loss {out.loss.item():.4f} "
                    f"({time.time()-t0:.0f}s elapsed)")
    return history


from transformers import LogitsProcessor, LogitsProcessorList


class NoRepeatGenerated(LogitsProcessor):
    """Forbid repeating any n-word sequence INSIDE THE GENERATED TEXT ONLY.
    (transformers' built-in no_repeat_ngram_size also counts the prompt, which would forbid copying
    multi-word values such as product names out of the MR.)"""
    def __init__(self, n, prompt_len):
        self.n, self.prompt_len = n, prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if self.n < 2 or gen.shape[1] < self.n - 1:
            return scores
        for b in range(gen.shape[0]):
            toks = gen[b].tolist()
            prefix = tuple(toks[-(self.n - 1):])
            banned = {toks[i + self.n - 1] for i in range(len(toks) - self.n + 1)
                      if tuple(toks[i:i + self.n - 1]) == prefix}
            if banned:
                scores[b, list(banned)] = -float("inf")
        return scores


@torch.no_grad()
def generate_all(model, tok, mrs, pad_id, bs=8, max_new_tokens=60, num_beams=1, max_src=100, device="cpu", no_repeat_ngram=0):
    model.eval()
    outs = []
    for b in range(0, len(mrs), bs):
        chunk = [encode(tok, linearize(m) + SEP)[:max_src] for m in mrs[b : b + bs]]
        L = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, c in enumerate(chunk):  # LEFT-pad for batched generation
            ids[r, L - len(c):] = torch.tensor(c)
            att[r, L - len(c):] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new_tokens,
                             num_beams=num_beams, do_sample=False, pad_token_id=pad_id,
                             logits_processor=LogitsProcessorList([NoRepeatGenerated(no_repeat_ngram, L)]) if no_repeat_ngram > 1 else None,
                             eos_token_id=tok.eos_token_id)
        for row in gen:
            text = tok.decode(row[L:].tolist(), skip_special_tokens=True)
            outs.append(text.strip().split("\n")[0])
    return outs


def _tok(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())


def score(hyps, refs_list, mrs, cover_slots=("name", "near", "food", "eattype", "area", "pricerange")):
    """BLEU / NIST / ROUGE-L / METEOR (if wordnet available) + slot coverage."""
    from nltk.translate.bleu_score import corpus_bleu
    res = {}
    H = [_tok(h) for h in hyps]
    R = [[_tok(r) for r in refs] for refs in refs_list]
    res["BLEU"] = 100 * corpus_bleu(R, H)
    from nltk.translate.bleu_score import SmoothingFunction
    res["BLEU_smoothed"] = 100 * corpus_bleu(R, H, smoothing_function=SmoothingFunction().method1)
    try:
        from nltk.translate.nist_score import corpus_nist
        res["NIST"] = corpus_nist(R, H, n=5)
    except Exception as e:
        res["NIST"] = f"failed: {e}"
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"])
    res["ROUGE_L"] = 100 * sum(max(sc.score(r, h)["rougeL"].fmeasure for r in refs)
                               for h, refs in zip(hyps, refs_list)) / len(hyps)
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        from nltk.translate.meteor_score import meteor_score
        res["METEOR"] = 100 * sum(meteor_score(rt, ht) for ht, rt in zip(H, R)) / len(H)
    except Exception as e:
        res["METEOR"] = f"skipped: {type(e).__name__}"
    # simple faithfulness proxy: share of MR slot values that appear verbatim in the output
    found = total = 0
    for h, mr in zip(hyps, mrs):
        for k, v in parse_slots(mr):
            if k.lower() in cover_slots:
                total += 1
                found += v.lower() in h.lower()
    res["slot_value_coverage_%"] = 100 * found / max(1, total)
    return res


def slot_report(hyps, mrs, cover):
    """Per-slot verbatim coverage, plus a 'missing slot values' string for every output."""
    per, missing_rows = {}, []
    for h, mr in zip(hyps, mrs):
        missing = []
        for k, v in parse_slots(mr):
            if k.lower() in cover:
                f, t = per.get(k.lower(), (0, 0))
                ok = v.lower() in h.lower()
                per[k.lower()] = (f + int(ok), t + 1)
                if not ok:
                    missing.append(f"{k}[{v}]")
        missing_rows.append("; ".join(missing))
    return per, missing_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--prefix_len", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--no_repeat_ngram", type=int, default=0,
                    help="forbid repeating any n-token sequence within the GENERATED text (e.g. 3); copying from the MR stays allowed; 0 = off")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_act", action="store_true",
                    help="include the dialogue act (inform, confirm, request ...) in the input (ViGGO)")
    ap.add_argument("--init_adapter", default=None,
                    help="folder of a saved prefix (e.g. results/e2e_short/adapter) to continue training from")
    ap.add_argument("--cover_slots", default="name,near,food,eattype,area,pricerange",
                    help="comma-separated slot names whose values should appear verbatim in the output")
    a = ap.parse_args()

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from peft import get_peft_model, PrefixTuningConfig, TaskType, PeftModel

    random.seed(a.seed); torch.manual_seed(a.seed)
    globals()["KEEP_ACT"] = a.keep_act
    os.makedirs(a.out_dir, exist_ok=True)
    logf = open(os.path.join(a.out_dir, "log.txt"), "w")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    tok = GPT2TokenizerFast.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    base = GPT2LMHeadModel.from_pretrained(a.model)
    if a.init_adapter:
        # peft refuses to load a prompt-learning adapter as trainable, so: build a fresh prefix model with the
        # same settings, then copy the saved prefix weights into it.
        from peft import PeftConfig, set_peft_model_state_dict
        from peft.utils import load_peft_weights
        saved = PeftConfig.from_pretrained(a.init_adapter)
        model = get_peft_model(base, PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                                        num_virtual_tokens=saved.num_virtual_tokens))
        set_peft_model_state_dict(model, load_peft_weights(a.init_adapter))
        print(f"warm-started from {a.init_adapter} (prefix length {saved.num_virtual_tokens})")
    else:
        model = get_peft_model(base, PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                                        num_virtual_tokens=a.prefix_len))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no GPU found - training will be slow)"))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"trainable {n_tr:,} / {n_all:,} = {100*n_tr/n_all:.4f}%")

    train_df = pd.read_csv(a.train_csv).dropna(subset=["mr", "ref"]).sample(frac=1, random_state=a.seed)
    train_df = train_df.head(a.n_train)
    log(f"training on {len(train_df)} (MR, reference) pairs for {a.epochs} epochs")
    hist = train(model, build_examples(train_df, tok, a.max_len), tok.eos_token_id,
                 a.epochs, a.bs, a.lr, log=log, device=device)

    test_df = pd.read_csv(a.test_csv).dropna(subset=["mr", "ref"])
    grouped = OrderedDict()
    for mr, ref in zip(test_df["mr"], test_df["ref"]):
        grouped.setdefault(mr, []).append(str(ref))
    mrs = list(grouped.keys())[: a.n_eval]
    refs = [grouped[m] for m in mrs]
    log(f"generating for {len(mrs)} unique test MRs (beams={a.beams})")
    hyps = generate_all(model, tok, mrs, tok.eos_token_id, num_beams=a.beams, device=device, no_repeat_ngram=a.no_repeat_ngram)

    cover = tuple(x.strip().lower() for x in a.cover_slots.split(","))
    metrics = score(hyps, refs, mrs, cover)
    per_slot, missing_rows = slot_report(hyps, mrs, cover)
    per_slot_pct = {k: f"{100*f/t:.0f}% ({f}/{t})" for k, (f, t) in per_slot.items()}
    log("COVERAGE BY SLOT: " + json.dumps(per_slot_pct))
    pd.DataFrame({"mr": mrs, "generated": hyps, "reference_1": [r[0] for r in refs],
                  "n_references": [len(r) for r in refs], "missing_slot_values": missing_rows}
                 ).to_csv(os.path.join(a.out_dir, "generations.csv"), index=False, encoding="utf-8-sig")
    log("METRICS: " + json.dumps(metrics, indent=2))
    model.save_pretrained(os.path.join(a.out_dir, "adapter"))
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(a), "trainable": n_tr, "total": n_all, "loss_history": hist,
                   "metrics": metrics, "coverage_by_slot": per_slot_pct,
                   "samples": [{"mr": m, "generated": h, "references": r[:2]}
                               for m, h, r in list(zip(mrs, hyps, refs))[:15]]}, f, indent=2)
    log(f"saved to {a.out_dir}")


if __name__ == "__main__":
    main()
