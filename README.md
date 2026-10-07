# COMP8240 Novel Project: Replicating and Extending Prefix-Tuning

Replication and extension of Li & Liang (2021)'s Prefix-Tuning method for
COMP8240 Applications of Data Science, Macquarie University (2026 Session 2).

## Paper

Xiang Lisa Li and Percy Liang. 2021. **Prefix-Tuning: Optimizing Continuous
Prompts for Generation**. In *Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing (ACL-IJCNLP 2021)*, pages 4582–4597.

- Paper: https://aclanthology.org/2021.acl-long.353/
- arXiv: https://arxiv.org/abs/2101.00190
- Official code: https://github.com/XiangLi1999/PrefixTuning

## Overview

Prefix-tuning freezes a pretrained language model and learns only a short
sequence of continuous, task-specific vectors (a "prefix") prepended at
every transformer layer, updating roughly 0.1% of parameters instead of
fine-tuning the full model. This project replicates the paper's table-to-text
setting on the E2E NLG dataset, then applies the method to two further datasets:
ViGGO (an existing dataset) and a product-description dataset built from
Flipkart listings.

All runs use GPT-2 small (124M parameters) with a 10-token prefix
(184,320 trainable parameters = 0.148%), through the Hugging Face `peft` library
(`PrefixTuningConfig`), on a Windows laptop CPU (no GPU).

## Repository structure

```
src/        scripts (training, evaluation, dataset building, analysis)
results/    one folder per run: log.txt, results.json, generations.csv, adapter/ (saved prefix)
data/       newdata/ and newdata_clean/: the Flipkart-based dataset (raw E2E, ViGGO and Flipkart files are not stored here)
prefix_adapter/   output of the first feasibility check (two toy examples)
```

| Script | Purpose |
|---|---|
| `train_prefix.py` | train a prefix and/or generate and score (BLEU, NIST, ROUGE-L, METEOR, slot coverage) |
| `build_flipkart.py` | build the product-spec-to-blurb dataset from the Flipkart sample |
| `filter_by_annotation.py` | remove rows annotated "no" from the splits |
| `get_viggo.py` | download ViGGO from Hugging Face and save it as CSV |
| `slot_check.py` | how often each slot value appears in the model's sentence vs. the human reference |
| `compare_runs.py` | metrics table and side-by-side sentences for several runs |
| `repetition_check.py` | count repetition loops and cut-off outputs |
| `act_check.py` | ViGGO: question vs. statement by dialogue act, and invented ESRB ratings |
| `peek_flipkart.py` | print statistics and sample rows of the Flipkart file |
| `build_dataset.py` | written for a first Kaggle file that turned out to contain no text; kept for the record |

## Setup

Python 3.11. Versions used: torch 2.14.1, transformers 5.18.0, peft 0.21.2, datasets 5.0.1.

```
pip install torch transformers peft datasets pandas nltk rouge-score
```

On Windows, install into a virtual environment with a short path (for example `python -m venv D:\venv`).
Installing into the Microsoft Store Python can fail with "filename too long".

## Data

| Dataset | Source | Size used |
|---|---|---|
| E2E NLG (original paper) | https://github.com/tuetschek/e2e-dataset | 42,061 training pairs (4,000 used), 630 unique test inputs |
| ViGGO (existing dataset) | Hugging Face `GEM/viggo` (`python src/get_viggo.py`) | 6,900 pairs (5,103 / 714 / 1,083), 9 dialogue acts; 2,000 training pairs and 100 test inputs used |
| Product descriptions (constructed) | Flipkart sample, Hugging Face `kenthua/flipkart-raw-demo` | 1,000 products -> 131 usable pairs |

## Reproducing the main runs

E2E (about 1 hour on a laptop CPU):
```
python src\train_prefix.py --train_csv data\e2e\trainset.csv --test_csv data\e2e\testset_w_refs.csv --n_train 4000 --epochs 2 --n_eval 200 --lr 0.03 --prefix_len 10 --out_dir results\e2e_short
```
Score the full test set from the saved prefix (no retraining):
```
python src\train_prefix.py --train_csv data\e2e\trainset.csv --test_csv data\e2e\testset_w_refs.csv --init_adapter results\e2e_short\adapter --epochs 0 --n_train 8 --n_eval 630 --out_dir results\e2e_full
```
Build the Flipkart dataset, then filter by annotation:
```
python src\build_flipkart.py --csv data\raw\flipkart.csv --out data\newdata --n_annotate 100
python src\filter_by_annotation.py --dir data\newdata --out data\newdata_clean
```

## Status: results for the Update Presentation (Oct 2026)

| Dataset | Setup | BLEU | ROUGE-L | METEOR | Slot coverage |
|---|---|---|---|---|---|
| E2E (all 630 unique test MRs) | 4,000 pairs, 2 epochs | 57.1 | 59.4 | 64.9 | 78.1% |
| ViGGO (100 test MRs) | from E2E prefix | 38.3 | 47.4 | 53.5 | 91.2% |
| ViGGO | from scratch | 32.7 | 44.9 | 48.4 | 76.2% |
| ViGGO | from E2E prefix + dialogue act | 37.0 | 49.2 | 53.3 | 82.3% |
| Flipkart (12 test items) | from E2E prefix | 3.8 | 16.5 | 19.2 | 40.9% |
| Flipkart | from scratch | 0.7 | 12.7 | 9.1 | 11.4% |

Notes:
- A first E2E evaluation on the first 200 test inputs gave BLEU 55.7, ROUGE-L 61.0, METEOR 66.1, coverage 80.9%.
  Those 200 inputs are not a representative sample (10 of 18 restaurant names), so the full-set row is the one to use.
- Scoring uses NLTK with simple tokenisation, so METEOR and NIST are not comparable with the paper's official scorer.
  Differences from the paper: GPT-2 small, 4,000 of 42,061 training pairs, greedy decoding, no reparameterisation MLP.
- Slot coverage = share of MR slot values that appear word for word in the output. The human reference sentences are checked the same way for a baseline.
- Each result is a single run; the Flipkart test set has only 12 inputs.

### Findings

- **E2E**: the model copies name, type and landmark 100% of the time, but mentions the area in only 23% of sentences (human references: 78%). Re-running from the saved prefix reproduced the earlier numbers exactly.
- **ViGGO**: starting from the E2E prefix gave higher slot coverage than starting from scratch (release year 97% vs 60%, developer 77% vs 40%) and fewer invented ESRB ratings (6 vs 21 of 78 inputs without a rating).
  Giving the model the dialogue act raised the rate of questions where the human asks one (54% to 67%) but lowered slot coverage (91% to 82%).
- **Flipkart**: only 131 of 1,000 products gave a usable pair (596 descriptions were sales boilerplate, 190 were spec dumps).
  100 pairs were annotated for whether the sentence is supported by its meaning representation: 25 yes / 66 partial / 9 no.
  Outputs often contain invented details and repetition loops; a constraint on repeated phrases within the generated text removed the loops (loops 2 -> 0 of 12) without affecting copying from the input.
  The first version of that constraint (the built-in Hugging Face option) also blocked copying product names (name coverage 42% -> 0%).

### Run folders

Reported results: `e2e_full`, `e2e_eval`, `viggo_warm`, `viggo_cold`, `viggo_warm_act`, `newdata_warm`, `newdata_cold`, `newdata_warm_nr3`, `newdata_warm_nr3g`.
Development runs kept for the record: `smoke`, `lr_1e-2`, `lr_3e-2`, `e2e_main`, `e2e_short`, `newdata_run`.

## Data sources and limitations

- The Flipkart data is a third-party sample, used here for coursework; its text is not my own.
- ViGGO's dialogue-act wrapper (for example `inform(...)`) is dropped from the model input except in the `--keep_act` run.
- Scores on 12 test sentences are indicative only.
