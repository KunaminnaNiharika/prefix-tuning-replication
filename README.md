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
| `agreement.py` | Cohen's kappa and confusion matrix between two annotation sheets |
| `score_authors_output.py` | score the authors' code output (`*_test_gold`, `*_test_beam`) with the same scorer as `train_prefix.py` |

## Setup

Python 3.11. Versions used: torch 2.14.1, transformers 5.18.0, peft 0.21.2, datasets 5.0.1.

```
pip install -r requirements.txt
```

On Windows, install into a virtual environment with a short path (for example `python -m venv D:\venv`).
Installing into the Microsoft Store Python can fail with "filename too long".

## Data

| Dataset | Source | Size used |
|---|---|---|
| E2E NLG (original paper) | https://github.com/tuetschek/e2e-dataset | 42,061 training pairs (4,000 used), 630 unique test inputs |
| ViGGO (existing dataset) | Hugging Face `GEM/viggo` (`python src/get_viggo.py`) | 6,900 pairs (5,103 / 714 / 1,083), 9 dialogue acts; 2,000 training pairs and 100 test inputs used |
| Product descriptions (constructed) | Flipkart sample, Hugging Face `kenthua/flipkart-raw-demo` | 1,000 products -> 131 usable pairs |

## Reproducing the runs

Every run's exact arguments are also stored in `results/<run>/results.json`.

```
pip install -r requirements.txt
git clone https://github.com/tuetschek/e2e-dataset data/e2e
python src/get_viggo.py                      # writes data/viggo/{train,dev,test}.csv
```

E2E (about 1 hour on a laptop CPU), then score the full test set from the saved prefix:
```
python src/train_prefix.py --train_csv data/e2e/trainset.csv --test_csv data/e2e/testset_w_refs.csv --n_train 4000 --epochs 2 --lr 0.03 --prefix_len 10 --n_eval 200 --out_dir results/e2e_short
python src/train_prefix.py --train_csv data/e2e/trainset.csv --test_csv data/e2e/testset_w_refs.csv --init_adapter results/e2e_short/adapter --epochs 0 --n_train 8 --n_eval 630 --out_dir results/e2e_full
```
Decoding ablation (same saved prefix, beam search instead of greedy):
```
python src/train_prefix.py --train_csv data/e2e/trainset.csv --test_csv data/e2e/testset_w_refs.csv --init_adapter results/e2e_short/adapter --epochs 0 --n_train 8 --n_eval 630 --beams 5 --out_dir results/e2e_beam5
```
ViGGO (from scratch, warm-started from the E2E prefix, and warm-started with the dialogue act):
```
python src/train_prefix.py --train_csv data/viggo/train.csv --test_csv data/viggo/test.csv --n_train 2000 --epochs 2 --lr 0.03 --max_len 192 --n_eval 100 --cover_slots name,release_year,developer --out_dir results/viggo_cold
python src/train_prefix.py --train_csv data/viggo/train.csv --test_csv data/viggo/test.csv --n_train 2000 --epochs 2 --lr 0.03 --max_len 192 --n_eval 100 --cover_slots name,release_year,developer --init_adapter results/e2e_short/adapter --out_dir results/viggo_warm
python src/train_prefix.py --train_csv data/viggo/train.csv --test_csv data/viggo/test.csv --n_train 2000 --epochs 2 --lr 0.03 --max_len 192 --n_eval 100 --cover_slots name,release_year,developer --init_adapter results/e2e_short/adapter --keep_act --out_dir results/viggo_warm_act
```
Flipkart (build, filter by annotation, train cold and warm, decode with the 3-gram constraint):
```
python src/build_flipkart.py --csv data/raw/flipkart.csv --out data/newdata --n_annotate 100
python src/filter_by_annotation.py --dir data/newdata --out data/newdata_clean
python src/train_prefix.py --train_csv data/newdata_clean/train.csv --test_csv data/newdata_clean/test.csv --n_train 2000 --epochs 20 --lr 0.03 --max_len 192 --n_eval 13 --cover_slots name,brand,color,fabric,material,pattern,type --out_dir results/newdata_cold
python src/train_prefix.py --train_csv data/newdata_clean/train.csv --test_csv data/newdata_clean/test.csv --n_train 2000 --epochs 20 --lr 0.03 --max_len 192 --n_eval 13 --cover_slots name,brand,color,fabric,material,pattern,type --init_adapter results/e2e_short/adapter --out_dir results/newdata_warm
python src/train_prefix.py --train_csv data/newdata_clean/train.csv --test_csv data/newdata_clean/test.csv --init_adapter results/newdata_warm/adapter --epochs 0 --n_train 8 --n_eval 13 --no_repeat_ngram 3 --cover_slots name,brand,color,fabric,material,pattern,type --out_dir results/newdata_warm_nr3g
```
(`data/raw/flipkart.csv` is `flipkart_com-ecommerce_sample.csv` from https://huggingface.co/datasets/kenthua/flipkart-raw-demo, CC BY-SA 4.0.)

Analysis and agreement:
```
python src/slot_check.py results/e2e_full/generations.csv
python src/repetition_check.py results/newdata_cold results/newdata_warm results/newdata_warm_nr3g
python src/act_check.py results/viggo_cold results/viggo_warm results/viggo_warm_act
python src/agreement_bootstrap.py data/newdata/annotation_sheet.csv data/newdata/rater2_batch1.csv data/newdata/rater2_batch2.csv
```
Official E2E scorer on the saved greedy outputs (needs Java and the Perl modules XML::Twig and Sort::Naturally; see the script's docstring):
```
git clone https://github.com/tuetschek/e2e-metrics
pip install -r e2e-metrics/requirements.txt
python src/official_e2e_scores.py results/e2e_full/generations.csv data/e2e/testset_w_refs.csv e2e-metrics results/e2e_full/official_e2e_metrics.json
```
Authors' code (Colab, T4 GPU): run `colab/original_code_attempt.ipynb` (`colab/original_code_run_with_outputs.ipynb` is the executed run, with the training log, timings and scores); it ends by scoring the output with
`python src/score_authors_output.py <name>_test_gold <name>_test_beam`.

## Results

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

### Inter-annotator agreement (Flipkart)

A second annotator labelled all 100 annotated pairs independently, in two batches of 50
(`data/newdata/rater2_batch1.csv`, `data/newdata/rater2_batch2.csv`), with only the label names as instruction:

```
python src/agreement_bootstrap.py data/newdata/annotation_sheet.csv data/newdata/rater2_batch1.csv data/newdata/rater2_batch2.csv
```

| Pairs | n | Raw agreement | Cohen's kappa | 95% bootstrap CI |
|---|---|---|---|---|
| First batch | 50 | 32% | 0.03 | -0.13 to 0.19 |
| Second batch | 50 | 28% | 0.06 | -0.07 to 0.18 |
| All | 100 | 30% | 0.03 | -0.07 to 0.13 |

Agreement is slight, and the disagreement changed direction between batches (first batch: many of my *partial* pairs
labelled *no*; second batch: labelled *yes*). The boundary of *partial* was undefined. Guidelines were written
afterwards (`docs/annotation_guidelines.md`). Both annotators then used them to label the 31 pairs that had never
been annotated (`data/newdata/guideline_round_annotator1.csv`, `data/newdata/guideline_round_annotator2.csv`):

```
python src/agreement_bootstrap.py data/newdata/guideline_round_annotator1.csv data/newdata/guideline_round_annotator2.csv
```

| Pairs | n | Raw agreement | Cohen's kappa | 95% bootstrap CI |
|---|---|---|---|---|
| With guidelines | 31 | 97% | 0.65 | 0.00 to 1.00 |

This is not evidence that the guidelines work. 30 and 29 of the 31 labels are *no*, so kappa rests on one or two
pairs and its interval spans 0 to 1. The labels also contradict the guidelines: step 2 says a description about the
right product that adds an unsupported fact is *partial*, and in at least 20 pairs the annotators' own notes describe
exactly that, yet the pairs were labelled *no*. The guidelines, as worded, did not fix the *partial* / *no* boundary.

How the two sheets were produced: `guideline_round_annotator1.csv` was filled in by me (annotator 1) and
`guideline_round_annotator2.csv` by the second annotator. Both annotators report that they labelled separately and by
hand, using `docs/annotation_guidelines.md`. The notes in the two sheets are closely similar in wording; this overlap is
unexplained and could not be verified further (report, Section 4.3). The repository history for these two files starts
when they were uploaded on 7 November 2026; it does not show when or how they were filled in. The original files, with
their saved dates, are kept by each annotator. After upload, the quote characters in both files were repaired after an
encoding mismatch (see the commit history); no labels or notes were changed. These 31 labels were not used to
filter the data: the 31 pairs stay in the experiments (26 train, 3 dev, 2 test).

### LLM-as-judge

72 outputs (40 E2E, 20 ViGGO, 12 Flipkart) rated for fluency, omissions, hallucination, repetition and dialogue-act fit;
rubric in `docs/llm_judge_rubric.md`, ratings in `results/llm_judge_ratings.csv`.

| Data | Fluency (1-5) | Omitted values | Hallucination | Repetition | Fully faithful |
|---|---|---|---|---|---|
| E2E (40) | 4.55 | 1.62 | 12% | 12% | 8% |
| ViGGO (20) | 3.90 | 1.05 | 40% | 15% | 25% |
| Flipkart (12) | 2.58 | 2.08 | 58% | 75% | 0% |

## Replication with the authors' original code (Google Colab)

`colab/original_code_attempt.ipynb` runs https://github.com/XiangLi1999/PrefixTuning (branch `cleaned`) on a Colab T4 GPU. The executed copy with all cell outputs is `colab/original_code_run_with_outputs.ipynb`: it shows 42,061 training examples, 8,413 steps, training from 10:01 to 10:33 on 2 November 2026, and the final scores (BLEU 67.39, NIST 7.24, ROUGE-L 68.82, slot coverage 98.27%).
Fixes that were needed:

1. Colab's Python 3.13 cannot build `tokenizers==0.8.1rc2` (needed by the bundled `transformers==3.2.0`): Python 3.8 environment with `uv`, plus `six`.
2. `train_e2e.py` starts `python run_language_modeling.py` with `os.system`, which picked up Colab's Python (`ImportError: cannot import name 'cached_path'`): call the 3.8 Python directly.
3. Hard-coded paths on the authors' servers (`/u/scr/xlisali/...`): GPT-2 medium downloaded from Hugging Face, paths patched with `sed`.
4. `gen.py` reads its settings from the model folder's name (`NameError: name 'tuning_mode' is not defined`, prefix length fixed at 20): call `run_generation.py` directly.
5. `run_generation.py` also reads the prefix format from the folder name (`AssertionError: ... need to specify a valid prefix mode`): link the model folder as `e2e_p10_e1_act_cat`.

Result (GPT-2 medium, the paper's general defaults: prefix length 10, batch 5, lr 5e-5; 1 epoch = 8,413 steps in 31 minutes on a T4, beam 5, 630 test MRs,
scored with the same scorer as my `peft` runs; `results/authors_code/scores.json`):

| System | BLEU | NIST | ROUGE-L | Slot coverage |
|---|---|---|---|---|
| Paper, GPT-2 medium prefix (5 epochs, batch 10, lr 8e-5, prefix 5; official scorer) | 69.7 | 8.81 | 71.4 | – |
| Authors' code, GPT-2 medium, 1 epoch (my scorer) | 67.4 | 7.24 | 68.8 | 98.3% |
| My `peft` run, GPT-2 small, 4,000 pairs, 2 epochs (my scorer) | 57.1 | 5.17 | 59.4 | 78.1% |
| Same `peft` outputs, official E2E scorer | 56.3 | 6.92 | 59.9 | – |

The authors' model mentions the area in 97% of outputs (my `peft` model: 23%).

### Run folders

Reported results: `e2e_full`, `e2e_eval`, `viggo_warm`, `viggo_cold`, `viggo_warm_act`, `newdata_warm`, `newdata_cold`, `newdata_warm_nr3g`.
Development runs kept for the record: `smoke`, `lr_1e-2`, `lr_3e-2`, `e2e_main`, `e2e_short`, `newdata_run`, `newdata_warm_nr3`.

**Which NR3 run is in the report.** Table 6's "E2E + NR3" row (BLEU 4.1, coverage 40.9%) is `newdata_warm_nr3g`, which uses the `NoRepeatGenerated` processor in `src/train_prefix.py`: no 3-gram may repeat *within the generated text*. `newdata_warm_nr3` (BLEU 5.4, coverage 25.0%) is an earlier, superseded attempt that used transformers' built-in `no_repeat_ngram_size=3`. That option also counts the prompt, so it stops the model copying multi-word values (product names, brands) out of the MR, which is why its slot coverage drops to 25%. It is not reported and cannot be re-created with the current script.

## Data sources and limitations

- The Flipkart data is a third-party sample, used here for coursework; its text is not my own.
- ViGGO's dialogue-act wrapper (for example `inform(...)`) is dropped from the model input except in the `--keep_act` run.
- Scores on 12 test sentences are indicative only.
