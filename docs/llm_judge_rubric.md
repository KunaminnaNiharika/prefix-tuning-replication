# LLM-as-judge rubric

Judge: Claude Opus 5.5, used through the Claude app on 2 November 2026, one pass, fixed rubric.
Sample: 40 E2E outputs and 20 ViGGO outputs (random seed 1) plus all 12 Flipkart outputs of the warm-start run.
Ratings: `results/llm_judge_ratings.csv` (one row per output, with a short note for every judgement that is not perfect).

For each item the judge sees the MR and the generated text and records:

| Column | Meaning |
|---|---|
| `fluency` | 1–5. 5 = natural and grammatical; 3 = understandable but awkward or repetitive; 1 = broken or looping |
| `omitted_values` | number of MR values the text does not express (E2E, ViGGO: every slot; Flipkart: type, colour, material, pattern, occasion, brand) |
| `hallucination` | 1 if the text states a fact that is absent from or contradicts the MR (opinion words such as "stylish" do not count), else 0 |
| `repetition` | 1 if a phrase or fact is repeated, else 0 |
| `act_ok` | ViGGO only: does the text fit the dialogue act (questions for request / verify_attribute / suggest / confirm, an opinion for give_opinion, a statement for inform)? |
| `judge_note` | short justification |

Limitations: a single judge in a single pass, not validated against human ratings on these outputs, and not blind to the aggregate metrics.
