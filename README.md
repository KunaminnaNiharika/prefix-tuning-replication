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
results on the E2E NLG dataset, then extends the method to new datasets and
a newly constructed dataset not used in the original paper.

## Repository structure
