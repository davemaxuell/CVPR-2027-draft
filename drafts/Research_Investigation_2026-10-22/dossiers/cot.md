# CoT Prompting

Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

Jason Wei; Xuezhi Wang; Dale Schuurmans; Maarten Bosma; Brian Ichter; Fei Xia; Ed H. Chi; Quoc V. Le; Denny Zhou. NeurIPS 2022. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [CoT Prompting.pdf](../../Paper/CoT%20Prompting.pdf). Pages reviewed: ['2–9', '15–22', '25–43'].

## Problem and operating conditions

Improve multi-step arithmetic, commonsense, and symbolic problem solving with a pretrained language model and a few demonstrations, without task fine-tuning. This is foundational prompting research, not a VLN architecture.

## Mechanism, training, and inference

Replace input/answer exemplars by input/intermediate natural-language rationale/answer triples. The model autoregressively generates intermediate tokens before its answer. Eight manually composed arithmetic demonstrations are reused across free-response datasets; AQuA uses four multiple-choice examples. Commonsense tasks use task-specific training examples, or hold out the first ten examples of evaluation-only BIG-bench tasks for prompts. Greedy decoding is used; LaMDA averages five exemplar-order seeds, while other model results use one ordering. No loss or training update is introduced. Ablations test equation-only output, dots as extra computation, and explanations after the answer. Symbolic tasks test two-step exemplars followed by longer inputs. SayCan is evaluated as language-to-action sequence generation rather than physical execution in this paper.

## Novelty and closest predecessors

Combines pretrained in-context learning with intermediate natural-language rationales, contrasting earlier rationale-supervised training and scratchpad/program approaches. It does not establish that a verbal rationale faithfully reports all internal computation or that longer text always improves a decision.

## Results and independent verdict

PAPER EXPERIMENT: Figure 2 PaLM 540B GSM8K solve rate roughly 18→57%; StrategyQA reaches 75.6 versus prior best 69.4. Benefits vary by scale and task; one-operation arithmetic gains are small or negative, and small models can produce fluent incorrect traces. The authors’ manual analysis includes correct final answers with invalid rationales; of 50 errors, 46% were described as nearly correct and 54% as major semantic/coherence errors. Appendix A explicitly notes parameter count is confounded with training compute. Dot and post-answer controls support usefulness of intermediate content in these setups, but do not isolate all possible extra-compute mechanisms. §6 explicitly disclaims guaranteed correct reasoning and leaves whether the network reasons as an open question.

APPENDIX CHECK: Table 7 includes negative transfers for the evaluated GPT-3 model on CSQA and StrategyQA; prompting benefits are not universal. Tables 13–19 exhibit symbol, commonsense and feasibility errors even when explanations sound coherent. The SayCan examples assume a supplied finite object/location/action inventory, and do not constitute robot execution. The supplied appendix’s correct-rationale count and the main-text count differ; this dossier uses their shared qualitative finding instead of choosing an unsupported corrected rate.

## Implementation and reproducibility

The full supplied appendix provides exact task prompts and worked outputs (pp. 30–43), and the official NeurIPS page links a 29-page supplementary PDF covering the same appendix topics. The official page exposes no executable supplementary archive in the checked links; the paper states that an input/target/output ZIP was supplied, but that archive was not located through the proceedings page and bounded title/author search. No official historical inference implementation was established. LaMDA and PaLM are proprietary, and historical GPT-3 endpoints are not reproduced. The prompt recipe is reviewable; original model pretraining and exact API execution are not independently reproducible here.


## Specific bottlenecks

**cot-faithfulness: A correct final answer does not guarantee a correct or faithful intermediate rationale.**

Conditions: Fluent semantic errors, lucky answers, or explanations that fail to reflect the causal computation. Cause: Autoregressive text is not a verified derivation and no correctness constraint is imposed. Evidence: AUTHOR ACKNOWLEDGEMENT: §6 and Appendix D.1; correct-answer cases with flawed rationales. Main and appendix counts differ, so no exact rate is adopted.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['VSI-Bench', 'VLN-MME', 'OmniAgent']

**cot-transfer: Reasoning-prompt gains do not automatically transfer to grounded sequential navigation.**

Conditions: An agent must collect observations, interpret geometry, and execute actions over time. Cause: Original evidence is primarily text reasoning with fixed inputs. Evidence: DEDUCTION from task scope.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['VLN-MME', 'VSI-Bench', 'UoT']

## Sources and limitations

- [Author manuscript](https://arxiv.org/abs/2201.11903)
- [Official NeurIPS 2022 proceedings and supplementary link](https://proceedings.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
- Historical APIs and proprietary pretrained weights prevent exact reproduction here. The reported output ZIP was not located through the checked official supplemental link; appendix prompts/examples were reviewed instead.
- Associated material: {"file": "Supplied main PDF, embedded appendices A–H", "pages_read": "15–22,25–43", "scope": "Prompt variants, result tables, error examples, resource and reproducibility statements reviewed; bibliography-only appendix pages not required for mechanism claims."}
