# Continual VLN

Continual Vision-and-Language Navigation

Seong Jun Jeong; Gi-Cheon Kang; Seongho Choi; Joochan Kim; Byoung-Tak Zhang. BMVC 2025. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [Continual VLN.pdf](../../Paper/Continual%20VLN.pdf). Pages reviewed: ['3–9'].

## Problem and operating conditions

Sequential scene-domain training causes forgetting. I-CVLN combines English R2R/RxR into 20 domains with over 600 training episodes each; D-CVLN groups CVDN scenes into 11 domains. Evaluation episodes come from the same domains with novel instructions. Domain identity is supplied during training, not inference.

## Mechanism, training, and inference

Perplexity Replay (PerpR) ranks completed episodes by exp(-mean log probability of agent actions), retains difficult high-perplexity trajectories and removes easy ones from a fixed buffer. Training sums supervised imitation losses on current and replay data. Episodic Self-Replay (ESR) stores previous action-logit sequences and reconstructs the corresponding trajectory with stored decisions, then adds squared logit distillation to current-domain and supervised replay losses; both replay coefficients are 0.2. PerpR and ESR are separate variants, with ESR using random reservoir-style episode selection. I-CVLN uses OSCAR-initialized recurrent VLN-BERT and 500 buffered episodes; D-CVLN uses randomly initialized HAMT and 100 buffered episodes. Three scene-order curricula are averaged. No human clarification policy is learned by D-CVLN: its inputs are recorded dialogue histories.

SUPPLEMENT CHECK: pp. 2–4 specify A3C plus imitation CE, batch eight, 4,000/5,000 iterations per domain on an A100. PerpR sorts new episodes by descending action perplexity and evicts low-perplexity stored episodes under a domain quota. ESR replays the stored logit-argmax action sequence to revisit corresponding states, then distills current versus stored logits while also using ground-truth action labels. The pseudocode shorthand applying a loss to argmax should not be interpreted as a differentiable literal CE formula; the text describes CE supervision.

## Novelty and closest predecessors

Adds explicit continual VLN benchmarks and adapts difficulty-based rehearsal and sequential logit distillation to navigation. It differs from IVLN’s within-environment episodic memory: this is parameter learning over scene domains with retained training examples.

## Results and independent verdict

Table 1: I-CVLN vanilla AvgSR/SPL 25.6/23.1, random replay 28.0/24.9, PerpR 28.9/26.1, ESR 31.9/28.2, joint training 42.6/40.1. D-CVLN PerpR GP 6.1 versus ESR 5.8 and joint 8.1. Results establish partial forgetting mitigation with a substantial joint-training gap. Table 3 reversing PerpR selection lowers I-CVLN SR 28.9→27.1. Table 4 removing logit replay lowers SR 31.1→24.9 in a single curriculum. Table 2 does not establish strictly monotonic buffer gains: D-CVLN PerpR GP falls 6.2→6.0 from 100 to 200 episodes. Calling vanilla/joint mathematical lower/upper bounds is too strong; they are empirical comparators, and some baselines score below vanilla.

## Implementation and reproducibility

No official implementation was located in the bounded search of the exact title, author names, supplied paper and official BMVC 2025 proceedings page as of 2026-10-22. The official supplementary ZIP was accessible and its six-page PDF was read in full; it provides algorithms, losses, schedules and benchmark details but not inspected model source. Consequently model/training/runtime equivalence cannot be checked and benchmark reproduction is not claimed.


## Specific bottlenecks

**continual-replay: Rehearsal improves retention but remains far below joint training and requires stored previous episodes.**

Conditions: Many scene domains, constrained storage, or domains whose rare skills are absent from the buffer. Cause: Finite buffer selection and sequential distillation preserve only stored behavior; confident errors may receive low perplexity and be underrepresented. Evidence: PAPER EXPERIMENT: Tables 1–4; DEDUCTION for confident-error selection.

Type: methodological. Status: Partially addressed. Confidence: High within the evaluated setting. Follow-ups: ['M3E']

## Sources and limitations

- [Official BMVC paper](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_84/paper.pdf)
- [Official proceedings entry](https://bmvc2025.bmva.org/proceedings/84/)
- [Official supplement: six-page PDF reviewed](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_84/supplementary.zip)
- Official model repository was not located through the paper, proceedings and bounded exact-title/author searches. Supplement pseudocode is not an inspected executable implementation. No benchmark executed.
- Associated material: {"file": "evidence/04/supplements/continualvln_0084_ sup.txt", "pages_read": "1–6", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
