# SkillNav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents**. Tianyi Ma, Yue Zhang, Zehao Wang, Parisa Kordjamshidi ACL 2026, long papers (2026).

Primary record: https://aclanthology.org/2026.acl-long.595/



Supplied main file: [SkillNav.pdf](../../Paper/SkillNav.pdf). Pages actually reviewed: 1,3–8,10,13–22. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Compositional generalization beyond R2R instruction styles/environments with specialized DUET experts and frozen VLM routing. Five execution experts; temporal skill is pretraining data and router reordering, not a sixth runtime expert.

## Method, training, inference, and equations

Section 4 separates data construction, expert training and routing. The authors synthesize 450 instruction/trajectory pairs for each of five skills: direction, vertical movement, stopping, landmarks and regions. They additionally synthesize 2,000 temporal-order examples. Paths have 4–7 steps and are filtered by skill-specific rules; GPT-4o supplies captions. A DUET backbone first learns from R2R plus ScaleVLN (or SRDF) and temporal data. Five copies are then specialized on the corresponding 450-example sets. Temporal reasoning is therefore a training and routing operation, not a sixth execution expert.

At inference, a frozen language model rewrites the instruction into execution order (Eq.2). A visual-language router uses the instruction, observed history and previous subgoals to infer the current subgoal and choose one expert (Eqs 3–5). The selected expert receives the ORIGINAL instruction, current observations and DUET map (Eq 6). This distinction matters for a rewriting control: changing the executor's instruction would test a different intervention. ScaleVLN uses CLIP-B/16 and BERT features; the SRDF variant uses InternViT-6B features.

Expert learning inherits supervised DUET behavior cloning/DAgger. In the inspected training path, each valid step contributes −log πθ(a*_t|I,O_t,M_t), summed over non-padding targets and scaled by the imitation weight and batch size. I is instruction text, O_t the visual observation, M_t the map, and a*_t an expert action over the current candidate set plus STOP. The frozen router has no gradient update. Five stored experts and the routing model still incur memory/API cost, even though only one expert executes at each step. Appendix F and released scripts disagree on some batch-size/learning-rate values, as recorded below.

## Novelty and closest predecessors

Task-grounded NavNuances-derived skills, small targeted synthetic data and specialist top 1 routing; closest SAME/MoE and planner-executor agents. Explicit temporal reordering and semantic stage alignment already overlap proposed text importance ideas. Compare against single backbone trained on union data and rewriting-only, random routing, oracle/learned routing.

## Results and independent verdict

Table 2 reports ScaleVLN-based R2R val-unseen SR/SPL rising from approximately 79/70 to 83/77, while the stronger SRDF baseline remains near 84/78. This makes backbone strength and training data essential controls. Appendix G/Table 13 provides the most important attribution check: Temporal DUET alone reaches 82.59 SR/75.40 SPL, while full SkillNav reaches 82.59/76.50. Most R2R improvement is already present before specialist routing.

Other settings show useful gains: GSA-N-Scene SPL rises roughly 43→48 for ScaleVLN and 43→45 for SRDF. Zero-shot RxR SR rises 26.8→30.3 and 21.4→31.8, respectively. Random routing gives 43.17 GSA-N-Scene SPL versus 47.96 for the Qwen router; instruction reordering adds 2.53 SPL. However, the mixed-skill agent outperforms the landmark specialist in Table 5 (48.90 versus 31.53 SR), so specialization is not uniformly beneficial. Appendix I.3's17-case analysis finds the landmark specialist optimal in 13 failures but selected correctly only once. These support grounded routing as a remaining problem, not a claim that routing is already solved.

## Implementation audit

INSPECTED CODE. At b56de12, skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py:720–735 rewrites text for the router while obtaining executor token IDs from the original observation. Lines 760–792 pass reordered text, past images and previous subinstructions to a local or remote router. The local default at 101–104 uses Qwen2.5-VL-7B for localization and Qwen3-8B for skill selection. The training agent r2r/agent.py:538–557,650–653 accumulates masked action CE and scales it by train_ml/batch_size; the supplied temporal script sets DAgger, ml_weight=.15 and precomputed 512-dimensional CLIP features. env.py supplies standard and NavNuances-specific evaluation paths.

A material control-flow discrepancy deserves testing before using this checkout as a baseline. _get_input_embeds at 647–664 assigns self.vln_bert to each batch item's selected expert while constructing its language/panorama features. After the loop it remains the last item's expert; rollout line 838 then applies that single navigation head to the whole batch. This is an inspected source deduction, not an executed failure or evidence about the paper's original runs. The released test script uses batch 8. A batch-size-one versus mixed-expert batch equivalence test is the cheapest diagnostic.

The released temporal training script uses batch 16 and lr 1e-5, unlike Appendix F's batch 32 and lr 5e-5; its router defaults also differ from a single-model description. Exact main-table reconstruction therefore needs configuration clarification. Appendix F supplies A 6000 training budgets and router context length; no checkpoint or simulator was executed.

Repository: https://github.com/HLR/SkillNav. Inspected snapshot commit: `b56de12a6af07ce09c9252938b1fb69e55d33103`. Retrieval status: Commit-pinned source retrieved through sparse git checkout. Project code was not executed.

- [skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py:98-104,281-295,595-684,688-879](https://github.com/HLR/SkillNav/blob/b56de12a6af07ce09c9252938b1fb69e55d33103/skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py)
- [skillnav/backbones/scalevln/maps_nav_src/r2r/agent.py:538-592,650-653](https://github.com/HLR/SkillNav/blob/b56de12a6af07ce09c9252938b1fb69e55d33103/skillnav/backbones/scalevln/maps_nav_src/r2r/agent.py)
- [skillnav/backbones/scalevln/maps_nav_src/r2r/env.py:342-352,386-414,447-710](https://github.com/HLR/SkillNav/blob/b56de12a6af07ce09c9252938b1fb69e55d33103/skillnav/backbones/scalevln/maps_nav_src/r2r/env.py)
- [skillnav/backbones/scalevln/maps_nav_src/scripts/train_r2r_b16_mix_temporal.sh:1-77](https://github.com/HLR/SkillNav/blob/b56de12a6af07ce09c9252938b1fb69e55d33103/skillnav/backbones/scalevln/maps_nav_src/scripts/train_r2r_b16_mix_temporal.sh)
- [skillnav/backbones/scalevln/maps_nav_src/scripts/test_r2r_b16_moe-top1.sh:1-98](https://github.com/HLR/SkillNav/blob/b56de12a6af07ce09c9252938b1fb69e55d33103/skillnav/backbones/scalevln/maps_nav_src/scripts/test_r2r_b16_moe-top1.sh)

## Bottlenecks and follow-up chain

- **SKILL-1: Router understands words but binds wrong visual entity or chooses coarser skill** Conditions: Cluttered scenes, landmark-heavy instructions, unseen styles Cause: Textual priors and unverified subgoal completion dominate routing Evidence: §5.3 error analysis 17 cases; Table 5 mixed skill landmark 48.90 vs specialist 31.53 Type: Methodological / protocol. Follow-ups:  ProFocus queries missing visual evidence; Progress-Think /IDEAL alignment; grounded routing still not resolved by style decomposition. Status: **Partially addressed**, confidence Moderate.

## Source/access record and remaining uncertainty

**Subsequent independent source execution:** assignment 10 extracted the actual `_get_input_embeds` and navigation-call statement at the same commit, replacing expert networks with distinct fake functions. For routing IDs [0,1,2,3,4,0,1,2], batch-one evaluation used each selected expert, while a batch of eight used head 2 for every item; reversing the batch used head 0. There were 6/8 batch-one versus batch-eight dispatch differences and 8/8 permutation differences. This strengthens the control-flow finding beyond the original illustrative toy. It does not measure learned outputs or identify the code/checkpoint used in published experiments. Full model and simulator execution remain absent. [Exact extraction, inputs and results](../experiments/10/source_check_results.json), [independent feasibility review](../evaluations/10_feasibility.md).

Appendices A–I (PDF pp.13–22) were inspected for skill definitions, data integrity, training and routing failures. Appendix G/Table 13 reports Temporal DUET alone at 82.59 SR/75.40 SPL, versus full SkillNav 82.59/76.50 in Table 8: most R2R improvement over ScaleVLN 79.23/69.96 already occurs after temporal-data fine-tuning. Appendix I.3 examines 17 failure cases, with the landmark expert optimal in 13 but correctly selected once. The authors' vocabulary-overlap analysis is not a proof against all leakage; they explicitly acknowledge HM3D scene overlap. No benchmark was executed. The inspected batch-routing issue and configuration differences limit reproduction confidence. Appendix Table 10, visually checked in the rendered PDF p.18 (saved as evidence/03/skillnav_table10.png), reports Region DUET SR 36.67 and SPL 43.02. Under standard identical-episode SPL, SPL cannot exceed SR; the discrepancy needs author clarification before that row is used. The audit does not identify its cause. A separate four-line toy dispatch example reproduces the last-expert-over-batch mechanism, but no SkillNav checkpoint or simulator was run.

