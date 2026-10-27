# Assignment 1 — Problem, bottleneck, and baseline audit

Cutoff: 2026-10-23, Asia/Seoul. Independent primary-source inspection; no earlier dossier verdict used as evidence. Original proposal files remain unchanged. This assignment investigates the original H=1, four-primitive-action design, not any later redesign.

## Decision

**The broad problem is valid; the proposed specific bottleneck remains a hypothesis.** Published interventions establish that future representations, goal evaluation, history use, execution format, and policy training can each affect CE navigation. They do not establish that an instruction-conditioned one-step future ranker is the dominant unresolved cause of failure in the three proposed small frozen policies.

The defensible question is: **At states actually reached by a fixed CE policy, do action-conditioned predicted next features reduce out-of-scene action-value regret and improve closed-loop navigation beyond a comparably trained direct action-value model, at the same sensing and execution schedule?** The supplied materials contain no CE branch outcomes answering that question. A larger world-model comparison should follow this bottleneck test, not stand in for it.

Evidence labels below: **[Specification]** supplied design; **[Published]** author-reported experiment; **[Code]** independently inspected official source; **[Deduction]** implication rather than measured result; **[Hypothesis]** untested proposed explanation.

## What the original proposal actually commits to

[Specification] The three policy trials are SmolVLM2-500M, LLaVA-OneVision-0.5B, and Qwen3-VL-2B. Added training is restricted to CE training scenes; pretrained components retain their external histories. Policies are trained with a StreamVLN-style IL/DAgger adaptation and then frozen. Four dynamics trials remain recorded options: transferred/fresh DINO dynamics and transferred/fresh CDiT dynamics. The scorer ranks FORWARD, LEFT, RIGHT, STOP, observes after one executed action, and learns utility from that action followed by the frozen one-action policy.

[Specification] Original utility is 0.70 success + 0.30 whole-path nDTW minus normalized remaining-action/collision penalties. No additional usefulness-aware world-model loss has yet been defined. Synthetic feature checks do not supply navigation evidence. Sources: [Approach](../../Research%20Approach/CE_World_Model_Approach.md), [Scorer Design](../../Research%20Approach/Scorer_Design.md).

[Deduction] This is a learned one-step policy-improvement experiment with a predicted-feature pathway. Calling it a demonstration that current CE failures arise from future ranking would reverse the order of evidence.

## Primary evidence that bears on the bottleneck

### DREAMWALKER: a valuable intervention with privileged goal information

[Published] ICCV 2023, full proceedings PDF pp.10877–10880, §§3.3–4.7, Table 2/Figure 4. Unseen SR is 49 for DREAMWALKER, 54 with actual future imagery, 42 for greedy waypoint selection, and 27 for copied-memory imagery. Replacing distance estimates by exact waypoint-to-goal distances yields about 70 SR at 20% replacement and 100 at full replacement. Its distance function drives reward, rollout selection and stopping; default search uses horizon four and 50 MCTS iterations. Table 4 gives unseen SR 42/47/49/48 for horizons 0/2/4/6. [Full paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DREAMWALKER_Mental_Planning_for_Continuous_Vision-Language_Navigation_ICCV_2023_paper.pdf)

[Deduction] This identifies sensitivity to privileged goal evaluation inside that planner. The two oracle interventions are not interchangeable measures of “perception error” and “scorer error”: exact goal distance contains information absent from an imagined view and influences several planning components. It is not a frozen modern VLM, primitive H=1, instruction-fidelity intervention. The 100% result cannot be transported as expected headroom for the proposed scorer.

[Code] Official repository at [ab7c9a5c9d07742e30e4cf99132cf1f61bb83de5](https://github.com/HanqingWangAI/Dreamwalker/tree/ab7c9a5c9d07742e30e4cf99132cf1f61bb83de5) contains only README.md. No released implementation verifies oracle insertion, randomization or exact evaluation settings. Figure 4/Table 2 were also visually checked in the downloaded PDF.

### HNR: predictive features help a waypoint lookahead model, with modest remaining oracle lift

[Published] CVPR 2024, §§3.2–3.3/4.3, Table 3: R2R-CE val-unseen SR/SPL is 57.21/49.15 for single view, 58.35/50.96 for generated RGB, 60.74/51.27 for HNR, and 63.13/54.59 for actual future CLIP features. The paper says rows 2–4 use the same lookahead VLN model; this does **not** explicitly establish one identical frozen checkpoint across all representation conditions. HNR predicts 12 views around candidate locations and uses a future-path tree, rather than four primitive next images. Figure 5 measures cosine similarity versus distance, not action-ranking regret. [Full paper, Table 3 and §§3.2.4/4.3](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Lookahead_Exploration_with_Neural_Radiance_Representation_for_Continuous_Vision-Language_Navigation_CVPR_2024_paper.pdf)

[Published] Pretraining uses 800 HM3D scenes and four RTX3090 GPUs; the navigation model starts from ETPNav. [Supplement, A.1–A.2](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wang_Lookahead_Exploration_with_CVPR_2024_supplemental.pdf)

[Deduction] HNR supports a real-feature diagnostic and shows that improved feature prediction can matter in one architecture. It neither demonstrates a language-value bottleneck nor proves H=1 sufficiency. Original Approach §7 should replace “under a fixed lookahead policy” with “within the same lookahead model design”; exact weight sharing was not verified.

[Code] [HNR repository, 6c935c1f6f0d715b1c6bef28724bdfabb48b159e](https://github.com/MrZihan/HNR-VLN/tree/6c935c1f6f0d715b1c6bef28724bdfabb48b159e): README releases HNR pretraining/checkpoints but redirects unreleased lookahead fine-tuning/checkpoints to g3D-LF. [run.py lines 11–12](https://github.com/MrZihan/HNR-VLN/blob/6c935c1f6f0d715b1c6bef28724bdfabb48b159e/run.py#L11) imports vlnce_baselines, absent from the inspected recursive tree. Presence of launch scripts therefore does not establish reproduction readiness for the full paper.

### StreamVLN: strong evidence for data and context confounders

[Published] Accepted ICRA 2026; inspected arXiv v2 dated 9 July 2026. Tables III–IV show substantial effects from co-training, DAgger and memory. Table III includes SR45.6 for initial R2R+RxR training, 50.8 after DAgger+VideoQA, 52.8 with MMC4, and 56.4 with ScaleVLN. Table IV increases SR37.3 to45.5 with a larger memory representation; retaining all context gives40.0. Its full recipe uses LLaVA-Video-7B, extra navigation/general multimodal data, and approximately1500 A100 GPU-hours. These results do not isolate action forecasting. [Paper §§III-D/IV-B/IV-D, Tables III–IV](https://arxiv.org/html/2507.05240v2), [acceptance/version record](https://arxiv.org/abs/2507.05240)

[Code] Official commit [e48f6ff7e9201d93aae003e8f64b04c00cec13bc](https://github.com/InternRobotics/StreamVLN/tree/e48f6ff7e9201d93aae003e8f64b04c00cec13bc):

- [streamvln_eval.py lines 290–344](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/streamvln/streamvln_eval.py#L290): generate only when the action queue empties; pop and execute queued actions otherwise. It records real frames every primitive step. One-action replanning changes the schedule.
- [stage-one training script](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/scripts/streamvln_train_slurm.sh) enables vision tower, projector and language-model tuning, four future actions, 32 frames, eight history frames. Proposed LoRA/frozen-vision small models are adaptations, not reproductions.
- [stage-two script](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/scripts/streamvln_stage_two_train_slurm.sh) and [co-training configuration](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/config/co-training_data.yaml) include additional data beyond the proposed CE-only scope.
- [README News](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/README.md): dataset-version update and a prior history-parameter evaluation bug materially affect results. Pin dataset/checkpoint/config together.

### Stronger recent alternatives to the diagnosis

| Work and verified status | Primary result / limitation | Consequence for this project |
|---|---|---|
| **SimpleVLN**, preprint 2610.23203v1, 16 Oct2026 | Table3: Qwen3-VL4B R2R-only ablation, primitive SR45.6 versus chunk53.6; multi-turn35.7 versus single-turn53.6. Table4: combined demonstrations57.9→65.4 after DAgger. Table1: 4B full-recipe SR73.8 R2R/75.1 RxR; that recipe includes ScaleVLN and RL. Model-only latency293ms per chunk excludes environment/preprocessing. [§§4.1–4.5, Tables1–4](https://arxiv.org/html/2610.23203v1) | Better interface/training is a serious rival explanation; aggregate SOTA scores are not matched CE-only small-policy controls. |
| **FutureNav**, preprint 2606.30367v1, 29 Jun2026 | Table2, no extra-navigation-data setting: policy-only SR50.1; +forward dynamics53.0; all auxiliary objectives55.1. Default inference decodes actions without dynamics/future heads. Full-data SR65.4 is a different setting. Table3 favors spatial targets over DINO/VAE targets in that setup. [§§3.3–4.3, Tables2–3](https://arxiv.org/html/2606.30367v1) | Prediction can help through training alone. Include a training-only prediction/distillation alternative; explicit deployment imagination is not yet justified. |
| **ReInsVLN**, preprint 2610.26545v1, 19 Oct2026 | Selected failure continuations: full812-instance history-review versus neutral net gain2.34pp, 95% trajectory-cluster interval[-0.99,5.68]. Selection requires exactly one of three navigators to solve an episode. Reference sequence length determines794 reminder suffixes. Review combines history prompting and task restatement; the paper acknowledges attribution limits. [§III-A–C, TableI](https://arxiv.org/html/2610.26545v1) | A useful rival history-utilization intervention, not proof that history alone causes failures. Use nonprivileged full-instruction reminders and matched neutral prompts as controls. |
| **SeekVLN**, preprint 2609.37353v1, 29 Sep2026 | AppendixA compares100 failed segments to matched expert-guided segments, with teacher-forced confidence. §3.3 branches seeking versus direct navigation from the same state for credit assignment. [§3.3; AppendixA](https://arxiv.org/html/2609.37353v1) | Missing observations/active sensing is another explanation. Its observational failure analysis does not establish the proposed ranker. Same-root counterfactual comparison is already used in adjacent CE work. |
| **VISTAv2**, preprint 2512.00041v1; primary page displays14 Nov2025 | §§3.2–3.3 explicitly score action-conditioned imagined futures and fuse values with a base planner. §4.3 describes a reflective-surface rollout that inflates traversability and misranks a candidate; its gating threshold is selected on val-unseen. [Full text §§3/4.3](https://arxiv.org/html/2512.00041v1) | Closely related architecture, and a concrete caution about overconfident predicted evidence. Treat numerical comparisons cautiously until protocol/release auditing; not independent proof of our mechanism. |

[Code] SimpleVLN [89b72a59c8f7feb805c0918e43dd5bfbdd7c48cc](https://github.com/simplevln/SimpleVLN/tree/89b72a59c8f7feb805c0918e43dd5bfbdd7c48cc) has evaluation code/checkpoint links but no training implementation in the inspected tree. [eval/vln/eval.py lines300–376](https://github.com/simplevln/SimpleVLN/blob/89b72a59c8f7feb805c0918e43dd5bfbdd7c48cc/eval/vln/eval.py#L300) samples up to eight history frames, queues at most two parsed commands with each expanded to at most three primitives. [Evaluation Settings](https://github.com/simplevln/SimpleVLN/blob/89b72a59c8f7feb805c0918e43dd5bfbdd7c48cc/eval/README.md) explicitly uses90° RxR HFOV, no separate local-history frame, and stochastic decoding(temperature0.3/top-p0.95). This differs from a literal reproduction of the paper's global-local design and from standard79° RxR sensing. Its README bibliography says2027, but the primary arXiv submission establishes2026; use the primary publication record.

[Code] FutureNav [ef4cb4358a0aeb7afed60ad79f7c8e85668a48f2](https://github.com/linglingxiansen/FutureNav/tree/ef4cb4358a0aeb7afed60ad79f7c8e85668a48f2) contains model/evaluation code; [README](https://github.com/linglingxiansen/FutureNav/blob/ef4cb4358a0aeb7afed60ad79f7c8e85668a48f2/README.md) marks training scripts/data pending. [eval/agent.py lines157–203](https://github.com/linglingxiansen/FutureNav/blob/ef4cb4358a0aeb7afed60ad79f7c8e85668a48f2/eval/agent.py#L157) generates and parses one action without candidate reranking. No weights were downloaded or results reproduced.

## The strongest objections, stated as falsifiable alternatives

1. **The ranker may simply be a better-trained second policy.** It receives extra recurrent memory, outcome labels, supervision and parameters. If direct Q(I,h,a) with the same branch labels matches it, explicit future features have not earned the causal claim.
2. **One-step effects may be too small, too aliased, or unrecoverable.** A turn changes viewpoint, not position; one forward step may leave nearly identical features. A bad frozen continuation can fail after every candidate. The all-four candidate set avoids proposal exclusion, but does not guarantee an advantageous first action.
3. **Real futures and predicted futures supply different kinds of information.** Actual branch observations reveal hidden scene details. Predicted features are computed from existing history/action and learned priors; an actual-future gain does not imply predictable/deployable gain.
4. **Restoring an interface can masquerade as model reasoning.** Cache inconsistency, discarded chunk tokens, different observation cadence or more policy calls can dominate a measured improvement.
5. **The original frozen-policy label is not the deployed policy's value.** Repeated overrides shift visitation and continuation. Good isolated branch ranking does not certify closed-loop utility or success.
6. **Higher composite utility may not mean better instruction following.** Success, nDTW and penalties are proxies; full-path nDTW can obscure small local differences and valid alternative routes. Report each outcome and route-order stress tests.
7. **Resource savings at the policy level may be spent several times over.** Three future predictions, DINO encoding, diffusion decoding, memory and scorer add latency/VRAM. A larger or better-trained direct policy at equal total cost can be stronger.
8. **Weak initialization can create artificial headroom.** If the small CE adaptation is undertrained, a world-model gain may repair poor basic supervision rather than a general bottleneck. Establish competitive within-budget policies before declaring the failure mode.

These are deductions/hypotheses, not observed failures of the proposed system.

## Required baseline dossier

All causal comparisons use the **same frozen checkpoint**, scenes/instructions, primitive action definitions, sensors, maximum episode budget, continuation protocol and seeds.

| Baseline | What it rules out |
|---|---|
| Native chunk execution | Establishes the actual policy baseline. |
| Same policy, H=1 replanning, no scorer | Separates observation/action cadence and KV-cache changes. |
| Same-capacity direct action-value model, null future | Separates extra labels, memory, training and capacity from prediction. Give it equal access to allowed real history and matched CE data. |
| Current-view copy / action-conditioned cheap predictor | Tests whether useful transformation beats identity and inexpensive local prediction. |
| Actual-next-view scorer | Privileged capacity/information diagnostic; never a deployment baseline. |
| Predicted-next-view scorer | Main treatment, assessed against direct Q and matched compute. |
| Non-STOP future permutation within root | Reliance diagnostic only; it creates mismatched inputs and is not an achievable competing method. |
| Training-only predictive auxiliary or distilled direct policy | Tests whether the benefit needs world-model execution at inference. |
| Neutral prompt versus full-instruction/history reminder | Low-cost rival history-use mechanism; avoid reference-derived suffixes. |
| Budget-matched stronger direct baseline | Tests whether added runtime/CE labels are better spent elsewhere. |

Preserve the three backbone and four dynamics trials as experimental options. Stage them: establish prediction value on one policy first; then compare dynamics under shared labels/scorer/protocol; confirm robustness on the other two policies. This audit does not authorize silently deleting agreed trials.

External contextual references should include StreamVLN, HNR/g3D-LF, SimpleVLN, FutureNav and the closest future-scoring work. Do not put their heterogeneous published scores in a table implying matched fairness.

## Data, compute and schedule controls

- **Scenes/data:** scene-disjoint training/development/evaluation; group shared R2R/RxR scenes. Count expert episodes, primitive transitions, DAgger expert calls, branch rollouts and external pretraining separately. Fixed weights do not mean equal information when a transferred dynamics model has extra data.
- **Simulator contract:** pin Habitat versions, dataset revision, HFOV/resolution, RGB versus depth/pose use, forward distance, turn angle, collision/sliding behavior and STOP/budget semantics. StreamVLN's [config](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/config/vln_r2r.yaml) specifies79°,0.25m,15°,500 actions; this cannot be silently mixed with other settings.
- **Execution:** equal replan frequency within causal comparisons; separate native chunk baseline. Force-action branches must restore simulator, history, cache, random and metric states; preserve remaining budget.
- **Labels:** stratify all-failure roots, near-ties, STOP decisions and recoverable errors. Keep expert-prefix and policy-visited distributions distinct. Duplicate roots do not create independent episodes.
- **Compute:** profile end-to-end p50/p95 decision latency, policy calls/episode, simulator throughput, peak allocated/reserved VRAM, total training GPU-hours and branch collection hours. Include world-model samples/denoising, VAE/DINO conversion and data movement.
- **Statistical selection:** choose thresholds on development training scenes; do not optimize on val-unseen. Bootstrap episodes/scenes, not correlated action branches. State RxR language subset explicitly; English-guide results do not establish multilingual robustness.
- **Matched comparisons:** one fixed scorer across world-model trials tests representation swaps; separately tuned scorers answer a different practical-system question. Match architecture and CE budget within transfer/fresh initialization pairs.

## Decisive bottleneck check before scaling

[Proposed test, not executed] Retain the original64-root/at-least-eight-episode pilot as a **collector debugging exercise only**. For inference about the bottleneck, expand to held-out training scenes with episode-level replication after measuring pilot effect sizes and dependence.

At each root execute all four candidates, then the identical frozen H=1 continuation. Record base-action utility, maximum measured utility, actual next features, all component metrics, and prediction outputs. Use repeated continuations when stochasticity makes the label order unstable. Never use a different continuation policy to manufacture oracle headroom.

Let G=mean[max_a U(a)-U(a_base)] on measured roots. Also report the proportion of base failures with an alternative successful continuation and the fraction of roots whose best/second-best gap exceeds the measured label noise. G is a diagnostic of candidate opportunity, not a deployable policy bound under repeated interventions.

| Observation | Interpretation / next decision |
|---|---|
| G negligible with stable repeated labels | H=1 candidate/continuation design lacks useful leverage. Do not scale dynamics training to rescue it. |
| G substantial, actual-future scorer fails out of scene | Scorer/history/labels are the immediate bottleneck; prediction quality cannot explain this failure. |
| Actual-future helps but predicted-future does not | Useful information is unpredictable from the available context, or dynamics/representation transfer is inadequate. Distinguish these before adding capacity. |
| Predicted scorer equals null-future Q | Added explicit foresight is unsupported; improvement belongs to value learning/memory/data. |
| Predicted scorer wins only against native chunk | Schedule remains a confound until it beats H=1 policy and H=1 direct Q. |
| Predicted scorer wins offline but fails closed loop | Frozen-continuation mismatch, distribution shift or compounding overrides are likely. |
| Closed-loop gain survives all controls but vanishes at matched runtime | Useful mechanism, weak lightweight-system case. |
| Gain survives held-out scenes, metric decomposition, seed uncertainty and matched cost | Empirical support for the narrow research question; still not automatic architectural novelty. |

The pilot should also log errors as stopping, collision/control, route-stage/history, landmark visibility/grounding, and off-route recovery. Categories require trace evidence and may overlap; do not infer them from an incorrect action alone.

## Search and source ledger

All searches performed in this assignment through2026-10-23; primary publications/code were used for claims. Secondary search hits were leads only.

| Search group | Actual query strings / navigation | Resolution |
|---|---|---|
| Initial diagnosis | “DREAMWALKER ICCV2023 oracle navigation future prediction”; “HNR CVPR2024 vision language navigation future”; “StreamVLN vision language navigation2025 2026” | Located proceedings/arXiv/official repositories. |
| Primary verification | site:openaccess.thecvf.com “DREAMWALKER” “Oracle”; site:openaccess.thecvf.com “Lookahead Exploration” “HNR” | Downloaded full proceedings PDFs; inspected tables/mechanism. |
| Release discovery | “DREAMWALKER ICCV2023 github Wang official”; “DREAMWALKER” “github.com” navigation Wang Hanqing; “HNR Lookahead Exploration github official”; “HNR” “github.com/MrZihan” | Connector inspected trees, pinned README and relevant files. |
| Strong contemporary alternatives | “SimpleVLN” arxiv2610; “What Limits Vision-and-Language Navigation” arxiv; “ReInsVLN official arxiv2610.26545”; “FutureNav2606.30367 navigation” | SimpleVLN/ReInsVLN/FutureNav primary texts inspected. StereoNav lead excluded from substantive claims: not sufficiently audited here. |
| Most recent boundary | site:github.com “SimpleVLN” “Hong”; “ReInsVLN” code github; “Seek Before You Move” “2609.37353”; “VISTAv2”2512.00041 | SimpleVLN official code reached from homepage; Seek/VISTA primary methods/failure sections inspected. No official ReInsVLN code established in bounded search. |
| Citation-following | HNR README→g3D-LF; original papers' own related work | g3D-LF official release link and commit9338aa53cf529f923f8845412c2a61a8dc82aa4d located; full downstream model not audited. Detailed novelty belongs to Assignment4. |

Source snapshots: downloaded full PDF/text for DREAMWALKER, HNR, StreamVLN, SimpleVLN, ReInsVLN under [evidence/a1](../evidence/a1/); selected pinned code snapshots and provenance metadata saved there. FutureNav, SeekVLN and VISTAv2 were inspected via full primary arXiv HTML, not third-party summaries.

Access limits: web PDF fetch returned403 for both CVF papers; direct PowerShell downloads succeeded and were converted with pdftotext. Default Python lacked requests/pypdf and its urllib certificate validation failed; no certificate bypass was used. The initial homepage PowerShell parser failed; UseBasicParsing succeeded. DREAMWALKER's release is README-only; HNR's exact lookahead release is incomplete; newer training recipes were not fully released in inspected trees. No absence of a search hit is evidence of novelty.

## Remaining uncertainties and evidence status

**Executed in this assignment:** original-document reading, primary full-text inspection, official GitHub connector inspection/pinning, PDF download/extraction and a visual table/figure check. **Not executed:** simulator navigation, branch collection, policy/dynamics/scorer inference, training, latency profiling or score reproduction.

No reviewed primary paper demonstrates the exact small-frozen-policy/H=1/CE-only/8GB claim. Nor does the literature refute that narrow possibility. The immediate missing evidence is the branch opportunity and matched direct-Q comparison on the intended policy, followed by closed-loop and runtime validation. Until those exist, the strongest justified verdict is **researchable hypothesis with an explicit go/no-go experiment**, not an established bottleneck or a validated CVPR contribution.

