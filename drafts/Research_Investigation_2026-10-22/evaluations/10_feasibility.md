# Independent feasibility and experimental-validity review

Assignment 10. Cutoff: **22 October 2026**. Reviewed all four proposals in assignment 07's completed draft, the selected original source entrypoints, and the coordinator's initial experiment plan. Decisions below concern whether a valid experiment can be built; they do not certify novelty or predict gains.

| Proposal | Decision | Practical first step | Principal obstacle |
|---|---|---|---|
| P1 Recovery labels | **Revise; proceed with diagnostic** | Annotated, fixed-checkpoint teacher comparison | Teacher replacement alone leaves endpoint termination, candidate masking and cached return inconsistent with the new task |
| P2 Validity and difficulty | **Revise; proceed with audit** | Independently label a fixed generated pool before training | Pool, visual evidence and policy difficulty scores are not yet available; equal examples do not equal annotation or training budget |
| P3 Joint future events | **Revise; defer model training** | Representation headroom and learnability gate | Requires new ordered-event supervision and a different task; the selected NavQ predictor is frozen during navigation |
| P4 Failure-aware memory | **Revise; release-level diagnostic only initially** | Log harmful writes, then compare fixed-representation write policies | Released memory differs from paper EEC; a complex model must beat simple quarantine and use deployable evidence |

## What was independently checked

I inspected actual DUET teacher, rollout, candidate masking, macro-action, data-loading and augmentation scheduling; RSI-Nav verification acceptance and later sample construction; NavQ target generation, predictor loss, checkpoint loading and REVERIE integration; SkillNav expert dispatch and test launcher; and PSG-Nav terminal feedback, RAG writes/retrieval, detection-side rejection and server requirements. The machine-readable [inspection manifest](../evidence/10/source_manifest.json) provides immutable commits, file hashes and inspected ranges. A fetched file is not treated as inspected outside those ranges.

Three isolated tests used **AST-extracted, reviewed source functions with explicit synthetic inputs/stubs**, without importing a project module. They tested SkillNav dispatch, PSG-Nav failure-write/read behavior, and DUET's endpoint teacher condition. [Setup and reproducible script](../experiments/10/source_checks.py), [results](../experiments/10/source_check_results.json), and [scope notes](../experiments/10/README.md). No learned model, simulator, benchmark, real detection, natural failure prevalence, or proposal training was executed. Memory file I/O was stubbed. Only the tensor `.cuda()` calls in the two CPU-tested methods were removed; this does not change their selection logic.

I also re-read PSG-Nav §3.4 and §4.1, PDF pp.6–7, to compare its memory contract directly with the newly retrieved source. Other scientific paper findings are attributed to the existing dossiers; this review does not claim a second full reading of the entire collection.

## Environment and access

The local R2R annotation hashes match the coordinator's inventory: 4,675 training paths in 61 scenes and 783 validation-unseen paths in 11 scenes; scene overlap is empty. There are 72 downloaded connectivity graphs. These files do **not** include visual observations or event labels. Raw annotations lack `instr_encodings`, which the pinned DUET loader requires. Tokenization must use the pinned tokenizer and truncation settings rather than silently mixing raw and encoded releases. [Independent asset check](../evidence/10/asset_check_results.json).

The physical RTX 4070 Laptop has 8,188 MiB VRAM, while installed PyTorch is CPU-only. Habitat, Habitat-Sim and MatterSim are absent. A usable local navigator checkpoint, visual feature collection and licensed scene asset installation have not been established. This is a bounded inventory result, not a claim that the user lacks access elsewhere. The already examined validation-unseen geometry should be treated as development evidence, with a separate untouched final test.

DUET's pinned README points to processed annotations/features/checkpoints and MatterSim setup. NavQ's remote Hugging Face listing was successfully retrieved independently: it lists Q-model and navigation checkpoints for original and ScaleVLN data. **Listing availability does not verify contents, checkpoint compatibility or reproduction.** No model files were downloaded. RSI-Nav requires separately prepared Qwen3.5-9B weights, images and Habitat 0.3.3; its distributed recipe is unnecessary for the first P2 audit. PSG-Nav's server loads both Qwen2.5-VL-7B and Qwen2.5-7B in bfloat16 on one device. Its unmodified server is not a credible fit on the present 8 GB GPU; quantization/offloading or another server would change the execution recipe and requires explicit accounting. No runtime or training-memory estimate has been measured.

## P1: Recovery labels

### Actual integration map

Use DUET commit `93e8b233164bc079a6db48b8a0a78d123ec8de41`.

| Component | Inspected source | Required change/control |
|---|---|---|
| Training examples | `map_nav_src/r2r/data_utils.py:5–46`, `main_nav.py:24–82` | New immutable example/annotation manifest; preserve instruction encoding, candidate provenance and scene/route-family split |
| Recovery teacher | `r2r/agent.py:239–285,404–423` | Return a mask over the **same** candidates; mask unresolved cases; replace scalar CE only in the specified teacher intervention |
| Candidate construction and masking | `agent.py:110–123`; `models/vilmodel.py:786–813` | Visited nodes may be represented but their global logits are masked. Allow any required visited destination identically for every recovery arm, or restrict the diagnostic to native feasible actions |
| Macro-action execution | `agent.py:287–303` | Recorded route may contain intermediate nodes, but the simulator is moved directly to the destination with `newEpisode`. Log the actual route contract; do not invent new intermediate visual observations |
| Training termination | `agent.py:453–455` | Teacher/sample modes stop on equality with the reference endpoint even when a new procedural obligation would remain. Replace this consistently across the semantic-task controls |
| Forced termination | `agent.py:459–466` | No-frontier and maximum-step conditions still end an episode. Report incomplete obligations at budget exhaustion; do not invent a successful label |
| Cached endpoint return | `agent.py:471–478` | A terminal path is extended to the highest cached STOP score. Re-evaluate completion **after** this path, including event updates and cost; specify a common return convention for all arms |
| State model | `models/vilmodel.py` navigation input/branches | Add the state feature only after label-only comparison; record extra parameters/compute and share annotation access with state baselines |

These references are [pinned source](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/agent.py). The existing routing mechanism can traverse visited nodes; the issue is selectable destinations and semantic state, not inability to backtrack.

The extracted `_teacher_action_r4r` returned STOP for both illustrative histories ending at the reference goal, regardless of whether history included landmark A. This only confirms the endpoint rule. It does not show that a real R2R instruction required A or that standard R2R evaluation is wrong. The state-estimation claim must be tested on independently grounded obligations, not reference-path deviations alone.

The product-graph teacher also needs an explicit macro-action transition rule. A pose visit is not always a perceptual witness: a panoramic cached image, a directional look and a commanded turn have different observation contracts. For the smallest study, use events identifiable from actually available observations and action records. If intermediate observations are newly acquired, provide and charge them to all controls. Disable cached return for all semantic-task arms or implement a shared semantics-aware return convention; do not fix only the proposed arm. An oracle state may be used for a separate upper bound, never the deployable baseline.

### Cheapest decisive experiment

The coordinator's [initial plan](../experiments/Initial_Experiment_Plan.md) is a sound diagnostic starting point, subject to the integration corrections above. Collect frozen-policy development traces once assets exist. Sample natural deviation checkpoints independently of teacher disagreement, and maintain separately constructed same-pose/different-history pairs. Two annotators should see the observations and instruction, remain blind to teacher/model output, and identify semantic feasibility separately from efficiency preference. The planned 200 checkpoints and 100 pairs are pilot quantities, not a power calculation.

A go decision requires an interpretable natural subset with stable adjudication and a material, scene-distributed error of geometric teachers **after** allowing legitimate routes. Report the denominator including exclusions. Stop or narrow if a simple language-aligned waypoint/reversible milestone teacher removes the difference. Only then compare identical policy architectures trained on a fixed rollout dataset, common sampled states, candidates, human labels and updates. If independent data are unavailable, an automaton/graph toy is an implementation check, not the required teacher audit.

Mandatory pre-training checks: completed versus unfinished histories at the same endpoint; a visited destination still required; an intermediate event on a macro path; no feasible accepted continuation; budget termination; and cached return that changes obligation satisfaction. All control arms must pass their declared common task contract. A separate native-DUET result preserves comparability to prior R2R papers.

## P2: Validity and difficulty

### Actual integration and confounders

The minimal intervention is a selected-pair manifest in DUET's augmentation loader, with policy loss unchanged. `main_nav.py:143–155` alternates one ordinary-data update with one augmentation update. Equal selected-pool size alone does not control exposure; fix this schedule, optimizer steps, batch construction, seeds and original-data mixing. `construct_instrs` splits each route's instructions into separate examples and generates IDs from `path_id` and instruction index. Preserve stable globally unique IDs and route-family provenance before subset selection; paraphrases and derived examples must remain in the same cross-fitting split.

RSI-Nav commit `5e8fa2c74cf8e7dc305b33088a963b200e20805d` is a useful **filter implementation comparator**, not the appropriate initial training stack. In `evo/evo_rollout.py:1517–1540`, acceptance is the union of two SR/SPL threshold rules. At `1578–1599`, accepted navigation/language examples are built from `item['trajectory']` source frames/actions, not automatically the verifier's realized route. At `evo/evo_utils.py:1035–1101`, accepted sample queues are combined with separately sampled spatial/STOP QA. A causal selection comparison must freeze those replay streams and distinguish the source route's validity from the verifier's trajectory. [Pinned filter](https://github.com/OpenGVLab/RSI-Nav/blob/5e8fa2c74cf8e7dc305b33088a963b200e20805d/evo/evo_rollout.py).

A balanced validity/difficulty sample cannot be drawn from truly unknown validity labels without an initial screen. First probability-sample the natural pool within prespecified model-difficulty strata; then label validity. A separately enriched case-control subset may improve diagnostics, but report its inclusion probabilities and never use its balanced fractions as natural prevalence. Model difficulty must name a frozen checkpoint; valid-hard for one navigator need not be valid-hard for another.

Use one immutable pool and label version. Separate valid, invalid and unresolved; retain missing-witness status. Do not call an unobserved object absent. Scorer training, quota fitting, pair annotation and final navigation evaluation need disjoint scenes or route families appropriate to the claim. Human effort should include clause segmentation, witness search, adjudication and rescoring—not just pair-label count. Match both accepted unique routes and optimizer exposure, then separately measure the effect of deliberately different semantic composition. Preserve a disjoint independently authored test rather than only the generator's language.

### Cheapest decisive experiment and gate

Before training a selector, quantify success-filter false acceptance and false rejection against blinded semantic labels, at matched acceptance rates. Include compatibility, ordinary clean-validation/data-quality and training-dynamics baselines; compare witness versus no-witness scoring with identical architecture and equal annotation-time budgets. If all high-difficulty examples are ambiguous/noisy, or a simple compatibility/validity baseline already matches the proposed selector, stop the method claim. A useful dataset audit remains possible.

A downstream go decision requires incremental selection quality that survives scene/route-family cross-fitting, then matched-size/compute navigation training. Report uncertainty by scenes and unique routes; multiple clauses or paraphrases are correlated observations. The first offline audit is cheaper than RSI-Nav retraining, but it is **not currently executable from geometry alone**: route images/features, the fixed generated pool, difficulty scores and independent annotations are missing. No selector or navigator experiment was run.

## P3: Joint future events

### Selected NavQ path and constraints

NavQ commit `9465fe43c9c1583bd7ba390b6be30996ea4f5a5c` provides a concrete target-construction baseline. `code/Q_pretrain/dataset.py:136–244` samples non-revisiting histories, forms candidate branches using stored shortest paths, and adds discounted 512-dimensional caption features with branch probabilities. `GPT_VL.py:236–243` minimizes summed feature MSE. Ordered event co-occurrence needs **trajectory-consistent labels under a declared continuation policy**, not multiplication of these marginal features. Equally weighted endpoint lists and equally weighted trajectories generally induce different distributions; publish the chosen sampler.

The downstream path is inspectable: `map_nav_src/models/vlnbert_init.py:72–81` loads the Q-model state; `models/vilmodel.py:724–752` runs it under `torch.no_grad()` and detaches its outputs. `reverie/agent_obj.py:419–437` inserts predicted features for frontier candidates and observed past features for visited nodes. `vilmodel.py:640–643` assumes 512 input dimensions; the future branch has separate action/progress heads. Merely adding a joint head to the navigation model will not train the frozen predictor through navigation loss. Pretrain it explicitly, adapt the projection and checkpoint schema, or declare and control a changed end-to-end training regime. [Pinned integration](https://github.com/woyut/NavQ_ICCV25/blob/9465fe43c9c1583bd7ba390b6be30996ea4f5a5c/code/map_nav_src/models/vilmodel.py).

The selected repository entrypoint is REVERIE object-goal navigation. Its ordinary labels do not establish the ordered-route task in P3. Moving to route-following DUET/RxR requires a separately validated port and ordered-event dataset. A fair initial study should use a frozen candidate interface and short, predeclared horizons; hold the continuation policy fixed across all target heads. If event queries are instruction-conditioned, every comparator receives them. Match labels, training trajectories, history inputs, parameter/readout capacity and candidate count. Report output width K+K² and the projector's cost rather than hiding it inside a 512-dimensional interface.

### Cheapest decisive experiment and gate

First compare **oracle representations** on independently labelled branch continuations: discounted features, marginal events, joint ordered events, full short sequences and direct future-QA. Use a common controlled readout and a separate scene/combination test. Reject if ordinary history or a capacity-matched sequence/QA head removes headroom. Then predict those labels from deployable observations; an oracle representation gain without observable predictive signal is a no-go. Grounding noise and rare ordered events can dominate the outcome, so publish event-label agreement, class prevalence and missingness.

Only after both gates should a navigator be trained. Evaluate candidate-ranking regret under the **same** continuation policy, event Brier/log loss and runtime per candidate. Then separately evaluate policy shift after closed-loop deployment. P3 is technically implementable but highest risk and lowest immediate priority: no ordered-event data, Q weights, visual features or ported route-following benchmark has been established locally. No learned P3 experiment was executed. Its conceptual first-moment counterexample is not evidence of real NavQ target collisions or benchmark gain.

## P4: Failure-aware memory

### What the actual write/read path establishes

At PSG-Nav commit `181d878ff207845a31e542a19a6592925c1e9bab`, `PSG_Nav.py:1483–1554` copies environment success and, at terminal failure or step 500 with a nonempty detection window, calls `add_false_positive` on recent detections. It does not distinguish control failure from identity error. `utils/object_detection.py:289–324,543–564` stores crops/caption/confidence in a five-entry default window; the stored caption is overwritten with the goal category. Resetting goal detection does not clear this window (`PSG_Nav.py:297–321`). Accordingly, the write may summarize recent rather than causally responsible detections. This is a source-contract finding, not a measured real-world mislabel rate.

Newly inspected `utils/rag_manager.py` is more specific than the original proposal's assumption:

- `185–254` stores negative examples, with duplicate checking and a default negative capacity of 20.
- `277–302` computes crop-feature similarity to category-keyed negatives and immediately returns false/confidence zero at cosine similarity ≥0.95, before consulting positives.
- `354–392` uses resized **per-channel color histograms**, not the paper's joint visual/group/room descriptor.
- `739–809` loads negatives across scene subfolders for the active dataset into category banks. Retrieval has no scene/track argument.
- `object_detection.py:618–662` turns the negative rejection into a spatial exclusion region and resets goal detection; `266–270` can skip a later detection in that region.

This is materially different from paper §3.4 Eqs.9–11, which use visual plus group/room context, an additive positive-minus-negative score, and capacity 10. Therefore an improvement over this source cannot automatically be described as an improvement over the published EEC mechanism. [Pinned RAG source](https://github.com/Chenrf1121/psg-nav/blob/181d878ff207845a31e542a19a6592925c1e9bab/utils/rag_manager.py), [caller](https://github.com/Chenrf1121/psg-nav/blob/181d878ff207845a31e542a19a6592925c1e9bab/utils/object_detection.py).

The isolated source test supplies a synthetic crop stipulated to be a correct category and terminal failure stipulated to be control-related. The actual write stores it; the actual verifier later rejects the same crop, even if a matching positive record is added. A rearranged crop with the same histogram is also rejected. The actual read-side caller clears the goal and marks 289 cells negative in the supplied 50×50 synthetic map. This verifies a **conditional harmful-write pathway**. It measures neither real crop correctness, occurrence frequency, natural spatial harm, nor published performance. Changing to a better representation alone could remove many collisions; that factor must be isolated from failure attribution.

### Can this release support the narrower hypothesis?

**Yes, as an instrumented release-level write-policy study; no, by itself as evidence of paper-method failure or novel memory architecture.** First keep detector outputs, visual features, bank capacity, retrieval, veto, scene exposure and feedback identical. Compare no write, quarantine all failed episodes, a simple observable controller-failure rule, witness-presence-only, and learned attribution. If only the learned arm receives richer image embeddings or new views, the attribution claim is confounded. A second factorial study may improve representations for every arm. A paper-faithful EEC reconstruction is a distinct baseline requiring validation and explicit labelling.

The existing bank does not retain a robust track identity, full controller trace, pose residuals or a contradiction view. Those inputs must be collected; the failure label alone cannot recover them. Begin in native category ObjectNav: a different instance of the requested category is valid. An instance-specific extension needs separate task labels. Genuine contradiction evidence must concern that category claim, with visibility/track reliability audited. No witness and no attributable cause means unresolved, not negative. Additional views change the sensing budget and must be shared or separately factored.

The novelty review's HiMeco/EmbodiSkill findings further narrow the contribution: failure diagnosis and gated memory editing already exist. P4 would need demonstrable incremental value from deployable, claim-specific evidence for **negative semantic writes**, beyond simple quarantine and an appropriate adaptation of those precedents. This review independently verifies the implementation premise, not the novelty of those external comparisons.

### Cheapest decisive experiment and gate

Start with log-only traces under fixed upstream policy and reset the on-disk bank per replicate. Independently identify correct-category/failed-approach, wrong-category/adequate-control, occlusion, track swaps and dynamic-scene cases; use synthetic perturbations only to test mechanisms, with a separate natural sample for prevalence. Replay the same observations and candidate sequence through all write/read policies, recording write precision, later false suppression and false acceptance. Offline replay establishes decision changes under a shared trace, not navigation SR: changed decisions would generate different later observations online.

Proceed only if natural harmful writes are meaningful and the attribution gate beats no-write/quarantine at matched recall, memory and feedback budget. Stop if identity correctness is not identifiable from deployable inputs, if simple quarantine suffices, or if preventing negatives leaves stale positives that erase the benefit. Then test randomized episode-order repeated navigation across scene-disjoint development/final partitions. Report fixed-memory, feedback-free, self-assessed and oracle-terminal-feedback protocols separately; count all episodes, including failures and budget overruns. Without new traces and a compatible simulator/model stack, this remains a planned experiment.

## Cross-proposal baseline integrity

The requested SkillNav check is now stronger than an illustrative pseudocode toy. I extracted `_get_input_embeds` and the exact navigation-call statement from commit `b56de12a6af07ce09c9252938b1fb69e55d33103`. Distinct fake expert functions expose which expert is actually dispatched. For selected experts `[0,1,2,3,4,0,1,2]`, batch-one calls use each selected expert, while batch eight uses navigation head 2 for all examples. Reversing the batch changes the common head to 0. Six of eight dispatches differ from individual evaluation; eight of eight change after permutation. [Source function and call](https://github.com/HLR/SkillNav/blob/b56de12a6af07ce09c9252938b1fb69e55d33103/skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py#L647), [test launcher batch 8](https://github.com/HLR/SkillNav/blob/b56de12a6af07ce09c9252938b1fb69e55d33103/skillnav/backbones/scalevln/maps_nav_src/scripts/test_r2r_b16_moe-top1.sh#L80).

This proves the inspected dispatch depends on other batch items when experts differ; it does not establish learned-output differences for a particular checkpoint or invalidate published tables. A baseline should use batch one or a separately identified dispatch correction, then test learned batch/permutation invariance before comparison. Preserve Temporal DUET and shared union-data controls; fixing dispatch is not proposal novelty.

I separately inspected SAME's `TaskMoeRouting` softmax over batch dimension in `src/models/vilmodel.py:979–984`, corroborating assignment 04's reason for a baseline routing check. M3E and slow4fast release blockers are documented by assignments 04/05; I did not rerun those audits and do not rely on them to infer paper performance. Their unavailable/incomplete mechanisms should not be used as supposedly reproduced weak baselines. Report exclusions or validated paper-derived implementations.

## Recommendation

Prioritize **P1's teacher audit**, then P2's selection audit. Both can falsify a research premise before extensive training and have a clear minimal DUET base. P4 has a verified conditional source mechanism but needs natural traces and stronger controls before method work; its negative-write premise is narrower than a general failure-aware memory contribution. Defer P3 until representation and learnability gates, task-port validation and novelty review justify the additional dataset/model effort.

All four proposals remain empirically unvalidated. The appropriate next commitment is an asset-verified, preregistered diagnostic—not an assertion that the laptop can train the complete models, that release fixes are new science, or that benchmark gains are already supported.
