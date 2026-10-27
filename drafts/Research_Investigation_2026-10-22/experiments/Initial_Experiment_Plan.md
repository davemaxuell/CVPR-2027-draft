# Initial experiment plan: recovery-label audit

**Planned, not executed.** This is the low-cost first gate for proposal P1; the final proposal and independent evaluations take precedence if they narrow its claims. It is not a training recipe with a promised gain.

## Question and primary measurement

Does the existing teacher's **complete proposed recovery continuation, including STOP**, fail an independently validated instruction requirement, after allowing legitimate alternative routes? The primary diagnostic is the fraction of independently adjudicated, interpretable checkpoints with such a failing continuation. Separately measure first-action disagreement with the cost-optimal acceptable set. In a reversible graph, going to the goal before a required landmark can remain recoverable until STOP; the first move alone is not necessarily a semantic violation. A valid longer detour is an efficiency disagreement. Report ambiguous and unannotatable checkpoints separately, not as errors or successes. Secondary measurements distinguish incomplete obligations, wrong grounding, irreversible violations and unreachable actions.

This is a label-quality question. A positive result does not yet establish a better learned navigator, the prevalence of an error in all VLN datasets, or the novelty of a teacher correction.

## What is available now

The workspace contains local R2R annotation files and 72 pinned connectivity graphs. Their geometry has already been inspected, including the validation-unseen split; that split is therefore not an untouched final test for this investigation. The isolated product-graph examples are executable on CPU. No usable navigator checkpoint, cached visual features, licensed scene assets or simulator environment has been established. No natural recovery rollout or human semantic annotation has been collected.

The physical machine has an 8 GB-class RTX 4070 Laptop GPU, but the installed PyTorch build is CPU-only. Establish a separate compatible environment and the required data access before attempting model inference. The research investigation has not installed a new navigation stack or measured training memory/runtime.

## Collection protocol

1. Pin the DUET checkpoint, source commit, tokenized instruction/preprocessing files, visual features, environment version and evaluation code. The available raw R2R annotations lack the `instr_encodings` expected by the inspected loader. Verify the native teacher and action candidates on a small trace before changing anything. Record every macro action's actual intermediate nodes and physical travel cost.
2. Freeze one navigator and collect natural rollouts on a development scene partition. Sample recovery checkpoints using a declared rule independent of whether the proposed teacher would win. Keep the full action/observation history, instruction, candidate actions and trace provenance. The suggested pilot is 200 natural checkpoints; this is a planning quantity, not a power calculation or a completed sample.
3. Separately construct 100 controlled pairs ending at the same pose with different relevant histories. Verify physical reachability and instruction meaning. These pairs test the mechanism and must never estimate natural prevalence. If identical observations cannot be arranged, record the mismatch rather than claiming a perfect paired intervention.
4. Two annotators, blinded to teacher outputs and model scores, label remaining requirements and acceptable actions using a fixed guideline. Adjudicate disagreements with reasons. Track annotation time and uncertainty; avoid judging language solely from a reference path. Review representative groundings with actual images, not only graph geometry.
5. Apply the frozen geometric, nDTW, language-aligned waypoint, reversible-progress and product-state teachers to the same checkpoints. Begin with a fixed common primitive-neighbor menu plus explicit STOP on the static annotated graph. Compute costs and complete continuations under that same interface; do not silently use a physical edge the policy cannot choose. If a baseline requires an adapted/reconstructed teacher, label and validate it rather than calling it a native model reproduction. Candidate expansion and additional event labels must be shared or isolated as separate factors. Native DUET integration additionally needs observed-map/menu state and consistent treatment of endpoint-forced teacher/sample termination and cached STOP return. Replacing only the teacher function does not change the full completion behavior. Preserve a separate native-protocol result.

## Annotation record

Use [the empty CSV template](recovery_annotation_template.csv); it contains no measured data. Store large histories, observations and action sets as separate hashed JSON records referenced by ID. Required fields include scene/episode/checkpoint identity, natural versus constructed origin, instruction and history references, independent annotation and adjudication, acceptable states/actions, unresolved grounding, and the teacher outputs filled only after annotation. Privacy-sensitive identifiers are unnecessary.

An acceptable-action label must specify its convention: eventual semantic viability, feasibility within a declared remaining budget, or optimal cost under the stated oracle. P1's set Y uses efficient acceptable actions with zero slack. If budget B is part of the task, record it before comparison and use it for every arm; do not choose it to manufacture violations. Report complete-continuation satisfaction independently of Y. STOP is a terminal decision under an explicit completion contract, and unreachable or ungrounded states must not receive fabricated labels.

For the first diagnostic, restrict instructions to independently validated node-landmark events. Turning or heading-dependent clauses require additional physical state and belong in a later extension. Terminal acceptance must include the current required goal location as well as completed historical obligations: a past visit to the goal cannot authorize STOP after leaving it. Treat unresolved action and state labels as missing in both corresponding losses.

## Analysis and decision

Report natural and constructed results separately. For natural checkpoints, retain sampling probabilities and scene/episode clusters; repeated checkpoints from one route are not independent samples. Give annotation agreement, exclusion rates, paired teacher disagreement and uncertainty by scene/episode. Define all thresholds before looking at comparative results. Do not promise a precise minimum effect size or sample power without a variance/prevalence estimate.

Stop or reframe P1 if the natural discrepancy is negligible, annotations are unreliable, or a language-aligned waypoint/reversible milestone baseline removes the gap. If the diagnostic is positive, run a label-only training comparison on identical sampled states, initialization, features, candidates, optimizer updates and annotation budget. Add a state-estimation module only after separating its benefit from corrected labels and extra supervision.

A later closed-loop test should use common externally specified disturbances and independently measured ordered completion, ordinary SR/SPL, route fidelity and extra distance. Keep self-induced deviations as a separate analysis. Use scene-disjoint development/calibration/final test partitions, and preserve a new untouched final test after the already inspected development work. No simulator or learned-policy result is implied by this plan.
