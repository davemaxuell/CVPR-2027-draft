# Controls required before interpreting an improvement

These are the coordinator's experimental-design requirements, informed by the reviewed protocols. They are proposed controls, not executed benchmark results. Assignment 10 independently audits the concrete proposal plans.

## Data and language

A data-scaling claim needs the same architecture, visual features, initialization, optimizer, action interface, and evaluation protocol. Separate increasing data diversity from increasing optimizer updates. A useful factorial comparison holds updates fixed in one contrast and exposes every example equally in another. Report instruction count, unique paths, unique scenes, and filtering yield; millions of paraphrases are not millions of independently grounded routes. The DUET, ScaleVLN, and SRDF systems are informative baselines but are not by themselves a controlled experiment on data volume.

Language interventions need ground-truth semantic consequences at the evaluated state. Use both meaning-changing pairs requiring different valid decisions and meaning-preserving paraphrases. Hold the observable history and action set fixed for an immediate-decision diagnostic. If the model caches language-conditioned history, either reconstruct that history consistently or explicitly label the experiment as a late-instruction intervention. Higher action-distribution sensitivity alone is not better instruction following.

Off-path analyses must separate distance to the reference path, ordering violations, time since deviation, action-set size, ambiguity, and which recovery actions remain valid. A goal-shortest-path teacher and a reference-faithful teacher provide different objectives. A teacher's use of privileged graph or route information is valid training supervision only when its test-time absence and the benchmark's intended objective are clear.

Text-rewriting controls must match where the rewrite is consumed. SkillNav's router and its execution experts do not receive identical text inputs. Comparing a router-only rewrite with a baseline that rewrites the executor's input changes two mechanisms. Match the backbone and union of specialist training data before attributing a gain to routing.

## Endpoint selection and verification

First test endpoint ranking on fixed recorded trajectories. Report whether a successful endpoint was available, the rank of that endpoint, calibration and discrimination of the verifier, and the false-stop cost. Then test the closed-loop policy: changed stopping decisions alter future observations and route length. Account for DUET's existing return to a cached endpoint, and charge the return path to the trajectory according to the evaluator's actual rules.

An OSR–SR difference is a descriptive conditional-failure opportunity, not a diagnosis of softmax competition. Candidate duplication or score rescaling can test an algebraic invariance, but a trained-policy causal claim requires realistic interventions and a control for new information in the candidate set. Compare simple calibration and existing endpoint-ranking methods before introducing a complex new module.

## Prediction and adaptation

Evaluate a predictive model at multiple horizons and under the actual querying policy. Report both prediction error and decision utility. An oracle-future-feature ablation can locate headroom caused by prediction quality, but oracle views must never enter the deployable method. Include a no-imagination policy, equal-compute extra-observation or reasoning controls where feasible, and a matched planner using the same available observations.

Treat generated views as model outputs with provenance. If the planner uses them as evidence, identify the conditional independence assumptions rather than silently counting repeated predictions as independent measurements. Distinguish calibration of a perceptual event, calibration of action success, and coverage of an entire trajectory.

Freeze the model before fitting a held-out calibration rule, unless the statistical argument explicitly covers adaptation. Split by environment or episode as appropriate; repeated frames from one trajectory are not independent held-out trials. An exchangeability-based bound does not automatically survive policy-induced distribution shift, sequential selection, or updating the model after calibration.

For adaptive reasoning or prediction horizons, compare the state-dependent gate with a replayed or shuffled schedule having the same empirical call and horizon budget. Include fixed-budget alternatives and report the quality-cost curve. This distinguishes useful state-dependent allocation from benefits due only to more calls, shorter horizons, or a favorable average schedule. The current follow-up audit records a particularly relevant schedule-replay control in Adaptive Rollout Truncation; its precise evidence belongs in the follow-up matrix.

For persistent adaptation, report memory reset boundaries, episode order, number of prior episodes, access to success labels or ground-truth paths, and all parameter updates. Compare feedback-free, self-assessed, and oracle-feedback conditions separately. Preserve an untouched evaluation split for any threshold, stopping rule, prompt, or memory policy chosen during development.

## Reporting and falsification

Before using an official release as a baseline, verify the model path selected by its actual entrypoint and the evaluator used to produce the metric. Run small invariance or unit checks for a suspected release defect, then compare against a corrected baseline where appropriate. Keep the upstream snapshot and the correction identifiable. A defect in the inspected release does not establish that the published experiment used that version, and fixing it is not by itself a new scientific mechanism.

Use paired episode comparisons for systems evaluated on the same tasks. Aggregate uncertainty at the scene or episode level rather than pretending all action steps are independent. Separate model-seed variation from evaluation-sample uncertainty. Report task-appropriate path fidelity, grounding, efficiency, collisions or control failures, and end-to-end latency alongside success. Include model/API version and all inference calls in cost accounting.

Before a full training run, specify the cheapest test that could invalidate the proposed mechanism. A failure of that test should narrow or stop the proposal, even if a larger system could obtain a higher aggregate score for unrelated reasons. Never fill the report with predicted numerical gains that have not been measured.

Relevant source detail is in the [dossier index](../inventory/Dossier_Index.md), [paper bottleneck ledger](../evidence/paper_bottlenecks.json), and [audit of earlier premises](Previous_Idea_Claim_Audit.md).
