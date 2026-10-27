# Step 6 complete — measured scorer labels

Completed: 2026-10-26T14:09:12.486467+00:00

Each root has four actual simulator outcomes: forced STOP/FORWARD/LEFT/RIGHT, then the fixed refined SmolVLM2 policy for movement branches. The selected policy is checkpoint 512 with STOP logit bias −0.25, native saved preprocessing, and batch-one greedy decoding.

| Group | Purpose | Roots | Branches | Distinct source episodes | Base successful branches | Best-utility successful branches | Mean best-candidate utility gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B | Fitting | 1,024 | 4,096 | 128 | 220 | 410 | 0.198693 |
| C | Development | 256 | 1,024 | 33 | 64 | 109 | 0.183798 |

The best candidate uses measured outcomes and is a privileged oracle diagnostic. These are branches from mixed expert/policy roots, not end-to-end episode success rates or scorer improvements. Results by dataset and root source, including successes, timeouts, label gaps and distinct outcomes, are in the verification report.

Group B and C are scene-disjoint. All four dataset/prefix-source strata have equal root counts within each group. Episode queues and evenly spaced decision indices were fixed independently of branch outcomes. Valid expert prefixes require successful low-drift replay; policy failures are retained. Short eligible histories are retained, and exclusions are listed in [exclusions.json](exclusions.json).

Every branch restores the original episode, random seed, actual prefix, metric state and remaining budget by verified replay. nDTW includes the full earlier trajectory. Counts include the candidate and any executed terminal STOP, with only forward collisions charged. STOP executes through the evaluator and uses the current RGB as its next-view identity.

[Live runtime checks](check_runtime.json) and [complete label audit](full/verification.json) passed. The audit independently recomputed every utility and nDTW, checked every continuation decoding decision and budget, and hashed all indexed image files. The saved policy, failed-safeguard selection record, Step 2 control and Step 5 models are preserved.

The DINO world models were not used for action selection or label construction. Step 7 will extract frozen features and each world model’s predictions for the same measured labels. The NWM/CDiT pair remains a later repeat.

Corpus: `/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/scorer_labels_smolvlm2_refined512_v1`. `catalog_B.jsonl` is for fitting; `catalog_C.jsonl` is for development. Both record canonical action IDs and immutable image/label references. See [methods](../../docs/Step6_Run.md), [locked protocol](protocol.json), and [result.json](result.json).
