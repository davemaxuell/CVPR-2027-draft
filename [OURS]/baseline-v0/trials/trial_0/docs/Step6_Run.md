# Step 6 — measured scorer labels

Run: `runs/step6_smolvlm2_refined512/`. The [locked protocol](../runs/step6_smolvlm2_refined512/protocol.json)
and [episode queues](../runs/step6_smolvlm2_refined512/selection.json) define collection. Group B supplies
fitting labels and group C supplies separate development labels. No group-A/E roots are included.

The continuation policy is the [versioned Step 4 reference](Step4_Run.md): refined SmolVLM2 update
512, STOP logit bias −0.25, native saved preprocessing and batch-one greedy four-action decoding.
Its identity is `729480bf5c8402b07cad653a7ef0f7124be212d50255077290316ba686777ca7`.
The user’s deliberate selection despite the failed stopping safeguard and the Step 2 control remain
unchanged. The trained DINO models do not select actions or supply labels during this step.

## Completed corpus

Completed on 26 October 2026 at 23:09 KST. The full collection took 11 hours 26 minutes and
reused the 64 pilot roots unchanged. The [results](../runs/step6_smolvlm2_refined512/RESULTS.md)
and [machine-readable handoff](../runs/step6_smolvlm2_refined512/result.json) record completion.

| Group | Purpose | Roots | Branches | Source episodes | Scenes |
| --- | --- | ---: | ---: | ---: | ---: |
| B | Fitting | 1,024 | 4,096 | 128 | 9 |
| C | Development | 256 | 1,024 | 33 | 8 |

All allowed B/C scenes are represented, with equal root counts in the four dataset/prefix-source
strata. Two B/RxR expert carriers failed the predeclared replay filter and were excluded; all
exclusions are recorded in [exclusions.json](../runs/step6_smolvlm2_refined512/exclusions.json).
The C/RxR expert stratum uses nine episodes because short eligible histories were retained.

The [full audit](../runs/step6_smolvlm2_refined512/full/verification.json) passed for all 5,120
branches, hashing and checking headers of 34,203 image files and fully decoding 6,727 selected
images for raw-pixel verification. Policy identity, STOP bias, preprocessing, original budgets,
continuation decoding and full-path metrics passed; the Step 2 control and Step 5 bundles are intact.

`catalog_B.jsonl` and `catalog_C.jsonl` in the corpus below are the verified Step 7 inputs.
The corpus manifest records their SHA-256 hashes and canonical action IDs. Mean measured
best-candidate utility gain over the base action is 0.198693 on B and 0.183798 on C.
These are privileged branch diagnostics from mixed roots, not navigation success rates or
measured scorer improvements. No scorer has been trained in this step.

## Collection protocol

Each dataset has disjoint expert-prefix and selected-policy episode queues, balanced across scenes
with seed 6. Policy trajectories are retained regardless of success. Expert prefixes qualify after
actual replay succeeds, consumes the complete expert schedule, and stays within 0.5 m of reference
forward locations. Failed expert replays and histories that terminate before four observations are
recorded explicitly. Short eligible policy histories remain eligible.

Choose up to eight evenly spaced decision indices per trajectory, from t=3 to the final decision.
These include late decisions and STOP decisions when present. Preserve every actual prefix frame,
executed action, pose, collision, metric and raw RGB hash. Images are lossless 640×480 PNGs; policy
inference sees the live RGB with the saved native processor.

For each root, reset the original episode and seed, replay every prefix action, and check exact
RGB hashes, poses, metrics, collisions and the elapsed action counter. Execute each of STOP,
FORWARD, LEFT and RIGHT once. After a movement, the same frozen policy continues on the actual
resulting observations and executed-action history. Each branch ends at STOP or the original
500-action R2R / 700-action RxR limit. RxR retains the experiment’s declared 15-degree turns.

Each action record stores its first-next RGB, complete remaining action/pose/collision trace,
raw continuation logits, terminal metrics and utility. STOP executes through Habitat, reuses the
current view, consumes one action and has zero remaining collisions. Only forward collision events
contribute to costs; the immediate collision label applies only to the FORWARD candidate.

Utility is `0.70 * success + 0.30 * nDTW - 0.01 * n/Bmax - 0.02 * c/Bmax`, using the unchanged
[label helper](../Scorer/labels.py) conventions. Success is the configured Habitat terminal success.
nDTW executes the pinned [VLN-CE evaluator method](https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/habitat_extensions/measures.py#L218-L268)
on the complete actual trajectory with exact DTW, Euclidean cost, consecutive duplicate suppression
and 3 m normalization. Its compiled distance backend is checked against an independent full-matrix
implementation and the existing policy evaluator. References and realized poses are used only for
label construction and verification.

## Pilot, scaling and verification

The [64-root pilot](../runs/step6_smolvlm2_refined512/pilot/RESULTS.md) completed and passed its audit:
16 roots per dataset/prefix-source combination, eight source episodes and 256 measured branches.
The base candidate succeeded at 27 roots; the best measured candidate at 41 roots. Mean measured
utility gain was 0.231794. This two-scene pilot establishes nondegenerate labels, not generalization.

The [locked larger budget](../runs/step6_smolvlm2_refined512/plan.json) is **1,024 group-B fitting roots
and 256 group-C development roots**, giving 5,120 branches. Each group has equal counts in the four
dataset/prefix-source strata. At most eight roots per trajectory implies at least 128 distinct fitting
episodes and 32 development episodes. Queues span all allowed scenes; actual coverage is reported.
This is an initial experimental scorer dataset. The pilot estimate was roughly 12 hours of
collection; the full stage finished in 11 hours 26 minutes.

The larger stage reuses the pilot’s exact recorded roots. Changing stage quotas does not change
policy, root selection order or branch semantics. Run or resume it with the `rolevln` environment:
`python tools/run_step6_full.py`. [Execution status](../runs/step6_smolvlm2_refined512/status.json)
records collection, audit and final publication.

Four independent workers use physical GPU 0, each preserving native batch-one inference. Completed
carrier trajectories and action branches are atomic and resumable. Source files, dataset archives,
scene partition, policy and evaluator identities are hashed in the protocol. No model is trained here.

`tools/check_step6.py` checks full-path nDTW, STOP at a successful expert endpoint, native preprocessing
after a forced action, branch order/repeat identity, blocked forward costs and all actions at the
last permitted budget step in both datasets. `tools/verify_step6.py` checks every completed label,
continuation logit decision, count, metric and image hash, and reports successes, timeouts, distinct
outcomes, label gaps and best-measured-candidate advantage over the base action. That advantage is
a privileged branch diagnostic, not a measured scorer or closed-loop navigation improvement.

The original pilot audit compared stored SDTW with an independently recomputed exponential using
exact equality. All stored labels were internally exact; 27 independent values differed by at most
1.11e−16. The [audit-only revision](../runs/step6_smolvlm2_refined512/pilot/audit_revision.json) retains
exact stored-value consistency and uses the existing 1e−12 tolerance for independent recomputation.
The original source and log are preserved; no collection semantics or measured labels changed.

The output corpus is
`/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/scorer_labels_smolvlm2_refined512_v1/`.
Step 7 will extract frozen policy/visual features and predictions for these measured labels.
