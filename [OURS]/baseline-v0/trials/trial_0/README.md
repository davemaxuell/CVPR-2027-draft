# trial_0 — SmolVLM2 + DINO world models

**Complete end-to-end pilot, steps 0–10.** Final reporting completed on
28 October 2026 at 05:59 KST (27 October at 20:59 UTC). Organized and checked on
28 October 2026. [Trial manifest](trial.json) · [Step index](steps.csv) ·
[Final report](results/RESULTS.md) · [CSV metrics](results/metrics.csv) ·
[Organization audit](verification.json).

This trial uses SmolVLM2-500M with decoder LoRA, a transferred DINO-WM PointMaze
predictor and a matched freshly initialized predictor, one pooled scorer, and a
matched learned-null-future scorer. Both predictors use the same frozen DINOv2
encoder. Data are R2R-CE and English-guide RxR-CE, with A for policy/world-model
fitting, B for scorer fitting, C for development, and E for final evaluation.

## Steps and evidence

| Step | Work | Completed output | Record |
| --- | --- | --- | --- |
| 0 | Environment and interface checks | R2R/RxR expert replay, policy forward/backward, scorer design and model compatibility checks | [Setup](steps/00_setup/README.md) |
| 1 | Fix the experiment and prepare experts | Frozen settings and A/B/C/E partition; 21,333 clean expert episodes / 2,218,976 SFT samples | [Manifest](steps/01_experiment/experiment_manifest.yaml), [expert data](data/expert_groupA/) |
| 2 | Policy SFT | 4,096 updates; selected update 3,072; C64 success 4/64 | [Results](steps/02_policy_sft/RESULTS.md) |
| 3 | DAgger refinement | 256 A episodes, 125,730 corrective labels, 2,048 updates; Step 2 retained as control | [Results](steps/03_dagger/RESULTS.md) |
| 3b | STOP refinement and selection | User selected refined update 512 with STOP logit bias −0.25; C224 success 29/224 | [Study](steps/03b_stop_refinement/result.json), [selection](steps/04_world_model_data/selection_decision.json) |
| 4 | World-model transition data | 2,488,382 movement transitions; 2,422,469 matched training samples | [Results](steps/04_world_model_data/RESULTS.md) |
| 5A | DINO transfer/fresh training | 37,852 updates per model; both selected update 36,864 | [Results](steps/05_dino_world_models/RESULTS.md) |
| 6 | Measure scorer labels | 1,024 B roots + 256 C roots; all four actions; 5,120 branches | [Results](steps/06_scorer_labels/RESULTS.md) |
| 7 | Cache frozen features | 1,280 roots, 28,859 real-prefix updates, 25,020 distinct images | [Results](steps/07_scorer_features/RESULTS.md) |
| 8 | Train pooled and null scorers | 2,048 updates each; training audit passed | [Training audit](steps/08_scorer_training/training_verification.json) |
| 9 | Calibrate on C and freeze | Pooled update 1,536; null update 1,408; both margins 1,000,000; base C224 actions reproduced | [Selection](steps/09_calibration/result.json) |
| 10 | Evaluate five systems on E | 224 episodes per system; 1,120 records and 363,790 action events audited | [Final results](results/RESULTS.md), [final audit](results/verification.json) |

The original training specification numbers steps 1–10. `00_setup` is an
organizational label for the earlier environment/interface checks, not an extra
training stage. Step 3b is retained because it explains the policy actually used
in steps 4–10. The pretrained C64 comparison is in
[runs/smolvlm2_pretrained_baseline_C64/](runs/smolvlm2_pretrained_baseline_C64/).

## Selected checkpoints

| Component | Trial path | Selection |
| --- | --- | --- |
| Navigation policy | [checkpoints/policy/](checkpoints/policy/) | Refined update 512; STOP bias −0.25 is required |
| SFT control | [checkpoints/policy_control/](checkpoints/policy_control/) | Step 2 update 3,072, frozen after Step 3 |
| DINO encoder | [checkpoints/encoder/](checkpoints/encoder/) | Shared frozen DINOv2 ViT-S/14 |
| Transferred dynamics | [checkpoints/dinowm_transfer/](checkpoints/dinowm_transfer/) | Update 36,864; C feature MSE 1.347414 |
| Fresh dynamics | [checkpoints/dinov2_fresh_dynamics/](checkpoints/dinov2_fresh_dynamics/) | Update 36,864; C feature MSE 1.320767 |
| Pooled scorer | [checkpoints/scorer_pooled/](checkpoints/scorer_pooled/) | Update 1,536; override margin 1,000,000 |
| Null-future scorer | [checkpoints/scorer_null/](checkpoints/scorer_null/) | Update 1,408; override margin 1,000,000 |

The policy bundle contains its adapter and processor; the base SmolVLM2 weights
remain a shared model dependency recorded in `manifest/model_revisions.json`.
Load the selected policy with `policy.experimental_reference.ExperimentalReference`
so its saved inference settings and STOP bias are applied. Original per-run source
snapshots and provenance remain accessible through `runs/`.

## Final E pilot

Every system ran the same 112 R2R + 112 RxR episodes across all 11 E scenes.

| System | Successes | SR | SPL | nDTW | Overrides |
| --- | ---: | ---: | ---: | ---: | ---: |
| Policy only | 21/224 | 9.375% | 5.779% | 25.430% | 0 |
| Learned-null-future scorer | 21/224 | 9.375% | 5.779% | 25.430% | 0 |
| Actual-next-view scorer (privileged) | 21/224 | 9.375% | 5.779% | 25.430% | 0 |
| DINO-WM transfer + pooled scorer | 21/224 | 9.375% | 5.779% | 25.430% | 0 |
| Fresh DINO dynamics + pooled scorer | 21/224 | 9.375% | 5.779% | 25.430% | 0 |

Calibration selected a conservative fallback that preserves base-policy actions.
The pipeline completed successfully, but **this trial demonstrates no navigation
gain from the scorer or predicted futures**. The feature-MSE improvements in Step 5
are prediction results, not navigation improvements.

This is a completed pilot: policy SFT used 262,144 samples, approximately 0.1181
passes of the available clean data, and E evaluation used 224 of the 5,508 official
unseen episodes. It is not a full-data or convergence training claim. The selected
policy failed the original stopping safeguard; the user's explicit selection and
the historical control are preserved. RxR uses 15° turns rather than the official
30° protocol. Five unreachable-goal episodes remain in every denominator; mean
navigation error is infinite, with failed native NaN SPL normalized to zero under
the documented reporting amendment.

## Where everything lives

```text
trial_0/
  README.md, trial.json, steps.csv    overview and experiment identity
  steps/                             ordered access to setup and steps 1–10, including 3b
  checkpoints/                       selected frozen bundles and policy control
  results/                           copied final report, metrics, protocol, and audit
  manifest/                          copied experiment settings and model revisions
  docs/                              protocol-document snapshot
  Scorer/                            scorer source and checks referenced by the design docs
  runs/                              links to the full original runs and source snapshots
  data/                              links to the corresponding processed datasets
  provenance/                        recorded stop of later automatic continuations
  checksums.sha256, verification.json organization and artifact verification
```

Large artifacts use relative symbolic links; original files stay in place so
embedded paths and migration tools continue to work. Copied documents retain their
historical stage-specific wording; this README and `trial.json` give the final
trial status. The final report is copied from `recovery_v2/`, including its amended
protocol; `results/original_protocol.json` preserves the initial protocol. The
initial partial evaluation remains under Step 10's `evaluation/`; authoritative
completed episode records are linked at [results/episodes/](results/episodes/).

The organization audit checks recorded stage completion, frozen-bundle hashes,
final episode hashes/counts/metrics, snapshot equality, and links. It does not
rerun training or simulator evaluations, or rehash the complete image/feature
corpora; their saved stage audits are preserved. Recheck the catalog from this
directory with `sha256sum --quiet -c checksums.sha256`.

NWM/CDiT (Step 5B) and the four-model repeats (8b–10b) are outside this completed
trial. They were stopped by user request on 28 October at 06:28 KST; automatic
continuation is disabled, as recorded in
[provenance/continuation_stop.json](provenance/continuation_stop.json).
Future work starts under a new trial ID; [trial_1 is available](../README.md).
