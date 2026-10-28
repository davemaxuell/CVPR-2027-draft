# Step 10 — controlled unseen-scene pilot

Completed: 2026-10-27T20:59:37.048038+00:00

All five systems ran the same 224 episodes: 112 R2R and 112 English-guide RxR episodes across all 11 E scenes. This is a scene-balanced pilot, not the complete 5,508-episode unseen benchmark. Checkpoints, margins and preprocessing were frozen on C before any E results.

Refined SmolVLM2 checkpoint 512 and STOP bias −0.25 remain the reference. Its deliberate selection despite the original stopping-safeguard failure remains recorded. The Step 2 control and upstream frozen artifacts are intact. RxR uses the agreed 15-degree turns, differing from official 30-degree RxR-CE.

| System | SR % | SPL % | NE m | nDTW % | SDTW % | Forward collision % | Mean actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| policy | 9.38 | 5.78 | ∞ | 25.43 | 5.81 | 21.66 | 324.8 |
| null | 9.38 | 5.78 | ∞ | 25.43 | 5.81 | 21.66 | 324.8 |
| actual (privileged) | 9.38 | 5.78 | ∞ | 25.43 | 5.81 | 21.66 | 324.8 |
| dinowm_transfer | 9.38 | 5.78 | ∞ | 25.43 | 5.81 | 21.66 | 324.8 |
| dinov2_fresh_dynamics | 9.38 | 5.78 | ∞ | 25.43 | 5.81 | 21.66 | 324.8 |

Reporting amendment: an unreachable RxR goal exposed native SPL = 0*(infinity/infinity), producing NaN. Every selected episode is retained. Such failed episodes receive SPL zero with the original NaN recorded; infinite navigation error is reported as infinity (JSON null plus an explicit flag). The 184 previously completed finite episodes and their actions were preserved exactly. No checkpoint, margin, preprocessing, episode selection or decision rule changed.

The actual-view control previews real simulator outcomes and is privileged. It uses the same pooled scorer and margin as the DINO pair. The null control was trained separately with matched architecture, initialization, root order and update budget.

| System | SR 95% scene interval | ΔSR vs policy, points (95% interval) | Decision mean / p95, ms | Sampled process peak, GiB |
| --- | --- | --- | ---: | ---: |
| policy | 5.39–13.88% | — | 159.6 / 168.9 | 5.75 |
| null | 5.39–13.88% | +0.00 (+0.00, +0.00) | 190.9 / 217.9 | 5.84 |
| actual | 5.39–13.88% | +0.00 (+0.00, +0.00) | 240.2 / 277.7 | 5.84 |
| dinowm_transfer | 5.39–13.88% | +0.00 (+0.00, +0.00) | 229.4 / 262.0 | 5.92 |
| dinov2_fresh_dynamics | 5.39–13.88% | +0.00 (+0.00, +0.00) | 230.5 / 262.2 | 5.92 |

The final report audit was rerun with a versioned fix for a local variable that overwrote the expected episode-ID set while checking imported records. All original checks remain active; no evaluation records or experimental settings changed. The selected conservative margins caused zero overrides in all controls, with no measured navigation gain over the base policy.

Intervals resample scenes with pairing preserved (2,000 resamples, seed 10). Eleven scenes limit uncertainty estimates. Decision latency includes contention from four concurrent workers. GPU process peaks include CUDA/Habitat allocations sampled every two seconds; exact PyTorch allocator peaks and sampling records are also retained. These H100 measurements do not establish an 8 GB deployment fit.

C measured-branch regret is development evidence and is recorded with checkpoint/margin selection; this E pilot reports closed-loop outcomes and does not add counterfactual E branch labels. Full event records include overrides, warm-up and prediction/scoring fallbacks.

[Machine-readable results](result.json), [all checks](verification.json), [CSV metrics](metrics.csv), [locked protocol](protocol.json), and [episode selection](selection.json).
