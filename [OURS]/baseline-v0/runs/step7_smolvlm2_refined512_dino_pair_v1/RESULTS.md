# Step 7 complete — frozen scorer features

Completed: 2026-10-26T20:44:30.080209+00:00

The cache preserves refined SmolVLM2 checkpoint 512, STOP bias −0.25, its native saved processor and batch-one decoding. Its deliberate selection despite the failed stopping safeguard remains recorded. Step 2 and Step 5 artifacts are unchanged.

There are 1,024 B fitting roots and 256 C development roots from 161 episodes, with 28,859 real observation updates and 25,020 distinct encoded images. All 5,120 measured action labels are retained. Each DINO model supplies 3,840 movement predictions; STOP is exact current-view identity.

Instruction token states and each actual prefill context are stored in BF16. Every instruction is complete within the saved context limit. Real DINO patches use the frozen FP32 encoder, FP16 raw storage, and FP32 2×2 pooling to 49×384. Encoding uses one image per native forward: large batches can select TF32 cuDNN patch convolution, documented in [the diagnostic](dino_batch_diagnosis.json). Both world models use the same actual history and canonical commands.

The cache includes every real prefix needed to recompute trainable scorer memory. It contains no cached scorer memory. Actual next views and both predicted views carry identical measured labels. The initial comparison includes the DINO transfer/fresh pair; NWM/CDiT requires a newly versioned pooled dataset and scorer.

[Runtime checks](check_runtime.json) and [full verification](verification.json) passed, including exact native policy replay, exact sampled encoder/prediction replay, all cache hashes and STOP identities, and the scorer memory/shape interface. The scorer API check used discarded random weights; no scorer was trained.

| Model | B movement pooled MSE | C movement pooled MSE |
| --- | ---: | ---: |
| dinowm_transfer | 0.674154 | 0.660589 |
| dinov2_fresh_dynamics | 0.651973 | 0.638014 |

These feature errors describe mixed expert/policy roots, not navigation success or scorer gains. Both frozen models are retained.

Cache: `/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/scorer_features_smolvlm2_refined512_dino_pair_v1`. `catalog_B.jsonl` is for fitting and `catalog_C.jsonl` for development. `scorer_config.json` records the initial pooled pair and equal model weights. Use `scorer_features.dataset.FrozenScorerDataset` to load prefixes, candidates and labels; recompute memory under current scorer weights in Step 8.

See [machine-readable results](result.json), [locked protocol](protocol.json), and [methods](../../docs/Step7_Run.md).
