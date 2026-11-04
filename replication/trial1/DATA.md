# Data required on the receiving machine

GitHub carries the training code and metadata, not the datasets or rendered trajectories. Obtain the licensed assets through authorized access or transfer from the original server; do not put scenes, rendered frames or model caches into this repository.

Under `DATA_ROOT` provide:

```text
datasets/R2R_VLNCE_v1-3_preprocessed/
datasets/RxR_VLNCE_v0/
scene_datasets/mp3d/
processed/baseline_v0/expert_groupA/
processed/baseline_v0/expert_groupC_600/
processed/baseline_v0/trial_1/full_train_english_guide/expert/
processed/baseline_v0/trial_1/full_train_english_guide/val_seen_diagnostics_600/
```

The annotation releases must include train/val-seen/val-unseen episodes and ground truth. Include the RxR guide/follower ground-truth files consumed by the official nDTW loader, while evaluated/trained instructions remain **English guide only**. Scene GLBs and navmeshes must resolve under the paths used by the episode loaders.

The four processed directories are model-independent expert renders. Prefer transferring the existing directories intact, including indexes, render manifests, metadata and RGB frames. `DATA_IDENTITIES.json` records the original index/manifest hashes for transfer checks. Do not transfer old 4B DAgger data as the new 2B training round.

| Data | Usable episodes | Windows |
|---|---:|---:|
| Group-A experts | 21,330 | 79,814 |
| Group-C diagnostics | 596 | 1,736 |
| Full official train experts | 30,221 | 109,784 |
| Official val-seen diagnostics | Audit after transfer/rendering | 1,860 |

If renders are missing, use the isolated, path-adjusted working copies produced by `prepare_sources.py`. After sourcing the generated `environment.sh`, render the missing roots sequentially on the admitted GPU:

```bash
"$SIM" "$CODE_A/tools/render_expert.py" --datasets r2r rxr \
  --split train --group A --workers 24 --gpu 0 --out "$EXPERT_A"

"$SIM" "$CODE_A/tools/render_expert.py" --datasets r2r rxr \
  --split train --group C --workers 16 --gpu 0 --max-episodes 300 \
  --selection scene_balanced --seed 0 --out "$VAL_C"

"$SIM" "$CODE_FULL/tools/render_expert.py" --datasets r2r rxr \
  --split train --group FULL_TRAIN --workers 24 --gpu 0 --out "$EXPERT_FULL"

"$SIM" "$CODE_FULL/tools/render_expert.py" --datasets r2r rxr \
  --split val_seen --group FULL_VAL_SEEN --workers 12 --gpu 0 \
  --max-episodes 300 --selection scene_balanced --seed 0 --out "$VAL_SEEN"
```

The source render plans differ in task chunk sizes; preserve the source selection/seed and actual selected episode IDs. A regenerated manifest may differ in paths, workers and timing, so its byte hash need not match the source manifest. Re-run the substantive source/episode/frame/action/window audits and document regeneration instead of claiming an identical transfer.

Audit group-A/C with `CODE_A/tools/audit_stream_data.py` and full-train/val-seen with `CODE_FULL/tools/audit_stream_data.py`. The latter has explicit full-split guards. Both must pass with the expected counts, exclusions, frame alignment and token budget.

The new 2B group-A collection selects 21,333 episodes but retains a policy-dependent number. S2's budget follows that audited result. F includes the entire full expert set, including previously used group-A examples. Original group C cannot serve as held-out F validation.

Full navigation evaluation coverage is R2R seen 778, R2R unseen 1,839, and RxR English-guide unseen 3,669 episodes. Regenerate the RxR split audit using the packaged audit implementation bound to local paths; verify scene/role/language and ordered episode identity, not just counts.
