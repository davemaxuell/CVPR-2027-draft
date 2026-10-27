# RoleVLN (CVPR 2027 draft)

Vision-and-language navigation in continuous environments (R2R-CE, RxR-CE): a policy → world model → scorer
pipeline, the prior-work baselines it is compared against, and an interactive navigation explorer.
Private working repository. Datasets are not included.

**Moving to a new machine? Start with [MIGRATION.md](MIGRATION.md).**

| Document | What it covers |
|---|---|
| [MIGRATION.md](MIGRATION.md) | What is in git / in the release / not included; final sync on the old server; full setup on a new machine |
| [TRAINING_STATE.md](TRAINING_STATE.md) | What has been trained, results, what was running at the snapshot, diagnoses and next steps, 16 GB VRAM notes |
| [DATA_MIGRATION.md](DATA_MIGRATION.md) | Every dataset and public weight file: needed or not, size, where to download it |
| [`[OURS]/baseline-v0/README.md`](%5BOURS%5D/baseline-v0/README.md) | The vanilla baseline in detail (layout, environment, every step's run record) |

## Layout

```text
[OURS]/baseline-v0/           our pipeline: policy (SmolVLM2-500M + LoRA), DINO/NWM world models, scorer, Habitat sim
  docs/ manifest/ policy/ wm/ Scorer/ sim/ tools/ env/ third_party/ runs/ logs/
[prior works]/
  ScaleVLN/                   ScaleVLN/DUET (patched) + MatterSim; analysis/vln_viz = the navigation explorer
  ETPNav/                     ETPNav on R2R-CE/RxR-CE + tools/log_etpnav_steps.py (explorer logger)
  VLN-DUET/                   original DUET, kept for diffing
  CANDIDATE_MODELS.md         VLN-CE models to add next
data/                         dataset docs, download/verification tools, split lists (the data itself is not in git)
drafts/                       research investigation and review drafts (2026-10-22, 2026-10-23)
migration/                    export_repo.py, release_assets.py, relocate_paths.py, manifests
```

Our checkpoints: the selected and frozen ones are committed under `[OURS]/baseline-v0/runs/`. All the
others, and the explorer's run logs, are in the GitHub release `assets-v1`:
`python3 -I migration/release_assets.py download`.
