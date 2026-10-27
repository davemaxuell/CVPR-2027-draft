# IR2R-CE local release

Downloaded from the [official IVLN-CE repository](https://github.com/jacobkrantz/IVLN-CE)
on 2026-10-22. The core benchmark's data dependencies are verified and present.
This is a dataset installation; the simulator and baseline model environment are separate.

| Split | Tours | Referenced episodes | Available episodes |
|---|---:|---:|---:|
| train | 222 | 10,668 | 10,668 |
| val_seen | 156 | 747 | 747 |
| val_unseen | 36 | 1,824 | 1,824 |
| envdrop (optional augmentation) | 3,283 | 142,940 | 142,835 |

- `tours.json`: original tour ordering for all four splits.
- `gt_ndtw.json`: original target trajectories for all 414 core tours.
- `known_maps/`: 142 extracted NPZ maps, covering 71 scenes with both ground-truth
  and predicted semantics. `known_maps.zip` preserves the original archive.
- `episodes/`: relative link to the existing `../R2R_VLNCE_v1-3` release.
- `preprocessed/`: relative link to `../R2R_VLNCE_v1-3_preprocessed`, including EnvDrop.
- `scene_datasets/`: relative link to the existing shared scene assets.

The upstream configurations expect paths relative to a project containing `data/`:

```yaml
DATASET:
  TYPE: Iterative-VLN-CE
  SPLIT: train
  DATA_PATH: data/datasets/R2R_VLNCE_v1-3/{split}/{split}.json.gz
  SCENES_DIR: data/scene_datasets/
  TOURS_FILE: data/tours.json
```

`../../tours.json`, `../../gt_ndtw.json`, and `../../known_maps` link to this release
so the authors' default paths work without another copy. The preprocessed path for
EnvDrop is `data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz`.

The published EnvDrop ordering references 105 IDs absent from the corresponding
published v1-3 preprocessed episode file. All available matching IDs also match the
correct scene. See [envdrop_missing_episode_ids.json](envdrop_missing_episode_ids.json).
The original ordering is preserved; no episodes were invented or substituted.
This affects optional augmentation only; every core benchmark episode and target is present.
The combined `joint_train_envdrop` file uses different IDs and is not a replacement.

All three downloads match their source sizes. Both JSON files were parsed and their
core tour/episode/scene/target references checked. The outer map ZIP, extracted files,
and all inner NPZ members passed CRC checks. Drive does not publish a checksum for
these files; locally computed SHA-256 receipts are in [source_manifest.json](source_manifest.json).
The full audit is [ce_expansion_completion_report.json](../../splits/ce_expansion_completion_report.json).

The original documentation and license are `README.upstream.md` and `LICENSE`.
Model checkpoints are not part of this dataset installation.
