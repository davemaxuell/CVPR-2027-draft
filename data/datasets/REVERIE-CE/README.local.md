# REVERIE-CE local release

This is the authors' continuous navigation conversion released with
[Dynam3D](https://github.com/MrZihan/Dynam3D), downloaded on 2026-10-22 from
[MrZihanWang/Dynam3D](https://huggingface.co/datasets/MrZihanWang/Dynam3D).
Revision: `0fe50d663bb4098adabd02006851f87b61bb88e7`.

| Split | Episodes | Scenes | Local entry point |
|---|---:|---:|---|
| train | 9,626 | 58 | `train/` (per-scene JSON lists) |
| val_unseen | 3,189 | 10 | `val_unseen/val_unseen.json.gz` |
| test placeholders | 6,292 | 16 | `test/test.json.gz` |

All required Matterport3D meshes and navmeshes are present under the shared
`../../scene_datasets/` directory. `scene_datasets/` links to that directory.
Use `SCENES_DIR: data/scene_datasets/` from the RoleVLN project root.
All scene paths in the annotations retain the authors' `mp3d/<scan>/<scan>.glb` form.

Original release bytes remain in `original/data/datasets/`:

- `reverie_training_data/`: 58 per-scene training files; `train/` links here.
- `reverie_val_unseen_data.json` and `reverie_val_unseen_gt.json`.
- `reverie_test_data.json` and `reverie_test_gt.json`.

The split folders contain deterministic gzip copies of the original data and GT JSON
files for loaders expecting `.json.gz`. The GT files are named
`val_unseen/val_unseen_gt.json.gz` and `test/test_gt.json.gz`. Decompressed copies were
checked against the original bytes. Training files are per-scene JSON lists and need
the authors' directory loader or an equivalent loader, not a single-file gzip loader.

Release limitations:

- All 6,292 published test episodes have a one-point path, zero geodesic distance,
  and start equal to goal. These are placeholders, not valid full test ground truth.
  Use `val_unseen` for public evaluation.
- The release does not publish `val_seen` or a separate `reverie_train_gt.json`.
  Training episodes do include reference paths.
- This conversion describes continuous navigation. It does not add separate
  object-selection annotations. The original discrete REVERIE annotations remain
  separately available in `../REVERIE`.

All 62 original release files match the authors' pinned Git-blob or LFS checksums.
Episode IDs, poses, instructions, reference paths, target coverage, and scene files
were checked. This is a data audit, not a simulator/baseline execution test.

See [source_manifest.json](source_manifest.json), [required_scenes.json](required_scenes.json),
and the [completion report](../../splits/ce_expansion_completion_report.json).
`README.upstream.md`, `README.hub.md`, and `upstream/` preserve the source documentation
and conversion scripts. The Hub metadata declares Apache-2.0; scene assets retain
their own source terms. No model checkpoints were downloaded.
