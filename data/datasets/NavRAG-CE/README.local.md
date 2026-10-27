# NavRAG-CE local release

This is the authors' continuous navigation conversion released with
[Dynam3D](https://github.com/MrZihan/Dynam3D), downloaded on 2026-10-22 from
[MrZihanWang/Dynam3D](https://huggingface.co/datasets/MrZihanWang/Dynam3D).
Revision: `0fe50d663bb4098adabd02006851f87b61bb88e7`.

**All published annotations and oracle paths are verified. The HM3D training-scene
download is booked for 2026-10-23 at 05:00 Asia/Seoul. Approved Matterport credentials
must be configured before then. Validation data dependencies are ready; training
requires the missing 612 HM3D scenes.**
See [required_scenes.json](required_scenes.json) and the latest
[completion report](../../splits/ce_expansion_completion_report.json) for current status.

| Split | Episodes | Scenes | Local entry point |
|---|---:|---:|---|
| train | 406,365 | 612 HM3D | `train/` (per-scene JSON lists) |
| val_unseen | 6,932 | 11 Matterport3D | `val_unseen/val_unseen.json.gz` |

Original release bytes remain in `original/data/datasets/`:

- `navrag_training_data/`: 612 per-scene JSON files; `train/` links here.
- `navrag_train_gt.json`: target paths/actions for all 406,365 training episodes;
  `train_gt.json` links here.
- `navrag_val_unseen_data.json` and `navrag_val_unseen_gt.json`.

The validation folder contains deterministic gzip copies of the original data and GT
JSON files. Its GT path is `val_unseen/val_unseen_gt.json.gz`. Decompressed copies were
checked against the original bytes. Training files are per-scene JSON lists and need
the authors' directory loader or an equivalent loader, not a single-file gzip loader.
This release publishes no separate `val_seen` or test split.

`scene_datasets/` links to the shared `../../scene_datasets/` directory. Set
`SCENES_DIR: data/scene_datasets/` when working from the RoleVLN project root.
Expected paths include:

```text
data/scene_datasets/hm3d/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb
data/scene_datasets/hm3d/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.navmesh
data/scene_datasets/mp3d/<scan>/<scan>.glb
data/scene_datasets/mp3d/<scan>/<scan>.navmesh
```

All validation MP3D files exist. Existing GSA-R2R HM3D features and connectivity graphs
do not supply the HM3D rendering meshes needed here. The scheduled download uses
approved Matterport access as described in the
[official Habitat HM3D instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d).
The credential file is `/home/bufsgpu/.config/rolevln/hm3d.netrc` (owner-only access).
Do not paste credentials into documentation or chat. See the
[schedule](../../splits/navrag_hm3d_download_schedule.json),
[current job status](../../splits/navrag_hm3d_download_status.json), and
[setup notes](../../tools/NAVRAG_HM3D_SCHEDULE.md).

After installing the required scenes under the shared scene root, refresh the audit
and inventory from the RoleVLN root:

```bash
/usr/bin/python3 data/tools/complete_ce_expansion.py
```

All 615 original files match the authors' pinned Git-blob or LFS checksums. Episode IDs,
poses, instructions, reference paths, and train/validation target coverage were
checked. This is a data audit, not a simulator/baseline execution test.

See [source_manifest.json](source_manifest.json). `README.upstream.md`, `README.hub.md`,
and `upstream/` preserve the source documentation and conversion scripts. The Hub
metadata declares Apache-2.0; scene assets retain their own source terms.
No model checkpoints were downloaded.
