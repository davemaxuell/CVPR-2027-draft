# Completed VLN dataset inventory

Verified at: 2026-10-21T17:51:35.936465+00:00

All paths below are relative to this data directory. VLN-CE covers R2R-CE and RxR-CE.

| Dataset | Location | Coverage |
|---|---|---|
| R2R | `datasets/R2R` | Original discrete navigation release |
| RxR | `datasets/RxR` | Original multilingual release, text features and pose traces |
| REVERIE | `datasets/REVERIE` | v1 and v2 annotations with matching bounding boxes |
| SOON | `datasets/SOON` | All four publicly linked original FAO/SOON splits |
| CVDN | `datasets/CVDN` | Public train, val_seen and val_unseen dialog splits |
| NDH | `datasets/NDH` | Navigation instances from dialog history, including public test |
| HANNA | `datasets/HANNA/hanna` | Five task splits and assistance assets |
| R2R-CE | `datasets/R2R_VLNCE_v1-3`, `datasets/R2R_VLNCE_v1-3_preprocessed` | VLN-CE: minimal and preprocessed releases, including EnvDrop augmentation |
| RxR-CE | `datasets/RxR_VLNCE_v0` | VLN-CE: continuous episodes with shared RxR text features |
| R4R | `datasets/R4R` | Train, val_seen and val_unseen constructed with the official recipe |
| GSA-R2R | `datasets/GSA-R2R` | 28 official annotation files, ScaleVLN graphs and CLIP ViT-B/16 features |

Shared assets: `scene_datasets/mp3d`, `mattersim/v1/scans`, `connectivity`, `img_features`.

Verification reports (including split counts): `splits/dataset_completion_report.json`, `splits/additional_datasets_completion_report.json`.

Public-release limitations:

- RxR publishes six empty follower pose files, preserved exactly as released; see splits/rxr_upstream_empty_objects.json.
- SOON val_seen_instrs is not published on the official download page.
- The official CVDN dialog downloader publishes no test dialog file.
- The official R4R recipe covers train, val_seen and val_unseen; no separate test release is published there.
- Current official R4R inputs/generator yield 25,921 train and 5,026 val_unseen paths, differing from the README example (25,930 and 5,018). All eligible pairs and composed instructions are independently verified; original annotations and generator are unchanged.
- GSA-R2R includes the official instructions, ScaleVLN graphs and CLIP ViT-B/16 features required by its baseline. Optional checkpoints, the regenerable pano_inputs.h5 cache and raw HM3D rendering meshes are not included.
- Private evaluation ground truth and trained model weights are outside this dataset mirror.

Other existing directories, not audited by the VLN checks:

- `datasets/PRM800K`
