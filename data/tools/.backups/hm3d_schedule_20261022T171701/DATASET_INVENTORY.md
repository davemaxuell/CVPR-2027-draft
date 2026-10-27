# VLN dataset inventory

Updated at: 2026-10-22T04:12:19.935873+00:00

All paths below are relative to this data directory. VLN-CE covers R2R-CE and RxR-CE.

| Dataset | Location | Coverage | Status |
|---|---|---|---|
| R2R | `datasets/R2R` | Original discrete navigation release | Verified |
| RxR | `datasets/RxR` | Original multilingual release, text features and pose traces | Verified |
| REVERIE | `datasets/REVERIE` | v1 and v2 annotations with matching bounding boxes | Verified |
| SOON | `datasets/SOON` | All four publicly linked original FAO/SOON splits | Verified |
| CVDN | `datasets/CVDN` | Public train, val_seen and val_unseen dialog splits | Verified |
| NDH | `datasets/NDH` | Navigation instances from dialog history, including public test | Verified |
| HANNA | `datasets/HANNA/hanna` | Five task splits and assistance assets | Verified |
| R2R-CE | `datasets/R2R_VLNCE_v1-3`, `datasets/R2R_VLNCE_v1-3_preprocessed` | VLN-CE: minimal and preprocessed releases, including EnvDrop augmentation | Verified |
| RxR-CE | `datasets/RxR_VLNCE_v0` | VLN-CE: continuous episodes with shared RxR text features | Verified |
| R4R | `datasets/R4R` | Train, val_seen and val_unseen constructed with the official recipe | Verified |
| GSA-R2R | `datasets/GSA-R2R` | 28 official annotation files, ScaleVLN graphs and CLIP ViT-B/16 features | Verified |
| IR2R-CE | `datasets/IR2R-CE` | 414 core tours; tour targets and 142 known maps; shared R2R-CE episodes | Ready; optional augmentation caveat |
| REVERIE-CE | `datasets/REVERIE-CE` | 9,626 train and 3,189 val_unseen episodes; public test placeholder files retained | Ready; test placeholders documented |
| NavRAG-CE | `datasets/NavRAG-CE` | 406,365 train and 6,932 val_unseen episodes; training oracle paths included | Validation ready; HM3D training scenes skipped |

Skipped scene downloads:

- NavRAG-CE: download of 612 HM3D training scenes skipped at the user's request. Annotations and oracle paths are verified; val_unseen uses existing MP3D scenes. Training cannot run without the omitted HM3D scene assets.

Shared assets: `scene_datasets/mp3d`, `mattersim/v1/scans`, `connectivity`, `img_features`.

Verification reports (including split counts): `splits/dataset_completion_report.json`, `splits/additional_datasets_completion_report.json`, `splits/ce_expansion_completion_report.json`.

Public-release limitations:

- RxR publishes six empty follower pose files, preserved exactly as released; see splits/rxr_upstream_empty_objects.json.
- SOON val_seen_instrs is not published on the official download page.
- The official CVDN dialog downloader publishes no test dialog file.
- The official R4R recipe covers train, val_seen and val_unseen; no separate test release is published there.
- Current official R4R inputs/generator yield 25,921 train and 5,026 val_unseen paths, differing from the README example (25,930 and 5,018). All eligible pairs and composed instructions are independently verified; original annotations and generator are unchanged.
- GSA-R2R includes the official instructions, ScaleVLN graphs and CLIP ViT-B/16 features required by its baseline. Optional checkpoints, the regenerable pano_inputs.h5 cache and raw HM3D rendering meshes are not included.
- Private evaluation ground truth and trained model weights are outside this dataset mirror.
- IR2R-CE optional EnvDrop tours reference 105 episode IDs absent from the published R2R-CE v1-3 preprocessed EnvDrop file. Core train/validation tours are complete. Original files are preserved; missing IDs are listed in datasets/IR2R-CE/envdrop_missing_episode_ids.json.
- REVERIE-CE is the Dynam3D continuous navigation conversion. No separate object-selection labels are added by this conversion; original discrete REVERIE annotations remain available separately.
- Dynam3D publishes no REVERIE-CE val_seen split or separate dense training oracle file. Training episodes include reference paths.
- All 6,292 published REVERIE-CE test episodes have one-point paths, zero distance and start=goal placeholders. These files are preserved for provenance, not valid full test ground truth; use val_unseen for public evaluation.
- The selected Dynam3D NavRAG-CE release publishes train and val_unseen; no separate test or val_seen.

Other existing directories, not audited by the VLN checks:

- `datasets/PRM800K`
