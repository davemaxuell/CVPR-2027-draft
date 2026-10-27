# RoleVLN datasets

All paths below are relative to this `data/` directory. Dataset files are shared
with the `researchers` group. Utilities and logs live alongside the data.

```text
datasets/
  R2R/                          Original discrete R2R: R2R_{split}.json
    source_manifest.json        Source URLs, SHA-256 hashes, and split counts
  R4R/                          Queued official construction from R2R after resume
    R4R_{train,val_seen,val_unseen}.json
    upstream/                   Pinned original Google Research generator
    generation_manifest.json    Input, graph and output hashes; generation settings
  R2R_VLNCE_v1-3_preprocessed/   R2R continuous/Habitat release and EnvDrop augmentation
  RxR/                          Original gs://rxr-data layout
    rxr_*.jsonl.gz              Guide/follower annotations and upstream augmentations
    text_features/rxr_*/        Multilingual BERT features (en, hi, te)
    pose_traces/rxr_*/          Guide and follower camera traces
    README.upstream.md          Official RxR documentation
    LICENSE                    Official RxR license
  RxR_VLNCE_v0/                RxR continuous/Habitat annotations
    text_features -> ../RxR/text_features
  REVERIE/
    {train,val_seen,val_unseen,test}/{split}.json   Original v1 annotations
    bbox/                      Original v1 object boxes (10,567 JSON files)
    v2/{split}.json            v2 annotations
    v2/bbox/                   v2 object boxes (10,567 JSON files)
    v2/bbox_source.json        v2 archive provenance and target-box verification
  SOON/                        Original FAO/SOON annotations (queued after RxR)
    {train,val_unseen_instrs,val_unseen_house,test_release}.json
    test.json -> test_release.json
  CVDN/                        Full dialogs: train, val_seen, val_unseen
  NDH/                         Navigation instances: train, val_seen, val_unseen, test
    test_cleaned.json -> test.json
  HANNA/
    hanna/                     Five task splits, routes, action space, vocabulary
    connectivity -> ../../connectivity
    scans -> ../../mattersim/v1/scans
    img_features -> ../../img_features
  R2R_VLNCE_v1-3/              Minimal original R2R-CE release (queued)
  VLN-CE/                     Documentation and links to the three CE releases and scenes
  .downloads/                  Original CE archives and REVERIE_BBoxes_v2.zip
img_features/                  Shared ResNet-152 ImageNet TSV (queued)
scene_datasets/mp3d/{scan}/    Habitat meshes, navmeshes, house and semantic files (90 scans)
connectivity/                 MatterSim navigation graphs (90 scans)
mattersim/v1/scans/{scan}/     Skybox images and camera parameters (90 scans)
.downloads/v1/                Complete original Matterport scan archives and Habitat archive
splits/                       Inventories, download status, and verification reports
tools/                        Dataset download and verification utilities
logs/                         Download and verification logs
processed/navvem_episodes/    Previously generated NavVEM episodes (see permissions below)
```

## Completion status

**Downloads resumed at the user's request on 2026-10-21 at 06:16 UTC (15:16 KST).**
Both background services were restarted with the saved partial files. RxR runs
first; the additional datasets wait for its successful final verification. The
pause and resume record is `splits/dataset_download_pause.json`; live state is
recorded in the two download status JSON files.

Original R2R, REVERIE v2 boxes, and the four additional MatterSim rendering
environments were added on 2026-10-21. Their checks are recorded in
`splits/r2r_reverie_completion_report.json`.

The full original RxR release contains **582,090 data files, 190,669,133,548 bytes
(177.57 GiB)** in the recorded upstream inventory. Completion of this large transfer
is recorded in `splits/rxr_original_download_status.json`. Only `phase: complete`
means that all downloads and the final dataset checks have passed. A symlink makes
the same text features available to the RxR-CE baseline after the mirror completes.

Each RxR object is checked against Google's published MD5 and byte size before it
receives its final filename. Interrupted objects retain `.part` suffixes; large
objects also use temporary range chunks. The exact object generations and checksums
are stored in `splits/rxr_original_download_manifest.jsonl`.

```bash
# Inspect progress from the RoleVLN project directory:
cat data/splits/rxr_original_download_status.json
tail -n 10 data/logs/rxr_completion.log
systemctl --user status rolevln-rxr-download.service

# Resume the full mirror and automatically run final verification:
python3 data/tools/complete_rxr_download.py --shards 4 --connections 64

# Verify everything after downloading:
python3 data/tools/verify_completed_datasets.py
```

Do not start a second downloader while an existing one is active. The supervisor
uses disjoint file lists for its workers. Its final report is
`splits/dataset_completion_report.json`; detailed output is saved to
`logs/dataset_completion_verification.log`.

Download tuning on 2026-10-21 keeps four workers and 64 HTTP/1.1 connections per
worker. Batches now contain 4,096 objects. Transfers retry when they stay below
1 KiB/s for 30 seconds, or reach 120 seconds, with two immediate retries before
the existing checksum/resume passes. Partial files and published MD5 validation
are retained. Larger HTTP/2 tests stalled, so HTTP/2 was not enabled. The tuning
record and benchmark locations are in `splits/rxr_download_tuning.json`.

The RxR source release contains six empty follower pose files. The mirror keeps
them exactly as published, including their empty-file MD5 checksums; these files
do not contain usable pose arrays. Their names and source metadata are listed in
`splits/rxr_upstream_empty_objects.json`. An empty-file resume error discovered on
2026-10-22 KST was fixed and checked against all six upstream objects. Worker
exits are now exposed separately from file failures in the aggregate status.

## Splits

Counts below are from the local annotation files. RxR-CE training and validation
counts combine guide and follower episodes; its test challenge contains guides only.

| Dataset | Train | Val seen | Val unseen | Test |
|---|---:|---:|---:|---:|
| Original R2R paths | 4,675 | 340 | 783 | 1,391 |
| R2R-CE episodes | 10,819 | 778 | 1,839 | 3,408 |
| RxR-CE episodes | 119,052 | 13,317 | 21,853 | 9,557 (challenge) |
| REVERIE v1 paths | 4,150 | 515 | 1,328 | 2,304 |
| REVERIE v1 instructions | 10,466 | 1,423 | 3,521 | 6,292 |
| REVERIE v2 paths | 4,071 | 501 | 1,292 | 2,244 |
| REVERIE v2 instructions | 10,290 | 1,384 | 3,433 | 6,144 |

The REVERIE instruction counts use `instructions`. The separate `instructions_l`
field in v1 contains inherited R2R instructions and is not the REVERIE instruction
count. REVERIE v2 must use `REVERIE/v2/bbox`, not the original `REVERIE/bbox`.

Official test annotations intentionally omit private evaluation ground truth.
Original RxR has both `test_standard` and `test_challenge`; the CE release has
`test_challenge`.

## Simulator paths

Habitat scene root: `data/scene_datasets`. MatterSim rendering root:

```bash
export MATTERPORT_DATA_DIR="$PWD/data/mattersim/v1/scans"
```

All 90 MatterSim graphs now have extracted skybox images and camera parameters.
Additional panoramas outside navigation graphs are valid surplus images.

## Additional dataset queue

On 2026-10-21, SOON, CVDN, NDH, HANNA, VLN-CE completion, and R4R preparation were
queued. CVDN was already covered; **R2R-CE** is explicitly covered by the VLN-CE job
and its preprocessed release is already present. R4R was added during the pause.
The user has since resumed both services. The additional worker starts downloads **only after**
the RxR status is `complete`, its verified-file count matches the inventory,
and `dataset_completion_report.json` records a successful RxR verification.
While stopped, `additional_datasets_status.json` reports `paused`. After resume,
it reports `waiting_for_rxr` until the prerequisite passes.

The queue downloads the public annotations first, then the shared ResNet-152
ImageNet features (3.06 GB compressed). It also extracts all 90 existing
`house_segmentations.zip` files into `mattersim/v1/scans/{scan}/house_segmentations`
(32.57 GB uncompressed). HANNA needs these house files in addition to the skyboxes
and camera parameters. The existing scene archives are reused locally.

The R2R-CE preprocessed release, RxR-CE annotations, and all 90 Habitat scene
directories are already present. The queue adds the small minimal R2R-CE release,
reuses the completed RxR text features, and checks the CE files against their
source archive CRCs and scene references. The `datasets/VLN-CE` directory groups
these releases through relative links. Habitat-Sim software and trained model
weights are separate from these datasets.

R4R's official release provides code to construct the dataset from R2R. The queued
job downloads the reviewed generator pinned to Google Research commit
`5b09c22d73a9d35eb6c5d2a99b95677a45053466` and checks its SHA-256 before executing it.
It creates `R4R_train.json`, `R4R_val_seen.json`, and `R4R_val_unseen.json` from our
existing R2R files with a 3.0 m join distance and no heading filter. It checks the
published path counts (25,930 / 115 / 5,018) and instruction counts
(233,613 / 1,035 / 45,162), records source and output hashes, and reuses shared
scenes and features. The original release does not publish a separate R4R test split.
Generation uses the existing `/home/bufsgpu/yes/envs/sw/bin/python3` environment
with NumPy, NetworkX and Matplotlib; it does not modify that environment.

SOON, CVDN, NDH, HANNA, and R4R each get relative links to shared `connectivity`,
`scans`, and `img_features`. HANNA preserves its upstream `hanna/` subdirectory,
so its data path is `data/datasets/HANNA/hanna`. All new dataset files are
group-readable. Original annotations are preserved without reformatting.

```bash
cat data/splits/additional_datasets_status.json
systemctl --user status rolevln-additional-datasets.service
tail -n 20 data/logs/additional_datasets.log

# Resume only after the user asks, and only if the service is not running:
/usr/bin/python3 data/tools/download_additional_datasets.py

# Run the final checks again after the queue completes:
/usr/bin/python3 data/tools/download_additional_datasets.py --verify-only
```

The worker resumes interrupted downloads, retries transfer failures, validates JSON
and ZIP integrity, checks navigation references and feature coverage, and records
SHA-256 hashes and provenance. A failure in one dataset does not stop attempts on
the others. The service retries failed passes after five minutes. The queue uses
Python's standard library and `curl` for transfers. R4R generation additionally
uses the existing scientific Python environment specified above.

Queue and results:

- `splits/additional_datasets_plan.json`: exact URLs, destinations, order, and scope.
- `splits/additional_datasets_download_receipts.json`: downloaded sizes and SHA-256 hashes.
- `splits/additional_*_manifest.json`: source information and extracted-file CRCs.
- `splits/additional_datasets_completion_report.json`: final split counts and checks.
- `DATASET_INVENTORY.md` and `splits/dataset_inventory.json`: the requested final
  dataset list, written automatically only after both dataset groups pass their
  checks. These list the ten requested VLN datasets, locations, versions, shared
  assets, verification reports, and public-release limitations. R2R-CE and RxR-CE
  are listed individually under the VLN-CE family. Other existing dataset folders
  are recorded separately without claiming that they were audited.

Only overall `phase: complete` means all queue jobs and verification passed.
`retry_pending` records an unsuccessful pass and retains the errors in the log.

Public-release scope: the SOON project page publishes four files; the separate
`val_seen_instrs.json` referenced by older baseline code is not provided there.
The CVDN official script publishes three dialog splits, while NDH additionally
publishes `test_cleaned`, saved as `test.json` per its official script. Private
evaluation labels are not treated as downloadable missing files. SOON's old
ResNet download link returns HTTP 403; the queue uses the current official
Matterport3DSimulator link, also used by HANNA.

## Existing derived-file permissions

The previously generated files in `processed/navvem_episodes/` are owned by
`team_a` and have mode `0600`. They are not readable by the current `bufsgpu`
account. Their owner can grant group read access with:

```bash
chmod g+r data/processed/navvem_episodes/manifest.json \
  data/processed/navvem_episodes/{r2r,rxr,reverie}/*.jsonl.gz
```

Newly downloaded files are group-readable. The source datasets can be used
independently of these old derived files.

## Sources and licenses

- [Original R2R download script](https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/data/download.sh)
- [Official R4R construction and published split counts](https://github.com/google-research/google-research/tree/master/r4r)
- [Original RxR release](https://github.com/google-research-datasets/RxR)
- [VLN-CE releases and feature layout](https://github.com/jacobkrantz/VLN-CE)
- [REVERIE v2 bounding-box download](https://github.com/YuankaiQi/REVERIE/blob/master/tasks/REVERIE/data_v2/download%20BBoxes%20v2.txt)
- [SOON original dataset downloads](https://scenario-oriented-object-navigation.github.io/)
- [CVDN and NDH releases](https://github.com/mmurray/cvdn)
- [HANNA dataset and scene requirements](https://github.com/khanhptnk/hanna/tree/master/data)
- [Shared ResNet-152 image features](https://github.com/peteanderson80/Matterport3DSimulator#precomputing-resnet-image-features)

Matterport3D scenes are covered by the
[Matterport3D Terms of Use](https://kaldir.vc.cit.tum.de/matterport/MP_TOS.pdf).
The RxR license is retained at `datasets/RxR/LICENSE`. Dataset terms still apply
to local copies and derived data.
