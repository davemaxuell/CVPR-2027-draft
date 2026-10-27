# Dataset retention plan

Checked on 2026-10-26 against the local inventory, directory sizes, dataset links, baseline configuration, and simulator/viewer source. Sizes use decimal GB/TB and do not count symlink targets again. This document is advice: no datasets have been moved, deleted, or relinked.

The scope is the current RoleVLN experiment, the previously requested CE and discrete benchmarks, and the existing Habitat, MatterSim, and trajectory-viewer workflows. The current pilot directly loads R2R-CE and RxR-CE. Other requested benchmarks are retained for planned experiments and comparisons; they are not all inputs to the current pilot.

**Recommendation: retain all 15 requested navigation datasets and the extracted shared assets. Move original Matterport archives and completed download caches off the working disk. About 1.206 TB is eligible for that archival step.** The unrelated PRM800K dataset can be archived separately. Preserve generated research data.

All paths below are relative to the project root.

| Requested dataset | Keep location | Current size | Reason and qualification |
|---|---|---:|---|
| R2R | `data/datasets/R2R` | 5.8 MB | Original instructions/routes, discrete comparison and R4R source. |
| RxR | `data/datasets/RxR` | 191.89 GB | Instructions, multilingual text features, and pose traces. Shared with RxR-CE and the viewer. |
| REVERIE | `data/datasets/REVERIE` | 0.37 GB | Keep v1/v2 annotations and their matching bounding boxes for object-navigation comparisons. |
| SOON | `data/datasets/SOON` | 22.9 MB | Requested object-navigation benchmark. |
| CVDN | `data/datasets/CVDN` | 10.7 MB | Requested dialogue-navigation benchmark. |
| NDH | `data/datasets/NDH` | 21.0 MB | Navigation-from-dialogue instances; complementary to CVDN. |
| HANNA | `data/datasets/HANNA` | 0.37 GB | Requested assistance-based navigation benchmark and auxiliary annotations. |
| R4R | `data/datasets/R4R` | 0.11 GB | Requested long-route benchmark. |
| GSA-R2R | `data/datasets/GSA-R2R` | 16.80 GB | Keep annotations, ScaleVLN graphs, and extracted CLIP features for the requested GSA baseline. |
| R2R-CE | `data/datasets/R2R_VLNCE_v1-3` and `data/datasets/R2R_VLNCE_v1-3_preprocessed` | 0.265 GB combined | Current training/evaluation inputs; also supplies IR2R episodes. Keep expert paths and augmentation. |
| RxR-CE | `data/datasets/RxR_VLNCE_v0` | 0.364 GB | Current training/evaluation inputs. Linked RxR text features are counted in RxR above. |
| IR2R-CE | `data/datasets/IR2R-CE` | 0.301 GB | Planned iterative CE experiments; keep tours, targets, and known maps. Core release is ready; optional EnvDrop references have documented upstream gaps. |
| REVERIE-CE | `data/datasets/REVERIE-CE` | 23.1 MB | Planned continuous navigation experiments. Published test paths are placeholders; the conversion does not replace original REVERIE object labels. |
| NavRAG-CE | `data/datasets/NavRAG-CE` | 0.678 GB | Keep training/validation episodes and oracle paths. MP3D validation is ready; 612 HM3D training scenes are still missing. |
| HA-VLN 2.0 / HA-R2R CE | `data/datasets/HA-VLN` | 23.34 GB now; about 15.01 GB after archiving only the HAPS ZIP | Keep episodes, extracted HAPS motions, human placements, auxiliary annotations, and the requested published CMA checkpoint. |

`data/datasets/VLN-CE` is a small documentation/link directory (about 29 KB), not another large independent scene collection. Keep it and its links.

The ordinary R2R, RxR, REVERIE, SOON, CVDN/NDH, HANNA, R4R, and GSA-R2R benchmarks are not prerequisites for every CE run. Retaining their annotations preserves the previously requested benchmark scope at modest additional cost, except for the explicitly listed feature/trace assets.

| Shared or generated asset | Decision | Current size | Why it stays |
|---|---|---:|---|
| `data/scene_datasets/mp3d` | Keep all 90 scenes | 22.34 GB | Shared Habitat scene collection. Keep `.glb`, `.navmesh`, `.house`, and `_semantic.ply` files for navigation and semantic/map/object extensions. |
| `data/mattersim/v1/scans` | Keep | 51.30 GB | Panorama rendering and the current 3D viewer. Contains extracted skyboxes, camera parameters, and house segmentations. |
| `data/connectivity` | Keep | 24.0 MB | Discrete navigation graphs and viewer routes. |
| `data/img_features` | Keep | 4.16 GB | Shared extracted ResNet feature table for requested discrete baselines. |
| `data/datasets/RxR/text_features` | Keep; included in RxR total | 170.50 GB | Target of `RxR_VLNCE_v0/text_features`; needed by the published cached-text-feature CE baseline. The current raw-text VLM pilot does not require these for every run. |
| `data/datasets/RxR/pose_traces` | Keep; included in RxR total | 20.44 GB | The local viewer's `rxr_subtasks.py` uses guide pose traces to align instruction segments to routes. Keep the full verified collection rather than selectively pruning it here. |
| `data/datasets/HA-VLN/HAPS2_0` | Keep; included in HA total | 14.82 GB | Extracted animated human meshes and object configurations. These are runtime assets. |
| `data/datasets/IR2R-CE/known_maps` | Keep; included in IR2R total | 81.5 MB | Iterative navigation map inputs. |
| `data/processed/baseline_v0` | Keep | About 394.84 GB; growing | Generated expert RGB observations, DAgger data, world-model features/corpora, and scorer labels referenced by current research scripts. Not an expendable download cache. |
| `data/splits`, `data/tools`, `data/logs`, manifests, licenses, and README files | Keep | Small | Provenance, validation results, reproduction tools, and records. |
| `data/known_maps`, `data/tours.json`, `data/gt_ndtw.json`, and dataset shared-asset links | Keep | Negligible | Existing aliases to shared task assets. |

The Habitat package is the scene representation used by VLN-CE. The upstream setup requests the Habitat task data and 90 extracted scene meshes; it does not require all original scan modalities. [Official VLN-CE setup](https://github.com/jacobkrantz/VLN-CE#scenes-matterport3d).

The 51.30 GB MatterSim/viewer tree is separate from the Habitat tree for a reason: the viewer reads original house PLY meshes and skybox images. Its loader cannot simply substitute Habitat semantic PLY meshes. The full house PLY and Habitat semantic PLY layouts differ.

For NavRAG-CE training, add the required HM3D scene meshes, navigation meshes, and scene configuration files once approved credentials are available. Retain them under the shared scene root. Existing GSA/ScaleVLN features are not replacements for those renderable scenes. This missing dependency is excluded from all current-size totals.

| Can leave working storage | Size released from working storage | Treatment |
|---|---:|---|
| `data/.downloads/v1` | 1,182.00 GB | Archive externally. Original scan ZIPs plus the Habitat ZIP; inspected training and rendering paths use extracted copies instead. |
| `data/datasets/.downloads` | 15.33 GB | Completed source archives and GSA feature-transfer chunks. Can be archived, or discarded after retaining verified extracted outputs and provenance. |
| `data/.downloads/scalevln` | 0.55 GB | Downloaded preprocessing ZIP. Retain extracted graphs/annotations in the dataset collection. |
| `data/datasets/HA-VLN/original/huggingface/HAPS2_0/HAPS2_0.zip` | 8.33 GB | Only this ZIP can leave; keep the extracted `HAPS2_0` runtime tree. |
| **Total for the four non-overlapping entries above** | **1,206.21 GB / 1.206 TB** | About 63% of the current `data/` allocation. |

Archiving means moving to a different storage volume or external backup; moving within the same filesystem does not reclaim its space. Original-release ZIP/CRC audits and automatic extraction/resume tools may still expect these archives. Retain their source/hash manifests and record archive destinations if the move is implemented.

Do not remove all of `HA-VLN/original`, `NavRAG-CE/original`, or `REVERIE-CE/original`: active dataset paths point into these directories. Likewise, `RxR_VLNCE_v0/text_features` points into the original RxR directory. A directory called `original` is not automatically disposable.

For completeness, these are all 17 original Matterport scan-archive categories under `data/.downloads/v1/scans/<scan>/`. Sizes below are compressed file payload totals across all 90 scans, so they differ slightly from allocated disk sizes.

| Original ZIP category | Size | Role in the inspected workflows |
|---|---:|---|
| `sens` | 349.78 GB | Raw sensor recordings; not read by current navigation training, Habitat, or the viewer. Archive. |
| `undistorted_normal_images` | 307.82 GB | Dense normal-image products; not read by these workflows. Archive. |
| `matterport_depth_images` | 163.16 GB | Original measured depth images; not needed by the inspected Habitat workflow. Archive. |
| `undistorted_depth_images` | 140.72 GB | Not used by current Habitat runs; needed if later generating measured-depth MatterSim skyboxes. Preserve in external archive for that option. |
| `matterport_color_images` | 61.52 GB | Original color-image collection; current Habitat uses textured meshes and MatterSim/viewer uses skyboxes. Archive. |
| `matterport_hdr_images` | 27.83 GB | Original HDR imagery; not read by these workflows. Archive. |
| `poisson_meshes` | 27.32 GB | Alternative reconstruction representation; not read by the inspected loaders. Archive. |
| `undistorted_color_images` | 21.15 GB | Rectified imagery; not read by current loaders. Can be useful for future RGB/depth alignment work; archive. |
| `matterport_skybox_images` | 18.55 GB | ZIP can leave because extracted skyboxes stay in `data/mattersim/v1/scans`. |
| `region_segmentations` | 16.28 GB | Region-level reconstruction products; not read by inspected loaders. Archive. |
| `house_segmentations` | 16.22 GB | ZIP can leave because extracted house meshes/annotations stay in the MatterSim/viewer tree. |
| `image_overlap_data` | 10.30 GB | Camera-image overlap data; not read by these navigation loaders. Archive. |
| `matterport_mesh` | 5.15 GB | Original mesh representation; current loaders use the retained Habitat/house assets. Archive. |
| `matterport_camera_poses` | 72.4 MB | Original camera pose archive; current loaders use retained episode/graph/camera data. Archive. |
| `matterport_camera_intrinsics` | 12.2 MB | Original camera calibration archive; not read by inspected loaders. Archive. |
| `undistorted_camera_parameters` | 11.1 MB | ZIP can leave; extracted camera parameters stay in the MatterSim tree. |
| `cameras` | 41 KB | Camera-list archive; not read by inspected loaders. Archive. |
| **All 1,530 scan ZIPs** | **1,165.91 GB** | All can leave working storage while the required extracted trees stay. |

`data/.downloads/v1/tasks/mp3d_habitat.zip` contributes another 16.09 GB of compressed payload. Its extracted 22.34 GB scene tree stays. Together with directory/block overhead this explains the 1,182.00 GB archive-directory allocation.

MatterSim's measured-depth mode is the exception to treating raw depth as unnecessary: it preprocesses `undistorted_depth_images` with camera parameters into depth skyboxes. That is a different data path from Habitat scene rendering. This plan archives those sources rather than declaring them useless for every possible simulator configuration. [Official MatterSim data and depth requirements](https://github.com/peteanderson80/Matterport3DSimulator#depth-outputs).

The completed dataset-cache directory consists of the following; each extracted output stays:

| Cache item in `data/datasets/.downloads` | Approximate size |
|---|---:|
| `GSA-R2R-clip-b16.chunks/` | 10.88 GB |
| `ResNet-152-imagenet.zip` | 3.06 GB |
| `ScaleVLN-r2r-preprocess.zip` | 0.547 GB |
| `RxR_VLNCE_v0.zip` | 0.364 GB |
| `R2R_VLNCE_v1-3_preprocessed.zip` | 0.262 GB |
| `GSA-R2R.zip` | 0.116 GB |
| `HANNA_data.zip` | 51.7 MB |
| `REVERIE_BBoxes_v2.zip` | 45.8 MB |
| `R2R_VLNCE_v1-3.zip` | 2.6 MB |

| Other existing data | Recommendation | Reason |
|---|---|---|
| `data/datasets/PRM800K` — 0.494 GB | Archive outside the navigation working set | Its README describes mathematical process-supervision labels. No reference was found in the inspected baseline, DUET training code, or viewer. It is not a simulator dependency. |
| `data/processed/navvem_episodes` — 0.773 GB | Hold pending provenance review | Legacy derived research episodes. No current-loader reference found, but their manifest is unreadable to the current account; unused status cannot be established confidently. |
| `data/hime_task_data` — about 1.9 MB | Optional historical archive | Old task screenshots/interaction logs, not a benchmark input. Preserve experiment history; negligible storage saving. |

At measurement, `data/` occupied about **1.907 TB**. After archiving only the four main archive/cache entries, it would occupy about **700 GB**, including roughly **395 GB of current generated baseline data** and about **305 GB of datasets, shared assets, and other records**. Model weights, code environments, and outputs outside `data/` are additional. HM3D is still missing. These are a storage snapshot, not a fixed upper bound while training continues.

Evidence: `data/DATASET_INVENTORY.md`; `data/splits/dataset_inventory.json`; `data/splits/matterport_storage_audit.json`; `data/splits/dataset_retention_snapshot.json`; the existing dataset/HA/GSA/CE completion reports; `[OURS]/baseline-v0/sim/config.py`; `[OURS]/baseline-v0/manifest/experiment_manifest.yaml`; `[OURS]/baseline-v0/tools/train_policy_sft.py`; `[prior works]/ScaleVLN/analysis/vln_viz/build_meshes.py`; and `[prior works]/ScaleVLN/analysis/vln_viz/rxr_subtasks.py`. This pass inspected dependencies and sizes and reused recorded integrity results; it did not rerun all dataset hashes or simulator benchmarks.
