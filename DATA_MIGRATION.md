# Data to bring when moving RoleVLN to a new machine

Checked on 2026-10-27 against the code that reads data: the navigation explorer
(`[prior works]/ScaleVLN/analysis/vln_viz`), our training (`[OURS]/baseline-v0`), and evaluation of the
other models (ScaleVLN/DUET, ETPNav, and the VLN-CE candidates in `[prior works]/CANDIDATE_MODELS.md`).
All paths are relative to the `RoleVLN/` folder. Sizes are from `du` (binary units: G = GiB).

**Fastest route while the old server is up:** copy all of this from it with `migration/pull_from_server.py`
([RETRIEVE_FROM_SERVER.md](RETRIEVE_FROM_SERVER.md)); its item list mirrors this document. The public sources
below are for when the server is gone.

None of the datasets go into git, apart from a few small public annotation files the explorer reads
(R2R/FGR2R/REVERIE enc annotations, RxR val guide files, connectivity graphs). Every file in sections 1–5 is either a dataset under its own license
(Matterport3D Terms of Use, RxR license, ...) or larger than GitHub's 100 MB file limit. Our own checkpoints and
run logs are a separate matter: they are in git or in the GitHub release (section 6, [MIGRATION.md](MIGRATION.md)).

| | Size |
|---|---:|
| **Bring (sections 1–4)** | **~100 G** (~86 G without HA-VLN; 15 G less if you delete `_semantic.ply`) |
| Skip (section 5) | ~1.4 T |
| Generated outputs (section 6): in git / the release, regenerated, or carried by disk | ~400 G |
| On disk now, `data/` + `[prior works]` + `[OURS]` | ~1.9 T |

Status: **Required** = something listed above fails without it. **CE** = kept because every VLN-CE
dataset stays, whether or not any code uses it yet. **Optional** = needed only for the job named in its row.

---

## 1. Matterport3D scenes (shared by everything)

Both need a signed Matterport3D Terms of Use, which gets you `download_mp.py`
(https://niessner.github.io/Matterport/).

| Path | Size | Status | Get it with | Used by |
|---|---:|---|---|---|
| `data/scene_datasets/mp3d/<scan>/` (90 scans: `.glb`, `.navmesh`, `.house`, `_semantic.ply`) | 21 G | Required | `python download_mp.py --task habitat -o data/scene_datasets/mp3d/` (mp3d_habitat.zip, then unzip) | Habitat: baseline-v0, ETPNav, all CE datasets, explorer `build_navmaps.py` |
| `data/mattersim/v1/scans/<scan>/matterport_skybox_images/` | 18 G | Required | `download_mp.py --type matterport_skybox_images` (unzip per scan) | Explorer panoramas and front view, "Photo at agent" |
| `data/mattersim/v1/scans/<scan>/house_segmentations/` | 31 G | Required | `download_mp.py --type house_segmentations` (unzip per scan) | Explorer: `.house` gives floors and room names; `.ply` (25 G of the 31 G) is the source of the 3D building meshes |
| `data/mattersim/v1/scans/<scan>/undistorted_camera_parameters/` | 41 M | Optional | `download_mp.py --type undistorted_camera_parameters` | Nothing here reads it (DUET uses precomputed features, not MatterSim rendering). Tiny; get it if you ever render or make depth skyboxes with MatterSim. |

The `_semantic.ply` files (15 G of the 21 G) come inside `mp3d_habitat.zip`; no current run reads them,
so you can delete them after unzipping. Habitat itself doesn't need them for RGB/depth navigation.

## 2. VLN-CE datasets (all of them)

| Path | Size | Status | Source | Used by |
|---|---:|---|---|---|
| `data/datasets/R2R_VLNCE_v1-3_preprocessed/` (train, val_seen, val_unseen, test, envdrop, joint_train_envdrop, embeddings) | 251 M | Required | VLN-CE README (https://github.com/jacobkrantz/VLN-CE), "R2R_VLNCE_v1-3_preprocessed" Google Drive link | baseline-v0 `sim/config.py` (`{split}.json.gz`, `{split}_gt.json.gz`); StreamVLN; IR2R-CE links into it |
| `data/datasets/R2R_VLNCE_v1-3/` | 2.5 M | CE | VLN-CE README, "R2R_VLNCE_v1-3" | IR2R-CE `episodes` link |
| `data/datasets/RxR_VLNCE_v0/` (train, val_seen, val_unseen, test_challenge; guide + follower; `_gt`) | 347 M | Required | VLN-CE README, "RxR_VLNCE_v0" | baseline-v0 (English guide episodes); ETPNav nDTW ground truth |
| `data/datasets/RxR/text_features/` (linked as `RxR_VLNCE_v0/text_features`) | **159 G** | Optional (CE) | `gsutil -m cp -R gs://rxr-data/text_features data/datasets/RxR/` | **Nothing we run.** Only VLN-CE's own CMA/Seq2Seq RxR baselines read these precomputed BERT features. ETPNav has that line commented out (`habitat_extensions/sensors.py`). baseline-v0, StreamVLN and Uni-NaVid read raw text. Recommendation: skip. |
| `data/datasets/VLN-CE/` | 18 K | CE | Relative links + upstream README/LICENSE; recreate the links | Groups the three releases above |
| `data/datasets/IR2R-CE/` (`tours.json`, `gt_ndtw.json`, `known_maps/`) | 288 M | CE | https://github.com/jacobkrantz/IVLN-CE (Google Drive links in `IR2R-CE/source_manifest.json`) | Planned iterative-CE experiments. `known_maps.zip` (78 M) duplicates the extracted `known_maps/`. Root links `data/{tours.json,gt_ndtw.json,known_maps}` point here. |
| `data/datasets/REVERIE-CE/` | 22 M | CE | HF `MrZihanWang/Dynam3D` @ `0fe50d66`, `data/datasets/reverie_*` (URLs in `source_manifest.json`) | Planned. Test split is placeholders; evaluate on val_unseen |
| `data/datasets/NavRAG-CE/` | 643 M | CE | HF `MrZihanWang/Dynam3D` @ `0fe50d66`, `data/datasets/navrag_*` | Planned. Training also needs 612 HM3D scenes, which we never had (no HM3D credentials) |
| `data/datasets/HA-VLN/` (HA-R2R episodes, extracted `HAPS2_0/`, Multi-Human-Annotations, CMA checkpoint) | 22 G (14 G after deleting the zip) | CE | HF dataset `fly1113/HA-VLN` @ `2632eeac` + GitHub repo (URLs in `source_manifest.json`) | Planned. Unzip `HAPS2_0/HAPS2_0.zip` into `HA-VLN/HAPS2_0/`, then the 7.8 G zip can go |
| `[prior works]/ETPNav/data/datasets/R2R_VLNCE_v1-2_preprocessed{,_BERTidx}/`, `RxR_VLNCE_v0_enc_xlmr/` | 46 M | Required | `modelscope download --model admagic/ETPNav --local_dir ./data` (same download as its weights, section 4) | ETPNav eval and `tools/log_etpnav_steps.py` (ETPNav uses R2R-CE v1-2 with BERT ids, RxR-CE with XLM-R ids) |

## 3. Navigation explorer (discrete DUET/ScaleVLN runs)

The CE runs in the explorer only need sections 1, 2 and the ETPNav logs in section 6.

| Path | Size | Status | Source | Used for |
|---|---:|---|---|---|
| `[prior works]/ScaleVLN/VLN-DUET/datasets/R2R/connectivity_mp3d/` | 23 M | Required | HF `OpenGVLab/ScaleVLN` (R2R data zip) | Navigation graphs for every discrete run, `build_meshes.py`, `build_navmaps.py` |
| `.../VLN-DUET/datasets/R2R/annotations/R2R_{train,val_seen,val_unseen,val_train_seen,test}_enc.json` | 17 M | Required | HF `OpenGVLab/ScaleVLN` | Ground truth for R2R runs; also DUET eval |
| `.../VLN-DUET/datasets/R2R/annotations/FGR2R/` | 17 M | Required | https://github.com/YicongHong/Fine-Grained-R2R | R2R sub-tasks (also used by CE R2R runs) |
| `.../VLN-DUET/datasets/R2R/annotations/scanvp_candview_relangles_with_hm3d_gibson.json` | 132 M | Optional | HF `OpenGVLab/ScaleVLN` | Only to rebuild `analysis/vln_viz/cache/view_index.json` (2 M). Not needed if that cache file is copied. |
| `[prior works]/ScaleVLN/VLN-DUET-RVR/datasets/REVERIE/annotations/REVERIE_{val_seen,val_unseen}_enc.json`, `BBoxes.json` | 10 M | Required | HF `OpenGVLab/ScaleVLN` (REVERIE zip) | Zero-shot REVERIE runs (goal zone = viewpoints where the object is visible) |
| `data/datasets/RxR/rxr_{val_seen,val_unseen}_guide.jsonl.gz` | 21 M | Required | `gsutil cp gs://rxr-data/rxr_val_*_guide.jsonl.gz` (https://github.com/google-research-datasets/RxR) | Zero-shot RxR runs (`parse_trajectories.RXR_DIR`), `log_duet_steps.py --anno` |
| `data/datasets/RxR/rxr_train_guide.jsonl.gz` | 72 M | Optional | same bucket | Only to log DUET on RxR train (never done so far) |
| `data/datasets/RxR/pose_traces/rxr_{val_seen,val_unseen}/` | 4.1 G | Optional | `gs://rxr-data/pose_traces/rxr_val_*` | Only to rebuild `cache/rxr_subtasks/` (4.9 M). Not needed if that cache folder is copied. |

## 4. Weights for training and for testing other models

These aren't datasets, but they're also too big for git and must exist on the new machine.

| What | Where it goes | Size | Source |
|---|---|---:|---|
| DUET/ScaleVLN eval features `clip_vit-{b16,h14}_mp3d_original.hdf5` | `[prior works]/ScaleVLN/VLN-DUET/datasets/R2R/features/` | 1.6 G | HF `OpenGVLab/ScaleVLN` `features.zip` (take just these two members; eval reads only these, see `r2r/parser.py`) |
| ScaleVLN fine-tuned `duet_vit-{b16,h14}_ft_best_val_unseen` | `.../VLN-DUET/datasets/R2R/trained_models/finetune/` | 4.0 G | HF `OpenGVLab/ScaleVLN` |
| ETPNav `ckpt.iter12000.pth` (R2R-CE), `ckpt.iter19600.pth` (RxR-CE), `wp_pred/`, `ddppo-models/`, `connectivity_graphs.pkl` | `[prior works]/ETPNav/data/` | 6.2 G | `modelscope download --model admagic/ETPNav --local_dir ./data` (skip `pretrained/`, not needed for eval) |
| CLIP ViT-B/32 (ETPNav) | `~/.cache/clip` | small | downloaded automatically by `clip.load` |
| SmolVLM2-500M-Video-Instruct @ `7b375e1b`, `facebook/dinov2-small` @ `ed25f3a3` | HF cache (`HF_HOME`, here `/data/shared/hf_cache`) | ~2 G | `HF_HOME=... python -I env/download_models.py manifest/model_revisions.json` in baseline-v0 |
| `facebook/nwm` (`cdit_s_100000.pth.tar` @ `0821a1a7`, gated: request access again), `gaoyuezhou/dino_wm` | HF cache | — | Hugging Face |
| Backbone trials `Qwen/Qwen3-VL-2B-Instruct`, `lmms-lab/llava-onevision-qwen2-0.5b-ov` | HF cache | — | Hugging Face (`manifest/Backbone_Trials.json`) |
| VLN-CE candidates (Uni-NaVid, InternVLA-N1, StreamVLN, NaVILA) | — | ~15 G each | Not downloaded yet; repos and weights in `[prior works]/CANDIDATE_MODELS.md`. They need only R2R-CE, RxR-CE and the mp3d scenes, all listed above. |

## 5. Skip: not needed for the explorer, training or testing

| Path | Size | Why it's not needed |
|---|---:|---|
| `data/.downloads/v1/` | 1.1 T | Original Matterport scan ZIPs + `mp3d_habitat.zip`. Everything we read is the extracted copy in section 1. |
| `data/datasets/.downloads/`, `data/.downloads/scalevln/` | 16 G | Download caches of archives that are already extracted |
| `data/datasets/RxR/text_features/` | 159 G | See section 2: no code we run reads it |
| `data/datasets/RxR/pose_traces/rxr_train/` | 15 G | Nothing reads train pose traces (sub-tasks only exist for val) |
| `data/datasets/RxR/rxr_marky_*`, `rxr_landmarks_*`, `rxr_*_follower.jsonl.gz`, `rxr_test_*_public_guide.jsonl.gz`, `gibson_navgraphs.gz`, `rxr_val_seen_guide.json`, `rxr_val_seen_guide_by_path.json` | 0.8 G | Upstream augmentation / landmark / follower / test files; not read |
| `data/datasets/GSA-R2R/` | 16 G | GSA-VLN baseline only; no code here uses it |
| `data/img_features/` (ResNet-152 TSV) | 3.9 G | Old discrete baselines (HANNA, Speaker-Follower); not used |
| `data/datasets/{R2R,R4R,REVERIE,SOON,CVDN,NDH,HANNA}/` | 0.8 G | Discrete benchmarks with no runner here. The explorer reads DUET's copies of R2R/REVERIE (section 3), not these. Small, so keep them if convenient. |
| `data/connectivity/`, `data/scans.txt` | 23 M | Linked only from the skipped discrete benchmarks; the explorer uses `connectivity_mp3d` (section 3) |
| `data/datasets/PRM800K/` | 472 M | Unrelated (math process-supervision labels) |
| `data/processed/navvem_episodes/`, `data/hime_task_data/` | 0.7 G | Old derived/legacy files, not read by any current code |
| `[prior works]/ScaleVLN/VLN-DUET/datasets/R2R/features/*_mp3d_hm3d_gibson.hdf5`, the three `clip_vit-h14_mp3d_img_*` EnvEdit files | 35 G | Only to **retrain** ScaleVLN (fine-tuning with augmentation). Eval uses `*_mp3d_original.hdf5`. |
| `.../R2R/annotations/R2R_scalevln_ft_aug_enc.json`, `pretrain_map/`, `R2R/connectivity/` (HM3D/Gibson graphs), `trained_models/pretrain/`, `datasets/pretrained/LXMERT/` | 6.9 G | ScaleVLN pretraining / augmented fine-tuning only |
| `[prior works]/ScaleVLN/VLN-DUET-RVR/datasets/REVERIE/features/`, `trained_models/`, `annotations/pretrain/`, `ade20k_*.jsonl`, `REVERIE_speaker_aug_enc.json`, `scanvp_candview_relangles_new*.json` | 9.5 G | REVERIE pretraining / fine-tuning only (the explorer's REVERIE runs are zero-shot R2R checkpoints) |
| `[prior works]/ScaleVLN/runs/` | 46 G | Setup smoke tests, verification checkpoints and old demo exports |

To retrain ScaleVLN later, download the full HF `OpenGVLab/ScaleVLN` release (~48 G) instead of the
eval subset in section 4.

## 6. Generated outputs (not datasets)

These are our own results. [MIGRATION.md](MIGRATION.md) §1 shows how each one moves; in short:

| Path | Size | How it moves |
|---|---:|---|
| `[OURS]/baseline-v0/runs/` | 13 G | Records, results and the 15 selected/frozen checkpoints are in **git**. The other 89 checkpoints and optimizer states are **release assets** (`migration/release_assets.py download`). 2 G of `*.jpg` frame dumps are left out. |
| `[prior works]/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/` (eval preds, step logs, what-if results) | 891 M | **git**, except the two `steps_train.jsonl` over 100 MB (**release** + server item `explorer_train_steplogs`) |
| `[prior works]/ETPNav/data/logs/explorer/` (episode logs + cube-face renders) | 3.8 G | episode logs in **git**; the 138k cube-face renders are Matterport-derived: server item `etpnav_explorer_views` |
| `[prior works]/ScaleVLN/analysis/vln_viz/cache/` | 3.7 G | `rxr_subtasks/`, `view_index.json`, `navmaps/` in **git**; meshes are Matterport-derived: server item `explorer_meshes`, or rebuild with `build_meshes.py --detail`; row caches rebuild on start |
| `data/processed/baseline_v0/expert_groupA/`, `wm_dino_features_refined512_v1/` | 319 G | **regenerate** (`tools/render_expert.py`, about 20 min on the H100; then the DINO feature pass) |
| `data/processed/baseline_v0/{dagger_*,scorer_labels_*,scorer_features_*,wm_visits_*,wm_groupA_*,expert_groupC_600}` | 64 G | **server items** `processed_*` if you want to continue without re-running rollouts |

## 7. Things that differ on the new machine

- **Nested git repositories** (`ScaleVLN`, `ETPNav`, `VLN-DUET`, `Matterport3DSimulator`, habitat-lab v0.1.7)
  are handled by `migration/export_repo.py`. It copies their files without their `.git` folders, so our
  untracked work inside them (the explorer, `ScaleVLN/tools`, `run.sh`, the ETPNav logger) and the 21 patched
  ScaleVLN files are in the repo as plain files. Their upstream commits: ScaleVLN `3bdff0da`, ETPNav `160e65fe`,
  VLN-DUET `93e8b233`, Matterport3DSimulator `589d091b`, habitat-lab `d6ed1c0a` (tag `challenge-2021`, v0.1.7).
- **Absolute paths.** 38 code/config files hardcode `/data/team_a/dave-workspace/RoleVLN`,
  `/data/shared/hf_cache` or the old conda base `/home/bufsgpu/yes` (`migration/hardcoded_paths.txt`).
  Use the same path on the new machine, or run `migration/relocate_paths.py`. The two absolute ETPNav symlinks
  became relative in the export.
- **Conda environments** (`rolevln`, `scalevln`, `etpnav`) don't travel with git; MIGRATION.md §3.4 has the recipes
  and the MatterSim build workarounds.

## Checklist for the new machine

```text
RoleVLN/
  data/
    scene_datasets/mp3d/<90 scans>/                     section 1
    mattersim/v1/scans/<90 scans>/{matterport_skybox_images,house_segmentations,undistorted_camera_parameters}/
    datasets/
      R2R_VLNCE_v1-3_preprocessed/   R2R_VLNCE_v1-3/   RxR_VLNCE_v0/   VLN-CE/   section 2
      IR2R-CE/   REVERIE-CE/   NavRAG-CE/   HA-VLN/
      RxR/rxr_{val_seen,val_unseen}_guide.jsonl.gz                     section 3
      (RxR/text_features/ only if you decide to keep it: 159 G)
    tours.json, gt_ndtw.json, known_maps  -> datasets/IR2R-CE/...  (relative links)
  [prior works]/
    ScaleVLN/VLN-DUET/datasets/R2R/
      connectivity_mp3d/                                   section 3
      annotations/{R2R_*_enc.json, FGR2R/}
      features/clip_vit-{b16,h14}_mp3d_original.hdf5      section 4
      trained_models/finetune/duet_vit-{b16,h14}_ft_best_val_unseen
    ScaleVLN/VLN-DUET-RVR/datasets/REVERIE/annotations/{REVERIE_val_*_enc.json, BBoxes.json}
    ScaleVLN/VLN-DUET-RVR/datasets/R2R -> ../../VLN-DUET/datasets/R2R  (relative link)
    ETPNav/data/{datasets/, logs/checkpoints/release_{r2r,rxr}/, wp_pred/, ddppo-models/, connectivity_graphs.pkl}
    ETPNav/data/scene_datasets -> RoleVLN/data/scene_datasets
    ETPNav/data/datasets/RxR_VLNCE_v0 -> RoleVLN/data/datasets/RxR_VLNCE_v0
  HF cache: SmolVLM2-500M, dinov2-small, facebook/nwm, dino_wm, Qwen3-VL-2B, llava-onevision-0.5b
```
