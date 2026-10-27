# Migration guide: RoleVLN → new machine

This repo (`github.com/davemaxuell/CVPR-2027-draft`, public) is a snapshot of the `RoleVLN/` folder on the
old H100 server, taken 2026-10-27. Three documents cover the move:

- **this file**: what is where, the final sync on the old server, and the setup on the new machine, step by step
- [TRAINING_STATE.md](TRAINING_STATE.md): what we trained, results, what was running, what to do next
- [DATA_MIGRATION.md](DATA_MIGRATION.md): every dataset and public weight file: needed or not, size, source

## 1. What is where

| Tier | Contents | Size | How you get it |
|---|---|---:|---|
| **git** | All code (ours + the patched ScaleVLN/DUET, ETPNav, MatterSim and habitat-lab v0.1.7 sources), docs, drafts, run records and results, logs, the explorer (`[prior works]/ScaleVLN/analysis/vln_viz`) with its small caches, and the **selected/frozen checkpoints** (15 files, every one < 100 MB) | ~2.5 G | `git clone` |
| **release** `assets-v1` | Every other checkpoint of ours (89 files: intermediate checkpoints, optimizer/trainer states, NWM EMA weights) and the explorer run logs (ScaleVLN eval/step logs, ETPNav R2R-CE/RxR-CE logs + renders) as split archives | ~12 G | `python3 -I migration/release_assets.py download` |
| **datasets / public weights** | Matterport3D scenes, VLN-CE episodes, DUET features, released DUET/ETPNav/HF checkpoints | ~100 G | download from the sources in DATA_MIGRATION.md |
| **regenerate** | Expert renders, DINO features, explorer 3D mesh cache | ~320 G | scripts, §3.6 |
| **carry (optional)** | Processed data that needs simulator rollouts to rebuild (DAgger data, WM visits, scorer labels/features) | ~62 G | disk / rsync, §2.3 |

`migration/release_assets.json` lists every release asset with its original path and sha256.
`migration/export_report.json` records what the export left out and why. The two folders that matter in git:
`[OURS]/baseline-v0` (our pipeline) and `[prior works]` (baselines + explorer). Directory names contain
spaces and brackets, so quote them in the shell.

Not in git on purpose: datasets (licenses: Matterport3D Terms of Use, RxR), files over GitHub's 100 MB limit,
22k simulator frame dumps (`runs/**/*.jpg`, 2 G of debug images), the upstream DINO-WM checkpoints
(re-download), `.git` folders of the upstream clones, a virtualenv, and an upstream file with an API key in it
(`drafts/.../CAST/.../gamma_api.txt`).

## 2. On the old server: the final sync

The repo was exported while the automatic Step 9 → Step 10 → NWM → four-model queue was still running (see
TRAINING_STATE.md). When it has finished, or right before you stop using the server, refresh the repo:

### 2.1 Refresh git

```bash
cd /data/team_a/dave-workspace/RoleVLN
python3 -I migration/export_repo.py --build-archives --stage   # mirror into ../CVPR-2027-draft, repack logs, git add
cd ../CVPR-2027-draft
git commit -m "Sync from server $(date +%F)" && git push
```

`export_repo.py` never touches `RoleVLN/`. It copies changed files into the separate working copy
`../CVPR-2027-draft` and deletes files that are gone from the source. Edit `TIER1` at the top of the script to
change which checkpoints are committed. Anything else under `runs/` that ends in `.pt/.pth/.safetensors`
becomes a release asset automatically.

### 2.2 Upload the release assets

The deploy key used for `git push` can't create releases; that needs the GitHub API and a token.
Create a **fine-grained token** (GitHub → Settings → Developer settings → Fine-grained tokens) with
*Repository access: only CVPR-2027-draft* and *Permissions → Contents: Read and write*. Then run this in your
own terminal on the server, so the token stays out of shell history and logs:

```bash
cd /data/team_a/dave-workspace/CVPR-2027-draft
read -rs GITHUB_TOKEN && export GITHUB_TOKEN        # paste the token, press Enter
python3 -I migration/release_assets.py upload       # ~12 GiB; resumable, re-run after any failure
unset GITHUB_TOKEN
```

Uploads read the staged files in `../CVPR-2027-draft-assets/`. Checkpoints there are hardlinks, so they take no
extra space; the explorer archives are real files (4 GiB). Assets already on the release with the right size
are skipped, so re-running after a re-export uploads only what changed. At this server's ~6 MB/s, a first
upload takes about 35 minutes. Delete the token afterwards.

Fallback if you prefer the GitHub CLI: `gh release create assets-v1 ../CVPR-2027-draft-assets/*`
(asset names = file names; `release_assets.py download` works with either).

### 2.3 Data to carry by disk (optional)

To continue from exactly the current state instead of re-running rollouts, copy these from
`data/processed/baseline_v0/` with rsync or a disk (they are not datasets and not in the release):

| Folder | Size | Rebuild cost if you don't carry it |
|---|---:|---|
| `scorer_labels_smolvlm2_refined512_v1` | 11 G | Step 6 rollouts (hours). **Needed** for the planned relabel-and-retrain scorer test |
| `scorer_features_smolvlm2_refined512_dino_pair_v1` | 6.1 G | `tools/cache_step7.py` (GPU, ~1 h) |
| `dagger_groupA_round1_smolvlm2`, `dagger_groupA_round2_smolvlm2_stop2m` | 14 G | DAgger collection with the policy (hours) |
| `wm_visits_groupA_smolvlm2_refined512_v1`, `wm_groupA_smolvlm2_refined512_v1` | 26 G | Step 4 policy episodes (hours) |
| `expert_groupC_600` | 4.0 G | `tools/render_expert.py --group C` with the selection in `tools/run_step2.py` (minutes) |
| `expert_groupA`, `wm_dino_features_refined512_v1` | 319 G | don't carry: re-render (20 min on the H100) and recompute |

```bash
rsync -a --info=progress2 /data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/scorer_labels_smolvlm2_refined512_v1 \
      newhost:/path/to/RoleVLN/data/processed/baseline_v0/
```

## 3. On the new machine

### 3.1 Clone

The repo is public, so cloning needs no credentials. To push from the new machine, add its SSH key to your
GitHub account (or as another deploy key). The deploy key `rolevln-cvpr2027-deploy` belongs to the old server;
remove it under Settings → Deploy keys once you're done there.

**Recommended: use the same absolute path as the old server.** Code, configs and run records refer to
`/data/team_a/dave-workspace/RoleVLN` (38 code/config files, listed in `migration/hardcoded_paths.txt`), and
resuming old jobs needs it:

```bash
sudo mkdir -p /data/team_a/dave-workspace && sudo chown "$USER" /data/team_a/dave-workspace
git clone https://github.com/davemaxuell/CVPR-2027-draft.git /data/team_a/dave-workspace/RoleVLN
cd /data/team_a/dave-workspace/RoleVLN
```

If that's not possible, clone anywhere and rewrite the paths. The dry run lists the changes first:

```bash
python3 -I migration/relocate_paths.py --conda-base ~/miniforge3 --hf-home ~/hf_cache          # dry run
python3 -I migration/relocate_paths.py --conda-base ~/miniforge3 --hf-home ~/hf_cache --apply
```

It replaces `/data/team_a/dave-workspace/RoleVLN` → the clone, `/data/shared/hf_cache` → `--hf-home`, and
`/home/bufsgpu/yes` (the old conda base) → `--conda-base`. Run records under `runs/` keep the old paths as history.

### 3.2 Our checkpoints and the explorer logs

No token is needed while the repo is public (press Enter at the token prompt):

```bash
python3 -I migration/release_assets.py download          # everything; or --kind checkpoint, or --match 'step5_*'
python3 -I migration/release_assets.py verify            # sha256 of every checkpoint
```

If you make the repo private later, downloads need a fine-grained token with *Contents: Read-only*
(`read -rs GITHUB_TOKEN && export GITHUB_TOKEN` first).

Checkpoints land at their original paths under `[OURS]/baseline-v0/runs/`. The archives unpack into
`[prior works]/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/` and `[prior works]/ETPNav/data/logs/explorer/`.
If you moved `CVPR-2027-draft-assets/` by disk instead: `release_assets.py download --from-dir /path/to/it`.

### 3.3 Datasets and public weights

Follow [DATA_MIGRATION.md](DATA_MIGRATION.md) §1–4 and its checklist. In short:

1. Matterport3D (Terms of Use → `download_mp.py`): `--task habitat` → `data/scene_datasets/mp3d/`; skyboxes,
   house segmentations and camera parameters → `data/mattersim/v1/scans/`.
2. VLN-CE episodes: R2R-CE v1-3, RxR-CE v0, IR2R-CE, REVERIE-CE, NavRAG-CE, HA-VLN → `data/datasets/`
   (the folders already exist with their `source_manifest.json` / README files, holding the exact URLs).
3. ETPNav: `cd "[prior works]/ETPNav" && modelscope download --model admagic/ETPNav --local_dir ./data`
   (episodes, released checkpoints, waypoint predictors).
4. ScaleVLN: from HF `OpenGVLab/ScaleVLN`, the R2R annotations/connectivity, the two `*_mp3d_original.hdf5` eval
   features and the two fine-tuned checkpoints into `[prior works]/ScaleVLN/VLN-DUET/datasets/R2R/`; the
   REVERIE annotations + `BBoxes.json` into `VLN-DUET-RVR/datasets/REVERIE/annotations/`.
5. Hugging Face models into `$HF_HOME` (use the same revisions):
   `cd "[OURS]/baseline-v0" && HF_HOME=... python -I env/download_models.py manifest/model_revisions.json`;
   plus `facebook/nwm` (gated: request access again for the new account/token), `gaoyuezhou/dino_wm`, and the
   backbone trials in `manifest/Backbone_Trials.json`.
6. Upstream DINO-WM checkpoints (left out of git): `gaoyuezhou/dino_wm` →
   `[OURS]/baseline-v0/runs/step5_smolvlm2_refined512/assets/outputs/{point_maze,pusht,wall_single}/checkpoints/model_latest.pth`
   (only needed to repeat the Step 5 transfer from scratch; the trained models are in git).

### 3.4 Environments

Three conda environments, each recorded where it was built:

| Env | For | Recipe |
|---|---|---|
| `rolevln` | our pipeline (Python 3.9, habitat-sim/lab 0.2.4, torch 2.8 cu128, transformers 4.57.1, peft 0.17.1) | `bash "[OURS]/baseline-v0/env/setup_env.sh"`, after fixing its `source …/conda.sh` line (or `relocate_paths.py`) |
| `scalevln` | DUET/ScaleVLN eval, the explorer's DUET logger and what-if worker (Python 3.9, torch 2.5.1 cu121, transformers 4.12.5, numpy 1.23.5) | `[prior works]/ScaleVLN/LOCAL_SETUP.md` + `requirements-runtime.txt`. Build MatterSim in `ScaleVLN/Matterport3DSimulator/build`; the build needs the workarounds below |
| `etpnav` | ETPNav and `build_navmaps.py` (Python 3.8, habitat-sim 0.1.7 headless, torch 2.4.1 cu121) | `[prior works]/ETPNav/LOCAL_SETUP.md`; habitat-lab v0.1.7 is in `third_party_habitat-lab-v0.1.7/`, put on the path with a `.pth` file |

MatterSim build workarounds (from the old server):
- Build through a symlink without spaces: the `[prior works]` path breaks the compiler flags.
- Pass `-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON`.
- Add `$CONDA_PREFIX/include/jsoncpp/json -> ../json`.
- With CMake 4 / OpenCV 5, add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` and
  `CMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -std=c++17 -include cstdint -DCV_LOAD_IMAGE_ANYDEPTH=cv::IMREAD_ANYDEPTH -DCV_L2=cv::NORM_L2"`.
- Build only the target `MatterSimPython`.

Editable `pip install -e` fails on paths containing `[OURS]` / `[prior works]` (pip reads the brackets as
extras). That's why habitat-lab lives in `$CONDA_PREFIX/src` (rolevln) or is added with a `.pth` file (etpnav).

### 3.5 GPU / rendering

Habitat renders through EGL. Add your user to the `video` and `render` groups, or EGL may see only one GPU
(on the old server only CUDA device 0 worked). Pass Habitat's `gpu_device_id` (`--gpu`) rather than remapping
with `CUDA_VISIBLE_DEVICES` for rendering jobs. For the 16 GB VRAM limits (workers, SFT batch), see
TRAINING_STATE.md.

### 3.6 Regenerate

```bash
# expert renders for policy training (group A; ~200 G, 20 min on an H100 with 24 workers)
cd "[OURS]/baseline-v0" && python tools/render_expert.py --out ../../data/processed/baseline_v0/expert_groupA --workers 8 --chunk 150
# explorer 3D meshes (needs data/mattersim/.../house_segmentations; ~3.6 G; the server also builds a missing one on demand)
cd "[prior works]/ScaleVLN" && python3 analysis/vln_viz/build_meshes.py --detail      # whole buildings + per-floor detail
```

`analysis/vln_viz/cache/{navmaps,rxr_subtasks,view_index.json}` are in git. The row caches rebuild on first
start (~50 s).

### 3.7 Check that everything works

```bash
python3 -I migration/release_assets.py verify
cd "[OURS]/baseline-v0" && conda activate rolevln
python tools/check_ce_replay.py --dataset r2r --gpu 0 --group A --episodes 20 --out logs/check_ce_replay_r2r.json   # 20/20
python tools/check_ce_replay.py --dataset rxr --gpu 0 --group A --episodes 20 --out logs/check_ce_replay_rxr.json   # 20/20
cd Scorer && python -B check_design.py && python -B check_compatibility.py                                          # 13/13, 10/10
cd "../../../[prior works]/ScaleVLN" && bash run.sh eval --model b16      # val_unseen SR 79.40 / SPL 69.97 (B/16)
python3 analysis/vln_viz/visualizer_app.py   # port 8765; DUET, REVERIE, RxR and ETPNav CE runs should all list
```

The Scorer checks overwrite their `*_results.json`; `git diff` shows whether anything changed.

## 4. Troubleshooting

- **`release_assets.py` says the release is not reachable**: the release `assets-v1` has not been uploaded yet
  (§2.2), or the repo was made private and the token is missing or lacks *Contents* permission.
- **A checkpoint fails `verify`**: re-run `download` (it re-fetches mismatched files). A checkpoint that
  changed on the server after the export gets new sha256 values on the next export + upload.
- **Dangling symlinks right after cloning** (`data/tours.json`, `data/datasets/*/scene_datasets`,
  `[prior works]/ETPNav/data/scene_datasets`, …) are expected. They point into datasets you haven't downloaded yet.
- **The explorer lists no CE runs**: the ETPNav archive hasn't been unpacked (§3.2), or `--no-ce` was passed.
- **Old automatic jobs don't resume**: their `job.json` files hold absolute paths and in-progress state.
  Restart the stage from its `tools/run_step*.py` / `tools/auto_*.py` entry point (TRAINING_STATE.md).
