# Retrieving what is not on GitHub (from the old server)

**Written for the agent setting up the new machine.** GitHub holds the code, docs, run records, the explorer
with its run logs, and the selected checkpoints. Everything else (datasets, Matterport-derived files, public
weights, the rest of our checkpoints, processed training data) is still on the old H100 server. Copy it from
there with `migration/pull_from_server.py`, which rsyncs over SSH, using the item list in
`migration/server_manifest.json` (exact server paths, rsync filters, sizes measured 2026-10-27).

**Can't reach the old server?** Use the GitHub releases instead: the public one for checkpoints and scorer
features (MIGRATION.md §3.2b), the private one for the Matterport-derived processed data, explorer renders and
NWM image index (§3.2c), and public sources for datasets and weights (§3.3).

## Rules

- **Never commit pulled data.** Datasets and Matterport-derived files (scenes, skyboxes, house meshes, habitat
  renders, mesh caches) may not be redistributed (Matterport3D Terms of Use), and this repo is public.
  `.gitignore` covers every destination below; never `git add -f` inside them.
- Pull `required` first, then `recommended`. Don't pull `skip` items unless the user asks for one by name.
- The server's address is deliberately not written in this public repo. Ask the user for the SSH host
  (`user@host`, port, key), or use their `~/.ssh/config` alias.
- The server's internet link is slow (~6 MB/s measured for downloads; upload speed unknown). `required` is
  ~74 GiB, about 3.5 h at 6 MiB/s. Run pulls in the background and re-run after any interruption: rsync resumes
  (`--partial`) and skips identical files.
- Don't touch anything on the server: no deletes, and don't pause the user's jobs there. `pull` only reads.

## Steps

```bash
cd <repo root>                                   # ideally /data/team_a/dave-workspace/RoleVLN (MIGRATION.md §3.1)
export HF_HOME=<hf cache on this machine>        # where the hf_* items go (default ~/.cache/huggingface)
H=user@old-server                                # from the user; add --ssh "ssh -p PORT -i KEY" if needed

python3 -I migration/pull_from_server.py list                                  # what exists, sizes
python3 -I migration/pull_from_server.py pull --host "$H" --dry-run            # shows every rsync command
python3 -I migration/pull_from_server.py pull --host "$H"                      # required  (~74 GiB)
python3 -I migration/pull_from_server.py pull --host "$H" --priority recommended   # + ~64 GiB
python3 -I migration/pull_from_server.py check --host "$H" --priority recommended  # "complete" for each item
```

Single items: `pull --host "$H" --item explorer_meshes processed_scorer_labels`. Items land at the same paths
relative to the repo as on the server. `hf_*` items go into `$HF_HOME/hub/`; `hf_bert_base_uncased` and
`clip_cache` go under `~/.cache/`.

After pulling:

1. If the repo is not at `/data/team_a/dave-workspace/RoleVLN`, run `migration/relocate_paths.py` (MIGRATION.md §3.1).
2. Checkpoint integrity: `python3 -I migration/release_assets.py verify` (sha256 of all 445 release files:
   107 checkpoints, 2 step logs, 336 scorer-feature files, which `our_checkpoints`, `explorer_train_steplogs`,
   `etpnav_data`, `duet_eval`, `ha_vln`, `processed_scorer_features` and the optional `duet_training` +
   `reverie_training` also provide).
3. Set up the environments and run the checks: MIGRATION.md §3.4–3.7.

Alternatives if the server is gone: all checkpoints, ours and third-party, come from the GitHub release (`release_assets.py download`,
once the user has uploaded it), and public data from the sources in the `alt` field of each manifest item and in
DATA_MIGRATION.md. The Matterport-derived explorer files can be regenerated: `etpnav_explorer_views` with
`ETPNav/tools/log_etpnav_steps.py`, `explorer_meshes` with `build_meshes.py --detail` after pulling or
downloading `mattersim_house_ply`. Processed data can be regenerated with the step scripts (TRAINING_STATE.md).

## Items

### required: 74.3 GiB
| Item | Size | What it is for | Server path (under RoleVLN/ unless absolute) |
|---|---:|---|---|
| `scenes_mp3d` | 6.6 G | Habitat scenes for everything CE: baseline-v0, ETPNav, all CE datasets, build_navmaps.py (_semantic.ply (15 G) left out: nothing reads it) | `data/scene_datasets/mp3d` |
| `mattersim_skyboxes` | 17.3 G | explorer panoramas / front view / "Photo at agent" | `data/mattersim/v1/scans` |
| `mattersim_house_meta` | 5.7 G | explorer floors and room names (.house, panorama_to_region) (the .ply meshes are the separate optional item mattersim_house_ply) | `data/mattersim/v1/scans` |
| `r2r_ce` | 0.2 G | baseline-v0 R2R-CE episodes + experts | `data/datasets/R2R_VLNCE_v1-3_preprocessed` |
| `r2r_ce_minimal` | 2 M | R2R-CE minimal release (IR2R-CE links) | `data/datasets/R2R_VLNCE_v1-3` |
| `rxr_ce` | 0.3 G | baseline-v0 RxR-CE episodes; ETPNav nDTW ground truth (its text_features link points at the skipped 159 G RxR text features (item rxr_text_features)) | `data/datasets/RxR_VLNCE_v0` |
| `ir2r_ce` | 0.2 G | CE dataset (planned iterative-CE experiments) | `data/datasets/IR2R-CE` |
| `reverie_ce` | 22 M | CE dataset (planned) | `data/datasets/REVERIE-CE` |
| `navrag_ce` | 0.6 G | CE dataset (planned; training also needs HM3D, never had) | `data/datasets/NavRAG-CE` |
| `ha_vln` | 13.7 G | CE dataset HA-VLN 2.0 (episodes, HAPS 2.0 humans, CMA ckpt) (the 7.8 G zip is left out; HAPS2_0/ is extracted) | `data/datasets/HA-VLN` |
| `etpnav_data` | 6.2 G | ETPNav eval: CE episodes, released checkpoints, waypoint predictors (the two dataset symlinks are already in git (relative)) | `[prior works]/ETPNav/data` |
| `etpnav_explorer_views` | 3.8 G | explorer CE runs: habitat cube faces at every map node (episode logs are already in git) (Matterport-derived renders, so not in the public repo) | `[prior works]/ETPNav/data/logs/explorer` |
| `duet_eval` | 5.6 G | DUET/ScaleVLN eval, explorer what-if and log_duet_steps.py | `[prior works]/ScaleVLN/VLN-DUET/datasets/R2R` |
| `explorer_train_steplogs` | 0.2 G | explorer: DUET train-split step logs (the two files over GitHub's 100 MB limit) (also a release asset) | `[prior works]/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map` |
| `our_checkpoints` | 9.5 G | all our checkpoints + optimizer states (15 are also in git; the rest also on the release) (includes the 3 upstream DINO-WM checkpoints under step5/assets) | `[OURS]/baseline-v0/runs` |
| `hf_smolvlm2` | 1.9 G | policy backbone (revision 7b375e1b) | `/data/shared/hf_cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct` |
| `hf_dinov2_small` | 84 M | world-model encoder | `/data/shared/hf_cache/hub/models--facebook--dinov2-small` |
| `hf_nwm` | 0.7 G | NWM/CDiT (gated on HF) | `/data/shared/hf_cache/hub/models--facebook--nwm` |
| `hf_dino_wm` | 0.9 G | DINO-WM transfer source | `/data/shared/hf_cache/hub/datasets--gaoyuezhou--dino_wm` |
| `hf_bert_base_uncased` | 0.4 G | ScaleVLN tokenizer (it runs with HF_HUB_OFFLINE=1 and the default cache) | `/home/bufsgpu/.cache/huggingface/hub/models--bert-base-uncased` |
| `clip_cache` | 0.3 G | ETPNav CLIP ViT-B/32 | `/home/bufsgpu/.cache/clip` |

### recommended: 63.8 GiB
| Item | Size | What it is for | Server path (under RoleVLN/ unless absolute) |
|---|---:|---|---|
| `explorer_meshes` | 3.5 G | explorer 3D building + per-floor detail meshes (Matterport-derived; otherwise rebuild with build_meshes.py --detail from mattersim_house_ply) | `[prior works]/ScaleVLN/analysis/vln_viz/cache` |
| `processed_scorer_labels` | 10.2 G | Step 6 labels + branch traces; needed for the relabel-and-retrain scorer test | `data/processed/baseline_v0/scorer_labels_smolvlm2_refined512_v1` |
| `processed_scorer_features` | 6.1 G | Step 7 scorer features (also on the release; else tools/cache_step7.py) | `data/processed/baseline_v0/scorer_features_smolvlm2_refined512_dino_pair_v1` |
| `processed_dagger` | 13.8 G | DAgger round 1 + STOP round 2 data | `data/processed/baseline_v0` |
| `processed_wm_corpus` | 26.3 G | Step 4 world-model corpus (visits + index) | `data/processed/baseline_v0` |
| `processed_expert_c600` | 3.9 G | group-C validation renders (SFT val) | `data/processed/baseline_v0/expert_groupC_600` |

### optional: 441.7 GiB
| Item | Size | What it is for | Server path (under RoleVLN/ unless absolute) |
|---|---:|---|---|
| `mattersim_house_ply` | 24.7 G | house meshes: only to rebuild the explorer mesh caches | `data/mattersim/v1/scans` |
| `mattersim_camera_params` | 41 M | MatterSim camera parameters (nothing here reads them) | `data/mattersim/v1/scans` |
| `scenes_mp3d_semantic` | 14.2 G | Habitat semantic meshes (unused) | `data/scene_datasets/mp3d` |
| `rxr_pose_traces_val` | 4.0 G | only to rebuild the explorer RxR sub-tasks | `data/datasets/RxR/pose_traces` |
| `rxr_annotations` | 0.9 G | original RxR annotations (val guide files are in git) | `data/datasets/RxR` |
| `duet_training` | 47.7 G | everything else for retraining ScaleVLN/DUET (features, aug, LXMERT) | `[prior works]/ScaleVLN/VLN-DUET/datasets` |
| `reverie_training` | 9.5 G | ScaleVLN REVERIE features/checkpoints/annotations | `[prior works]/ScaleVLN/VLN-DUET-RVR/datasets/REVERIE` |
| `processed_expert_a` | 195.5 G | group-A expert renders (re-render instead: tools/render_expert.py (20 min on the H100)) | `data/processed/baseline_v0/expert_groupA` |
| `processed_wm_dino_features` | 122.7 G | cached DINO features (recompute from the expert renders instead) | `data/processed/baseline_v0/wm_dino_features_refined512_v1` |
| `runs_frames_and_indices` | 2.2 G | simulator frame dumps + Step 5 data index | `[OURS]/baseline-v0/runs` |
| `discrete_benchmarks` | 0.8 G | R2R, R4R, REVERIE, SOON, CVDN, NDH, HANNA annotations | `data/datasets` |
| `discrete_shared` | 3.9 G | ResNet-152 features + MatterSim graphs for those benchmarks | `data` |
| `gsa_r2r` | 15.6 G | GSA-R2R annotations, graphs, CLIP features | `data/datasets/GSA-R2R` |

### skip: 1335.0 GiB
| Item | Size | What it is for | Server path (under RoleVLN/ unless absolute) |
|---|---:|---|---|
| `rxr_text_features` | 158.1 G | only VLN-CE CMA/Seq2Seq RxR baselines | `data/datasets/RxR/text_features` |
| `rxr_pose_traces_train` | 14.6 G | nothing reads it | `data/datasets/RxR/pose_traces/rxr_train` |
| `matterport_zips` | 1101.3 G | original Matterport/Habitat zips (extracted copies above) | `data/.downloads` |
| `dataset_download_cache` | 14.3 G | archives already extracted | `data/datasets/.downloads` |
| `scalevln_smoke_runs` | 45.5 G | setup smoke tests and demo exports | `[prior works]/ScaleVLN/runs` |
| `legacy` | 1.2 G | PRM800K, NavVEM episodes, hime task data, download logs | `data` |

