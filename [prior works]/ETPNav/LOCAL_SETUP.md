# ETPNav on this server

Upstream: https://github.com/MarSaKi/ETPNav @ 160e65f (cloned 2026-10-26; source unmodified). Weights from
ModelScope `admagic/ETPNav` (only what inference needs).

## Environment: conda `etpnav`

The upstream recipe (Python 3.6, torch 1.9.1+cu111) has no kernels for the H100 (sm_90), so:

- Python 3.8, `habitat-sim=0.1.7=py3.8_headless_linux_856d4b08…` (aihabitat channel), numpy 1.23.5
- habitat-lab v0.1.7 in `third_party_habitat-lab-v0.1.7/`, on the path through
  `$CONDA_PREFIX/lib/python3.8/site-packages/habitat_lab_v017.pth` (pip editable installs misread the `[prior works]`
  brackets as extras)
- torch 2.4.1 / torchvision 0.19.1 (cu121), transformers 4.12.5, tokenizers 0.10.3, pytorch-transformers 1.2.0,
  gym 0.17.3, CLIP (openai), lmdb 1.4.1, webdataset 0.1.40, plus the habitat-lab requirements
- no TensorFlow: `vlnce_baselines` imports it without using it; `tools/log_etpnav_steps.py` stands in
  tensorboard's `tensorflow_stub` (TF 2.13 would pin typing-extensions below what torch 2.4 needs)

Habitat EGL rendering sees only GPU 0 for this user.

## Data (`data/`)

| Path | Contents |
|---|---|
| `logs/checkpoints/release_r2r/ckpt.iter12000.pth` | R2R-CE fine-tuned ETPNav (2.1 GB) |
| `logs/checkpoints/release_rxr/ckpt.iter19600.pth` | RxR-CE fine-tuned ETPNav (4.1 GB) |
| `wp_pred/check_cwp_bestdist_hfov{90,63}` | waypoint predictors (R2R-CE, RxR-CE) |
| `ddppo-models/gibson-2plus-resnet50.pth` | depth encoder |
| `datasets/R2R_VLNCE_v1-2_preprocessed{,_BERTidx}/val_*` | R2R-CE episodes (BERT token ids) |
| `datasets/RxR_VLNCE_v0_enc_xlmr/val_*` | RxR-CE guide episodes (XLM-R token ids) |
| `datasets/RxR_VLNCE_v0` → `RoleVLN/data/datasets/RxR_VLNCE_v0` | RxR-CE ground-truth paths (nDTW) |
| `scene_datasets` → `RoleVLN/data/scene_datasets` | Matterport3D habitat scenes (90) |

Pre-training checkpoints are not downloaded (`MODEL.pretrained_path None`; the fine-tuned checkpoints hold all
771 policy tensors, checked at load). CLIP ViT-B/32 is cached in `~/.cache/clip`.

## Running

`tools/log_etpnav_steps.py` runs ETPNav's evaluation rollout and logs every decision for the navigation explorer
(`[prior works]/ScaleVLN/analysis/vln_viz`, see its README). Its docstring lists the options and the log format.
