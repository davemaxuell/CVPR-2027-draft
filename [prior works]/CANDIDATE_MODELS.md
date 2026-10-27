# Candidate prior works for later (VLN-CE)

Models to add next to ETPNav in the navigation explorer (`ScaleVLN/analysis/vln_viz`), none downloaded yet.
Checked against their repos on 2026-10-26: all have public weights and are evaluated on R2R-CE / RxR-CE in Habitat,
so the scenes and episodes already on disk apply. Numbers are as reported by the authors (val_unseen, NE / OSR /
SR / SPL, nDTW for RxR); for reference, ETPNav here: R2R-CE 4.75 / 64.0 / 57.15 / 49.18.

| Model | How it navigates | R2R-CE | RxR-CE | Weights | Simulator |
|---|---|---|---|---|---|
| [Uni-NaVid](https://github.com/jzhzhang/NaVid-VLN-CE) (RSS 2025) | 7B video-language model, RGB only, low-level moves, no map | 4.96 / 57.4 / 51.8 / 47.7 | 5.67 / 66.4 / 56.1 / 44.5 | HF `Jzzhang/Uni-NaVid` (`uninavid-7b-full-224-video-fps-1-grid-2`) | habitat 0.1.7, Python 3.8 |
| [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE) (RSS 2024) | Uni-NaVid's predecessor, same repo | 5.65 / 49.2 / 41.9 / 36.5 | 5.72 / 55.6 / 45.7 / 38.2 | HF `Jzzhang/NaVid` | habitat 0.1.7 |
| [InternVLA-N1 DualVLN](https://github.com/InternRobotics/InternNav) | VLM picks a goal in the image, a fast local policy drives to it | 4.05 / 70.7 / 64.3 / 58.5 | 4.58 / – / 61.4 / 51.8, nDTW 70.0 | HF `InternRobotics/InternVLA-N1-DualVLN` | newer Habitat, version unconfirmed |
| [StreamVLN](https://github.com/OpenRobotLab/StreamVLN) (2025) | streaming video LLM (LLaVA-Video, Qwen 7B) | 4.90 / 63.6 / 56.4 / 50.2 | 5.65 / – / 54.4 / 45.4, nDTW 63.7 | HF `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | habitat 0.2.4, Python 3.9, torch 2.1.2 |
| [NaVILA](https://github.com/AnjieCheng/NaVILA) (RSS 2025) | VILA 8B, outputs moves as text | SR ≈54 (paper) | in the paper; repo has no RxR script | HF `a8cheng/navila-llama3-8b-8f` | habitat 0.1.7 from source + NumPy hotfix |

Notes:

- Suggested order: Uni-NaVid first (same Habitat 0.1.7 stack and episode files as ETPNav, both datasets, the
  opposite design: no map, no panorama), then InternVLA-N1 DualVLN for the strongest current numbers.
- Training data differs from ETPNav: Uni-NaVid, StreamVLN and InternVLA-N1 use one checkpoint for both datasets;
  StreamVLN also trained on EnvDrop and ScaleVLN data; Uni-NaVid on other navigation tasks too. InternNav's table
  says "R2R-CE val" without seen/unseen. StreamVLN uses R2R-CE episodes v1-3 (ETPNav: v1-2), a small download.
- Cost: 7-8B checkpoints, roughly 15 GB each (~45 min at this machine's bandwidth). Run the model on GPU 1;
  Habitat renders on GPU 0 only. One forward pass per 25 cm / 15° move, so a full split takes hours.
- Explorer integration: these agents have no scored waypoints. Each decision is forward / turn / stop, so the
  logger would record per-step action probabilities, the panorama becomes the forward camera, and `ce_runs.py`
  needs a step-decision variant of its graph (no ghost waypoints). Track, sub-tasks, metrics and Compare with
  work as for ETPNav. The ETPNav logger (`ETPNav/tools/log_etpnav_steps.py`, `tools/explorer_env.py`) is the template.

Not on the list:

- [BEVBert](https://github.com/MarSaKi/VLN-BEVBert), [HNR](https://github.com/MrZihan/HNR-VLN),
  [g3D-LF](https://github.com/MrZihan/g3D-LF): CE models, but their repos still list releasing the navigation
  checkpoints as a to-do. Recheck later.
- [VLN-CE CMA](https://github.com/jacobkrantz/VLN-CE): only the RxR-CE weights are linked; ~30 SR, a lower bound only.
- Idea, not a prior work: run DUET / ScaleVLN H/14 in Habitat with ETPNav's waypoint predictor, as
  [Sim2Sim](https://github.com/jacobkrantz/Sim2Sim-VLNCE) (ECCV 2022) did for VLN↻BERT, to measure how much of the
  discrete success carries over to continuous environments.
