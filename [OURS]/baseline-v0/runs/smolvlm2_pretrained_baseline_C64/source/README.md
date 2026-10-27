# baseline-v0 — vanilla baseline

The navigation policy → world model → scorer pipeline trained stage by stage with ordinary supervised objectives,
before any research contribution. Spec: [docs/Vanilla_Training.md](docs/Vanilla_Training.md).
First trial: SmolVLM2-500M policy + DINO-feature world model.

## Layout

```text
baseline-v0/
├── docs/            design spec: Vanilla_Training, CE_World_Model_Approach, Architecture_Compatibility, Scorer_Design
│   └── references/  the cited dossiers and reviews (reading copies)
├── manifest/        what defines the experiment
│   ├── Backbone_Trials.json, World_Model_Trials.json   locked policy trials / agreed world-model trials
│   ├── source_pins.json                               saved HF configs + revisions of the three policies
│   ├── experiment_manifest.yaml                       every setting, tagged [verified] / [proposed] / [TBD]
│   ├── scene_partition.json                           A/B/C (61 train scenes) + E (val_unseen) pilot split
│   └── model_revisions.json                           downloaded checkpoint revisions
├── Scorer/          scorer, labels, interfaces, world-model adapters, runtime + their checks
├── third_party/     pinned upstream source, read-only (see below)
├── sim/             habitat config built from the manifest; RxR-CE dataset loader (RxRVLN-v1)
├── policy/          prompt + history window (common), SmolVLM2 inputs/model (smolvlm2), per-step SFT samples (data)
├── tools/           scene split, expert rendering, SFT trainer, simulator and policy checks
├── env/             conda env build, model downloads
└── logs/            check reports, build/download logs, sample frames
```

## Environment

```bash
source /home/bufsgpu/yes/etc/profile.d/conda.sh && conda activate rolevln
export HF_HOME=/data/shared/hf_cache
```

Python 3.9, habitat-sim 0.2.4 (headless, bullet), habitat-lab v0.2.4 (in `$CONDA_PREFIX/src/habitat-lab`),
torch 2.8.0+cu128, transformers 4.57.1, peft 0.17.1. Built by `env/setup_env.sh`.

**GPUs:** this account is not in the `video`/`render` groups, so Habitat's EGL renderer sees only CUDA device 0.
Pass `--gpu 0` (Habitat's `gpu_device_id`) and do not remap with `CUDA_VISIBLE_DEVICES`. GPU 0 is free for this project;
GPU 1 also hosts another team's services (~10 GB). Adding the account to those groups
(`sudo usermod -aG video,render bufsgpu`, then log in again) should expose GPU 1 to EGL as well; this has not been tested.

## Checks run (2026-10-23)

| Check | Result |
| --- | --- |
| `python tools/check_ce_replay.py --dataset r2r --gpu 0 --group A --episodes 20 --out logs/check_ce_replay_r2r.json` | 20/20 R2R-CE experts (spread over 11 scenes) succeed; 0.0 m deviation from stored locations; action order `[stop, move_forward, turn_left, turn_right]`; RGB 480×640 at 1.25 m; sliding on |
| `python tools/check_ce_replay.py --dataset rxr --gpu 0 --group A --episodes 20 --out logs/check_ce_replay_rxr.json` | Loader returns all 14,132 group-A English guide episodes; 20/20 experts (14 scenes) succeed with turns split to 15°; max drift 0.27 m (present in the RxR data at 30° too) |
| `CUDA_VISIBLE_DEVICES=1 python tools/check_policy_step.py --frames 8 --out logs/check_policy_step.json` | LoRA r=16 on decoder projections only (8.7M / 516M params); loss covers only `FORWARD<end_of_utterance>` (4 tokens); 64 tokens per frame; q_t width 960; 2.9 GB peak |
| `cd Scorer && python -B check_design.py` | 13/13 pass; scorer parameters 2,890,115 / 2,873,603 / 3,170,819 |
| `cd Scorer && python -B check_compatibility.py` | 10/10 groups pass; DINO predictor (20,122,600) and CDiT-S/2 (49,227,680) details identical to the Windows run |

These confirm interfaces only; they are not training results. Both Scorer checks overwrite their `*_results.json`;
the original Windows runs are kept as `Scorer/*_results.windows.json`.

Changes to the delivered Scorer code: `scorer.py` starts with `from __future__ import annotations` (its `Tensor | None`
hints need Python 3.10+), and `check_compatibility.py` reads `third_party/` and `manifest/source_pins.json`.

## Action space

One action space for both datasets, as in StreamVLN: FORWARD 0.25 m, LEFT/RIGHT 15°, STOP. RxR-CE's expert actions were
generated with 30° turns, so `sim.config.load_expert` splits each into two 15° turns (exact: turns do not move the agent).
RxR results are therefore not on the official 30° RxR-CE protocol; state this when reporting.
Step limits (B_max): R2R-CE 500, RxR-CE 700 (the longest 15° RxR expert is 627 steps).

## Expert data (group A)

`/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupA/` (200 GB), written by
`python tools/render_expert.py --out <dir> --workers 24 --chunk 150` in 20 min on GPU 0 (resumable). Per episode:
`rgb/{t:04d}.jpg` (640×480, q95; frame t = view before action t) and `meta.json` (instruction, expert and executed
actions, poses, collision flags, drift from the stored path, metrics). `render_manifest.json` records settings and
source hashes; `{r2r,rxr}_index.jsonl` has one row per episode.

| | Episodes | Frames | Expert success | Drift ≥ 0.5 m |
| --- | --- | --- | --- | --- |
| R2R-CE | 7,489 | 450,005 | 100% | 0 |
| RxR-CE (en guide) | 14,132 | 1,815,393 | 98.5% (210 fail) | 288 |

RxR experts drifting ≥ 0.5 m average 27 wall collisions (27% succeed), so SFT excludes failed, truncated and drifting
experts (`--max-deviation 0.5`): 21,333 episodes, 2,218,976 per-step samples (STOP 1%, FORWARD 53%, LEFT 24%, RIGHT 22%).

## Policy SFT (Step 2)

One sample per (episode, t): the full instruction, up to 7 earlier views spread over the whole past (always incl. t−1),
the last 16 executed actions as text, and the current view; target ` ACTION<end_of_utterance>` (3–4 tokens), loss on
those tokens only. Decoding: argmax over the four actions' first-token logits at the prompt end (distinct first tokens),
one step per forward pass. `SmolVLM2Inputs` builds prompts for training and deployment alike.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_policy_sft.py --run-dir runs/<name> --max-steps <N> [--val-root <rendered group-C dir>]
```

Verified on a smoke run: loss falls on real data; deploy-time scores equal training-time prompt-end scores exactly;
resume restores weights, optimizer, schedule and data order (identical next-step loss with dropout off; LoRA dropout
masks differ after resume); batching vs batch 1 changes logits by ≤ 0.125 (bf16) with the same argmax.

## third_party

| Folder | Upstream | Commit | Used for | License file |
| --- | --- | --- | --- | --- |
| `dino_wm/` | gaoyuezhou/dino_wm | `0a9492fa` | ViT predictor (`models/vit.py`, hash-checked by `Scorer/world_models.py`) | not in snapshot |
| `nwm/` | facebookresearch/nwm | `3f6cd8e7` | CDiT (`models.py`) and diffusion sampler (`diffusion/`), hash-checked | `LICENSE.md` |
| `streamvln/` | InternRobotics/StreamVLN | `e48f6ff7` | Reference for policy SFT/DAgger code; not executed. `config/vln_r2r.yaml` is the source of our simulator settings | not in snapshot |

Each folder's `SNAPSHOT.json` records the source URL, commit and retrieval date. Files are unmodified. Add the missing
upstream license files before sharing this folder outside the team.

`docs/references/` holds reading copies of the cited dossiers and reviews, plus `resource_arithmetic.json` (weight and
optimizer-state sizes for the 8 GB target; arithmetic, not measured). The dossiers' own internal links (papers, audit
notes) point into the original research folder and do not resolve here.

## Step 2 pilot run

`runs/step2_smolvlm2/` contains the completed pilot. See [RESULTS.md](runs/step2_smolvlm2/RESULTS.md)
for the checkpoint comparison and `train/log.jsonl` for learning/validation metrics.
All four checkpoints were evaluated in the simulator. The selected checkpoint is **step 3,072**:
4/64 group-C successes (6.25%), macro SPL 0.0472, and offline action accuracy 76.4%.
Navigation performance remains weak. `best_checkpoint.json` records the selection;
`train/best_checkpoint` points to its adapter. The final training update was 4,096.

- Validation: 300 R2R + 300 RxR episodes, seeded and spread across all eight group-C scenes.
  596 episodes / 46,373 steps pass the expert-quality filter; offline validation uses 2,000 fixed steps.
- Speed: 21.11 samples/s at batch 16 with 24 data workers; full pass approximately 29.2 hours.
  CPU bf16 pixel conversion and selective response-position logits were checked against the original loss.
- Chosen pilot budget: 4,096 optimizer updates × 64 examples = 262,144 steps (0.1181 passes),
  approximately 3.5 hours plus validation and simulator evaluation. This is a partial-pass pilot.
- Closed-loop selection: the same 32 R2R + 32 RxR episodes at updates 1,024 / 2,048 / 3,072 / 4,096;
  highest dataset-macro success, then macro SPL, then lowest offline cross-entropy.
  Selection includes episodes regardless of expert quality. Group E is untouched.
- Evaluator checks: expert replay succeeds on all 16 checked episodes (eight scenes per dataset).
  Cached online pixel tensors exactly match native preprocessing at early and late history positions.
  Four independent simulator workers reproduce the serial expert metrics and four complete policy
  trajectories exactly; each policy decision still uses batch size one. Earlier checkpoint evaluations
  can overlap SFT through `tools/overlap_step2_eval.py`. The 3.5-hour training estimate assumes a
  dedicated GPU; sharing it with these rollouts increases the training stage's elapsed time.

Restart `python tools/run_step2.py` in the `rolevln` environment to resume a failed job from saved state.
The job rejects duplicate running instances and changed source files; failures are recorded in `status.json`.
Training explicitly exposes CUDA 0 to avoid PyTorch's mixed full-GPU/MIG enumeration issue on this host;
simulator evaluation retains physical EGL device numbering.

## Next

1. Step 3: choose the corrective expert/recovery rule, collect group-A DAgger data, refine, then freeze.
2. Remaining simulator checks from `docs/Architecture_Compatibility.md`: turn signs, blocked forward motion, branch replay with metric prefix.
