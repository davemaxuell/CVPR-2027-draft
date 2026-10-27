# Training state at migration (baseline-v0)

Snapshot of 2026-10-27, written for picking the work up on the new machine. The full record is
`[OURS]/baseline-v0/README.md`, the `docs/Step*_Run.md` files and each run's `RESULTS.md` /
`final_verification.json`. This page says what was done, what is still running, and where every artifact
is after the move: **git** (in this repo), **release** (GitHub release `assets-v1`, fetched with
`migration/release_assets.py`), **carry** (a server item in RETRIEVE_FROM_SERVER.md: `pull_from_server.py`),
or **regenerate**. Our release files are also in the server item `our_checkpoints`; the release also holds the
third-party checkpoints (MIGRATION.md §1).

## The pipeline in one paragraph

A vanilla policy → world model → scorer system on R2R-CE + RxR-CE (English guide), Habitat 0.2.4, 15° turns,
0.25 m steps. The policy is SmolVLM2-500M with LoRA r16 on the decoder; it reads the instruction, 7 past views,
the last 16 actions and the current view, and argmaxes over FORWARD/LEFT/RIGHT/STOP. Two DINOv2-feature world
models predict the next view's features for each action. A small scorer ranks the four actions from the
policy context plus the predicted futures and may override the policy when it beats it by a margin.
Scenes are split A (fit the policy) / B (fit the scorer) / C (development) / E (= val_unseen, final), see
`manifest/scene_partition.json`.

## What has been done

| Step | What | Result | Artifacts (where after the move) |
|---|---|---|---|
| Expert data | Render all group-A expert episodes (R2R 7,489 + RxR 14,132) | 2.27M frames; SFT uses 21,333 clean episodes / 2,218,976 steps | `data/processed/baseline_v0/expert_groupA` (196 G): **regenerate** with `tools/render_expert.py` (20 min on an H100, 24 workers) |
| 2 SFT | 4,096 updates × 64 samples (0.12 passes, LR to 0) | Selected update 3,072: 4/64 C success, SPL 0.047, offline action acc 76.4% | selected adapter: **git**; other checkpoints + optimizer states: **release** |
| 3 DAgger | 256 A episodes, 125,730 corrective labels, 2,048 refinement updates | Tied Step 2 (4/64) with lower SPL; Step 2 kept | `runs/step3_smolvlm2/frozen_reference`: **git**; DAgger data `dagger_groupA_round1_smolvlm2` (7.9 G): **carry** |
| 3b STOP study | STOP-focused refinement + STOP-bias calibration | **Selected policy: refined checkpoint 512, STOP bias −0.25** (user's choice 2026-10-26): 29/224 C success, 9.6% forward collisions, 149 wrong stops; fails the original stopping safeguard | adapter: **git**; round-2 data `dagger_groupA_round2_smolvlm2_stop2m` (6.1 G): **carry** |
| 4 WM corpus | 256 A policy episodes + real branch outcomes | 2,488,382 movement transitions, 2,422,469 matched DINO/CDiT samples | `runs/step4_smolvlm2_refined512/reference` (the selected policy bundle): **git**; `wm_groupA_*` (0.4 G), `wm_visits_*` (26 G): **carry** |
| 5A DINO dynamics | DINO-WM PointMaze transfer vs fresh dynamics, 37,852 updates each | Both select update 36,864; C feature MSE 1.347 / 1.321 (copy-current 4.611) | `runs/step5_smolvlm2_refined512/frozen/` (both models + encoder): **git**; all intermediate checkpoints: **release**; DINO features `wm_dino_features_refined512_v1` (123 G): **regenerate** from the expert renders |
| 5B NWM/CDiT | Same comparison with NWM transfer vs fresh CDiT | CPU assets and checks done; GPU work queued behind the DINO E pilot | `assets/nwm_cdit_s_ema.pt`: **release** |
| 6 Scorer labels | All 4 actual action outcomes at 1,024 B + 256 C roots | 5,120 measured branches, full audit passed | `scorer_labels_smolvlm2_refined512_v1` (11 G, includes the branch traces): **carry**. Needed for the cheap relabelling test below. |
| 7 Scorer features | Frozen policy context + DINO predictions for all 1,280 roots | Audit passed | `scorer_features_smolvlm2_refined512_dino_pair_v1` (6.1 G): **carry** (or recompute with `tools/cache_step7.py`) |
| 8 Scorer training | Pooled scorer + learned-null-future control, 2,048 updates each | Training verified; the 6 checkpoints shortlisted for Step 9 | shortlisted 6: **git**; the other 26: **release** |
| 9 C calibration | Navigation screen + margin calibration on C224 | **Running at export time.** Diagnosis so far: with the scorer 8/224 (transfer) and 6/224 (fresh) vs 29/224 for the policy alone; the scorer forces STOP early (197/224 episodes) | per-episode results in `runs/step9_*`: **git** (snapshot, re-sync when done) |
| 10 E pilot | 224 E episodes, 5 systems, frozen settings | Starts automatically after Step 9 | `runs/step10_*` |
| Four-model repeat | NWM pair + one shared scorer over 4 world models, then the 7-system E repeat | Queued (`tools/auto_four_*.py`, `auto_nwm_repeat.py`), each waiting on the previous stage | `runs/automatic_*/status.json` |

The `carry` folders are not checkpoints. Regenerating them means rerunning policy rollouts on the
simulator, so pull them if you want to continue from exactly this state: ~64 G, the `recommended` server
items `processed_*` (RETRIEVE_FROM_SERVER.md).

## What was running when this snapshot was taken

Started on the H100 server with automatic progression (authorized by the user on 2026-10-27):

- `tools/run_step9.py`: Step 9 C confirmation (the null-control job had finished 171 episodes).
- `tools/auto_smol_remaining.py`: waits for calibration, then freezes settings and runs Step 10 on E.
- `tools/auto_nwm_repeat.py`: waits for the DINO E pilot, then runs the NWM/CDiT fit (Step 5B).
- `tools/auto_four_features.py`, `auto_four_training.py`, `auto_four_calibration.py`: the four-model repeat,
  each waiting for the stage before.

**Recommendation:** let this queue finish on the H100 (it is GPU-heavy, and the 16 GB machine is much
smaller), then do the final sync in MIGRATION.md §2. If you migrate earlier, the jobs can't resume
elsewhere from their saved state: their `job.json` files hold absolute paths and in-progress state. Start the
stage again from its `tools/run_step*.py` / `tools/auto_*.py` entry point instead.

## Diagnoses from 2026-10-27 (what to do next)

Two analyses on the existing data, no code changed:

1. **The scorer hurts navigation because its labels favour STOP.** The Step 6 utility (0.7·success +
   0.3·nDTW − costs, single greedy continuation) makes STOP the best action at 50.5% of B roots, since
   quitting early keeps nDTW credit while the mostly failing continuation wanders. 72% of overrides are L↔R
   flips (oscillation). Branch regret is anti-correlated with closed-loop success, so it is a misleading
   selection metric. With the actual next views the scorer is no better, so the world model is not the
   bottleneck. Relabelling the same data: STOP-best 50.5% → 26.9% (nDTW only on success) → 12.7% (success only).
2. **The policy underfits.** Group-A (training buildings) success 10.9% ≈ C 12.9%. STOP validation accuracy
   was 0% at every Step 2 checkpoint (STOP is 1% of samples). The LR decayed to 0 after 0.12 passes while
   validation loss was still falling, and only 1.7% of the parameters train (LoRA on the decoder; vision and
   projector frozen). Closed loop it turns LEFT 43% vs RIGHT 15%.

Suggested order: (a) overfit test on ~200 A episodes; (b) longer policy training with STOP fixed
(class weighting/oversampling, constant-ish LR, possibly unfreezing projector/LLM); (c) relabel the Step 6
data (success-only STOP credit), retrain the scorer (~15 min) and rerun the C screen with STOP veto-only;
(d) a larger cross-fitted scorer set; (e) an anti-oscillation rule; only then the NWM/four-model repeats.
Switching to Qwen3-VL-2B (already in `manifest/Backbone_Trials.json`) comes after (a) and (b).

## Constraints of the new machine (16 GB VRAM)

Measured on the H100 (94 GB):

- One full Step 9 navigation worker (policy + DINO world model + scorer + Habitat) uses 4.5 GiB typical,
  5.6 GiB peak per process. On 16 GB, run 2 workers, not the 4 used here (`--workers`).
- Step 2 SFT (500M, LoRA, batch 16) peaked at **38.8 GB**. On 16 GB use micro-batch 2–4 with gradient
  accumulation to the same effective batch 64, and/or gradient checkpointing. Expect it to be several times
  slower than the 21 samples/s here.
- Full fine-tuning of the 500M LM + connector should fit with checkpointing (estimate, not measured).
  SmolVLM2-2.2B could run as one worker (~8–9 GB, estimated) with LoRA-only training. StreamVLN-7B is out.
- Front-load heavy training (longer SFT, DAgger) on the H100 before the move if you can.
- Habitat EGL rendering needs the GPU visible to EGL. On the old server only CUDA device 0 worked for this
  account (not in the `video`/`render` groups). Add your user to those groups on the new machine.

## How to resume on the new machine

After MIGRATION.md §3 (clone, assets, data, environment):

```bash
cd "[OURS]/baseline-v0" && conda activate rolevln && export HF_HOME=<your hf cache>
python -I ../../migration/release_assets.py verify              # all checkpoints present
python tools/check_ce_replay.py --dataset r2r --gpu 0 --group A --episodes 20 --out logs/check_ce_replay_r2r.json
cd Scorer && python -B check_design.py && python -B check_compatibility.py && cd ..
python tools/check_policy_step.py --frames 8 --out logs/check_policy_step.json
# then re-render experts if you need them for training:
python tools/render_expert.py --out /path/to/data/processed/baseline_v0/expert_groupA --workers 8 --chunk 150
```

The selected policy loads from `runs/step4_smolvlm2_refined512/reference/` through
`policy.experimental_reference.ExperimentalReference`. The STOP bias −0.25 is part of its identity.
The frozen world models are in `runs/step5_smolvlm2_refined512/frozen/`, and the shortlisted scorers in
`runs/step8_smolvlm2_refined512_dino_pair_v1/{pooled,null}/checkpoints/`.
