# trial_1: the StreamVLN recipe on Qwen3.5-4B

28 October 2026 · design record for the policy stage (steps 1–3) · status: **code prepared and
checked, nothing trained**; decided so far: backbone, recipe, full fine-tuning, one pass, visible
action history, final checkpoint (§4). Stage one is fully specified. Sources: `third_party/streamvln` @ e48f6ff7 plus, for the files that
snapshot lacks, `streamvln/streamvln_dagger.py`, `streamvln/dataset/vln_action_dataset.py`,
`streamvln/habitat_extensions/measures.py`, `scripts/streamvln_dagger_collect.sh` and
`config/vln_dagger.yaml` read at the same commit (not executed, not vendored).

## 1. Recipe mapping

| StreamVLN setting | StreamVLN value | trial_1 | Where |
| --- | --- | --- | --- |
| Backbone | LLaVA-Video-7B (Qwen2-7B + SigLIP-384) | **Qwen3.5-4B** (`Qwen/Qwen3.5-4B` @ 851bf6e8), user's choice | `stream_policy/qwen35.py` |
| Tokens per view | 27×27 SigLIP patches, 2×2 pooled → 196 | 512×384 view, 32-px merged patches → **192** | `StreamConfig.frame_*` |
| Dialogue window (`num_frames`) | 32 actions, then the context restarts | 32 | `layout.py` |
| Actions per answer (`num_future_steps`) | 4, one new view per answer | 4 | `layout.py` |
| Memory (`num_history`) | 8 views `arange(0, s, s // 8)` at each window start | 8 views `i*s//8` (identical when 8 divides s) | `layout.memory_indices` |
| Prompt, conjunctions, actions | fixed task prompt, 7 conjunctions, `↑ ← → STOP` | identical text; each action is one token | `config.py` |
| Training sample | one window, every assistant turn supervised | identical (checked on 1,375 random windows) | `data.py` |
| Supervised tokens | assistant turns (+ role headers and newlines, a LLaVA quirk) | actions + each turn's `<|im_end|>` | `qwen35.Qwen35Inputs.encode` |
| Augmentation | ColorJitter(0.2, 0.2), Posterize(4), Sharpness(1.5), Autocontrast | identical, `--augment` | `data.streamvln_augmentation` |
| Tuned parts | vision tower + projector + LLM | **`full`** (decided); `llm` and `lora` remain available | `qwen35.Qwen35Policy` |
| Optimiser | lr 2e-5, vision 5e-6, wd 0, bf16 (fp32 master weights in ZeRO-2), grad checkpointing | same; fp32 master weights, bf16 autocast | `--recipe streamvln_stage1` |
| Stage-one schedule | warmup 7.5%, cosine with floor 1.85e-5 (HF Trainer), 1 epoch, drop-last | same shape; **one pass** (decided) = 623 updates, 47 warmup | same |
| Effective batch | 2 × 2 accumulation × 32 GPUs = 128 | micro-batch 1 × 128 accumulation | same |
| Stage-one data | R2R, RxR, EnvDrop (+ ScaleVLN in the released model) | group-A R2R-CE + RxR-CE English guide | protocol (trial_0) |
| DAgger collection | stage-one model drives; expert takeover rule; keep detours | same rule, same thresholds | `controller.py`, `expert.py`, `tools/collect_stream_dagger.py` |
| Stage two | expert + DAgger + co-training (LLaVA-Video-178K, ScanQA/SQA3D, MMC4); warmup 3%, cosine to 0 | expert + DAgger roots (`--source-weights` optional); `--recipe streamvln_stage2 --init <stage one>` | trainer |
| Inference | sliding-window KV cache, depth-based voxel pruning of memory | per-decision re-encoding of the window, constrained decoding | `server.py` |
| Simulator | Habitat 0.2.4, 640×480, HFOV 79, 0.25 m, 15°, success 3 m | identical (shared `manifest/experiment_manifest.yaml`) | `sim/` |

### StreamVLN's DAgger rule, as implemented

The policy acts; with probability β per turn the expert drives a whole turn (released script: p = 0,
so β = 0). The expert is Habitat's ShortestPathFollower towards the episode's reference waypoints in
order (radius 1.8 m, final waypoint 0.25 m). It takes over after an error and keeps control until the
next waypoint: a policy STOP ≥ 3 m from the goal, more than 12 disagreements, or disagreements above
0.8 × (expert actions per reference segment). Disagreements reset only at a takeover. A policy STOP
before the final waypoint is replaced by the expert's action, so episodes end only at the goal or the
step limit. **Kept** episodes end < 0.5 m from the goal with path-length ratio < 0.93 after a takeover
or < 0.85 without one: recoveries and detours, not clean successes. They are trained on their
**executed** actions, final frame labelled STOP. Reference paths are sparse in both datasets (R2R-CE
median 6 points, RxR-CE English guide median 8, max 24), so the per-segment rule transfers as is.

## 2. Deviations and why

1. **Backbone.** Qwen3.5-4B is a hybrid model: 24 Gated DeltaNet (linear-attention) layers and 8
   full-attention layers, a native ViT (16-px patches, 2×2 merge, M-RoPE). It needs transformers ≥ 5
   (Python ≥ 3.10) while habitat-sim 0.2.4 exists only for Python 3.9, hence two environments and a
   policy server (`rolevln-qwen35` ↔ `rolevln`, `stream_policy/client.py`).
2. **No EnvDrop, ScaleVLN or co-training data.** The pilot protocol uses R2R-CE + RxR-CE only, and the
   co-training corpora are hundreds of GB over a ~6 MB/s link.
3. **No voxel pruning of memory.** It needs depth; the protocol is RGB-only. Memory views are full
   192-token views (16 views → 3.5k tokens per window).
4. **No KV-cache append between turns.** Hugging Face's Qwen3.5 reuses linear-attention state only for
   one-token steps; appending a multi-token turn (view + text) restarts those 24 layers from zero. The
   server re-encodes the window prefix at each decision and caches only the turn's own action tokens.
   Outputs are the same as a correct cache; only compute differs.
5. **Constrained decoding.** Argmax over the four action tokens instead of free generation and regex
   parsing; StreamVLN's "empty output → STOP" case cannot occur. A STOP logit bias is a server option.
6. **Expert labels.** Waypoints are snapped onto the agent's navmesh island, the follower is rebuilt per
   query (as trial_0's `sim/recovery.py`), and a follower failure is "no label" (episode aborted, not
   kept) instead of STOP, which StreamVLN would count as reaching the waypoint.
7. **Context after a takeover.** The dialogue shows the actually executed actions, so the policy's
   context always equals the training layout of the real trajectory. StreamVLN's cache keeps only the
   model's own turns and skips expert turns.
8. **Chat format.** Qwen markup without a thinking block, tokenized piece by piece identically in
   training and inference (StreamVLN also replaces the chat template). Inference picks each turn's
   conjunction by a fixed hash instead of `random.choice`.

## 3. Measured on the H100 server, 28 October 2026

| Quantity | Value | Record |
| --- | --- | --- |
| Tokens per window | 1.9k (first window, 8 views) / 3.5k (8 memory + 8 views); 40 supervised | `steps/00_setup/check_stream_backbone.json` |
| One pass over group A | 79,814 windows (58,484 with memory) from 21,330 episodes / 2,218,970 steps, after StreamVLN's 4-action minimum (drops 3 two-action RxR episodes) | `steps/01_experiment/data_audit.json` |
| LoRA r64 (122M, 2.7%) | 0.58 s per 3.5k-token window, **11.7 GiB** peak | same |
| `llm` (4.23B; vision frozen) | 0.61 s, 66.2 GiB peak | same |
| `full` (4.54B, StreamVLN) | 0.78–0.95 s, 70.2 GiB peak in a 3-update profile | same |
| `full` peak while accumulating 128 micro-batches | **≈ 74 GiB** of 93.6: 67.6 GiB weights + gradients + Adam, ≤ 6.2 GiB activations (longest window, 4.0k tokens). Autocast's cast cache is off: with fp32 master weights it held a bf16 copy of every weight (+4.9 GiB; 78.8 GiB before) | probe, 2026-10-28 |
| One pass in `full` mode, one H100 | **≈ 17 h**: 0.76 s per window (48-window group-A sample, reweighted to 73% memory windows) plus validation and saves; micro-batch 2 is 44% slower per window (padding) | probe |
| Resume | exact: a run resumed from update 1 reproduces update 2's loss to all digits (fp32 master weights, Adam state, per-draw seeded data stream); later updates drift only by GPU nondeterminism | trainer smoke runs |
| Checkpoints | 9.08 GB bf16 (tied `lm_head` stored once, Qwen's unused MTP head omitted); an untrained save equals the original weights bitwise and decodes identically | probe |
| fla kernels vs torch fallback | forward+backward ~4× faster (0.7 s vs 3.0 s in a probe; 1.1 s vs 4.0 s in the recorded check, which also copies gradients out); prefill 0.17 s vs 0.52 s | `kernels` in the backbone check |
| bf16 gradients vs an fp32 reference (one window) | fla at least as close as the torch fallback on every probed weight, and deterministic; language-model layers 0.92–1.0 cosine; **vision tower only 0.2–0.5** (merger ≈ 0.75) in both bf16 paths; losses within 0.01 | same |
| Cached decoding vs full re-encoding | same argmax; logits within 0.125 (one bf16 step) | backbone check |
| Training window vs inference prefix | same argmax; logits within 0.28 | backbone check |
| Closed loop (2 workers sharing GPU 0) | 0.11 s policy time per step; one model call per turn without overrides | `steps/00_setup/smoke_eval_summary.json` |

The Gated DeltaNet backward of fla 0.6.0 is refused on Hopper with Triton 3.4–3.7.0 (wrong gradients,
fla #640), so the env pins torch 2.13 + Triton 3.7.1 (cu126 build, which runs on the 550 driver).

## 4. Decisions

Decided by the user on 28 October 2026:

1. **Train mode: `full`**, as StreamVLN. It needs ~74 GiB, so stage one, DAgger and stage two run on the
   H100 before the move; the 16 GB machine can run the policy (~9 GB) but cannot train it in this mode.
   Kept as context: under bf16 one window's vision-tower gradient is mostly rounding noise (cosine 0.2–0.5
   to fp32), which a 128-window batch averages down if the noise is unbiased.
2. **Budget: one pass**, as StreamVLN: 623 updates of 128 windows, ≈ 17 h. Applied through
   `--recipe streamvln_stage1` with StreamVLN's optimiser, schedule, batch and augmentation (`[adopted]`).

3. **Action history: visible, as StreamVLN (option A).** The dialogue keeps the executed actions; no
   action dropout, no views-only placeholder, no change-point loss weighting. The risk this accepts,
   measured before the decision: inside a window the prompt shows every executed action since the window
   began (up to 31), and 96% of decisions see some. From that history alone, without any view, an 8-gram
   predicts 73.0% of group-A expert actions, and 74.3% of the first action after each new view
   (repeat-the-last-action: 60.4%; majority class: 53%). trial_0's SFT reached 76% offline while mostly
   copying (its swap probe); training shows the expert's actions as history, deployment the policy's
   own, so copying repeats its own mistakes (trial_0's forward collisions and spins). Reading results:
   offline accuracy cannot show visual grounding; compare offset-0 accuracy with 74.3% and judge the
   policy by a view/history swap probe and C closed-loop collisions, spins and success. Alternatives
   considered: train-time action dropout and views only (both swap the action tokens for a placeholder;
   linear attention rules out hiding them with an attention mask) and change-point loss weighting.
4. **Checkpoint: the final one** (update 623), as StreamVLN, which saves only at the end of its epoch.
   Updates 156, 312 and 468 are kept for diagnosis only (e.g. an early copying check or a C screen);
   they never replace the final checkpoint.

Open (recommendations are mine):

5. **DAgger round.** Size (StreamVLN iterates every training episode; group A has 21,330 usable), number of
   rounds, β, and label source (`executed`, StreamVLN; or `expert`, classic DAgger relabelling with the
   executed actions kept in the prompt). Recommendation: one round over all group-A episodes with the
   StreamVLN rule and `executed` labels.
6. **STOP.** One label per episode (about 1–2% of actions); trial_0 never learned it. The trainer logs
   per-class and per-turn-offset accuracy; decide on STOP weighting or a C-calibrated STOP bias after
   stage one.
7. **Memory view size.** 192 tokens is applied (`[adopted]`, StreamVLN encodes memory frames like current
   views); 256×192 = 48 tokens would cut a window from 3.5k to 2.0k tokens.
8. **Co-training data.** Not prepared; decide whether the general-skill preservation matters here.
9. **World model and scorer (steps 4–10).** They assumed one decision per step. With 4-action turns the
   scorer can act at turn starts or per step on the per-action logits the server returns.
10. **Checkpoint storage.** A `full` checkpoint is 9.1 GB (bf16); the newest also holds a 54 GB resume
   state (Adam + fp32 master weights), deleted when the pass completes. One pass leaves 4 checkpoints
   (≈ 36 GB). Everything under `runs/` becomes a release asset in `migration/export_repo.py`; decide what
   to keep before the next sync.

## 5. Commands

```bash
cd "[OURS]/baseline-v0"; export HF_HOME=/data/shared/hf_cache
Q=/home/bufsgpu/yes/envs/rolevln-qwen35/bin/python   # policy env;  H = rolevln's python for Habitat tools
# checks (CPU; GPU)
$Q tools/check_stream_layout.py --out logs/check_stream_layout.json
CUDA_VISIBLE_DEVICES=0 $Q tools/check_stream_backbone.py --out logs/check_stream_backbone.json
# step 2, stage one: full fine-tuning, one pass, StreamVLN settings (~17 h on GPU 0, ~74 GiB)
CUDA_VISIBLE_DEVICES=0 $Q tools/train_stream_policy.py --recipe streamvln_stage1 --run-dir runs/trial_1/step2_<name>
#   interrupted? the same command plus --resume runs/trial_1/step2_<name>/checkpoints/<newest step> continues exactly
# development screen on C (Habitat env; one policy server per worker)
python tools/eval_stream_policy.py --run-dir runs/trial_1/step2_<name> --weights runs/trial_1/step2_<name>/checkpoints/step_N \
    --episodes 32 --workers 4 --out runs/trial_1/step2_<name>/eval_C64_step_N
# step 3, one DAgger round on A, then stage two
python tools/collect_stream_dagger.py --run-dir runs/trial_1/step2_<name> --weights .../step_N --episodes 0 --workers 4 \
    --out ../../data/processed/baseline_v0/trial_1/dagger_r1_<name>
CUDA_VISIBLE_DEVICES=0 $Q tools/train_stream_policy.py --recipe streamvln_stage2 --run-dir runs/trial_1/step3_<name> \
    --init .../step_N --train-root <expert_groupA> <dagger_r1_name>
```
