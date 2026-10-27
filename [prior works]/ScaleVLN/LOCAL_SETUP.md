# Local demonstration and training

Run these commands from this ScaleVLN checkout. `run.sh` also works from another
working directory. It selects `/home/bufsgpu/yes/envs/scalevln/bin/python`, loads
the built MatterSim extension, and uses the cached BERT tokenizer/config offline.
An inherited Python virtualenv no longer takes precedence over this environment.

## Demonstration

Run a fresh demo and open the navigation visualizer automatically:

```bash
bash run.sh demo --model h14
bash run.sh demo --model b16 --episodes 5
```

After inference finishes, the new visualizer opens in your default browser and
starts replaying the model's route, with a top-down map and first-person panorama.
It uses the predictions from this demo run. Play/pause, step controls, and episode
selection stay available. The server remains running until Ctrl-C; inference has
already exited and released its GPU memory. The demo does not record per-decision
model scores; see [the visualizer README](analysis/vln_viz/README.md) to generate step logs.

On a remote/headless server, open the printed URL through an SSH tunnel instead:

```bash
# On the server (choose a free fixed port for forwarding)
bash run.sh demo --model h14 --no-browser --port 8766
# On your laptop
ssh -L 8766:127.0.0.1:8766 <server>
```

Open the printed URL on your laptop, including its `#...autoplay=1` suffix to
start playback automatically. Without `--port`, the demo chooses an available
port. Use `--export-only` to generate the portable offline player and exit without
starting a server.

The prepared H/14 demonstration is [runs/demo_h14/index.html](runs/demo_h14/index.html).
Open it in a browser. Play, pause, step through the route, select another episode,
and switch between scene views. The graph shows reference and predicted paths.
A portable copy is `runs/ScaleVLN-demo-h14.zip`.
The entire `demo_h14` folder can be moved together; its photos are local assets.

This is an offline replay of **fresh checkpoint inference**, not a live inference
server. `demo.json` and `predictions.json` contain the results. Three evenly spaced
instructions from R2R `val_unseen` were used without filtering for success. These
episodes demonstrate behavior; their metrics are not a benchmark estimate.
Photos show the visited viewpoints with an independently selectable view direction.

Generate an offline demo or run the complete released-checkpoint evaluation:

```bash
bash run.sh demo --model h14 --export-only
bash run.sh eval --model h14
bash run.sh eval --model b16
```

The previous full evaluations remain in `VLN-DUET/datasets/R2R/exprs_map/eval`.
Fresh evaluations use separate output directories under `runs/`.

## Training

These start full training jobs. Every launch records its exact command, dataset
root, configuration, interpreter, GPU selection, exit status, and console output
in a new directory under `runs/`. Set `--output /absolute/path` to choose one;
populated directories are rejected to protect earlier results.

```bash
# R2R fine-tuning, initialized from the released pretraining weights
bash run.sh finetune --model b16
bash run.sh finetune --model h14

# R2R pretraining, initialized from LXMERT
bash run.sh pretrain --model b16
bash run.sh pretrain --model h14

# REVERIE: HM3D augmentation and the released object features
bash run.sh finetune --task reverie
bash run.sh pretrain --task reverie
```

Navigation uses one GPU. Pretraining defaults to one GPU with batch 8 and gradient
accumulation: 32 for R2R (effective batch 256), 16 for REVERIE (effective batch 128).
This preserves the intended effective batch while fitting the available GPU.
These are practical local defaults; single-GPU accumulation does not establish
exact reproduction of the published multi-GPU training results. Training uses
FP32 by default; pretraining `--fp16` is optional. Image features load on demand,
and pretraining shares navigation graph calculations within a process.

Use `--gpu GPU-...` or `--gpu MIG-...` for an explicit CUDA device, otherwise the
current `CUDA_VISIBLE_DEVICES` is inherited. The current shell exposes one CUDA
device. On a setup exposing two compatible GPUs, distributed pretraining uses:

```bash
bash run.sh pretrain --model b16 --gpu 0,1 --num-gpus 2
```

`--batch-size`, `--accumulation` (pretraining), `--steps`, and `--log-every`
(fine-tuning) customize the run. R2R fine-tuning steps and logging intervals must
be multiples of 10 for its 1:9 original/augmented mixture; REVERIE uses multiples
of 2. Pretraining `--steps` is the total optimizer-update target. Fine-tuning
`--steps` is the number of additional iterations.

## Short checks and checkpoint continuation

Append `--smoke` to either training command for actual GPU forward/backward
updates, validation, and checkpoint saves using a separate tiny dataset view.
Original annotations and released checkpoints remain intact. R2R checks include
MP3D, HM3D, Gibson and PREVALENT trajectories. REVERIE checks include object
grounding. REVERIE pretraining smoke checks use 12 updates to exercise all three
training tasks; the other checks use two. Tiny checks establish execution, not
final training quality.

```bash
bash run.sh pretrain --model b16 --smoke
bash run.sh finetune --model h14 --smoke
bash run.sh pretrain --task reverie --smoke
bash run.sh finetune --task reverie --smoke
```

Pretraining saves `ckpts/model_step_N.pt` and `ckpts/train_state_N.pt`. The latter
contains the optimizer, update number and AMP scaler when used. Resume requires
both files. It continues the optimizer and learning-rate schedule; dataset
iteration restarts, so it is not an exact replay of the previous minibatch order.
Fine-tuning checkpoints include both model and optimizer states.

```bash
# Continue toward 200000 total pretraining updates
bash run.sh pretrain --model b16 --resume runs/MY_RUN/ckpts/model_step_10000.pt

# Run another 200000 fine-tuning iterations with the saved optimizer
bash run.sh finetune --model b16 --resume runs/MY_RUN/ckpts/latest_dict

# Initialize fine-tuning from your own pretrained model
bash run.sh finetune --model b16 --checkpoint runs/MY_PRETRAIN/ckpts/model_step_200000.pt
```

Fine-tuning can be evaluated with `bash run.sh eval --model b16 --checkpoint PATH`.
REVERIE evaluation requires a fine-tuned checkpoint, since the release here only
contains its pretraining weights: `bash run.sh eval --task reverie --checkpoint PATH`.

## Data scope and verification

R2R uses the released MP3D/HM3D/Gibson annotations, features and connectivity.
H/14 fine-tuning also uses all three EnvEdit feature variants.

**REVERIE pretraining uses HM3D only.** The local released object-feature LMDB
contains 112552 viewpoints in 800 HM3D scenes, and no Gibson object features.
All 519342 HM3D pretraining annotations have their published positive viewpoints
in this database. The upstream config referenced an unavailable merged object
database and included Gibson trajectories despite their missing object features.
The local config points to the released HM3D database and excludes those Gibson
trajectories explicitly. REVERIE fine-tuning uses the released HM3D augmentation.
This setup does not claim HM3D+Gibson REVERIE pretraining reproduction.

```bash
bash run.sh check --output runs/readiness-new.json
source env.sh
python tools/audit_data.py
```

`check` verifies runtime imports, CUDA, cached tokenizer/config, feature samples
and released weights. `audit_data.py` checks every configured pretraining and
R2R augmentation trajectory against image, graph and candidate keys, plus REVERIE
target-object coverage, and saves `runs/data_audit.json`.

Completed local checks are recorded in `runs/verification.json`; detailed logs
and checkpoints are in its referenced run directories. Full 200000-update jobs
are prepared for launch; they are not started automatically during setup.

## Environment record

Python is 3.9.23, PyTorch is 2.5.1 with CUDA 12.1, and MatterSim is built in
`Matterport3DSimulator/build`. `requirements-runtime.txt` records the installed
runtime versions used for verification. It complements the upstream README's
simulator build instructions. No image extraction or additional dataset download
is required for the feature-based agents or the prepared photo demonstration.
