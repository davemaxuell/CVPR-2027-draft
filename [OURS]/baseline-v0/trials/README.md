# Experiment trials

Each `trial_N` records one end-to-end experiment: its setup, data, model selections,
training, calibration, evaluation, and limitations. Numbering starts at zero.

| Trial | Policy | World models | Status | Final E result |
| --- | --- | --- | --- | --- |
| [trial_0](trial_0/README.md) | SmolVLM2-500M, refined checkpoint 512, STOP bias −0.25 | DINO-WM PointMaze transfer + matched fresh DINO dynamics | Complete pilot, steps 0–10 | 21/224 successes (9.375%); zero scorer overrides |

[index.json](index.json) is the machine-readable registry. `trial_1` is the next
available name; it has not been started. Model variants within one experiment
are recorded inside that trial, so the DINO transfer/fresh comparison belongs
together in `trial_0`.

## Layout and preservation

Each trial has a README, `trial.json`, ordered `steps/`, `runs/`, `checkpoints/`,
`data/`, configuration in `manifest/`, protocol documents in `docs/`, and final
`results/`. `steps.csv` provides a compact progress table.

For the existing trial, large outputs remain in their original locations and
relative symbolic links provide the organized view. This preserves paths embedded
in checkpoints, source snapshots, scripts, and migration manifests. The trial's
manifest, documents, setup reports, and final results are copied snapshots.
`checksums.sha256` records both snapshots and linked evidence; `verification.json`
records the organization audit. This is a catalog within the workspace, not a
standalone backup: retain the linked `runs/` and `data/processed/` directories when
moving it.

## Starting the next trial

Copy [`_template/`](_template/README.md) to `trial_1`, fill in its identity, scope,
parent trial, hypothesis, protocol, and budgets, and add it to `index.json` and the
table above. Use new output paths for every changed run or processed dataset.
Record any reused frozen artifacts explicitly; preserve completed trial records.
Select settings on C before evaluating on E, and report reuse of E when comparing
later trials. A trial number alone does not authorize or launch training.

The stopped NWM/CDiT and four-model continuation folders remain outside the
completed `trial_0`; their existence is not evidence of another completed trial.
