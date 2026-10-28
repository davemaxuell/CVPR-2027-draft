# trial_N — experiment title

Status: planned. Parent trial: record an existing trial ID or `null`.

Describe the hypothesis, what changes from the parent, and the intended comparison.
Fill in `trial.json` before starting. Use `trial_1` for the next experiment; this
template is not an active run.

## Protocol

Record policy and world-model identities, scorer variants, datasets and scene
partitions, action settings, training budgets, seeds, checkpoint-selection rules,
and the frozen evaluation protocol. Disclose any reused E evaluation data.

## Artifacts and completion

Create `steps/`, `runs/`, `checkpoints/`, `data/`, `manifest/`, `docs/`, and `results/`
as needed. Keep an ordered `steps.csv`; Step 0 covers setup and steps 1–10 follow
the training specification. Record optional substeps explicitly.

Give each new run and dataset a unique output path. Link reused frozen artifacts
and record the source trial and hashes. Do not overwrite a completed trial's files.
Save the final metrics, audit, and limitations when the experiment finishes, then
update the parent trial registry.
