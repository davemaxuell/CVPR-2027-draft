# Trial 1 replication handoff

This package captures the Trial 1 training implementation from the H100 server on 2026-11-04. It contains the group-A SFT/DAgger code, full-split continuation code, R2R/RxR evaluators, environment records, and an agent prompt for an independent **Qwen3.5-2B** reproduction.

The completed 4B sequence was **pretrained → group-A SFT → fresh group-A DAgger → stage-two SFT → F (one additional full-training expert pass)**. E/G/J are optional comparison branches, described in the prompt. They are not alternative names for the original stages.

## Start on the receiving machine

```bash
git clone --depth 1 --branch trial1-2b-replication-20261104 \
  https://github.com/davemaxuell/CVPR-2027-draft.git
cd CVPR-2027-draft/replication/trial1
python3 -B verify_package.py
```

Read [ENVIRONMENT.md](ENVIRONMENT.md) and [DATA.md](DATA.md), then give the other agent this instruction:

> Read `replication/trial1/AGENT_PROMPT_2B.md` in this checkout and execute that independent reproduction. Use this package's source snapshots rather than the older code elsewhere in the repository. Discover the local data, Python, cache and GPU paths, qualify the 2B model, and follow the staged workflow. Do not execute the historical 4B orchestration scripts directly.

The package is a **source and protocol handoff**, not a pretrained 2B policy. Target-machine GPU compatibility, 2B model loading, exact resume, and Habitat integration must pass before a real training launch. No new GPU training was performed to produce this export.

## Contents

| Path | Purpose |
|---|---|
| `AGENT_PROMPT_2B.md` | Complete receiving-agent task and fixed experiment |
| `prepare_sources.py` | Create isolated model/path-bound working copies; never launches training |
| `source/group_a/` | Original trainer, group-A collector/auditor, renderer, native policy and simulator |
| `source/full/` | Full official-training collector/auditor/renderer and the same trainer/native policy |
| `evaluation/r2r/`, `evaluation/rxr/` | Separate verified full-split evaluator implementations |
| `evaluation/vendor/vlnce/` | Pinned upstream metric source and its license |
| `reference/` | Historical recipes, guarded F/E/G/J orchestration and recovery repair; read as reference |
| `reference_runs/S1,S2,F/` | Actual 4B run arguments, counts, stream configs and completion metadata |
| `env/` | Setup recipes and recorded policy/Habitat environments |
| `DATA_IDENTITIES.json` | Hashes of source render indexes/manifests; contains no rendered images |
| `SOURCE_ORIGINS.json` | Exact source-file origins and hashes |
| `FILE_MANIFEST.json` | Integrity manifest for this package |
| `VALIDATION.json` | Export-time CPU checks and explicit untested target-machine checks |

The historical source snapshots intentionally retain original machine paths and 4B defaults. `prepare_sources.py` produces new working copies with explicit 2B/model revision, data and Python bindings. It does not edit the historical snapshots. The receiving agent still needs to instantiate new guarded stage records/receipts and qualify the resulting implementation.

## Boundaries that matter

- DAgger corrections must be collected with the **new 2B S1 policy**. Original 4B retained counts and its `990` update count are not fixed targets.
- F starts from the new S2 final, uses **full expert data only**, has a fresh optimizer, and collects no new DAgger data.
- F's validation uses official val-seen renders. Original group C is inside the expanded training scope.
- The group-A and full-split audit/collection scripts are different; do not interchange them.
- `tools/check_stream_layout.py` checks both stream layout and the StreamVLN controller. The legacy `tools/check_dagger.py` tests the older SmolVLM pipeline and is deliberately not part of this package's qualification sequence.
- Historical authorizations, source hashes, checkpoint hashes and status files describe 4B runs. They never authorize or prove completion of a new 2B run.
- The full evaluators bind their Python interpreter in `common.py`; setting only `ROLEVLN_QWEN_PYTHON` does not update that historical constant. The preparation script updates it in the new copies.
- Unseen evaluations are report-only. RxR uses the matched internal 15-degree/700-step protocol, not the official 30-degree protocol.

Datasets, Matterport scenes/renders, model weights, optimizer states, credentials, live queue state and training logs are excluded. Retrieve authorized data separately or regenerate it with the included rendering code. The older repository outside this directory contains other experiments; it is not the entry point for this handoff.
