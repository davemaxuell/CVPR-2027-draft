# Trial 1: six-policy RxR val-unseen comparison

Authorized by the user on 4 November 2026: “can you also test it on RXR unseen? (do this for all models too). we want to compare it corerctly.”

Evaluate the preserved 623 and 990 policies and the final F/E/G/J policies on the same complete RxR-CE English-guide val-unseen split. There are **3,669 episodes in 11 held-out scenes**: 1,223 en-US and 2,446 en-IN guide instructions. No evaluation episode is excluded. The source release also contains Hindi and Telugu guides; those languages and follower instructions remain outside the established Trial 1 scope. Split-local numeric episode IDs may recur in train and validation; the audit confirms no training-scene overlap and identifies records by dataset/split/episode ID.

The source annotations, guide/follower ground-truth maps used by upstream nDTW, all scene GLBs and navmeshes were audited and hashed. Selected guide IDs do not collide with follower ground-truth IDs. The real Habitat loader independently selects the exact audited episode set. A CPU test exercises the verbatim upstream nDTW class with a reference path and obtains 1.0 without creating a GPU simulator.

All six models use the identical existing Qwen3.5-4B native encoding, 32-step windows, eight historical views and four-action responses. Evaluation is greedy, uses RGB 640×480 / HFOV 79, 0.25 m forward steps, 15-degree turns, a 700-step limit, success distance 3 m, STOP bias 0, and no custom Saved places slot. Six workers share GPU 0. **The established 15-degree / 700-step RxR protocol differs from the official 30-degree protocol. These are matched internal comparisons, not official-protocol benchmark scores.** The currently available 623/990/F native configurations are identical; future models must match the same canonical configuration hash.

The execution order is:

1. Preserve and finish the active F full R2R evaluation and its report.
2. Full RxR val-unseen for F, then 990, then 623.
3. E training, checkpoint verification, existing full R2R seen/unseen evaluation, then full RxR unseen.
4. G follows the same order.
5. J's authorized frozen-F collection, audit, training, checkpoint verification, existing R2R evaluations, then full RxR unseen.
6. Verify and compare all six RxR results together with the available full R2R results for 990/F/E/G/J. Full R2R 623 results are not fabricated from its small diagnostic screen.

Training order, initializers, datasets, sampling, epoch budgets and the one fresh J collection remain unchanged. Unseen scores are report-only and never select checkpoints, recipes, extra rounds or model promotion. Future E/G/J checkpoint bindings are created only after real completion and verification receipts exist. No model's cached episodes are reused for another model.

The shared supervisor continues to own the existing `trial1_strategy_suite` job. Its completion gate now points to this directory's `completion.json`. `run.py` reuses the frozen original suite's training, collection, checkpoint verification and R2R functions, plus the verified child-identity repair. It adds RxR stages and a combined report; original source snapshots and training identities remain intact. The added evaluator is a separate snapshot: only full-split guards and attribution fields changed; simulator actions, model inference and metric implementations are retained. The exact evaluator diff is in `evaluation/CHANGES.diff`.

Stage state and logs remain under `../strategy_comparison_20261103/stages` and `logs`. This directory and the original suite directory both receive live progress, status and final comparison records. Evaluation outputs are separate per policy under `../../../runs/trial_1/rxr_unseen_comparison_20261104/`. Partial episodes resume atomically, live children are adopted, and the existing bounded retry rules remain in effect. MMC4 downloads, all Trial 2/unrelated holds, the stopped fresh-start hold, and original 623/990 checkpoints are preserved.

The comparison will report SR, SPL, nDTW, sDTW and the other measured navigation metrics, plus changes relative to 990. Full-coverage checks validate checkpoint hashes, loaded-worker identities, native encoding, protocol/episode identity, instructions, scene membership, action units, STOP handling, duplicate/missing episodes and metric aggregation. A missing result prevents completion. All training counts, updates, durations, recovery attempts and peak memory remain part of the final report. This remains a one-seed experiment with unequal update counts across some strategies; it cannot establish statistical significance across seeds.

[Live status](STATUS.md) · [Authorization](authorization.json) · [Protocol](protocol.json) · [Split audit](split_audit.json) · [Checks](checks.json) · [Source snapshot](SOURCE_SNAPSHOT.json)
