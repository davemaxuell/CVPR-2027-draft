# SmolVLM2 pretrained baseline: same 64 group-C episodes

Completed: 2026-10-24T02:17:45.277598+00:00

The original `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` was evaluated with **no navigation LoRA adapter**. The comparison uses the exact same 32 R2R-CE + 32 RxR-CE episodes, instructions, eight buildings, starting positions and simulator settings as the selected trained checkpoint (step 3,072).

| Dataset | Pretrained baseline successes | Trained step 3,072 successes | Pretrained SPL | Trained SPL |
| --- | ---: | ---: | ---: | ---: |
| R2R | 0/32 (0.000%) | 3/32 (9.375%) | 0.0000 | 0.0701 |
| RXR | 0/32 (0.000%) | 1/32 (3.125%) | 0.0000 | 0.0244 |
| **Overall** | **0/64 (0.00%)** | **4/64 (6.25%)** | **0.0000** | **0.0472** |

The trained checkpoint scores 6.25 percentage points higher on this development subset.

## Baseline behavior

Reached the goal region at any point: 0/64. Issued STOP: 0/64. Ended at the action limit: 64/64.

The baseline only turned LEFT or RIGHT; it never moved FORWARD or issued STOP under this prompt and four-action decoding setup.

Action counts: STOP 0, FORWARD 0, LEFT 12,700, RIGHT 25,700.

## Comparison protocol

- Frozen pretrained revision: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Same native image preprocessing, full instruction, up to seven earlier views, and sixteen recent actions.
- Same deterministic argmax over the first token of STOP / FORWARD / LEFT / RIGHT; one observation history per model forward pass.
- Same live RGB simulator observations, 0.25 m forward action, 15-degree turns, and limits of 500 R2R / 700 RxR actions.
- Success requires STOP within 3 m of the goal. No expert actions, goal-based stopping overrides, or weight updates were used.
- Four independent workers on GPU 0; evaluation elapsed 21.0 minutes.

This is a small group-C development comparison. The trained checkpoint was selected using this subset; group E remains untouched. RxR uses 15-degree turns, which differs from its official 30-degree protocol.

## Saved evidence

- `evaluation/summary.json`: baseline aggregate results.
- `evaluation/episodes/`: all 64 complete trajectories and per-episode metrics.
- `comparison.json`: both results, paired successes, action counts and consistency checks.
- `episode_comparison.csv`: aligned per-episode comparison.
- `source/`: isolated evaluation code; the original training source and checkpoints were not modified.
