# Step 3 — StreamVLN DAgger and stage two (ready; runs after stage one)

StreamVLN's DAgger stage, with its own settings (`[adopted]` in `manifest/trial_1_settings.yaml`):

| Part | Setting |
| --- | --- |
| Collecting policy | stage-one final checkpoint (`runs/step2_qwen35_4b_streamvln/checkpoints/step_000623`) |
| Round | one, over every validated group-A episode (7,489 R2R + 13,844 RxR), seed 1, 6 workers |
| Rule | `stream_policy/controller.py`: β = 0; expert takeover after a STOP ≥ 3 m from the goal, more than 12 disagreements, or more than 0.8 per reference segment, until the next waypoint; a policy STOP before the final waypoint is replaced; kept: ends < 0.5 m from the goal with path-length ratio < 0.93 after a takeover, < 0.85 without |
| Targets | executed actions, final frame STOP |
| Stage two | `--recipe streamvln_stage2 --init <stage-one final> --train-root <expert_groupA> <round>`: full fine-tuning, one pass over expert + DAgger windows, 3% warmup then cosine to 0; final checkpoint is the result; no co-training data |
| Diagnostics | group-C closed-loop screens (32 R2R + 32 RxR) of both finals, never used to choose anything |

Output paths: the round in `../../data/dagger_r1_qwen35_4b_streamvln` (= `data/processed/baseline_v0/trial_1/`,
Matterport-derived, never exported), stage two in `../../runs/step3_qwen35_4b_streamvln_dagger_r1`.
Expected durations on GPU 0: collection roughly 12–24 h (6 workers, ~11 GiB each), stage two about a day
(its length depends on how many episodes the rule keeps).

To hand the stage to another agent, give it [AGENT_PROMPT.md](AGENT_PROMPT.md): it waits for stage one,
then runs the whole stage, resumes after interruptions and records the outcome here. Check a round before
training on it with `tools/audit_stream_data.py --dagger-root <round>`.

Smoke runs during preparation (a few episodes per dataset) behaved as the rule predicts: episodes reached
the goal through takeovers, and efficient ones (path ratio 0.95, 0.98, 1.0) were not kept.

<!-- dagger-supervisor-run-record -->
## Run record

Completed 2026-10-30T23:40:40.557901+00:00. One round: 21,333 episodes, 11,660 kept, 46,907 DAgger windows. Stage two: 126,721 windows, 990 updates, 24.798 active hours, 0 resumes.

Final checkpoint: `runs/step3_qwen35_4b_streamvln_dagger_r1/checkpoints/step_000990`; SHA-256 `6033ff740ee701705038c8eb5d55150bb1a7d7f8388423d0bb55ee2de306cc4c`. [Full run record](../../runs/step3_qwen35_4b_streamvln_dagger_r1/RESULTS.md).

| Diagnostic C64 metric | Stage-one final | Stage-two final |
| --- | ---: | ---: |
| Macro success | 53.12% | 64.06% |
| Macro SPL | 0.44819 | 0.51814 |
| Mean collisions per episode | 44.828 | 32.719 |
| Total collisions | 2869 | 2094 |
| Mean steps per episode | 129.625 | 157.703 |
| STOP rate | 90.62% | 89.06% |
| Episode endings | {"STOP": 58, "step_limit": 6} | {"STOP": 57, "step_limit": 7} |

The 64 group-C development episodes are diagnostic only; they did not select the checkpoint.
