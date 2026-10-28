# Step 4 — actual transition corpus

Completed on 26 October 2026 (KST). Execution: `runs/step4_smolvlm2_refined512/`;
[results](../runs/step4_smolvlm2_refined512/RESULTS.md),
[final verification](../runs/step4_smolvlm2_refined512/final_verification.json),
[status](../runs/step4_smolvlm2_refined512/status.json).
The selected experimental reference is separately versioned in that run's `reference/` directory.

The 256 new policy episodes produced 88,306 executed actions, 2,848 branch roots and 8,544
actual F/L/R outcomes, covering all 44 available R2R and 42 available RxR group-A scenes.
After deduplicating the executed action at each branch root, this source contributes 93,848
movement transitions. The combined corpus contains 21,973 episodes, 2,488,382 movement
transitions, 2,422,469 matched samples and 65,913 separately indexed early DINO-only samples.
All 2,513,438 stored images were hashed and checked; 113,108 were fully decoded during indexing.
The final audit passed, including 273 paired-family sample checks. The corpus is ready for Step 5.

The user's 26 October 2026 handoff selects `step3b_smolvlm2_stop/train/checkpoints/step_000512`.
Decoding subtracts exactly 0.25 from STOP's first-token logit, then chooses the largest of
STOP/FORWARD/LEFT/RIGHT. The model configuration, tokenizer, image processor, prompt/history
conventions, native vision forward and batch-one decision schedule match the authoritative
confirmation evaluation. `policy.experimental_reference.ExperimentalReference` verifies the
bundle and runtime hashes, freezes all parameters, and applies the decoder correction.

This is a deliberate experimental selection despite the failed original stopping safeguard:
29/224 successes, 149 incorrect stops, and 9.6% forward collision rate on confirmation A.
The old result still records failure. The Step 2 reference in `runs/step3_smolvlm2/frozen_reference`
is preserved as a control. The new [selection decision](../runs/step4_smolvlm2_refined512/selection_decision.json)
records the authority, exception, and identities. Changing the checkpoint, bias, processor,
prompt or action order requires a new policy identity and new dependent data.

The corpus combines these group-A sources:

- The 21,333 validated original expert episodes used by SFT. Every non-STOP action has its real next image.
- All 256 original DAgger episodes, including policy failures, retaining the original collecting policy's provenance.
- All 128 STOP-refinement DAgger episodes, likewise retaining their historical provenance.
- 128 R2R and 128 RxR new visits from the selected reference, selected by seeded scene-balanced sampling (seed 4), unfiltered by success.

At new-policy times 3, 35, 67, …, execute FORWARD, LEFT and RIGHT independently from the actual
root. Each target is a real one-command simulator observation. Branches use the simulator without
advancing task metrics, episode steps, STOP state or policy history. Restore native body/sensor
translations and quaternions, cached collision state and simulator counters. Physics-enabled,
rigid-object and articulated-object scenes are rejected by this static-scene implementation.
The actual trajectory continues with the policy's decision; only that action enters its history.

[Runtime checks](../runs/step4_smolvlm2_refined512/check_runtime.json) compare eight R2R/RxR
replays with and without branching. Actual RGB, poses, metric prefixes and STOP termination match
exactly. Branch targets match the actual next observation when the same command is executed.
The checks cover signed 15-degree turns, blocked forward movement, saved/native preprocessing
equality, the −0.25 decoder adjustment and zero trainable parameters.

New frames are lossless 640×480 RGB PNGs, without resizing or cropping before storage. The policy
receives live uncompressed RGB. Existing expert/DAgger images retain their recorded JPEG quality
95 representation. This encoding difference is explicit provenance; both members of each paired
world-model comparison receive identical indexed examples and images.

`tools/build_wm_corpus.py` hashes every image, verifies format/dimensions, fully decodes every
31st frame plus each episode's last stored image, and builds `catalog.sqlite`. Episode metadata
and source configuration hashes are retained. Movement collisions and zero-displacement forward
commands are retained. STOP has no dynamics target and is excluded. At a branched root the
executed movement is indexed once using its actual trajectory outcome; other movements use the
recorded branch targets. A root where the policy chooses STOP can still supply three movement branches.

`wm.data.TransitionDataset(corpus, history_frames=3)` and `history_frames=4` expose the same
matched root/action IDs at times t≥3, with the last three or four actual observations and their
intervening executed commands. Targets are actual next RGB observations. The loader rechecks
accessed image and metadata hashes, and does not supply future poses or realized displacement
as inputs. `subset="dino_early_extra"` separately exposes t<3 for optional DINO-only experiments;
four-frame loading rejects that subset. Both transfer/fresh trials use the same matched catalog.

An additional [terminal-frame check](../runs/step4_smolvlm2_refined512/check_terminal_frames.json)
fully decoded all 256 new episodes' final trajectory PNGs. Each file hash and decoded pixel hash
matched the observation recorded live in the simulator. The collection environment's package,
Python and GPU versions are saved in [runtime_environment.json](../runs/step4_smolvlm2_refined512/runtime_environment.json).

Run/resume in the `rolevln` environment with `python tools/run_step4.py`. Completed episodes are
atomic, and catalog transactions commit per episode. The supervisor rejects duplicate jobs or
changed pinned sources. GPU work is restricted to physical GPU 0. The corpus references original
frame paths, so the source corpora must be retained.

This step collects actual transitions only. World-model fitting, group-B scorer labels and
group-E evaluation are outside this run.

One RxR episode (4838) exposed an unavailable geodesic goal diagnostic during collection.
The strict writer initially rejected its infinite value after 175 completed episodes. A
[metadata-only correction](../runs/step4_smolvlm2_refined512/nonfinite_diagnostics_fix.json)
stores unavailable distance/SPL values as `null` with explicit `inf`/`nan` tags. These values
are never policy or dynamics inputs. Completed episodes keep their original configuration hashes;
the archived and corrected collector configurations are both retained and checked for identical
policy, simulation, selection, and branch settings.
The [recovery verification](../runs/step4_smolvlm2_refined512/check_serialization_recovery.json)
confirmed all 175 completed records were unchanged and all 767 images from the failed episode
were reproduced byte for byte. The eight simulator replay checks also passed again.
