# Step 10 — fixed unseen-scene comparison

Run: `runs/step10_smolvlm2_refined512_dino_pair_v1/`.
The automatic continuation process starts this stage after Step 9 finishes and freezes both
scorers and their override margins. It cannot use E results to change those choices.

Started on 27 October 2026 at 14:50 KST. The initial attempt saved 184 base-policy episodes,
then stopped on RxR episode 5734: the simulator reported infinite geodesic distance from the
initial pose, and native SPL evaluated `0 * (infinity / infinity)` as NaN. A versioned reporting
recovery resumed at 16:36 KST in `recovery_v2/`; the original protocol and all initial records
remain intact. Every selected episode is retained. Failed episodes with this native NaN SPL
receive zero SPL with the exception recorded; infinite navigation error is represented as
JSON null with an explicit infinite-mean flag and printed as infinity. Other unexpected
nonfinite metrics still fail validation. No model, action, preprocessing, margin, or episode
selection changed. All 184 finite records and aggregates remained exact; the recovered
100-action failing episode also matched the original-logic diagnostic exactly. The inherited
reporting convention is used in the later four-model repeat.

All five systems finished their 224 episodes at 22:00 KST on 27 October. Final reporting then
stopped because an imported-record check reused a variable holding the expected episode-ID
set. The versioned `tools/complete_step10_audit_v3.py` correction retained every check and
passed all 1,120 episode records and 363,790 action events. Results were published at
05:59 KST on 28 October; the automatic NWM/CDiT continuation started immediately afterward.
The original sources, locked protocols, and evaluation records remain intact. The audit
amendment is recorded in `recovery_v2/finalization_v3.json`; publication can be resumed with
`tools/complete_step10_audit_v3.py`.

The [completed report](../runs/step10_smolvlm2_refined512_dino_pair_v1/RESULTS.md) records
21/224 successes (9.375%), SPL 5.779%, and forward collision rate 21.658% for every system.
All four controls reproduced the base policy's actions, poses, observation hashes, and
navigation outcomes exactly, with zero overrides at the C-selected conservative margins.
This run demonstrates no navigation gain from the scorers. Five unreachable-goal episodes
remain in every denominator, so mean navigation error is infinite.

The predeclared population is a scene-balanced 224-episode pilot: 112 R2R and 112 English-guide
RxR episodes across all 11 E buildings, selected with seed 10 and no outcome/expert filtering.
E is the official `val_unseen` split. This pilot is smaller than the complete 5,508-episode
unseen benchmark. The same episode IDs and original action budgets are used by every system.

Five systems run: the selected frozen policy alone, the matched learned-null-future scorer,
the privileged actual-next-view scorer, and the transfer/fresh DINO pair. The three systems
that use future views share the exact pooled scorer weights and margin. The null control has
its separately trained and C-selected weights/margin. The actual-view control uses reversible
one-action simulator branches and is explicitly privileged.

Checkpoint 512, STOP bias −0.25, saved policy preprocessing and batch-one decoding remain fixed.
Each decision executes exactly one primitive action. Memory updates use actual observations
only; the first three observations keep the base action. F/L/R receive one prediction each;
STOP uses the current view. Logged failures/nonfinite scores keep the base action.

Reports include SR, SPL, navigation error, nDTW/SDTW, collisions, actions, overrides/fallbacks,
latency and GPU memory. Confidence intervals use 2,000 paired scene-cluster bootstrap resamples
with seed 10. Eleven scenes limit uncertainty estimates. C branch regret is reported separately
as development evidence; no counterfactual E branch labels are fitted.

Four independent workers use native batch-one inference. Decision latency records contention
under that concurrency. PyTorch allocator high-water marks are exact for the allocator;
`nvidia-smi` samples complete process memory, including CUDA/Habitat allocations, every two
seconds. These samples can miss brief peaks, and H100 measurements do not establish an 8 GB fit.

The final audit checks every episode hash and event, policy decoding, chosen actions, budgets,
warm-up/fallback rules, matched episodes, shared scorer identities, and preserved upstream
controls. CSV and Markdown summaries accompany the full JSON records. RxR retains the agreed
15-degree turns, differing from its official 30-degree protocol.
