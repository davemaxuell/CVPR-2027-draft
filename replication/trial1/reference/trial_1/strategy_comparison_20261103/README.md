# Trial 1: four training strategies, E / F / G / J

The current F run is a valid coverage-expansion baseline. Its superiority is unproven. Expert-only updates could improve full-split navigation while weakening behavior learned from corrective trajectories; this is a hypothesis to test, not an observed regression. We retain F and add three distinct experiments, for four approaches in total.

| ID | Starting point and sequence | Main comparison | New collection |
|---|---|---|---|
| F | 990 → one epoch full expert SFT | Current coverage-expansion baseline | None; keep the running job |
| E | 623 → one epoch full experts + original 623-policy DAgger | Expanded stage-two baseline; compare with G to test initialization | None |
| G | 990 → one epoch full experts + original 623-policy DAgger | Compare with F to test replaying existing corrections | None |
| J | F final → frozen-policy collection → audit → one epoch full experts + fresh corrections | Test correction of the fully adapted policy's own errors | One round, frozen F |

E and G are independent runs from the preserved 623 and 990 anchors. G does not initialize from E. J uses F's verified final epoch checkpoint, irrespective of evaluation scores; it does not initialize from G. F is trained and evaluated only once. J uses fresh corrections rather than including the original correction dataset again.

The official StreamVLN implementation separates [collection from subsequent training](https://github.com/InternRobotics/StreamVLN/tree/e48f6ff7e9201d93aae003e8f64b04c00cec13bc#-training). Its [paper](https://arxiv.org/html/2507.05240v2) describes collecting corrective demonstrations after initial training, then adding them to the training mixture. This motivates the correction comparisons; it does not establish which sequence wins with our Qwen3.5 backbone. These are StreamVLN-style navigation-only experiments adapted to Qwen3.5-4B, not exact reproductions of the authors' model or multimodal mixture.

## Fixed settings and data

Qwen3.5-4B, native encoding, full language/vision/projection fine-tuning, BF16, gradient checkpointing, effective batch 128, LR 2e-5 (vision 5e-6), weight decay 0, 3% warmup then cosine decay, established augmentation, 32-step windows, eight historical views, four actions per response, 16,384 token budget. Seed 0 and one epoch per added training pass. Fresh AdamW per stage; retained recovery state is used only to resume that same run. No added losses, curriculum, arbitrary source weights, short-SFT fractions, extra language scope or custom Saved places slot.

The full source scope has 10,819 R2R and 19,996 RxR English-guide episodes across the official training scenes. Existing validity filters leave 10,819 R2R + 19,402 RxR expert trajectories and 109,784 windows. The 594 excluded RxR examples retain their reasons and metadata; the full-data audit and LookStep split comparison are preserved. These runs use English guides (en-US/en-IN), never validation/unseen episodes as training data.

E/G additionally use the original frozen-623 group-A DAgger dataset: 11,660 usable corrective trajectories, 46,907 windows. That original collection has not been expanded or relabeled. Combined data: 41,881 trajectory records, 156,691 windows, 1,224 updates / 156,672 draws, with a 19-window full-batch tail. E and G share exact root order, sampler seed, augmentation, schedule and budget.

F uses 857 updates / 109,696 draws, with an 88-window tail. Because the recipes use one epoch, F versus G changes both the corrective mixture and total update count. E versus G is the direct initialization comparison with matching data and budget. J adds collection and another training stage. Report training/collection time and updates alongside metrics; this is not an equal-compute causal study. One seed per strategy does not measure between-seed variance.

J's frozen collection plan selects all 10,819 R2R + 19,408 RxR episodes satisfying the established expert replay rule. The six RxR episodes too short for expert SFT remain eligible for collection under that unchanged rule. New corrective training still applies the minimum-four-action rule. Collection uses beta 0, seed 1, six workers on GPU 0, unchanged StreamVLN takeovers/keep rule, executed-action labels and STOP handling. Its final retained counts and training update budget are determined by the audited collection; no future count is claimed now.

## Execution and reporting

Keep active F training, verification and full R2R seen/unseen evaluation unchanged. The user's 2026-11-04 extension adds full RxR English-guide val-unseen evaluation: first F, then preserved 990 and 623; then E training → verification → fresh full R2R seen/unseen → RxR unseen; then G likewise; then J collection → audit → training → verification → full R2R seen/unseen → RxR unseen; then the combined comparison. The added evaluations use the existing single supervisor and GPU 0. [Extension protocol and authorization](../rxr_unseen_comparison_20261104/README.md).

All policies use fixed final-epoch checkpoints. Val-unseen is report-only: no selection, hyperparameter tuning or extra rounds in response to results. Every policy has separate evaluation output and checkpoint receipts. Full R2R evaluation is 778 seen + 1,839 unseen episodes per policy. The added RxR comparison evaluates all six policies (623, 990, F, E, G, J) on the identical 3,669 English-guide val-unseen episodes across 11 held-out scenes. It preserves Trial 1's 15-degree turns and 700-step RxR cap; these are matched internal comparisons, not scores under the official 30-degree RxR-CE protocol. The frozen original authorization/source files retain their historical R2R-only scope; the dated extension explicitly supersedes that evaluation exclusion without changing any training method or input binding.

The existing supervisor and lock serialize GPU 0 use. There is no duplicate F training/evaluation, no second supervisor, and no dependency on a downstream final-model receipt. Bounded recovery retains partial failure evidence, checkpoints and all logs. Each launch reserves storage for the actual audited update budget; insufficient space blocks that job without deleting existing data or reducing coverage. Physical corrective data remains under the organized data directory.

Trial 2 training and HM3D remain held; MMC4 downloads continue. The stopped pretrained restart stays held. Original 623/990 checkpoints and their results remain immutable, and no downstream model role is rebound until separately verified/authorized. The Saved places experiment is separate.

[Live suite status](STATUS.md) · [Fixed authorization](authorization.json) · [Existing-mixture audit](audits/existing_mixture.json) · [Frozen J collection plan](collection/plan.json) · [Implementation checks](evidence/implementation_checks.json) · [Source identity](SOURCE_SNAPSHOT.json)
