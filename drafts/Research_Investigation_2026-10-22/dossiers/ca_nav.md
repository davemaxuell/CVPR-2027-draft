# CA-Nav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Constraint-Aware Zero-Shot Vision-Language Navigation in Continuous Environments**. Kehan Chen, Dong An, Yan Huang, Rongtao Xu, Yifei Su, Yonggen Ling, Ian Reid, Liang Wang arXiv v4; final journal status not independently verified (2026).

Primary record: https://arxiv.org/abs/2412.10137v4

The PDF's 'Journal of LaTeX Class Files, August 2021' masthead is a template artifact, not a publication record. First preprint date is2024; the reviewed version is v4. Do not infer venue/year from the masthead.

Supplied main file: [CA-Nav.pdf](../../Paper/CA-Nav.pdf). Pages actually reviewed: 1, 3-8. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Training-free sequential instruction navigation in continuous space with narrow 79° RGB -D view and odometry. Outputs use 0.25 m forward and 30° turns. The task is not merely category object search, since temporal and directional constraints must hold in order.

## Method, training, inference, and equations

PAPER METHOD (§III): an initial LLM call decomposes instructions into object/location/direction constraints queued by subinstruction. GroundingDINO checks objects within 5 m, BLIP2 VQA checks locations, and odometry over five steps checks direction. All current constraints must be satisfied before advancing, except minimum 10/maximum 25-step switching heuristics. BLIP2 image-text similarity supplies a scalar value projected across the visible frustum of a semantic occupancy map. Cosine-squared angular confidence weights temporal fusion (Eqs.2–4). On subinstruction switches, previous map scores decay byγ=0.5 rather than resetting; a trajectory maskλ^visit_count withλ=.95 discourages revisits. SLIC clusters the value map; select the highest-mean superpixel centroid, then execute an FMM path. For the final target, RepViT -SAM projects an object mask centroid. No navigation optimizer/loss is introduced; foundation models are pretrained.

## Novelty and closest predecessors

Extends VLFM -style value mapping to an explicitly ordered constraint queue and superpixel rather than frontier selection. Relative to InstructNav , reduces repeated LLM calls and uses egocentric sensing. Object/location/direction ablations and map-update ablations support component utility; this is already explicit instruction constraint tracking.

## Results and independent verdict

PAPER EXPERIMENT: reviewed v 4 Table I p.7 reports R2R val-unseen 25.3 SR/10.8 SPL ; Table II p.8 gives English RxR 19.0/6.0. Abstract improvements of 12%/13% are relative to A 2Nav 22.6/16.8 SR, not +12/+13 percentage points. Table I separates panoramic InstructNav from its egocentric adaptation. Stronger success does not imply efficient path fidelity: RxR nDTW 13.5 and SPL 6.0 are important limitations. The queue's timeout bypass means 'constraint-aware' is not a hard correctness guarantee.

## Implementation audit

INSPECTED CODE. common/constraints.py:34–59 tests room membership with BLIP VQA yes/no ranking and object presence through Grounded-SAM class names;61–93 tests direction from pose displacement, accepts ambiguous language without a directional check and requires at least0.5 m displacement for recognized directions. ZS_Evaluator_mp.py:551–587 removes satisfied constraints and advances on either full satisfaction or a maximum-step timeout, after enforcing the minimum-step bound. Timeout advancement therefore is not evidence that the subinstruction was completed.

map/value_map.py:34–66,140–175 stores two map channels (confidence and BLIP2 image/text value), uses angular cosine-squared confidence and projects image/text matching scores onto floor cells. map/history_map.py:21–36 multiplies visited trajectory cells by 0.95 each update. RGB-D preprocessing is explicit in ZS_Evaluator_mp.py:152–157; model initialization uses Grounded-SAM and semantic mapping at 137–150; metrics at 115–133 use a 3 m threshold and distance-weighted SPL. No task-specific training loop is required for this proposed zero-shot policy. Dependencies include frozen perception models, poses, depth, parsed language and Habitat; source inspection does not independently reproduce their outputs.

Repository: https://github.com/Chenkehan21/CA-Nav-code. Inspected snapshot commit: `8430359ff930880ccc74131d4c31d74a598a9b6e`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [vlnce_baselines/common/constraints.py:1-113](https://github.com/Chenkehan21/CA-Nav-code/blob/8430359ff930880ccc74131d4c31d74a598a9b6e/vlnce_baselines/common/constraints.py)
- [vlnce_baselines/ZS_Evaluator_mp.py:115-157,322-343,551-617](https://github.com/Chenkehan21/CA-Nav-code/blob/8430359ff930880ccc74131d4c31d74a598a9b6e/vlnce_baselines/ZS_Evaluator_mp.py)
- [vlnce_baselines/map/value_map.py:34-66,140-175](https://github.com/Chenkehan21/CA-Nav-code/blob/8430359ff930880ccc74131d4c31d74a598a9b6e/vlnce_baselines/map/value_map.py)
- [vlnce_baselines/map/history_map.py:1-41](https://github.com/Chenkehan21/CA-Nav-code/blob/8430359ff930880ccc74131d4c31d74a598a9b6e/vlnce_baselines/map/history_map.py)

## Bottlenecks and follow-up chain

- **03-ca_nav-commit: Constraint checks can prematurely commit to the wrong instruction stage or timeout past an unmet requirement** Conditions: Repeated landmarks, detector/ VQA errors, long within-room subtasks Cause: Binary local checks and finite timeouts replace uncertainty-aware satisfaction evidence; map values are not explicit object relations Evidence: §III-B p.4 switching rules; Tables I–II low SPL ; deduction about false-positive checks Type: methodological. Follow-ups:  GC-VLN introduces explicit graph constraints and backtracking; Context-Nav verifies viewpoint-dependent object relations for instance goals.. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

The full v4 method and principal tables plus targeted constraint/value-map/evaluation code were reviewed. Remaining qualitative appendix examples were not exhaustively reviewed. Final journal acceptance was not verified. No benchmark was executed.

- https://proceedings.mlr.press/v305/yin25a.html
- https://openaccess.thecvf.com/content/CVPR2026/html/Jang_Context-Nav_Context-Driven_Exploration_and_Viewpoint-Aware_3D_Spatial_Reasoning_for_Instance_CVPR_2026_paper.html
