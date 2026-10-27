# Context-Nav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for InstanceNavigation**. Won Shik Jang, Ue-Hwan Kim CVPR 2026 (2026).

Primary record: https://openaccess.thecvf.com/content/CVPR2026/html/Jang_Context-Nav_Context-Driven_Exploration_and_Viewpoint-Aware_3D_Spatial_Reasoning_for_Instance_CVPR_2026_paper.html



Supplied main file: [Context-Nav.pdf](../../Paper/Context-Nav.pdf). Pages actually reviewed: 1-7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Find one text-described object instance among same-category distractors using RGB -D and odometry. HM3D InstanceNav / CoIN -Bench are instance-navigation tasks, not R2R route following. 'Training-free' means no TGIN -specific training; pretrained perception and PointNav remain.

## Method, training, inference, and equations

PAPER METHOD (§3): parse/canonicalize target/context categories; open-vocabulary detect, segment, then confirm COCO categories with a detector and open-set categories by VLM . Fuse detections into per-instance clouds through center proximity or voxel intersection/min-set-size overlap. RANSAC vertical walls form room components. GOAL- CLIP long-caption dense similarity is projected to a value map to rank frontiers; a one-time same-room exploration override seeks missing context. Depth-only pretrained PointNav executes motion. Candidate intrinsic attributes are queried with paraphrases and Yes/Unknown/No bins; unknowns are retried using the highest-similarity frame from the next five steps. Extrinsic verification samples 24 bearings at four radii around pair/global anchors. For each candidate observer pose, transform object centers into reference-aligned axes and check seven geometric relations. Accept only if ONE common same-room viewpoint satisfies ALL parsed relations (Eqs.2–5). No new navigation loss is trained.

## Novelty and closest predecessors

Combines full-caption exploration with explicit shared-viewpoint relation feasibility, beyond category VLFM and viewpoint-agnostic UniGoal checks. The common-viewpoint quantifier matters: individually satisfiable relations need not be jointly satisfiable. This is direct prior art for multi-relation verification before stopping.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p.6 gives InstanceNav 26.2 SR/9.1 SPL versus PSL 26.0/10.2 and UniGoal 20.2/11.4: highest SR is marginal and does NOT imply best SPL . CoIN unseen 11.3/5.2 versus UniGoal 2.6/2.2. InstanceNav success is 1 m/1,000 steps; CoIN .25 m/500 steps. Table 3 p.7 removes context verification, reducing synonym-split 20.3 to 12.0 SR. Baselines with category-only goals receive less information; separate description-conditioned baselines for fair mechanism comparisons.

## Implementation audit

INSPECTED CODE. vlfm/policy/_spatial_reasoning.py:11–60 evaluates left/right/front/behind/near relations in a candidate observer frame;62–87 restricts neighbors to a radius and inferred room;156–241 searches sampled joint viewpoints;243–323 combines candidate-instance assignments using depth-first search. It is a finite geometric test over a noisy map, not a proof that an object is the intended target. The implementation soft-passes unsupported relations and unavailable height values.

_ynu_defer.py:22–52 buffers masked views and chooses the highest CLIP-scoring image for deferred verification;131–142 blacklists negative attribute decisions. At 181–196 repeated Unknown decisions eventually count as passed after a configurable threshold. This fallback matters for any comparison claiming strict retention of uncertainty. base_objectnav_policy.py:118–121 loads a pretrained PointNav controller,306–310 consumes posed RGB-D, and 392–421 verifies relations around STOP and resumes exploration if necessary. distractor_success.py:35–60 separately checks STOP near a distractor using geodesic distance and a 0.1 m wall offset. No task-specific gradient training is introduced. Model server outputs, map construction quality and full benchmark execution remain unverified here.

Repository: https://github.com/AutoCompSysLab/ContextNav. Inspected snapshot commit: `d7ec95a03c1d8a31aaab8cab9ef460c0172cda81`. Retrieval status: Commit-pinned source retrieved through sparse git checkout. Project code was not executed.

- [vlfm/policy/_spatial_reasoning.py:11-323](https://github.com/AutoCompSysLab/ContextNav/blob/d7ec95a03c1d8a31aaab8cab9ef460c0172cda81/vlfm/policy/_spatial_reasoning.py)
- [vlfm/policy/_ynu_defer.py:15-196,236-309](https://github.com/AutoCompSysLab/ContextNav/blob/d7ec95a03c1d8a31aaab8cab9ef460c0172cda81/vlfm/policy/_ynu_defer.py)
- [vlfm/policy/base_objectnav_policy.py:118-121,170-178,306-310,392-421](https://github.com/AutoCompSysLab/ContextNav/blob/d7ec95a03c1d8a31aaab8cab9ef460c0172cda81/vlfm/policy/base_objectnav_policy.py)
- [vlfm/measurements/distractor_success.py:1-74](https://github.com/AutoCompSysLab/ContextNav/blob/d7ec95a03c1d8a31aaab8cab9ef460c0172cda81/vlfm/measurements/distractor_success.py)

## Bottlenecks and follow-up chain

- **03-context_nav-absence: Failure to observe a contextual object is not evidence it is absent** Conditions: Occlusion, poor viewpoint, incomplete wall/instance map Cause: Hard geometric predicates and finite viewpoint sampling conflate missing observation with infeasibility Evidence: §3.4 p.5 and low absolute success Table 1 Type: methodological. Follow-ups:  PSG-Nav models semantic uncertainty; PolarMem distinguishes uncertain from negative states, but neither validates this exact embodied relation test. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

Main paper reviewed; supplementary failure cases and thresholds not all inspected. No benchmark executed. Predicate feasibility does not prove the narrator's intended viewpoint or true instance identity.

