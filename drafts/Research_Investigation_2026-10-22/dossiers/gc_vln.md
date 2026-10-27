# GC-VLN

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation**. Hang Yin, Haoyu Wei, Xiuwei Xu, Wenxuan Guo, Jie Zhou, Jiwen Lu CoRL 2025 (2025).

Primary record: https://proceedings.mlr.press/v305/yin25a.html



Supplied main file: [GC-VLN.pdf](../../Paper/GC-VLN.pdf). Pages actually reviewed: 1, 3-8. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Interpret route instructions as executable geometric constraints in RGB -D continuous environments instead of selecting actions from unconstrained language prose. Requires pose estimates, object perception, BEV mapping, and an FMM controller.

## Method, training, inference, and equations

PAPER METHOD (§3): DeepSeek-R1 decomposes instructions into a DAG of movement stages, waypoints and object nodes, with one displacement per stage and each mentioned object assigned to one stage. Query a six-type constraint library for direction/distance and multi-object relationships. Topological sorting determines node-coordinate solve order. Grounded- SAM2 detects object candidates; waypoint coordinates maximize summed constraint satisfaction subject to nonnegative component constraints and separation from already generated solutions (Eqs.5–7). A navigation tree enumerates alternative assignments. If no feasible continuation exists, physically backtrack to an unexplored sibling; stop at the last graph stage. No learned navigation loss. DEDUCTION: feasibility is conditional on correct parsing and detections; graph claims to preserve all language are requirements, not proved guarantees.

## Novelty and closest predecessors

Explicit symbolic graph solving and branch backtracking go beyond CA-Nav 's sequential object/location checks. This directly occupies 'instruction-as-constraints plus backtracking' novelty. Library coverage is engineered rather than universal natural-language expressiveness.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p.6 gives R2R 33.6 SR/16.3 SPL , RxR 33.8/13.8. R2R SR exceeds InstructNav 31 but SPL is lower than 24; do not claim all-metric superiority. Table 2 p.7 removing backtracking gives 21.3 SR, random solver 7.0, relaxed constraints 21.5. Object-count ambiguity and expensive backtracking remain. Real-world figures demonstrate cases, not a large controlled success-rate experiment.

## Implementation audit

INSPECTED CODE. src/solver/constraint.py implements the four families of geometric masks. Direction tolerances differ between RxR and R2R (π/4 versus π/6), so the geometric prior is dataset-configured. agent_policy.py:115–143 receives simulator pose/context and the parsed instruction DAG;310–322 calls the navigation-tree constraint process. navigation_tree.py:76–121 topologically orders the instruction DAG and groups object constraints by stage. region_solver.py:72–85 filters detected instances by range,141–170 additionally uses detection count, and 199–215 converts semantic relations into geometric constraints. Its failure branches at 777–790 and 905–916 invoke backtracking, then bounded exploration/final fallback. These are finite heuristics over estimated instances; they do not certify instruction satisfaction.

navigation_planner.py:124–234 constructs an FMM distance field, follows its gradient and handles incomplete paths, including a geodesic-to-straight-distance threshold of 3. No navigation gradient training is introduced. The inspected path covers constraint representation, instance selection and local motion, while exact upstream detector outputs, instruction parsing reproducibility and complete benchmark aggregation were not independently reproduced. The printed directional equation was not treated as a proven error because coordinate/typographic conventions require visual disambiguation. No project code was executed.

Repository: https://github.com/bagh2178/GC-VLN. Inspected snapshot commit: `1beddca8d90d79855c110c5a433f7b8856ca263a`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [src/solver/constraint.py](https://github.com/bagh2178/GC-VLN/blob/1beddca8d90d79855c110c5a433f7b8856ca263a/src/solver/constraint.py)
- [src/agent/agent_policy.py:115-143,310-322](https://github.com/bagh2178/GC-VLN/blob/1beddca8d90d79855c110c5a433f7b8856ca263a/src/agent/agent_policy.py)
- [src/solver/navigation_tree.py:76-121](https://github.com/bagh2178/GC-VLN/blob/1beddca8d90d79855c110c5a433f7b8856ca263a/src/solver/navigation_tree.py)
- [src/solver/region_solver.py:72-85,141-170,199-215,777-790,905-916](https://github.com/bagh2178/GC-VLN/blob/1beddca8d90d79855c110c5a433f7b8856ca263a/src/solver/region_solver.py)
- [src/agent/navigation_planner.py:124-234](https://github.com/bagh2178/GC-VLN/blob/1beddca8d90d79855c110c5a433f7b8856ca263a/src/agent/navigation_planner.py)

## Bottlenecks and follow-up chain

- **03-gc_vln-hardconstraints: A misparsed or falsely grounded constraint can make the correct route infeasible** Conditions: Repeated objects, missing detections, ambiguous temporal language Cause: Hard constraint conjunction and single-stage object assignment do not represent epistemic alternatives in parsing Evidence: §3.2–3.3; Table 2; inspected constraint masks Type: methodological. Follow-ups:  PSG-Nav maintains categorical uncertainty; Context-Nav checks shared viewpoints; neither alone proves route-language completeness. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

Main equations and results read; printed angular Eq.2 appears algebraically different from conventional angle tolerance and needs independent visual equation verification before labeling a mathematical error. No benchmark executed.

