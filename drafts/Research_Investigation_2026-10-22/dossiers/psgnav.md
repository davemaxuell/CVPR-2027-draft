# PSG-Nav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making**. Rufeng Chen, Yue Chang, Xiaqiang Tang, Hechang Chen, Sihong Xie ICML 2026, identified by the PMLR 306 masthead (2026).

Primary record: https://arxiv.org/abs/2606.01313



Supplied main file: [PSG-Nav.pdf](../../Paper/PSG-Nav.pdf). Pages actually reviewed: 1,3–8,13–15,21. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Continual open-vocabulary ObjectNav under ambiguous detections. Posed RGBD , GLIP/ Grounded-SAM , occupancy,500 steps,1 m visible goal; not R2R route instruction. Adaptive variant uses ground truth terminal success across validation episodes, unlike fixed zero-shot baselines.

## Method, training, inference, and equations

§3 Eqs 1–3 object counts n_t=n_(t−1)+one hot label, normalized categorical belief; group configuration probability product of object marginals assumes independence. Retain top Kg combinations; LLM prunes commonsense conflicts; room level repeats. Three sampled'deterministic worlds' drive pairwise LLM landmark preferences. GVG /frontiers carry spatial unknown coverage/(πr²) and sum proximal object entropy; U_gain=I_spatial+I_sem; prune U<.1. Semantic entropy is not expected information gain because no future observation likelihood/expected posterior is computed. Score=average pairwise win rate across worlds +.5 U(Eqs 4–8), Qwen 2.5-7 B. EEC stores positive/negative goal contexts(visual crop, group semantic, room distribution); similarity=cos visual+w 1 cos group+w 2(1− JSD room), S_final=S_det+max pos−2 max neg; stop if>.61(Eqs 9–10). This score is not probability calibration. Banks update from GT feedback after each validation episode, capacity 10; evict most average similar entry(Eq 11). No gradient training but persistent labeled experience. Independence, repeat-vote correlation and LLM hard pruning can misrepresent uncertainty; unusual layout is not logical impossibility.

## Novelty and closest predecessors

Full categorical semantic memory, world-conditioned pairwise reasoning and positive/negative context verification for ObjectNav . Negative experience and uncertainty-aware exploration must cite it. PolarMem uses negative concept retrieval; PSG-Nav uses goal-instance feedback and spatial context. Its'calibrator'is retrieval score adjustment, not conformal or probability guarantee.

## Results and independent verdict

Table 1 correct dataset order HM3D 66.1 SR/32.1 SPL , MP3D 44.8/17.9, HSSD 67.9/33.4; abstract lists MP3D / HM3D opposite order before these values. Fixed no EEC 63.5 HM3D below ASCENT 65.4; adaptive 66.1 has extra feedback. Table 2 without probabilistic graph 58.4, without groups 58.8; Table 3 K 1→3 63.3→66.1. EEC adds 2.6 SR. Evidence supports useful uncertainty representation but not globally optimal decision or calibrated belief. Episode order/reset/ground truth availability must be reported.

Appendix A.2/Table 5 evaluates a deliberately selected320-episode false-positive subset: EEC intercepts 246 wrong stops, recovers 97 episodes (30.3% SR), and leaves 74 false-positive stops. This supports reducing that failure mode within the selected protocol, not a universal false-positive guarantee. Table 6 shows a small memory bank already helps (capacity 0/5/10/20 gives 63.5/65.2/66.1/66.3 SR). Table 7 changes the detector to WeDetect for OVON, so comparisons there cannot isolate graph reasoning alone. Appendix A.3 defines geometric visibility using72 rays; the semantic entropy term still lacks an observation-conditioned expected posterior. The printed overall algorithm returns Success whenever the threshold triggers STOP, whereas actual success still requires environment evaluation.

## Implementation audit

INSPECTED CODE. graph/scenegraph_combinations.py:60–125 sorts per-object categorical choices and enumerates high-probability combinations with a heap; log probabilities are added, implementing the paper's independence product. The search stops after at most10,000 visited configurations in that path. graph/enumerated_groups.py:252–263 invokes the LLM filter, so retained worlds still depend on semantic-prior judgments. PSG_Nav.py:1484–1489 reads Habitat outcome metrics. Its feedback path at 1497–1554 inserts successful detections into positive memory and, on a failed episode, puts recent detections into the false-positive database.

The latter is a potential credit-assignment problem: episode failure can result from control or stopping rather than false object recognition. The code alone does not show how often this mislabels memory, so an oracle-perception versus observed-feedback experiment is needed. calculate_avg_results.py:7–19 parses saved dictionaries and 41–52,72–82 averages SR/SPL, replacing NaN numeric values by zero. No task-specific gradient training is introduced, but persistent outcome-labeled updates are adaptation. Complete perception preprocessing, all planner branches and released-model outputs were not reproduced. No project code was executed.

Repository: https://github.com/Chenrf1121/psg-nav. Inspected snapshot commit: `181d878ff207845a31e542a19a6592925c1e9bab`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [graph/scenegraph_combinations.py:60-125](https://github.com/Chenrf1121/psg-nav/blob/181d878ff207845a31e542a19a6592925c1e9bab/graph/scenegraph_combinations.py)
- [graph/enumerated_groups.py:60-80,125,252-263](https://github.com/Chenrf1121/psg-nav/blob/181d878ff207845a31e542a19a6592925c1e9bab/graph/enumerated_groups.py)
- [PSG_Nav.py:1484-1554](https://github.com/Chenrf1121/psg-nav/blob/181d878ff207845a31e542a19a6592925c1e9bab/PSG_Nav.py)
- [calculate_avg_results.py:7-19,41-52,72-82](https://github.com/Chenrf1121/psg-nav/blob/181d878ff207845a31e542a19a6592925c1e9bab/calculate_avg_results.py)

## Bottlenecks and follow-up chain

- **PSG-1: Adaptation relies on labeled test outcomes and commonsense filters can reject valid unusual scenes** Conditions: Continual deployment without oracle feedback; atypical room layouts Cause: Success/failure banks require GT; top K hard logical pruning on fallible LLM priors Evidence: §3.4 last paragraph,§4.1 lifelong evaluation; Eq 3 Type: Methodological / protocol. Follow-ups:  EvoNav also oracle feedback; PolarMem does not have GT but perceptual negatives unreliable; calibrated self-supervision remains open. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

Main method, Appendix A technical/memory/false-positive results and the released world-enumeration/outcome-memory/aggregation code were inspected. Exact episode order and full detector outputs were not independently reconstructed. The selected false-positive subset does not measure calibration on all deployment states. No ObjectNav benchmark was run.

## Independent extension: actual retrieval and rejection path

Assignment 10 additionally inspected `utils/rag_manager.py` and `utils/object_detection.py` at the same commit. The released RAG path uses per-channel color histograms (`rag_manager.py:354–392`), a default negative-bank capacity of 20, and an immediate negative cosine veto at 0.95 before checking positives (`277–302`). It does not implement the paper's full visual/group/room descriptor, additive positive-minus-negative score and capacity-10 contract. Negative records are loaded across scene subfolders into category banks (`739–809`), without a scene/track key at retrieval. This limits claims of reproducing the published EEC from that release.

An independently executed, AST-extracted source test used a synthetic RGB crop, explicitly stipulating correct category identity and a control-caused terminal failure. The actual write stored a negative; the actual verifier rejected that crop even with a matching positive record; the actual caller reset the goal and marked 289 cells negative in the supplied synthetic map. File I/O and simulator/model dependencies were stubbed. This is a conditional source counterexample, not a real detection, natural harmful-write rate, or a benchmark result. The crop's real-world identity was stipulated rather than verified by perception. [Exact setup, functions and hashes](../experiments/10/source_check_results.json), [script](../experiments/10/source_checks.py), [full independent interpretation](../evaluations/10_feasibility.md).

A valid next study must hold feature representation, retrieval, veto, memory size and feedback fixed while changing write attribution. It must compare no-write/quarantine and witness-only policies before introducing a learned model. Changes to the representation or active sensing require separate matched controls. A different instance of the correct category is valid in native category ObjectNav; instance-specific claims require a separately specified task.

