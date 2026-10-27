# MapNav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**MapNav: A Novel Memory Representation via Annotated Semantic Maps for VLM-based Vision-and-Language Navigation**. Lingfeng Zhang, Xiaoshuai Hao, Qinwen Xu, Qiang Zhang, Xinyao Zhang, Pengwei Wang, Jing Zhang, Zhongyuan Wang, Shanghang Zhang, Renjing Xu ACL 2025 long papers (2025).

Primary record: https://aclanthology.org/2025.acl-long.638/



Supplied main file: [MapNav.pdf](../../Paper/MapNav.pdf). Pages actually reviewed: 1,3–8,12–15. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Raw RGB -history accumulation costs memory and repeated visual encoding. MapNav substitutes an online annotated semantic top-down map for most history. Despite RGB -only-looking check marks in Table 1, ASM construction explicitly consumes depth and pose (§3.2/Fig 3), so the whole system is not a monocular RGB -only sensor baseline.

## Method, training, inference, and equations

Construct M∈R^((n+4)×W×H): obstacle, explored, current-position, trajectory channels plus n semantic classes. Mask2Former segments RGB , depth gives point cloud, pose aligns successive observations, and projection updates BEV. Connected components over each semantic category above areaτ get centroid text labels, making a rendered ASM. Shared SigLIP-so400m encodes current RGB and ASM; spatial unpadding/patch merge Φ_spatial produces visual patches, separate 2 layer GELU projectors map to Qwen 2-7 B hidden space (Eqs 1–3). LLaVA-OneVision generates text commands; regex Ψ(text, patterns) maps synonyms to Forward/Left/Right/Stop (Eq 4). No differentiability through mapping/regex is claimed by its 'end-to-end' policy terminology. Training approximately 1 M total pairs, about 300 K oracle+200 K DAgger+25 K collision per R2R / RxR , with general VQA co-training;8 A 100 ×30 h. The 500 K R2R-only model is transferred zero-shot to RxR ; the best 2 history frame version trains both R2R / RxR . Appendix B freezes vision and fine-tunes the projector and Qwen2 for one epoch at 1e-6 using bfloat16; SigLIP features have width 1152 and Qwen2 hidden width 3584. Exact loss masks were not established in the released source audit. At inference maintain ASM, current RGB and optionally 2 history frames, then parse actions. Appendix A explicitly adds 600K general vision-language samples alongside500K navigation samples for the R2R recipe. Its expert-data prose says300K, while Eq 5 says approximately 200K; the total counts therefore have an internal ambiguity. Navigation-pair counts must not be treated as total training exposure.

## Novelty and closest predecessors

Linguistically annotated map pixels align spatial memory with VLM OCR/semantic priors. Semantic maps, DAgger and collision recovery existed; novel map rendering and learned policy integration need separable controls. It is external mapping-plus-trained VLM , not prompt-only MapGPT and not DUET .

## Results and independent verdict

Table 1 RGB -only within MapNav 27.1 SR/23.5 SPL →ASM 36.5/34.3 R2R ; adding 2 history frames and RxR training 39.7/37.2 and RxR 32.6/27.7. This supports ASM within backbone; changing training data limits attributing all best model gain to history. Table 4 semantic map 29.1 SR vs ASM 36.5, but prose says original map improves baseline although 26.4<27.3. Table 5 oracle 23.9→ DAgger 33.5→collision 36.5 SR shows substantial data contribution. Table 3 .17 MB constant representation vs NaVid 276 MB at 300 steps is stored observation memory, not full model/ GPU memory; .25 s vs 1.22 s needs whole pipeline accounting. Real 50 instructions five environments show semantic advantages but small samples. Appendix Tables 6–7 isolate input rendering and segmenter choice: directly adding depth reduces SR 27.1→23.1, whereas ASM reaches 36.5. YOLOv8/MobileSAM/Mask2Former yield31.5/34.8/36.5 SR, showing sensitivity to perception quality. This supports the representation interface but does not establish the same mechanism under equally accurate semantic inputs.

## Implementation audit

INSPECTED CODE. mapper.py:49–69 explicitly takes RGB, depth, position and rotation, projects depth to a world point cloud, associates semantic objects and accumulates the map. The full pipeline therefore depends on geometry even if the action model receives only rendered images. r2rnav_agent_nohis.py:180–245 formats current RGB plus a semantic-map image as a VLM action prompt;245–262 executes a parsed action with a maximum-step STOP. This supports the architectural interpretation that a map replaces much of the visual-history input.

A reporting helper requires particular care: caculate_metrics.py:9–14 turns failures within 3.2 m into successes and sets their adjusted SPL to 1.0 regardless of path length. It prints both original and adjusted metrics. This is nonstandard relative to 3 m SR/SPL and must not be used as the comparison evaluator. Its presence does not establish that the paper's tables used adjusted metrics. The inspected source does not establish the complete DAgger collection/training provenance or exact final configuration; data generation and VLM fine-tuning remain paper-level evidence. No project code was executed.

Repository: https://github.com/linglingxiansen/MapNav. Inspected snapshot commit: `4a3acc70270688630b4ca8f6df39f26585898e0d`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [mapper.py:13-69,120-177](https://github.com/linglingxiansen/MapNav/blob/4a3acc70270688630b4ca8f6df39f26585898e0d/mapper.py)
- [r2rnav_agent_nohis.py:180-274,387-473](https://github.com/linglingxiansen/MapNav/blob/4a3acc70270688630b4ca8f6df39f26585898e0d/r2rnav_agent_nohis.py)
- [caculate_metrics.py:1-43](https://github.com/linglingxiansen/MapNav/blob/4a3acc70270688630b4ca8f6df39f26585898e0d/caculate_metrics.py)

## Bottlenecks and follow-up chain

- **MAPNAV-1: Mapping and text annotation can destroy appearance, height, temporal and relational detail** Conditions: Repeated instances, depth/pose errors, map label clutter, long trajectories Cause: Fixed semantic categories and 2 D projection compress history; ASM raster/OCR must resolve instance identity Evidence: Fig 3 sensor pipeline; Table 1 adding 2 RGB history frames helps; Fig 5 Type: Methodological / insufficiently tested. Follow-ups:  NavFoM / StreamVLN retain visual history; PolarMem retrieves context; no matched memory budget proof of sufficient map representation. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

Main method/results and Appendix A–E data/training/sensor/ablation details were read, alongside the mapper/action/metric helper code. Complete DAgger regeneration and VLM loss masking were not independently reconstructed. Memory savings refer to the observation representation; no independent runtime or benchmark was measured.

