# Uni-NaVid

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks**. Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, He Wang Robotics: Science and Systems 2025 (2025).

Primary record: https://roboticsproceedings.org/rss21/p013.html



Supplied main file: [Uni-NaVid.pdf](../../Paper/Uni-NaVid.pdf). Pages actually reviewed: Alternate arXiv v2:1,3–12 (main method, task results and ablations). The local supplied PDF has no extractable text. Reviewed the authors' arXiv2412.06224v2 (6 February 2025), saved as evidence/03/uninavid_alternate.pdf. Numerical claims refer to that version; byte equivalence with the supplied proceedings PDF is not assumed.

Associated material:

- evidence/03/uninavid_alternate.pdf; reviewed pages/status: 1,3–12; version arXiv2412.06224v2

## Problem, motivation, and operating assumptions

Unify VLN, ObjectNav , EQA and human following rather than task specific policies, while processing long video efficiently. RGB single view and text in,4 low level actions out; EQA action and answer phases explicitly separated.

## Method, training, inference, and equations

SectionIV encodes each RGB frame with EVA-CLIP into256 patch features. Grid pooling retains 64 tokens for the current frame, four tokens per short-term frame (buffer size 64), and one token for each older frame. When adjacent old tokens have cosine similarity above 0.95, the paper merges them with a count-weighted mean; otherwise it appends a new token (Eqs 1–6). If all frames are sufficiently distinct, the long-term list grows linearly, so merging is not a hard memory bound. The released threshold and empty-cache handling differ from the displayed paper pseudocode, as recorded below.

A two-layer MLP projects visual features into Vicuna-7B. Special navigation markers distinguish action generation from question answering. The action vocabulary is forward 25 cm, left30°, right30° and STOP, with four actions generated per decision. Stage 1 aligns the projector on LLaMA-VID data; stage 2 tunes the projector and language model, keeping vision frozen. The total mixture is5.9M examples:3.6M navigation and 2.3M video QA. Navigation includes2.4M VLN,483K ObjectNav,240K EQA action examples plus 10K answers, and 544K following examples across861 scenes. Training takes about 40 H800 GPUs×35 hours.

For EQA, the agent first navigates, then removes the navigation marker and answers from observed history after STOP. This separates action and answer modes; it does not supply future-state prediction. Real deployment generates four actions in approximately 0.2s on a remote A 100 and uploads new observations asynchronously. Robot heights/radii are varied in training, but the authors explicitly limit claims to the trained range rather than arbitrary embodiments.

## Novelty and closest predecessors

Unified multi-task navigation training and online current/short/long token hierarchy; weighted adjacent merging saves real-time cost. Does not predict future observations/world states despite future action chunking. NavFoM extends cross camera/embodiment and trajectory output; StreamVLN adds KV reuse.

## Results and independent verdict

Table II R2R^47 SR/42.7 SPL vs NaVid 37.4/35.9; Table III RxR^48.7/40.9 uses RxR training while NaVid 23.8/21.2 is R2R -only: not matched cross-dataset comparison. Table IV HM3D 73.7/37.1; Table V OVON unseen 39.5/19.8 zero-shot to OVON but trained ObjectNav . Scene overlap checked by authors§VI; no independent audit. Figure 8 compares single-task and multi-task learning and data scale, supporting synergy under the authors' recipe; it does not by itself equalize every task's effective data/compute exposure.

TableX (alternate p.11) reports VLN SR 9.61 with only the current frame,39.7 with current/short memory and 48.7 with long memory added. Each ablation retrains the model. Removing VQA reduces EQA accuracy47.3→1.19, showing that preserving general language knowledge is a major part of the mixture. EQA TableVI distinguishes autonomous navigation47.3 from ground-truth-trajectory54.4. The memory ablation therefore supports retaining history, but not guaranteed preservation of rare landmarks. Additional external supplemental videos/results were not reviewed.

## Implementation audit

INSPECTED CODE. uninavid/model/uninavid_arch.py:113–155 implements an online cache with 64 short-term frames. Older frames are averaged into one token and merged with the latest long-term token only above a cosine threshold. The release default is0.985, differing from the0.95 value in the reviewed paper. The code correctly handles an initially absent long-term cache at 150–151, resolving the pseudocode empty-cache ambiguity. When consecutive summaries stay dissimilar, the long-term cache appends without a hard capacity bound. The offline process_tensor path at 168–210 applies the same merging idea.

language_model/llava_llama_vid.py:103–113 uses ordinary shifted token CE. This confirms memory compression and action-language supervision; it does not establish all dataset-construction, multitask-balancing or physical-robot results. The reviewed alternate arXiv text and the released source may represent different versions. No training, simulator or robot code was executed.

Repository: https://github.com/jzhzhang/Uni-NaVid. Inspected snapshot commit: `79ef5ea3fea14c205342d1ab070563d84c7a966a`. Retrieval status: Commit-pinned source retrieved through sparse git checkout. Project code was not executed.

- [uninavid/model/uninavid_arch.py:113-155,168-210](https://github.com/jzhzhang/Uni-NaVid/blob/79ef5ea3fea14c205342d1ab070563d84c7a966a/uninavid/model/uninavid_arch.py)
- [uninavid/model/language_model/llava_llama_vid.py:103-121](https://github.com/jzhzhang/Uni-NaVid/blob/79ef5ea3fea14c205342d1ab070563d84c7a966a/uninavid/model/language_model/llava_llama_vid.py)

## Bottlenecks and follow-up chain

- **UNINAVID-1: Token merging is unbounded worst-case and may discard rare relevant detail** Conditions: Diverse visual histories, task-switches, dynamic following Cause: Similarity threshold+global pooled old frame; only adjacent merge Evidence: Eqs 2–6; B 64/τ.95; data/protocol Table III Type: Methodological / protocol. Follow-ups:  StreamVLN fixed window, NavFoM BATS budget; instruction-aware utility retention not established. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

An accessible author arXiv version, including main task results and TableX ablations, was reviewed instead of the local image-only proceedings PDF. Separate external supplemental videos and any proceedings-only changes were not reviewed. The released merge implementation was inspected; no benchmark or physical robot run was executed.

