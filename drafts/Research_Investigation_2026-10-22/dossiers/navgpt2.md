# NavGPT-2

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models**. Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, Qi Wu ECCV 2024 (2024).

Primary record: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01143.pdf



Supplied main file: [NavGPT-2.pdf](../../Paper/NavGPT-2.pdf). Pages actually reviewed: 1, 4–10. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Frozen prompted LLM s reason verbally but lag supervised VLN. NavGPT2 adds trained visual alignment and a learned graph policy while retaining optional linguistic generation. It uses discrete simulator candidates, not end-to-end physical navigation.

## Method, training, inference, and equations

§3: frozen EVA-CLIP ViT-g/14 image features plus instruction-conditioned Q-Former 32 queries∈R^(32×768) per view, linear projection into FlanT5 / Vicuna hidden space. Prompt includes candidate angle and special image/instruction delimiters. Stage 1 instruction-tunes Q-Former /projector on 10 K GPT-4V reasoning examples sampled at reference path intermediate points, autoregressive token CE; LLM /vision frozen. Stage 2 freezes VLM and trains DUET global branch-style policy: last encoder(or decoder)hidden image and text features, MLP merge 32 image queries per view, average pool node observations, add direction/visit step embeddings. Crossattention to instruction then graph self-attention softmax(Q Kᵀ/√d+A(E))V (Eq 2; A is learned spatial affinity over pairwise distances). Global FFN scores all observed frontiers plus Stop; visited nodes masked; shortest graph path moves agent to chosen frontier. BC and DAgger CE losses L=λL_BC+L_DAG(Eq 3), with on-policy shortest-goal pseudoexpert. Stage 1 AdamW 200 K steps batch 8 lr 1 e−5, wd.05; stage 2 batch 2 lr 1 e−5; single A 100 ; best uses PREVALENT synthetic data. Action policy consumes latent features, not necessarily generated rationale text; optional reasoning does not guarantee explanation faithfulness.

## Novelty and closest predecessors

Directly exposes frozen LLM visual-language latents to a specialized graph policy, rather than choosing actions solely through text. This is explicitly DUET global branch-derived, outside neither DUET lineage nor train-free category. The large LLM is frozen but Q-Former and policy are trained.

## Results and independent verdict

Table 1 FlanT5XXL encoder 5 B+PREVALENT 72 SR/61 SPL unseen,71/60 test versus DUET 72/60,69/59; R2R -only 71/60. DUET without local branch 68/59. ScaleVLN 79/70 has more data. Supports closing gap at similar navigation data scale, not best large-data performance. Reasoning data 10 K alone is not total navigation training data. Architecture-stage freezing limits memory but frozen 5 B+vision still costly.

## Implementation audit

INSPECTED CODE. The actual policy implementation is map_nav_src/models/NavGPT_model.py, rather than the generic retained vilmodel.py. NavGPTThought:400–597 computes Q-Former-conditioned LLM states and extracts image-token and instruction embeddings. Without target thoughts, its Flan-T5 branch runs only the encoder at 574–584; forward_thought_per_step:916–925 explicitly passes no text-output target. Thus the navigation policy can act from latent LLM features without generating or consuming a textual rationale. NavGPTAction's global branch at 854–859 scores unvisited graph nodes. The released run_r2r_xxl.sh:25–61 sets global fusion, DAgger, frozen Q-Former, EVA-CLIP-G1408-dimensional features and a 4096-dimensional LLM context.

The separate reasoning-tuning model blip2_t5_instruct_nav.py:78–130 freezes vision/T5 while learning Q-Former/projection;144–238 replaces image tokens with projected queries and optimizes output-token CE. r2r_navgpt_ft_xxl.yaml:33–66 specifies20 epochs, lr 1e-5, batch 1 and accumulation8. Navigation agent.py:380–420 passes latentstates through panorama/graph policy;450–461 applies supervised action CE and optional generation loss. This is a trained DUET-derived global graph policy with a foundation-model representation, not a frozen prompt-only agent. Main feature databases, checkpoints and exact outcomes were not executed or reproduced.

Repository: https://github.com/GengzeZhou/NavGPT-2. Inspected snapshot commit: `1f535cbeedfe2aad0a94925a0f9a613fe857f211`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [map_nav_src/models/NavGPT_model.py:283-351,400-597,854-859,890-944](https://github.com/GengzeZhou/NavGPT-2/blob/1f535cbeedfe2aad0a94925a0f9a613fe857f211/map_nav_src/models/NavGPT_model.py)
- [map_nav_src/models/lavis/models/blip2_models/blip2_t5_instruct_nav.py:78-238](https://github.com/GengzeZhou/NavGPT-2/blob/1f535cbeedfe2aad0a94925a0f9a613fe857f211/map_nav_src/models/lavis/models/blip2_models/blip2_t5_instruct_nav.py)
- [map_nav_src/r2r/agent.py:32-46,380-461](https://github.com/GengzeZhou/NavGPT-2/blob/1f535cbeedfe2aad0a94925a0f9a613fe857f211/map_nav_src/r2r/agent.py)
- [map_nav_src/models/lavis/projects/blip2/train/r2r_navgpt_ft_xxl.yaml:1-66](https://github.com/GengzeZhou/NavGPT-2/blob/1f535cbeedfe2aad0a94925a0f9a613fe857f211/map_nav_src/models/lavis/projects/blip2/train/r2r_navgpt_ft_xxl.yaml)
- [map_nav_src/scripts/run_r2r_xxl.sh:1-70](https://github.com/GengzeZhou/NavGPT-2/blob/1f535cbeedfe2aad0a94925a0f9a613fe857f211/map_nav_src/scripts/run_r2r_xxl.sh)

## Bottlenecks and follow-up chain

- **NAVGPT2-1: Reasoning text and actions need not agree; graph geometry remains privileged** Conditions: Contradictory rationales, continuous physical deployment Cause: Policy consumes LLM encoder latents and geometric graph, not causally necessary rationale Evidence: §3.1–3.3/Fig 2; visited mask and shortest path control Type: Methodological / evaluation. Follow-ups:  SkillNav adds router over expert policies; NaVid / NaVILA remove graph candidate assumption; faithfulness unresolved. Status: **Partially addressed**, confidence Moderate.

## Source/access record and remaining uncertainty

Main method, principal results and targeted Q-Former/LLM/policy/loss/configuration code were inspected. Full feature preprocessing, all appendix experiments and exact pretrained artifacts were not independently reproduced. No project code was executed.

