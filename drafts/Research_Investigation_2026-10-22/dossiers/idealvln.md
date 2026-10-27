# IDEAL-VLN

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Instruction Decomposition and Action Alignment for Vision-and-Language Navigation**. Zihao Xin, Wentong Li, Yixuan Jiang, Bin Wang, Piji Li, Jianke Zhu, Jie Qin, Shengjun Huang ICML 2026, PMLR 306 (paper masthead) (2026).

Primary record: https://raw.githubusercontent.com/mlresearch/v306/main/assets/xin26a/xin26a.pdf



Supplied main file: [IDEAL-VLN.pdf](../../Paper/IDEAL-VLN.pdf). Pages actually reviewed: 1, 3–8. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Full-instruction VLM policies repeatedly encode long histories and can choose an action relevant to a different instruction stage. IDEAL-VLN targets this stage-alignment and inference-cost problem in monocular R2R / RxR-CE . Training has privileged reference paths and aligned sub-instructions; deployment uses RGB and language.

## Method, training, inference, and equations

Sections 3.1–3.3, Eqs.1–6: first infer semantic anchor S_t ~ Pθ(S|O_1: t, L,ρ_decomp), then action A_t ~ Pθ(A|I_t, S_t,ρ_act), withholding global L and history from the second prompt. This enforces the implemented input separation, not by itself an identifiable causal effect: writing do(S_t) does not establish causal identification. The displayed conditional independence assumes no hidden reuse of L through cached states beyond S_t. The trigger called 'entropy-based' is actually two deterministic action-sequence predicates: both Left and Right appear among N predicted micro-actions, or k same-direction rotations occur. It does not calculate Shannon entropy. During correction, word-overlap precision R=|W(S_pred)∩W(S_gt)|/|W(S_pred)| uses nearest reference-waypoint segments and replaces predictions when R<τ=.8. A degenerate short generic phrase can have high precision without covering required intent; empty-denominator behavior is unspecified in reviewed text. Expert action intervention is Bernoulli with P_int(d)=sigmoid(3(d−1.5)), where d is geodesic cross-track distance in meters, so k has inverse-meter units. This is a soft sampling policy, not a safety guarantee. The 160 K alignment pairs span 61 MP3D scenes. Qwen 2.5-VL 7 B freezes vision/projector, AdamW batch 128, lr 2 e−5 cosine, warm up.03,8 H100 . Extra-data version adds ScaleVLN 150 K and 80 K LLaVA-Video QA. Token-level supervised optimization is implied by fine-tuning; exact loss masking requires appendix/release inspection. Inference uses current RGB plus compact anchor; reanchoring uses history. Table 6 estimates 147 K→77 K tokens/episode. Concurrent ~2 s thought is masked by 3–5 s physical action execution, conditional on that timing assumption.

## Novelty and closest predecessors

An explicit semantic mediator that is the only language input to action selection, plus joint thought/action correction with distance-dependent expert mixing. Related AwareVLN / AdaNav already gate reasoning; CorrectNav already collects recovery trajectories. The distinctive input bottleneck should be evaluated independently of those data improvements. Its causal terminology is stronger than the experimentally established architectural separation.

## Results and independent verdict

Tables 1–2: base R2R 56.7 SR/51.6 SPL and RxR 55.9/48.2; extra-data 59.8/52.9 and 59.1/50.7. Figure 2 RxR^42.7→51.1 with Think-Before-Act,55.9 with correction,59.1 extra data. Table 3 LLM confidence trigger 35.8 SR, SigLIP 51.5, action-pattern 51.1; efficiency is competitive, not uniquely strongest accuracy. Table 4 DAgger 51.6, STC 53.2, full 55.9. This supports decomposition plus correction on these settings, not proof that spurious causal correlations were removed. As of cutoff the SOTA prose is not a global ranking: supplied CorrectNav / AwareVLN report higher SR under different training/data.

## Implementation audit

Advertised project https://allenxinn.github.io/IDEALVLN returned 404 during this investigation. No official implementation was inspected or executed; InternNav references are not treated as this paper's released code.


## Bottlenecks and follow-up chain

- **IDEAL-1: Semantic mediator errors can erase essential global constraints** Conditions: Ambiguous alignment, off-path recovery, instructions whose future condition changes current action Cause: Compressed S_t may not be a sufficient statistic; word-overlap precision is not semantic equivalence Evidence: §3.3 explicitly says large deviations invalidate semantic support; Eq.5 and soft-gate Eq.6 Type: Methodological / insufficiently tested. Follow-ups:  CorrectNav / AwareVLN address recovery; none establishes mediator sufficiency or calibrated semantic validity. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

The supplied main paper was reviewed. The advertised project URL was inaccessible, so no official implementation, exact dataset pipeline or training run was inspected. The causal language is assessed as a modeling claim, not an identification theorem.

