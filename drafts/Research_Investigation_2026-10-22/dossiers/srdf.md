# SRDF

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Bootstrapping Language-Guided Navigation Learning with Self-Refining Data Flywheel**. Zun Wang, Jialu Li, Yicong Hong, Songze Li, Kunchang Li, Shoubin Yu, Yi Wang, Yu Qiao, Yali Wang, Mohit Bansal, Limin Wang. ICLR 2025 (PDF header).

Primary paper: [Bootstrapping Language-Guided Navigation Learning with Self-Refining Data Flywheel](https://github.com/wz0919/VLN-SRDF). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27].

Associated files: SRDF.pdf (Main PDF; 27 pages; SHA256 402aabf3bae55f83f70139ea5e3ad5f3c980122bdac5738628f87787e25384a6); SRDF (alternate).pdf (Alternate / duplicate PDF; 27 pages; SHA256 402aabf3bae55f83f70139ea5e3ad5f3c980122bdac5738628f87787e25384a6).

The second supplied file has the exact same SHA256 as the main PDF and adds no independent evidence. Main includes appendices; Algorithm 1 and extra scorer/generator experiments on pp 20–21 were inspected.

## 2. Problem and operating assumptions

Generate abundant aligned path-language supervision without new human annotation in the iterative loop. Assumes an initial human seed set, unlabeled simulator trajectories, pretrained multimodal generator and a trainable navigator. It is not training-free or free of human-created supervision.

## 3. Mechanism, training, and inference

AUTHOR METHOD (§3 pp 4–6): LoRA-finetune Mantis-8B-SigLIP with interleaved image/action trajectory prompts. Greedy captions form candidate generator-training data; top-k sampled diverse captions train DUET. Train/fine-tune navigator, rerun generated instructions, retain generator candidates satisfying SPL=1 and navigator samples with nDTW≥0.9. Regenerate failed samples rather than changing the trajectory pool; repeat 3 rounds, plus fourth generator refinement. Dataset 20,417,874 instructions across 860 environments. InternViT-6B encodes images except CLIP-B/16 in CE; navigator 45k iterations, batch 1024, learning rate 5e-5, 8 A100s, then R2R 6k iterations batch 16 (§4.1 p 7). At inference only trained navigator is used. No theorem establishes monotonic improvement or verifier correctness.

Appendix A.1, PDF p 19, clarifies that RxR-English and R4R use recovery to a reference-path viewpoint maximizing nDTW rather than a universal shortest-path expert. R2R and other tasks use shortest-path teacher actions. This distinction is essential when evaluating whether the flywheel preserves route semantics. Generator training updates LoRA layers on Mantis-8B-SigLIP-Llama 3; the round 3 navigator and generator reuse round 2 initialization while earlier rounds restart specified base models.

## 4. Novelty and evidence

Relative to Speaker-Follower/ScaleVLN, jointly improves generator and navigator through navigation-consistency filtering and targeted recaptioning. Unlike mere scaling, quality changes at fixed underlying trajectory pool; Table 3 tracks rounds, Tables 4/11 compare scorers, Table 5 instruction count controls. The approach resembles a mutual verifier/generator self-training loop, not independent ground-truth validation.

## 5. Results and independent verdict

PAPER EXPERIMENT: Table 3 p 7 under shared InternViT features reports baseline 78.6 SR/69.9 SPL→round 3 84.4/77.6 val-unseen. Final Table 7 p 9 test 85 SR/78 SPL, versus human 86/76: superior SPL is not superior human success. Table 5 p 8 compares 1, 3, 6, 12 instructions per path, showing saturation after 6; Table 11 p 21 navigator-filtered generator SPICE 25.2 versus 23.9 CLIP and 23.6 self-score. The main evaluation emphasizes navigation on REVERIE/SOON (§4.1); Appendix A.5–A.6 and Tables 16–17 additionally report object-grounding metrics. The full paper therefore should not be described as omitting grounding evaluation. Verdict: strong data-quality evidence, but semantic fidelity is measured indirectly. Larger vision model, PanoGen and task-specific fine-tuning are confounders in cross-paper leaderboard comparisons.

Appendix Tables 14–18, pp 23–24, reports R4R nDTW 56.5/SDTW 44.6 and RxR-English nDTW 74.4/SDTW 66.3 on validation unseen, improving the cited route-fidelity baselines as well as endpoint success. REVERIE test SR 61.4 is accompanied by RGS 35.6, below GOAT 38.3, while RGSPL 27.1 slightly exceeds GOAT 26.7. Thus data scaling does not uniformly solve object grounding. Appendix qualitative claims about rare objects, OCR and outdoor scenes are examples, not controlled tests of those abilities.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/wz0919/VLN-SRDF](https://github.com/wz0919/VLN-SRDF); inspected snapshot **1835eca4bd90958b9b42c210236006e40ed6aa32**. Snapshot metadata and source are under `repositories/01/srdf`.

The pinned official Git tree exposes README.md but no executable model/training/filter code. Therefore exact filter implementation, selected checkpoints, SPL equality tolerance and released training configuration cannot be inspected. The paper algorithm is the evidence, not a reconstructed code claim.

The original SRDF snapshot is README-only. Later HLR/SkillNav specialist implementations reported by assignment 03 may support experiments with SRDF checkpoints; their availability must not be conflated with release of the complete original flywheel pipeline.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

**Execution status:** No model source inspection is claimed beyond the explicit scope above. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **srdf-B1 — Still unresolved in reviewed evidence**. SPL=1 is not equivalent to exact path following: equal-length alternative routes and goal-radius stopping can satisfy it. nDTW similarity likewise does not prove language entailment. Navigator-based acceptance may preferentially retain instructions compatible with that navigator rather than counterfactually necessary wording. Evidence: AUTHOR filter §3.2 p 5; DEDUCTION from SPL definition; no benchmark failure measured. Type: methodological. Follow-ups: CAST counterfactual augmentation (lead for follow-up audit). Confidence: high for metric counterexample; medium for selection-bias impact.
- **srdf-B2 — Shifted into compute/verification costs**. The flywheel requires repeated large generator/navigator training; study reports 3 navigator rounds but no unlimited-round stability guarantee. Additional rounds can amplify verifier bias. Evidence: AUTHOR compute §4.1; Alg 1 p 20; HYPOTHESIS about long-run drift. Type: methodological. Follow-ups: SkillNav. Confidence: medium.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://arxiv.org/abs/2508.13446; https://hlr.github.io/SkillNav/
