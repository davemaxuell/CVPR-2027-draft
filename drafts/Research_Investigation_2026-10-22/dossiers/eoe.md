# EOE

## Bibliography and reviewed material

**Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection**. Chentao Cao; Zhun Zhong; Zhanke Zhou; Yang Liu; Tongliang Liu; Bo Han. ICML 2024 (2024). Research cutoff: **2026-10-22**.

Primary publication/source: https://proceedings.mlr.press/v235/cao24d.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- EOE.pdf; Main PDF; SHA256 `8c304bdd2cb9e14048722f8844d76e3303a77232e2002da3f4dcd6aa88deb8d7`; 31 pages.

## Problem, task and assumptions

Zero-shot image out-of-distribution detection from known class names and pretrained CLIP. The supplied EOE paper is not a navigation or world-model method; the acronym must not be conflated with a navigation approach.

The user confirmed on 22 October 2026 that this is the intended EOE reference in the novelty shortlist. Its identity is settled; the scope of transferring its semantic novelty score to navigation remains a separate scientific question.

## Method, training and inference

Section 3 and Algorithm 1, pp. 3–4: prompt an LLM for outlier class names that are visually related but distinct from known classes. Far, near and fine-grained settings use different prompts. Remove lexical overlap, embed known and envisioned labels with CLIP, and compare them to the normalized image embedding. Equation 3 subtracts beta times the largest envisioned-outlier softmax probability from the largest known-class probability; the softmax denominator includes both sets and beta is 0.25. Threshold the resulting scalar for ID/OOD classification. Evaluation reports false positives at a threshold chosen to retain 95% of ID examples. There is no task-specific model training, dynamics model, action policy or future rollout. CLIP pretraining and offline LLM generation remain external dependencies.

## Novelty and closest predecessors

Compared with MCM's closed set of known labels, ZOC's learned caption generator and CLIPN's extra encoder training, EOE uses prompted negative-label augmentation and an explicit outlier penalty. Semantic novelty detection does not establish calibrated navigation-action risk.

## Results and independent verdict

Table 1, p. 5 averages four ID datasets and four OOD datasets: FPR95 0.21/AUROC 99.92 versus MCM 2.68/99.40. CLIPN uses extra data and scores 0.67/99.80. Later results are substantially harder: the ImageNet far-OOD result on pp. 6–8 has FPR95 30.09/AUROC 92.96; near-OOD average is 9.07/98.59; the fine-grained average is 60.52/81.82 versus CLIPN 56.05/83.18. Thus near-perfect small-dataset numbers must not be generalized to all settings. These are paper experiments, not executed reproduction.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/tmlr-group/EOE. Snapshot commit: `20fa95e5baff9242b93f0957bb4564db3d36d1f2`. Retrieval: 2026-10-22.

INSPECTED CODE: `eval_ood_detection.py:23–49` loads a frozen image-text model, constructs ID and envisioned OOD labels, and evaluates scores. `utils/detection_util.py:98–174` computes normalized image/text similarity, joint softmax, and the maximum ID probability minus beta times maximum envisioned-OOD probability for the EOE branch. AUROC/FPR95 are computed by its metric utilities, not navigation metrics. Lines 203–236 implement semantic filtering at cosine 0.85 specifically for near-OOD ImageNet20, while the default threshold is 1.0. `utils/generate_llm_class.py:187–227` loads, lowercases and deduplicates generated class names. Thus it is inaccurate to describe all filtering as exact-string-only. This method requires no detector fine-tuning, but does rely on pretrained CLIP and LLM class generation; the released candidate lists improve repeatability without guaranteeing coverage of unknown classes.

Inspected files: `eval_ood_detection.py`, `utils/detection_util.py`, `utils/generate_llm_class.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Generated negative labels cannot enumerate all unknown appearances. The ImageNet20 near-OOD setting filters semantic cosine overlap at 0.85, but this does not guarantee coverage of unseen categories or semantic disjointness elsewhere. LLM versions and prompts alter candidate lists. An unsafe or instruction-inconsistent navigation action may involve familiar objects, so transferring this score into action-risk estimation requires navigation-specific labels and tests.

Relevant follow-up relationship: Use EOE only as adjacent semantic novelty-detection prior art. Its confidence-like score should be compared with, rather than equated to, transition uncertainty and navigation risk.

## Access and uncertainty record

Full text is preserved under `corpus/eoe/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/eoe/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Coordinator supplementary review

Following the user's identity confirmation, the coordinator additionally read PDF page 9 and all appendix pages 13–31 from the page-labelled extraction on 22 October 2026. This is a separate review from assignment 02's main-paper and source inspection. Reference-only pages 10–12 were not counted, and figure images were not independently rendered in this supplementary pass.

The appendix strengthens the evidence for an image-detection method but also defines the limits more precisely. Tables 12–18 report large-scale near-OOD and domain-shift experiments; ImageNet-C results use severity level 1 (§C.2, p.20), so they do not establish robustness at every corruption severity. Table 12 reports mean near-OOD FPR95 79.81 versus MCM 84.06: a relative improvement with substantial residual false-positive error. Table 13 reports ImageNet-Sketch FPR95 42.94 versus 70.13. These are author experiments, not reproduced results.

Tables 5–9 vary the number of generated classes and the outlier penalty. Benefits are not uniform across datasets or hyperparameters. For example, Table 7, p.17, with 500 outlier labels changes average ImageNet FPR95 from 37.54 at beta=0 to 30.09 at beta=.25, while Texture FPR95 changes from 55.87 to 57.53. Tables 20–21 show that average improvements extend across several backbones; this does not make the score a calibrated probability.

The paper's page 9 limitation acknowledges reliance on known ID class names and on choosing an OOD task type. Table 25, p.28, tests a universal far-OOD prompt: it can underperform MCM on particular settings, including ImageNet-10 near-OOD FPR95 5.47 versus 5.00 and Oxford-Pet-18 fine-grained FPR95 67.27 versus 63.92. Task-specific prompt selection and the candidate-label list are therefore material parts of the method. Appendix F supplies concrete prompts and generated-label examples; neither those examples nor the t-SNE illustration in Appendix A establish exhaustive coverage of unseen classes. No appendix experiment supplies action-conditioned navigation risk labels, so the transfer question below remains open within the reviewed evidence.

Review scope is recorded in [the coordinator supplement record](../evidence/eoe_supplement_review.json).

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-eoe-risk_transfer

**Uncertain because evidence is insufficient — Insufficiently tested.** Image semantic novelty scores are not established estimates of unsafe or instruction-inconsistent navigation actions.

Conditions: Familiar objects with unsafe transitions, or unknown categories absent from generated negative labels. Cause: The score compares CLIP similarity to generated class descriptions, without a transition/action outcome variable or exhaustive unknown-class coverage.

Evidence: PAPER EXPERIMENT: pp.6–8 fine-grained FPR95/AUROC 60.52/81.82, far-ImageNet 30.09/92.96; INSPECTED CODE: cosine overlap filter at 0.85 for ImageNet20. DEDUCTION: semantic novelty differs from navigation risk.

Follow-up assessment: No matched navigation-risk follow-up was reviewed; EOE must remain adjacent OOD prior art. Confidence: Medium.
