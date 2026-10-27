# Explore until Confident

Explore until Confident: Efficient Exploration for Embodied Question Answering

Allen Z. Ren; Jaden Clark; Anushri Dixit; Masha Itkina; Anirudha Majumdar; Dorsa Sadigh. RSS 2024. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [Explore until Confident.pdf](../../Paper/Explore%20until%20Confident.pdf). Pages reviewed: ['3–12'].

## Problem and operating conditions

Explore an unknown indoor scene until there is sufficient visual evidence to answer a multiple-choice question. Inputs include RGB-D, camera intrinsics, localization, and a collision-free local planner. Simulation assumes perfect odometry and movements of at most 3 m between queried viewpoints.

## Mechanism, training, and inference

Prismatic VLM scores answer options, image-question relevance, and exploration directions marked on the RGB image. Local semantic value normalizes scores among three projected free-space points; global semantic value scores whether any direction is worth exploring. Their exponentially scaled sum is smoothed into a TSDF-derived map and weights frontier sampling. The map is built online from depth. For stopping, rho_t(y)=Rel_t*(p_t(y)-1), and episode score is min_t rho_t(y). A conformal threshold calibrated on full exploration episodes gives per-step answer sets whose running intersection retains the correct answer simultaneously over all prefixes with marginal probability at least 1-epsilon under the same scenario distribution and policy. Stop when the intersection is a singleton. At timeout or an empty intersection, choose the most likely answer at the most relevant view: this fallback is outside the singleton guarantee. Low relevance gives scores near zero, intentionally preventing irrelevant views from removing many answers. The main method has no additional VLM training but uses 300 labelled calibration episodes; an optional separate fine-tuning experiment is reported.

FULL-VERSION APPENDIX CHECK: arXiv v2, PDF p. 15, proves that the trajectory-minimum score exceeds a calibrated threshold exactly when every step retains the true answer in its prediction set. Thus the simultaneous-coverage event carries to any stop within that fixed scenario trajectory, under the scenario exchangeability and calibration assumptions in the main paper. This is an answer-set statement, not a guarantee of collision-free control, accurate localization or finite-time singleton termination. The same appendix uses a cropped explored-view update and frontier weights exp(SV + SV_normal) with a 3 m lookahead along frontier normals. The released repository audit below still did not locate the CP calibration/intersection path.

## Novelty and closest predecessors

Combines visually prompted semantic frontier exploration with sequence-calibrated answer stopping. KnowNo already supplies multi-step conformal calibration; here the set is intersected over repeated observations of a common answer rather than used to request human action labels. This is not a general proof of optimal exploration or conditional accuracy on stopped episodes.

## Results and independent verdict

HM-EQA contains 500 GPT-4V-generated, human-filtered questions across 267 HM3D scenes. Exploration comparisons use all 500 at fixed horizons; stopping comparisons split 300 calibration/200 test. Figure 9, p. 10: about 58% answer success requires 71% of the maximum steps versus 85% for a relevance-threshold baseline. Main raw VLM success saturates near 60%; §VII-E reports a separate fine-tuned model 56.2→68.1%. Six hardware scenarios yield four correct answers for this method and relevance baseline; several relevant views still produce wrong answers. These results support efficient exploration and calibrated set retention, not universal high answer accuracy. §VIII acknowledges orientation is omitted from semantic values, single-view reasoning is limited, and prediction sets do not yet guide exploration.

FULL-VERSION APPENDIX CHECK: pp. 16–17 find that extra exploration does not reliably help small-object state questions, and report no gain for identification questions requiring stronger semantic location inference. A view-orientation heuristic raises reported success from 58.4% to 59.3%; replacing the VLM raises it to 65.3% with LLaVA 1.6 or 73.9% with GPT-4V. These are separate component changes, not evidence that the stopping mechanism itself caused all gains.

## Implementation and reproducibility

INSPECTED CODE: The official explore-eqa release provides a Habitat exploration pipeline, TSDF mapping, VLM answer/relevancy scoring, semantic frontier selection and question-generation scripts. run_vlm_exp.py:197–213 scores answer options and confidence about the current view, while 339–366 aggregates the recorded trajectory by relevancy-weighted or highest-relevancy prediction. A full-tree search finds no conformal calibration/prediction-set intersection or calibrated stopping implementation in the released Python source, and the pinned tree contains no notebooks. README documents semantic-exploration and CLIP exploration commands and the 500-question HM-EQA dataset. Therefore the paper’s conformal stopping guarantee remains paper evidence; the released pipeline alone does not reproduce that guarantee. This is bounded to the pinned official release, not a claim no code exists elsewhere. HM3D access, VLM weights and simulator execution were not attempted.

Inspected release reference: [https://github.com/Stanford-ILIAD/explore-eqa at 18381da3370c5f7729594f10ebc49b5644bbb88c](https://github.com/Stanford-ILIAD/explore-eqa/tree/18381da3370c5f7729594f10ebc49b5644bbb88c). File-level inspection is restricted to the paths listed below.

- README.md
- run_vlm_exp.py:197–213,339–366
- Pinned repository tree and source search for conformal/calibration/quantile

## Specific bottlenecks

**explore-answer: A useful view can still produce a confident incorrect answer, and forced fallback answers lack the conformal retention guarantee.**

Conditions: VLM perception/spatial reasoning errors; timeout or empty-set fallback; distribution or exploration-policy change. Cause: The decision mechanism calibrates scores but does not improve the underlying visual reasoning; fallback returns a point answer without retained-set coverage. Evidence: PAPER EXPERIMENT: §VII-C/H and Figure 10; METHOD: §V-B.

Type: methodological. Status: Partially addressed. Confidence: High within the evaluated setting. Follow-ups: ['AIUTA', 'VSI-Bench', 'OmniAgent']

## Sources and limitations

- [Official project](https://explore-eqa.github.io/)
- [Author manuscript](https://arxiv.org/abs/2403.15941)
- [Official code](https://github.com/Stanford-ILIAD/explore-eqa)
- [Complete 17-page author version; appendices A–C pp. 15–17 reviewed](https://arxiv.org/pdf/2403.15941v2)
- Complete arXiv v2 appendix reviewed. CP calibration/intersection/stopping code was not located in the pinned repository; this limits code verification of the paper guarantee. No robot experiment executed.
- Associated material: {"file": "evidence/04/supplements/explore_confident_full.txt", "pages_read": "15–17", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
