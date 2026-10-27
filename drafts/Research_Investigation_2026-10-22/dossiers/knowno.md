# KnowNo

Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners

Allen Z. Ren; Anushri Dixit; Alexandra Bodrova; Sumeet Singh; Stephen Tu; Noah Brown; Peng Xu; Leila Takayama; Fei Xia; Jake Varley; Zhenjia Xu; Dorsa Sadigh; Andy Zeng; Anirudha Majumdar. CoRL 2023; PMLR volume 229. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [KnowNo.pdf](../../Paper/KnowNo.pdf). Pages reviewed: ['3–8', '12–22'].

## Problem and operating conditions

An LLM planner must choose actions under ambiguous language and request human clarification only when necessary. This is manipulation/planning research rather than an R2R route-following method. Context includes already grounded textual observations; the theorem assumes correct low-level execution and accurate human help.

## Mechanism, training, and inference

An LLM proposes four candidate actions and an appended other-action option. A second prompt scores the answer letters. Calibration uses nonconformity 1 minus the true-label score and a finite-sample conformal quantile. At inference, labels above the threshold form a prediction set; a singleton is executed, otherwise a human selects a valid option or halts. For a fixed calibration dataset the paper uses a beta-quantile tolerance bound with N=400 and delta=0.01. Multi-step calibration uses the minimum true-label score across an entire correct-action sequence, then constructs per-step sets causally with the same threshold. Appendix A3 handles several acceptable actions by following the highest-scored acceptable branch during both calibration and human-assisted deployment. No model fine-tuning is required; labelled calibration scenarios and human intervention are required. Most experiments use PaLM-2L; ensemble baselines require 20 queries.

## Novelty and closest predecessors

Extends conformal prediction to language-planning choices and dependent multi-step execution, contrasting prompt-only uncertainty and cumulative-softmax thresholding. Coverage does not require accurate model probabilities; minimal expected set-size claims require the additional assumption that the relevant score models true conditional probabilities. Minimal set size is a different objective from minimizing the probability of requesting help or human response effort.

## Results and independent verdict

PAPER EXPERIMENT: Hardware Table 1, p. 7, KnowNo plan success 0.76, task success 0.74, per-step help 0.58 versus Simple Set 0.76/0.72/0.72 in 50 trials. Hardware mobile manipulation Table 2, p. 8, plan success 0.87 and physical task success 0.76 with help 0.67; matched-plan-success Simple Set needs help 0.81. The planning versus physical success gap directly demonstrates the limit of the execution assumption. Appendix A4 has only ten bimanual trials. DEDUCTION: the coverage guarantee is distributional, not a guarantee conditional on each particular task or on only the no-help subset. Sequence-level minimum scores are not generally joint sequence probabilities, so do not infer optimal help rate from calibrated step scores. The paper’s own §6 acknowledges grounding/control uncertainty and inaccurate human responses are outside scope.

APPENDIX SCOPE: pp. 17–19 constrain hardware scenarios to avoid visually confusable detections; some challenging mobile tasks use known object locations and scripted execution policies. Figures A9–A11 retain multiple acceptable answers in some scenarios. Calibration concerns the specified scenario/label distribution and successful high-level option selection with the stated help assumptions; it does not independently certify the learned detector or low-level controller.

## Implementation and reproducibility

INSPECTED CODE: The official google-research/language_model_uncertainty release consists of three executable notebooks and README. MobileManipulation cells 23–27 explicitly temperature-scale the five option scores (temperature 5), use minimum nonconformity across acceptable labels, compute the finite-sample corrected quantile, and include options above 1-qhat. Its test loop sets simulated task success equal to set coverage and assumes the human always selects a correct included option; help triggers for non-singletons and the extra option. This precisely supports the stated ideal-human assumption, not physical execution reliability or conditional error control among autonomous actions. The demo uses a stored threshold; tabletop notebook includes PyBullet manipulation primitives and separate calibration. No notebook cells, API calls or physical/simulated controller were executed. Proprietary original model weights and real robot deployment are outside the release.

Inspected release reference: [https://github.com/google-research/google-research at 0d94f8c4d92678ce5f7a0eacae725f79174cfb03](https://github.com/google-research/google-research/tree/0d94f8c4d92678ce5f7a0eacae725f79174cfb03). File-level inspection is restricted to the paths listed below.

- language_model_uncertainty/KnowNo_Demo.ipynb: prediction-set cells
- language_model_uncertainty/KnowNo_MobileManipulation.ipynb: cells23–27 (extracted text373–481)
- language_model_uncertainty/KnowNo_TabletopSim.ipynb: environment reset/motion and calibration references

## Specific bottlenecks

**knowno-grounding: Planning coverage does not cover perception errors, failed low-level control, or a changed scenario distribution.**

Conditions: Ambiguous or incorrect textual grounding, physical execution failures, or deployment shifts. Cause: Calibration is over grounded planning choices from an exchangeable scenario distribution; correct human help is assumed. Evidence: AUTHOR ACKNOWLEDGEMENT: §6; PAPER EXPERIMENT: Table 2 plan 0.87 versus task 0.76.

Type: insufficiently tested. Status: Still unresolved in the reviewed literature. Confidence: High within the evaluated setting. Follow-ups: ['Explore until Confident', 'AIUTA']

## Sources and limitations

- [Official proceedings](https://proceedings.mlr.press/v229/ren23a.html)
- [Official project and repository link](https://robot-help.github.io/)
- Notebooks were read as text, not executed; external model APIs and robots were not run. Guarantee scope depends on scenario exchangeability, acceptable-label definitions and assumed execution/help reliability.
