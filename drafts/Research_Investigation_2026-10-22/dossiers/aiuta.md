# AIUTA

Collaborative Instance Object Navigation: Leveraging Uncertainty-Awareness to Minimize Human-Agent Dialogues

Francesco Taioli; Edoardo Zorzi; Gianni Franchi; Alberto Castellini; Alessandro Farinelli; Marco Cristani; Yiming Wang. ICCV 2025. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [AIUTA.pdf](../../Paper/AIUTA.pdf). Pages reviewed: ['3–8'].

## Problem and operating conditions

Find a particular object instance in an unknown continuous HM3D scene using an initial object category and dialogue with a user who knows the exact target. The agent does not see the target image; the simulated user does. Multiple same-category distractors make ordinary ObjectNav insufficient.

## Mechanism, training, and inference

AIUTA wraps VLFM frontier navigation, BLIP-2 value mapping, and Grounding DINO detection. At a candidate object, LLaVA-1.6 Mistral 7B generates a caption and GPT-4o proposes attribute questions. The visual model scores yes/no/unknown and normalized three-way entropy selects uncertain questions with threshold 0.75. Object-presence checks may continue exploration. A language model compares accumulated facts to user answers on a 0–10 scale: high alignment (at least 7) permits stopping, low alignment (below 5) skips the candidate, and intermediate scores trigger more questions. The method allows up to four dialogue rounds and filters uncertain attributes. CoIN contains 1,649 episodes: 831 seen categories, 359 synonyms, 459 unseen categories; at least two distractors, start distance 5–20 m, success within 0.25 m of a goal viewpoint, and 500 actions. The main navigation method uses pretrained models without task fine-tuning, but depends on a truthful target-aware user and tuned decision thresholds.

## Novelty and closest predecessors

Adds instance-specific collaborative verification and uncertainty-filtered dialogue to open-vocabulary object navigation. KnowNo calibrates action sets with formal distributional coverage; AIUTA instead uses heuristic entropy and alignment scores. Its user access changes the information available relative to category-only VLFM.

## Results and independent verdict

PAPER EXPERIMENT: Table 2 seen-category SR/SPL/success-conditioned question count is 7.42/2.92/1.67; synonym 14.38/7.99/1.36; unseen 6.67/2.30/1.13. PSL has better seen SR 8.78 and OVON better synonym SR 15.88. The method performs best on unseen instances among reported baselines, but absolute success remains low. Dialogue ablation reports SR 9.21→14.47 and question count 3.57→1.68 when combining self-questioning and candidate skipping. Question count is calculated on successful episodes only, so it does not measure total failed-run burden. The 20-person human study uses 40 episodes preselected to have detectable targets; 42.5% human and simulator success cannot be generalized to all 1,649 episodes. IDK-VQA has 502 questions over 102 images with three annotators; entropy improves abstention utility relative to max probability in that benchmark, without proving calibrated uncertainty under navigation shift.

SUPPLEMENT CHECK: the target image is available to the user, not the agent (p. 1). The human study covers 20 participants and 40 episodes, two per participant; failure to find a significant human-versus-simulated-user difference is not an equivalence proof (p. 5). Three component times on an RTX 4090 are reported as 11.3, 8.29 and 6.13 seconds over 20 episodes (p. 4), without an executed timing reproduction here. IDK-VQA uses 102 images and three annotators (p. 6); robustness around a tuned threshold does not establish probability calibration. The prompts request information gain, but the inspected routine does not explicitly optimize a probabilistic information-gain objective. Dialogues are capped at four rounds (p. 9).

## Implementation and reproducibility

INSPECTED CODE: vlfm/brain/llm_brain_history.py:159–183 computes normalized entropy over exactly three yes/no/uncertain token probabilities, with an optional offset, and labels responses using a threshold. The object policy passes tau 0.75 and offset 0.05 for false-positive filtering, rejecting no, uncertain and do-not-know responses (base_objectnav_policy.py:408–446). The LLM alignment/question routine parses YAML similarity_score and questions, conditioned on stored facts (llm_brain_history.py:234–259). This verifies model-score thresholding and prompted questions rather than a learned or conformally calibrated error probability. The released policy inherits VLFM and pretrained PointNav; training-free refers to AIUTA’s added decision logic. API calls and Habitat navigation were not executed; model/version/prompt dependence remains a reproduction prerequisite.

Inspected release reference: [https://github.com/intelligolabs/CoIN at b750a1a6cf6485e513aa4b24d6942fea56f50a51](https://github.com/intelligolabs/CoIN/tree/b750a1a6cf6485e513aa4b24d6942fea56f50a51). File-level inspection is restricted to the paths listed below.

- vlfm/brain/llm_brain_history.py:159–183,234–259
- vlfm/policy/base_objectnav_policy.py:408–446

## Specific bottlenecks

**aiuta-verify: Uncertainty-filtered dialogue does not ensure correct instance recognition or stop decisions.**

Conditions: Unseen object categories, poor views, misleading visual captions, or a confident visual mistake. Cause: Entropy measures the model score distribution and alignment is another model judgment; neither is a calibrated truth test. Evidence: PAPER EXPERIMENT: Table 2 low absolute SR; METHOD: heuristic thresholds.

Type: methodological. Status: Partially addressed. Confidence: High within the specified evidence scope. Follow-ups: ['KnowNo', 'Explore until Confident']

**aiuta-cost: Reported question efficiency excludes failed episodes.**

Conditions: Comparing user effort when methods have different success rates. Cause: The NQ denominator includes only successful episodes. Evidence: METRIC DEFINITION: experiments section.

Type: insufficiently tested. Status: Uncertain because evidence is insufficient. Confidence: High within the specified evidence scope. Follow-ups: ['KnowNo']

## Sources and limitations

- [Supplied main paper](https://openaccess.thecvf.com/content/ICCV2025/html/Taioli_Collaborative_Instance_Object_Navigation_Leveraging_Uncertainty-Awareness_to_Minimize_Human-Agent_Dialogues_ICCV_2025_paper.html)
- [Official ICCV project](https://intelligolabs.github.io/CoIN/)
- [Official repository](https://github.com/intelligolabs/CoIN)
- [Official supplementary or complete author version; reviewed scope listed in associated material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Taioli_Collaborative_Instance_Object_ICCV_2025_supplemental.pdf)
- Supplement sections A–J and relevant source reviewed. Human truthfulness is assumed; small human comparison is not an equivalence guarantee. No benchmark executed.
- Associated material: {"file": "evidence/04/supplements/aiuta_supp.txt", "pages_read": "1–10", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
