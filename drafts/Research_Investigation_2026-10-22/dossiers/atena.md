# ATENA

Active Test-time Vision-Language Navigation

Heeju Ko; Sung June Kim; Gyeongrok Oh; Jeongyoon Yoon; Honglak Lee; Sujin Jang; Seungryong Kim; Sangpil Kim. NeurIPS 2025 (volume 38; author repository BibTeX uses 2026 publication year). Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [ATENA.pdf](../../Paper/ATENA.pdf). Pages reviewed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20].

## Problem and operating conditions

ATENA adapts pretrained navigation policies after episodes while limiting external binary-feedback requests. Its input is a test stream plus occasional outcome labels, and it outputs both a navigation policy update and a predicted success/failure label. Low-entropy episodes use self-predicted labels rather than human labels. It is not a no-training navigation system, and its reported human oracle is simulated from benchmark outcomes in released validation code.

## Mechanism, training, and inference

For chosen action a*, form q=λ onehot(a*)+(1−λ)π. Minimize episode-average entropy H(q) for success and maximize it for failure (Eqs 1–4). The one-hot term sharpens the selected action, but its choice is treated as fixed for differentiation. An episode’s average original action entropy selects the feedback source. High entropy requests an external label; low entropy uses sigmoid(fφ(mean_t s_t))>.5. A BCE loss trains this outcome head on either true feedback or its own hard pseudo-label; joint policy/head adaptation occurs after termination (Eqs 5–8).

The source DUET implementation pools concatenated global/local current-state features (width 2×768), computes mixture entropy over nonzero candidates, and uses outcome-dependent sign. Released REVERIE config sets batch 1, λ=.5, threshold .1, external/self learning rates 5e−6/1e−7 and BCE weight .25. AdamW is recreated every episode, so its momentum does not persist. Inference still uses DUET’s graph, including return to the node with best cached STOP score.

SupplementA.7 correctly derives a (1−λ) chain-rule factor but then compares only log-probability gaps when claiming stronger updates. A larger log-ratio gap alone does not imply a larger parameter gradient. At λ→1 the gradient vanishes; isolated binary examples demonstrate nonuniform strength. Entropy also depends on candidate count and is not itself a calibrated success probability. This narrows the mechanism’s interpretation without rejecting its empirical ablations.

## Novelty and closest predecessors

Compared with FeedTTA’s feedback REINFORCE, ATENA uses signed mixture entropy and selective feedback/self-labeling. Compared with ordinary entropy TTA, the binary outcome chooses whether to sharpen or flatten. Generic outcome prediction and uncertainty-based query selection are established components; the contribution is their navigation-specific combination and mixture objective.

## Results and independent verdict

PAPER EXPERIMENT: Table 1/p7 reports DUET unseen REVERIE SR 46.98→68.11/SPL 33.73→45.82, whereas hidden test SR 52.51→54.28. SupplementA.2 says hidden test uses GPT-4o with FeedTTA prompts, changing feedback quality. Table 4 controls feedback access and gives MEO without SAL 63.70 SR versus active Tent 55.69; Table 5 adds SAL to 68.11. SupplementA.4’s outcome-head confusion matrix gives 908 true positives,216 false positives,98 false negatives and 541 true negatives:82.19% accuracy, but positive predictive value is about 80.8%, not 90.26% (the latter is recall).

Main Table 1’s HAMT+ATENA seen row prints OSR 52.92 below SR 57.34, incompatible with the stated shared success criteria; treat as an unresolved table issue, not corrected data. SupplementTable 8 also prints RGSPL 32.26 where main Table 1 gives 31.26. Smaller gains on R2R-CE (ETPNav57→58 SR,49→49 SPL) and IVLN-CE (35→36 SR,32→34 SPL) temper broad generalization claims. Authors acknowledge lack of step-level credit assignment. Verdict: useful feedback-efficient adaptation in the tested streams; confident self-label errors and dependence on the oracle remain.

## Implementation and reproducibility

INSPECTED CODE: Official kuai-lab/NeurIPS25_att_vln includes DUET and ETPNav variants. In ATENA_DUET/map_nav_src/reverie/agent_obj_atena.py:480–490, original entropy is logged and q is formed from chosen action plus nav_probs. agent_base_atena.py:153–177 pools stored features, obtains an outcome logit, chooses external/self feedback and backpropagates signed entropy plus BCE. The external branch reads env.gt_trajs and obj2vps to test whether the final path node belongs to goal viewpoints; it does not implement a real human interface or the hidden-test GPT-4o oracle in that path. The self branch trains against its own thresholded label.

The model adds a 2D-input self_prediction_head at vilmodel.py:680; scripts/run_reverie_atena.sh fixes the concrete parameters above. The policy is evaluated in eval mode while gradients are enabled, which is appropriate for differentiable adaptation without dropout. Released assets depend on DUET/ETPNav data and checkpoints. A paper-equivalent hidden-test reproduction additionally needs the unpublished oracle integration and exact split-specific configurations. No project code or navigation benchmark was executed.

Inspected release reference: [https://github.com/kuai-lab/NeurIPS25_att_vln at f6903760c2dccdc9fb3524aa3b83f62ee9331e35](https://github.com/kuai-lab/NeurIPS25_att_vln/tree/f6903760c2dccdc9fb3524aa3b83f62ee9331e35). File-level inspection is restricted to the paths listed below.

- ATENA_DUET/map_nav_src/reverie/agent_base_atena.py:105–190
- ATENA_DUET/map_nav_src/reverie/agent_obj_atena.py:323–380,445–515
- ATENA_DUET/map_nav_src/models/vilmodel.py:660–700
- ATENA_DUET/map_nav_src/scripts/run_reverie_atena.sh

## Specific bottlenecks

**atena-B1: Low-entropy failures may be reinforced by the self-labeling feedback loop.**

Conditions: A confident but wrong policy episode is routed away from external feedback. Cause: The outcome head is trained partly on its own hard prediction and receives only pooled state information. Evidence: Eqs 5–8; source self-label branch; supplementary confusion matrix contains false positive outcomes.

Type: methodological. Status: Partially addressed. Confidence: High for mechanism and remaining errors; cascade frequency unmeasured.. Follow-ups: Future work needs independently validated outcome evidence or selective calibration under policy drift; generic self-verification alone does not resolve this.

**atena-B2: Episode-level outcome feedback cannot localize which action or instruction clause failed.**

Conditions: Long routes with many correct actions and one decisive wrong action. Cause: The same success sign weights entropy at every step, without step-level blame. Evidence: Eq 3 and main limitations, PDF p9.

Type: methodological. Status: Still unresolved in the reviewed literature. Confidence: High for credit-assignment ambiguity; bounded to this method.. Follow-ups: Step/segment feedback and retrospective correction are relevant; their feedback budgets and oracle access must be matched.

## Sources and limitations

- [Supplied main paper](https://proceedings.nips.cc/paper_files/paper/2025/hash/3f510f82323c1293ae3e343893dd77b1-Abstract-Conference.html)
- [Official project and repository link](https://kuai-lab.github.io/neurips2025atena/)
- [Official supplementary appendix,7 PDF pages read](https://proceedings.nips.cc/paper_files/paper/2025/file/3f510f82323c1293ae3e343893dd77b1-Supplemental-Conference.zip)
- No benchmark executed. The separate seven-page supplement was reviewed. Hidden-test oracle access and all-backbone equivalence are not established; reported table inconsistencies remain unresolved.
- Associated material: Official supplementary 5981_SUPPLE.pdf, all7 pages read; saved under evidence/04/supplements.
