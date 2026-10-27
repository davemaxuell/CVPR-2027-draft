# Additional preparation sources

Assignment 07, cutoff **2026-10-22**. These are scoped overlap checks for proposal construction, not replacements for independent assignment 09. Downloaded PDFs, page-labelled text and SHA256 provenance are in [the manifest](additional_source_manifest.json). No code for these additional sources was inspected or executed. The supplied corpus and follow-up assignments supply the main evidentiary base.

## 07-planning_sufficiency

Rongzhe Wei, Hans Hao-Hsun Hsu, Peizhi Niu, Yifan Li, Pan Li. **What Must a World Model Distinguish for Planning?** arXiv 2609.33030v1, 26 September 2026. [Primary paper](https://arxiv.org/abs/2609.33030v1). Read PDF pp.1,3–8; remaining results/appendices/proofs were not fully audited.

Sections 3–6 distinguish preservation of dynamics, candidate costs and selected actions. Their studies separate final selection from candidate search, compare compact value/planner/hybrid supervision, and vary where a changing query enters proposals and prediction. This is direct prior art against claiming that decision-relevant compression or fixed-candidate regret analysis is new. P3 therefore requires a narrower ordered-event target and must compare an equivalent task-specific future-question baseline. No independent verification of the paper's theorems or benchmark results is claimed.

## 07-compactnav

Yihao Wu, Chenyi Xu, Liqi Yan, Chenhuan Cai, Geyong Min, Bin Lin, Fangli Guan, Jianhui Zhang, Pan Li. **Towards Minimal Sufficient Representation for Vision-Language Navigation**, reviewed PDF arXiv 2607.23181v1, 25 July 2026. [Primary record](https://arxiv.org/abs/2607.23181v1). Read PDF pp.1,3–6. The abstract-page search title used “Dual-Brain”/BrainNav, while the downloaded v1 PDF/HTML uses CompactNav; retain this version distinction.

Section 3 cascades text-then-vision attention, predicts CP-factorized latent representations, and uses an action-conditioned prior/posterior world model with sparsity and policy losses. This already occupies an instruction-guided low-rank world-representation proposal. Its sufficiency/robustness wording was not adopted as a proved guarantee; the appendix proofs, complete experiments and code remain outside this scoped check.

## 07-semantic_world_models

Jacob Berg, Chuning Zhu, Yanda Bao, Ishan Durugkar, Abhishek Gupta. **Semantic World Models**, reviewed arXiv 2510.19818v1, 22 October 2025. [Primary paper](https://arxiv.org/abs/2510.19818v1), [author project](https://weirdlabuw.github.io/swm/). Read PDF pp.1–5.

Section 3 constructs state/action/future-question/answer examples using future-state supervision. An action-conditioned PaliGemma model learns answer-token cross-entropy; desired-answer likelihoods score action sequences, with sampling/gradient planning discussed. Thus semantic future QA and planner-oriented semantic prediction are established. P3 must test whether joint temporal events supply anything beyond this interface with matched supervision and compute. Quantitative experiments and repository behavior were not audited here.

## 07-ground_compose_reinforce

Andrew Li, Toryn Klassen, Andrew Wang, Parand Alamdari, Sheila McIlraith. **Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data**, reviewed arXiv 2507.10741v2, 26 October 2025; arXiv record labels NeurIPS 2025. [Primary paper](https://arxiv.org/abs/2507.10741v2). Read PDF pp.1,3–6.

Sections 3–6 formalize reward machines, grounded propositional labels and policies over physical/instruction state. Learned grounding supplies self-generated reward and compositional shaping. This directly precedes P1's generic automaton and state-estimation ingredients. P1's potential contribution must therefore concern independently validated off-path teacher labels and their controlled effect, not inventing task automata or language-grounded state. Remaining experiments, extensive appendices and implementation were not audited by assignment 07.

## 07-negative_information

Jan Hoffmann, Michael Spranger, Daniel Göhring, Matthias Jüngel. **Negative Information and Proprioception in Monte Carlo Self-Localization for a 4-Legged Robots**, author-hosted paper, 2005 source path; exact workshop record not independently established. [Primary full text](https://faculty.kfupm.edu.sa/COE/mayez/ps-coe499/core/Vision-Localization/2005-08-ijcaj05ws-HoffmannSprangerGoehringJuengel-NegativeInformation.pdf). Read pp.1,5–6.

Section 4 uses expected non-detection conditioned on pose, camera field of view and occlusion in a Bayesian localization update. It explicitly distinguishes absence from an inability to sense a target. P4 cannot claim visibility-aware negative evidence or comparison of intended/observed motion as new. The narrower candidate is learned attribution of terminal navigation feedback to persistent claim types, with simple non-writing baselines. No historical implementation or full experiment reproduction was attempted.

## Other primary records consulted

- **Dataset Cartography** (Swayamdipta et al., EMNLP 2020): [proceedings record](https://aclanthology.org/2020.emnlp-main.746/). Abstract/metadata only. Training-dynamics data-quality selection is a mandatory P2 comparison; no full-paper review claimed.
- **Learning to Reweight Examples for Robust Deep Learning** (Ren et al., ICML 2018): [proceedings record](https://proceedings.mlr.press/v80/ren18a.html), [author method page](https://www.mengyeren.com/research/2018/learning-to-reweight-examples-for-robust-deep-learning/). Primary abstract and displayed method/experiment portions consulted. Clean-validation gradient reweighting precedes a generic quality/difficulty weighting claim; code and full appendix were not audited.

These sources constrain novelty. They are not evidence that any of the four proposed interventions improves navigation.
