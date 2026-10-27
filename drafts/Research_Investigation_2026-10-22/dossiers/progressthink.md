# Progress-Think

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation**. Shuo Wang, Yucheng Wang, Guoxin Lian, Yongcai Wang, Maiyue Chen, Kaihui Wang, Bo Zhang, Zhizhong Su, Yutian Zhou, Wanting Li, Deying Li, Zhaoxin Fan CVPR 2026, identified by the main-paper watermark (2026).

Primary record: https://arxiv.org/abs/2511.17097



Supplied main file: [Progress-Think.pdf](../../Paper/Progress-Think.pdf). Pages actually reviewed: 1,3–8. Associated local versions are recorded below with SHA256 identifiers. Exact duplicate bytes were not reread. Distinct alternatives were checked for the relevant method/results; their complete bibliographies were not separately reread.

Associated material:

- Progress-Think (alternate).pdf; reviewed pages/status: 4–7; SHA256 `f2d32f86f4e6444132f885f3f63c42516f7265a70e097a9930a39ced1b5918c9`

## Problem, motivation, and operating assumptions

Predict which instruction prefix has been completed rather than just numeric distance progress. Uses monocular CE and two NVILA 2 B models, weak trajectory prefix supervision and action correction.

## Method, training, inference, and equations

Section 3 uses two NVILA-2B models: a progress reasoner and an action policy. Let I be an instruction with |I| tokens and O_t the observed history. The reasoner predicts completed instruction text. For each candidate prefix length k∈{1,…,|I|}, Eq.2 defines p(k) proportional to exp(−CE_k/τ), where CE_k compares the predicted text with I's first k tokens and τ>0 is a temperature. The expected prefix length is k̂_t=Σ_k k p(k). The prefix loss is −τ logΣ_k exp(−CE_k/τ). A second loss, max(0,k̂_early−k̂_late), penalizes decreases between ordered observations (Eqs 3–5). The paper does not resolve whether prefix CE is summed or averaged, which affects length bias; no zero-length prefix is listed. A constant estimate satisfies monotonicity, so these constraints alone do not identify true progress.

Stage 2 freezes the progress model and trains the policy on the full instruction, images and predicted progress, using cross-entropy for the next three actions (Eqs 6–7). Stage 3 samples four progress/action alternatives and jointly updates both models with a clipped GRPO objective. The action reward is the number of consecutive correct actions before the first error; the format reward checks syntax. The progress-length reward gives 1 to any text no longer than I and penalizes only excess length (Eqs 8–11). Thus a short but wrong progress statement receives the full length reward. Eq.14 multiplies progress-model and action-model probability ratios, forming a joint importance ratio; zero-variance group handling is not specified in the displayed advantage equation.

Training uses R2R/RxR/ScaleVLN state-action pairs plus 500K DAgger examples; the wording around1.2M examples per dataset is ambiguous. Stages 1–2 run one epoch at 1e-5; stage 3 runs3,000 updates at 1e-6, clipping 0.28 and KL coefficient 0. Eight H20 GPUs are used for roughly 8+60+60 hours. At inference the policy predicts and executes three parameterized actions (25/50/75 cm forward or 15/30/45° turns). Completed-history progress can be monotone, but the robot's current stage can regress after an error; treating the former as sufficient for action is an assumption.

## Novelty and closest predecessors

Differentiable latent prefix alignment plus separately trained semantic progress and joint progress/action RL. Semantic stage alignment overlaps IDEAL-VLN / AwareVLN ; distinguishing feature weak annotation and prefix structural objective. This is not training-free or zero external pretraining:0 K external refers additional navigation fine-tuning data, while NVILA pretraining is external.

## Results and independent verdict

Table 1 R2R 60.1 SR/53.6 SPL ; Table 2 R2R -only RxR 27.5/22.7 distinct model/protocol. Ablation baseline 33/28.3→prefix 39.4/34.6→mono 41.4/36.5→action+format RL 40/35→length reward 43.8/38.5(Table 4): RL without length degrades. Ablation full 43.8 not main 60.1, so reduced setting must be identified before comparisons. Table 6 successful-episode-only time 36.60 s(2 B+2 B)vs NaVILA 56.84 s 8 B, not matched model size or unconditional runtime. Monotonic loss improves reported performance, not proven progress identifiability.

## Implementation audit

INSPECTED CODE. The official RoboOrchardLab snapshot is shared with Aux-Think. progress_model.py and action_model.py each subclass the common MonoDream implementation and load separate checkpoints. projects/progress_think/infer.py:193–205 initializes both pipelines;306–326 first predicts partial_instruction from images and then supplies that text, the full instruction and images to the action model. Thus the released inference path exposes the two-stage text interface directly. analyze_results.py:47–64 averages stored Habitat success/SPL metrics, replacing NaN/Inf numeric values with zero.

The inspected release provides inference classes, processors and an evaluator wrapper. The paper-specific latent-prefix alignment and joint-GRPO training objectives are not established by these thin classes, and the project README documents inference rather than the complete training recipe. Those training mechanisms remain paper evidence, not verified code behavior. No model or Habitat evaluation was executed.

Repository: https://github.com/HorizonRobotics/RoboOrchardLab. Inspected snapshot commit: `48ac43987df9ac1ba40fef72dd7a2b4679c592b4`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [robo_orchard_lab/models/progress_think/progress_model.py:32-56](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/robo_orchard_lab/models/progress_think/progress_model.py)
- [robo_orchard_lab/models/progress_think/action_model.py:32-56](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/robo_orchard_lab/models/progress_think/action_model.py)
- [projects/progress_think/infer.py:193-205,306-326](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/projects/progress_think/infer.py)
- [projects/progress_think/analyze_results.py:27-66](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/projects/progress_think/analyze_results.py)
- [projects/progress_think/README.md:38-119](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/projects/progress_think/README.md)

## Bottlenecks and follow-up chain

- **PROGRESS-1: Weak prefix objectives admit degenerate or semantically incorrect progress** Conditions: Repeated clauses, loops, partial completion, recovery Cause: Monotonic it y and length bounds constrain form but not observational alignment; action reward can ignore truthful progress Evidence: Eqs 2–5,8–14; Table 4 RL degradation without length Type: Methodological / evaluation. Follow-ups:  IDEAL-VLN adds privileged alignment thought correction; AwareVLN sparse oracle thoughts; faithful progress under recovery still uncertain. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

The paper objective and the released inference interfaces were inspected. Exact prefix-CE reduction, empty-prefix treatment and joint-GRPO training implementation were not established by the thin released wrappers. No benchmark was executed.

