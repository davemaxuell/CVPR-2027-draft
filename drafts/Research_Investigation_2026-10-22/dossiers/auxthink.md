# Aux-Think

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation**. Shuo Wang, Yongcai Wang, Wanting Li, Xudong Cai, Yucheng Wang, Maiyue Chen, Kaihui Wang, Zhizhong Su, Deying Li, Zhaoxin Fan NeurIPS 2025 (2025).

Primary record: https://proceedings.neurips.cc/paper_files/paper/2025/hash/2c90bf5bffe6497730ecade3a3a37458-Abstract-Conference.html



Supplied main file: [Aux-Think.pdf](../../Paper/Aux-Think.pdf). Pages actually reviewed: 1-10, 13-16. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Test whether explicit generated chain-of-thought helps a monocular continuous navigator under limited navigation supervision. The paper distinguishes learning from reasoning traces from generating them during deployment. Inputs are the instruction, current RGB frame, and eight uniformly sampled history frames.

## Method, training, inference, and equations

PAPER METHOD (§3): Qwen 2.5-VL-72 B annotates 320 K oracle-trajectory states using observations, instruction, and known correct action; these are rationalizations conditioned on the answer. NVILA-lite 8 B ( SigLIP , projector, Qwen 2) is supervised on three prompt-switched tasks: action sequence prediction, CoT generation, and reconstruction of the trajectory's instruction. Eqs.2–5 sum negative log likelihood for these tasks. CoT never becomes a required prefix for the action task. During inference only action prediction is activated; predict a short action horizon and execute its first action before observing again. Horizon three is best in Table 5. Forward increments are 25/50/75 cm and turns 15/30/45°. Training uses eight H20 GPUs, one epoch about 60 h, learning rate 10^-5; the larger variant adds RxR , DAgger , and web data.

DEDUCTION: the action objective's summation k=0..n in Eq.4 is off by one against text saying n actions. More substantively, post-action tokens cannot causally alter an earlier action token within the same causal inference pass; their training gradients and later-history conditioning can, so the proposed 'hidden state interference' explanation is a hypothesis rather than established mechanism.

## Novelty and closest predecessors

A controlled comparison of No/Pre/Post/ Aux-Think plus navigation CoT data is the contribution. Auxiliary reasoning itself has predecessors (AuxRN and language multi task learning). The distinctive deployment choice decouples reasoning generation from action context, contrasting AdaNav / AwareVLN sparse explicit reasoning. It blocks a broad claim that 'CoT as train-only auxiliary loss' is new.

## Results and independent verdict

PAPER EXPERIMENT: Table 3 p.9 reports No-Think 35.1 SR/30.2 SPL /1.25 s; Pre-Think 11.4/8.6/30.62 s; Post-Think 29.0/23.8/28.97 s; Aux CoT 41.3/35.8/1.25 s. Table 4's full three-task model reaches 46.0/40.5; enlarged-data Table 1 reaches 54.8 SR/46.9 SPL on R2R . RxR Table 2 gives 52.2 SR but 40.2 SPL , below NaVILA 44.0 SPL , so success gains trade against path efficiency. Introductory Figure 1 and Table 3 use different variant values; cite the specific table. Results establish the tested data/architecture setting, not universal harm from inference reasoning.

## Implementation audit

INSPECTED CODE. RoboOrchardLab's robo_orchard_lab/models/aux_think/model.py:72–110 loads the VILA model and forwards training inputs or calls generate_content for inference; processor.py:33–156 loads RGB PIL images, formats navigation prompts and wraps generated text. These are usable integration components, but this wrapper does not itself implement or prove the paper-specific auxiliary-CoT training objective. projects/aux_think/README.md:33,55–90 links a separate markinruc/VILA training repository and markinruc/VLN_CE evaluation environment. Thus the pinned RoboOrchard snapshot covers deployment wiring, not the complete original training/evaluation provenance. Training-only CoT is a paper mechanism; it must not be inferred from this wrapper's forward method. No model, training script or simulator was executed.

The linked training repository was also inspected at markinruc/VILA commit f5853256e9e2e796ee9245c72a17a6f853743d45: llava/data/dataset_impl/llava_cot.py:32–99,130–133 substitutes annotated CoT conversations, while llava/model/language_model/llava_llama.py:122–149 forwards embeddings/labels and exposes a generic optional soft-token loss. These files do not by themselves establish the full Aux-Think-specific objective or original data pipeline. The linked markinruc/VLN_CE commit3a6259b2336b1164ee6f25a43a13f52041cbfdff, habitat_extensions/config/vlnce_task_aux_think_r2r.yaml:1–49, specifies RGB640×480,15° turns,0.25m forward steps, sliding enabled and 3m SR/SPL. Its task sensors also register oracle-progress/shortest-path information for the environment; this registration alone does not show that the policy consumes those fields. This is a partial training/evaluation provenance audit, not a claim of exact reproduction.

Pinned related sources: [CoT dataset transform](https://github.com/markinruc/VILA/blob/f5853256e9e2e796ee9245c72a17a6f853743d45/llava/data/dataset_impl/llava_cot.py), [language-model forward](https://github.com/markinruc/VILA/blob/f5853256e9e2e796ee9245c72a17a6f853743d45/llava/model/language_model/llava_llama.py), [Habitat task configuration](https://github.com/markinruc/VLN_CE/blob/3a6259b2336b1164ee6f25a43a13f52041cbfdff/habitat_extensions/config/vlnce_task_aux_think_r2r.yaml).

Repository: https://github.com/HorizonRobotics/RoboOrchardLab. Inspected snapshot commit: `48ac43987df9ac1ba40fef72dd7a2b4679c592b4`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [robo_orchard_lab/models/aux_think/model.py:32-40,72-110](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/robo_orchard_lab/models/aux_think/model.py)
- [robo_orchard_lab/models/aux_think/processor.py:33-156](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/robo_orchard_lab/models/aux_think/processor.py)
- [projects/aux_think/README.md:1-110](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/projects/aux_think/README.md)

## Bottlenecks and follow-up chain

- **03-auxthink-causal: CoT quality and correctness are not separately identified from extra supervision and action-conditioned annotation** Conditions: Out-of-distribution states and long trajectories Cause: Oracle-only traces encode known answers; auxiliary tasks may regularize without learning the claimed reasoning process Evidence: §3.2 annotation; §3.3 failure analysis; Tables 3–5 Type: insufficiently tested. Follow-ups:  CorrectNav adds error-state supervision; AwareVLN adds key-event/correction traces; Progress-Think aligns semantic progress.. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

No benchmark execution. Checklist/reference pages are not additional empirical evidence; remaining appendix generation prompts were not fully audited.

- https://arxiv.org/abs/2508.10416
- https://gwxuan.github.io/AwareVLN/
- https://horizonrobotics.github.io/robot_lab/progress-think
