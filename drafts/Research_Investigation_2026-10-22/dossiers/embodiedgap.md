# VLN-PE

Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities

Liuyi Wang; Xinyuan Xia; Hui Zhao; Hanqing Wang; Tai Wang; Yilun Chen; Chengju Liu; Qijun Chen; Jiangmiao Pang. ICCV 2025. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [VLN-PE.pdf](../../Paper/VLN-PE.pdf). Pages reviewed: ['3–8'].

## Problem and operating conditions

VLN-CE movement abstractions omit locomotion, falls, controller effects, camera differences, and lighting. VLN-PE evaluates ego-centric agents in physically simulated humanoid, quadruped, and wheeled embodiments using GRUtopia/Isaac Sim.

## Mechanism, training, and inference

Ninety MP3D scenes are converted to USD and mesh holes manually repaired. Stairs are removed because controllers cannot reliably negotiate them, leaving 8,679/658/1,347 R2R train/seen/unseen episodes. Ten synthetic GRScenes and one Gaussian-splat laboratory add rendering/domain conditions. Baselines include Seq2Seq, CMA, NaVid, an adapted VLMaps with frontier exploration, and the new RDP. RDP encodes RGB/text with LongCLIP and depth with ResNet-50, uses cross-modal attention and GRU memory, and predicts eight relative (delta x, delta y, delta yaw) actions with a Transformer DDPM head. Noise-prediction MSE plus stop-progress MSE weighted 10 trains it. A separate stop head is necessary because diffusion actions do not reliably imply stopping. Metrics add fall and stuck rates to navigation success/path efficiency; the map-based comparison uses only 200 sampled episodes.

SUPPLEMENT CHECK: Appendix A supplies the RDP details: LongCLIP RGB/text features plus depth and a point goal, five RGB tokens of width 512, GRU history, four previous actions and relative coordinates. Diffusion predicts eight future actions and executes four; a heuristic stops when action magnitudes are below 0.1 or progress exceeds 0.8. Training is reported as about two days on four RTX 4090s, batch eight and lr 1e-4. Fall/stuck definitions and controller choices are operational benchmark components, not abstract navigation correctness.

## Novelty and closest predecessors

Primarily a platform and controlled transfer study, plus a recurrent diffusion-action baseline. Its physical simulation is more realistic than graph navigation or idealized CE, but does not itself prove real-robot reliability. Data collected with different controllers and embodiments test concrete sources of transfer error.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 NaVid zero-shot H1 unseen SR/SPL 22.42/18.58; RDP trained in VLN-PE 21.98/16.44; fine-tuned CMA 23.31/19.66. These compare different data/parameter budgets. Table 4 controller-consistent CMA training/evaluation gives 18.78 SR versus 13.51 when trained without locomotion but tested with it; falls remain 18.63%. Table 5 adding all embodiments and CE data raises H1 SR 18.78→26.87 but confounds diversity and data amount. Table 6 NaVid SR falls 22.42→9.95 under dim light, while RGB-D RDP is stable near 22%; modality differs. Authors report NaVid rotating over 25 steps near goals in 70% of episodes. Gaussian-lab NaVid SR 5.81 is an observed failure, while rendering-noise attribution is their hypothesis. The study exposes concrete embodiment/stopping fragility while excluding stairs.

SUPPLEMENT CHECK: Appendix C, p. 4, reports a 14-episode Unitree Go2 CMA test improving SR from 7.14% to 28.57%, with a custom STOP probability threshold above 1e-4 because the transferred policy did not stop robustly. This small physical test is not real-world validation of RDP. Evaluation is restricted to egocentric methods, and the Jetbot camera is raised in simulation (Fig. 14). These conditions matter when interpreting embodiment generalization.

## Implementation and reproducibility

INSPECTED CODE: The author project links the maintained InternNav repository; the inspected pin is a current framework snapshot, not a verified historical paper release. internnav/trainer/rdp_trainer.py:109–145 implements diffusion noise/sample MSE plus progress and stop-progress MSE with configured stop loss weight. internnav/agent/rdp_agent.py:300–352 uses no_grad recurrent denoising, caches an action chunk, pops actions sequentially and returns evaluator action records. This supports chunked diffusion control and explicit stopping supervision. The framework also contains newer unrelated agents; no claims about those were inferred from this inspection. Exact paper checkpoint/config equivalence, controller latency and hardware transfer were not executed.

Inspected release reference: [https://github.com/InternRobotics/InternNav at 7a5c62400ac45b313d9b709c740b64191556a242](https://github.com/InternRobotics/InternNav/tree/7a5c62400ac45b313d9b709c740b64191556a242). File-level inspection is restricted to the paths listed below.

- internnav/trainer/rdp_trainer.py:109–145
- internnav/agent/rdp_agent.py:300–352
- README.md and author project link

## Specific bottlenecks

**vlnpe-physical: Navigation success does not transfer across controller and sensor conditions without substantial degradation.**

Conditions: Legged locomotion, changed camera height, dim lighting, or Gaussian-splat rendering. Cause: Training observations and action consequences differ; depth availability and control stability vary by baseline. Evidence: PAPER EXPERIMENT: Tables 4–6.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['NaVILA', 'HumanoidVLN']

**vlnpe-stairs: Stair traversal remains outside the evaluated platform split.**

Conditions: Multi-floor routes requiring reliable legged stair control. Cause: Controller failure led to filtering stair episodes. Evidence: AUTHOR ACKNOWLEDGEMENT: §3 datasets.

Type: implementation-specific. Status: Still unresolved in the reviewed literature. Confidence: High within the stated experimental scope. Follow-ups: ['HumanoidVLN']

## Sources and limitations

- [Supplied main paper](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Wang_Rethinking_the_Embodied_Gap_in_Vision-and-Language_Navigation_A_Holistic_Study_ICCV_2025_paper.pdf)
- [Official project](https://crystalsixone.github.io/vln_pe.github.io/)
- [Author manuscript](https://arxiv.org/abs/2507.13019)
- [Official supplementary or complete author version; reviewed scope listed in associated material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wang_Rethinking_the_Embodied_ICCV_2025_supplemental.pdf)
- Supplement and current author-linked InternNav source reviewed; the maintained commit is not verified as the historical paper code. Simulator/controller installation and robot trials were not executed.
- Associated material: {"file": "evidence/04/supplements/embodiedgap_supp.txt", "pages_read": "1–7", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
