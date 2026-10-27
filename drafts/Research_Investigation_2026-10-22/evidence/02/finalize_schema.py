"""Paper-specific bottleneck rows, preserving evidence boundaries and scoped statuses."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
B = {}
UN = 'Uncertain because evidence is insufficient'
ST = 'Still unresolved in the reviewed literature'
PA = 'Partially addressed'
SH = 'Shifted into another cost or limitation'
RE = 'Resolved within a clearly specified setting'

def row(p, name, description, conditions, cause, evidence, kind, followups, status, confidence='Medium'):
    B.setdefault(p, []).append(dict(id='02-'+p+'-'+name, description=description,
        conditions=conditions, cause=cause, evidence=evidence, kind=kind,
        followups=followups, status=status, confidence=confidence))

row('chasing_ghosts','control','Improved instruction-goal localization does not translate into competitive complete navigation.',
    'R2R with RGB-D panoramas, supplied poses and discrete reachable graph nodes.',
    'The ghost endpoint belief is consumed by a separate local graph-node controller; localization quality alone does not solve route selection and stopping.',
    'PAPER EXPERIMENT: Tables 1–2, PDF pp.7–8: localization 44.9% versus 32.2%, but unseen navigation SR/SPL 35/31. DEDUCTION: interface bottleneck rather than measured causal attribution.',
    'Methodological','DUET/topological planners change control and representation; HNR/SALI add predicted content. No matched belief-versus-controller intervention was reviewed.',UN)
row('chasing_ghosts','belief','Normalized ghost beliefs have no demonstrated calibration under pose error or ambiguous instructions.',
    'Variable-length instructions, incorrect pose/depth or multiple plausible instruction routes.',
    'Learned motion/observation likelihoods, a fixed six-step ghost rollout, and normalized products do not establish valid likelihood independence or coverage.',
    'DEDUCTION: Sections 3–4, PDF pp.3–6; tracker/modules/navigators.py:142–189,279–286 and tracker/modules/loss.py:20–36 at the dossier commit.',
    'Insufficiently tested','Later map-based and imagined-view papers in this assignment do not report matched posterior-coverage tests.',UN)

row('cotvla','subgoal','Generated visual subgoals can fail to specify a useful reachable action target.',
    'Novel manipulation tasks requiring visual subgoals outside fine-tuning support.',
    'Language/image-conditioned generation does not explicitly enforce the action-conditioned reachability of the target.',
    'PAPER EXPERIMENT: Table 3, PDF p.8, five trials on each of two novel tasks: generated goals 20%/0% versus demonstration goals 60%/40%. Small sample and privileged targets limit attribution.',
    'Methodological','FantasyVLN and WorldVLN use different navigation tasks; neither supplies a matched manipulation reachability test.',UN)
row('cotvla','latency','Visual-token generation slows decisions and action chunks postpone observation feedback.',
    'Ten-action execution chunks, especially when the environment changes within a chunk.',
    'Autoregressive generation of 256 visual tokens precedes action prediction; the policy obtains a new real image at chunk boundaries.',
    'AUTHOR ACKNOWLEDGEMENT: Section 5, PDF p.8, approximately sevenfold slowdown at chunk length ten and missing high-frequency feedback.',
    'Methodological','FantasyVLN removes explicit visual CoT at deployment and reports faster actions (Table 2); this shifts burden into training and does not verify within-chunk control in the same task.',SH)

row('cvaml','estimator','Finite-sample variance correction can retain bias and raise sampling costs.',
    'Stochastic transitions with a finite number of next-state samples.',
    'The inspected estimator uses population variance divided by N; with iid samples and fixed value, that differs from unbiased sample variance divided by N. Sample-dependent action selection adds further dependencies.',
    'INSPECTED CODE: CVAML loss discussion and immutable links in 02_repository_audit.md. EXECUTED TEST: evidence/02/logic_check_results.json enumerates 16 N=4 ±1 samples; corrected mean loss 0.0625 versus approximately zero with unbiased variance. This is a restricted arithmetic counterexample, not a benchmark result.',
    'Implementation-specific','No inspected follow-up supplies a correction for this exact estimator and dependency structure.',ST,'High for the stated iid fixed-value counterexample; uncertain for benchmark impact')
row('cvaml','goal','Value equivalence for one value function does not preserve all future instruction-dependent distinctions.',
    'Changing language goals or rewards after learning a task-specific latent model.',
    'The objective preserves expected values for its selected value function, allowing state distinctions irrelevant to that function to disappear.',
    'DEDUCTION: main objective and Appendix proof, PDF pp.18–20; the existence argument assumes a compact connected setting and a suitable value mapping. Appendix C.4 p.23 shows auxiliary BYOL loss remains important.',
    'Methodological','NavQ predicts goal-dependent semantic successors; WoMAP learns detector-derived rewards. Neither establishes universal goal-change preservation or action-risk calibration.',UN)

row('dinowm','language','Image-goal feature planning does not itself provide language-goal reward semantics.',
    'Tasks specified by text rather than an available target image.',
    'The planner optimizes distance to a specified visual target in frozen DINO feature space.',
    'AUTHOR METHOD: Section 3 and task evaluation, PDF pp.3–6; Table 2 shows patch features help PushT but does not validate language understanding.',
    'Methodological','WoMAP adds VLM proposals and language-conditioned distilled object rewards, with physical object-reaching tests. This adds detector and VLM errors and is not broad instruction-following validation.',PA)
row('dinowm','exploitation','Feature similarity can reward predicted outcomes that are not achievable or task-correct.',
    'Out-of-support actions, aliased views or multimodal futures during CEM search.',
    'A deterministic feature predictor and feature-distance goal objective need not preserve all physical constraints; optimizing them can select model errors.',
    'DEDUCTION from Section 3 and inspected planner; Table 1 task successes do not test calibrated confidence or adversarial model exploitation.',
    'Methodological','WoMAP extends the reward interface but retains optimizer/model/reward error; no matched intervention establishes resolution.',UN)

row('dreamer','planning_cost','Avoiding online candidate search requires learning an actor and value model on imagined trajectories.',
    'Tasks where latency at action selection matters but training compute is available.',
    'Dreamer amortizes PlaNet-style latent planning through actor/value optimization rather than solving the search anew at deployment.',
    'AUTHOR METHOD and PAPER EXPERIMENT: Sections 3–4 and Figure 6, PDF pp.3–6; mean score 823 after 5M steps on 20 DMC tasks. This does not establish equal compute.',
    'Methodological','Dreamer addresses PlaNet online search cost; later Dreamer versions extend scaling. The cost moves into behavior learning and value approximation.',SH)
row('dreamer','model_bias','Imagined actor improvement can exploit model or value errors outside collected support.',
    'Long imagined trajectories or policy updates into poorly observed regions.',
    'Policy gradients optimize returns supplied by learned dynamics and bootstrapped values; neither provides a general off-distribution correctness guarantee.',
    'DEDUCTION: Dreamer latent rollout/value objectives; World Models provides a direct exploitation example, while Dreamer benchmark gains do not isolate this failure mode.',
    'Methodological','Iterative data collection and later Dreamer models mitigate distribution mismatch, but no reviewed evidence resolves all out-of-support goal-conditioned navigation errors.',ST)

row('dreamwalker','scoring','Distance prediction remains a stronger bottleneck than future image fidelity in the paper intervention.',
    'Unseen continuous VLN episodes requiring branch ranking.',
    'The distance evaluator scores imagined states; a visually reasonable branch can still receive the wrong task-progress score.',
    'PAPER EXPERIMENT: Table 2 p.7 gives SR 49 normally versus 54 with oracle views; Figure 4 reaches 100% with full privileged distance replacement. Oracle interventions are not deployable methods.',
    'Methodological','HNR improves fixed-policy future features; NavQ adds semantic/distance supervision. Neither supplies a matched resolution of DREAMWALKER distance calibration.',PA)
row('dreamwalker','reproduction','The printed distance/reward signs cannot be resolved against released model code.',
    'Attempting a faithful reimplementation of Eqs.11–12.',
    'The text describes progress toward a goal, while printed reward/softmax signs appear to favor larger distance; retrieved official repository contains documentation rather than model source.',
    'UNVERIFIED: Eqs.11–12 versus prose, PDF method section; official repository README at ab7c9a5c9d07742e30e4cf99132cf1f61bb83de5.',
    'Implementation-specific','No inspected correction or executable official implementation resolves the sign ambiguity.',UN,'High for the documentation gap; unknown for intended implementation')

row('eoe','risk_transfer','Image semantic novelty scores are not established estimates of unsafe or instruction-inconsistent navigation actions.',
    'Familiar objects with unsafe transitions, or unknown categories absent from generated negative labels.',
    'The score compares CLIP similarity to generated class descriptions, without a transition/action outcome variable or exhaustive unknown-class coverage.',
    'PAPER EXPERIMENT: pp.6–8 fine-grained FPR95/AUROC 60.52/81.82, far-ImageNet 30.09/92.96; INSPECTED CODE: cosine overlap filter at 0.85 for ImageNet20. DEDUCTION: semantic novelty differs from navigation risk.',
    'Insufficiently tested','No matched navigation-risk follow-up was reviewed; EOE must remain adjacent OOD prior art.',UN)

row('fantasyvln','completion','Full multi-part navigation success remains very low despite relative gains.',
    'The evaluated long complete-task benchmark rather than isolated subgoal success.',
    'A long sequence of grounding/action decisions can fail even when per-subtask completion improves; the experiment does not isolate the causal distribution of these failures.',
    'PAPER EXPERIMENT: Table 1 p.6, SR 2.44, ISR 11.01, CSR 9.64; more than 97% of complete tasks fail. No independently reproduced decomposition.',
    'Methodological','MonoDream/NavForesee/WorldVLN evaluate different task/sensing settings, so their headline successes do not establish a solution on this protocol.',ST)
row('fantasyvln','faithfulness','Action-token alignment does not establish that visual reasoning is faithful or necessary at inference.',
    'Attributing improvements to actual reasoning rather than auxiliary representation learning.',
    'Training aligns branches, while direct deployment can act without explicit visual CoT; multiple training changes share credit.',
    'PAPER EXPERIMENT: Table 3, visual-only non-CoT ISR 11.19 versus full 11.01; alignment removal SR 0. INSPECTED CODE: trainer ranges cited in dossier confirm non-CoT teacher to CoT action alignment.',
    'Insufficiently tested','MonoDream and VLN-SIG also use auxiliary prediction. No matched counterfactual intervention establishes faithful reasoning across these systems.',UN)

row('genex','provenance','Long self-consistent generated trajectories need not describe the actual occluded environment.',
    'Using simulated views to answer questions about hidden physical facts or update navigation confidence.',
    'Conditional on a fixed real observation history and fixed model, additional generated samples are computed from that information; loop closure can be satisfied by a consistently false world.',
    'DEDUCTION: generative pipeline; PAPER EXPERIMENT: Table 2 p.9 IELC 0.105 streets/0.092 indoors and synthetic GenEx-EQA do not measure truth of hidden real environments.',
    'Fundamental under fixed-history conditioning','SALI persists imagined nodes; WorldVLN assimilates later real observations. Real observations add evidence, but no reviewed study establishes reliable confidence for hidden facts from generation alone.',ST,'High for information distinction; medium for system-level practical impact')

row('hnr','extrapolation','Semantic feature prediction degrades with distance and occlusion.',
    'Longer lookahead branches entering unobserved regions in R2R-CE/RxR-CE.',
    'Rendered features must extrapolate geometry/semantics not directly contained in the observed feature field.',
    'PAPER EXPERIMENT: Figure 5 p.7 shows declining similarity with distance; Table 3 predicted SR 60.74 versus oracle 63.13 under fixed lookahead policy; Table 6 soft targets outperform hard targets.',
    'Methodological','NavMorph adds contextual adaptation and improves HNR; SC²-WM adds calibration and delayed real-observation adaptation. Gains do not eliminate the equal-sensing occluded-region oracle gap.',PA)
row('hnr','branch_selection','Maximizing descendant scores can favor an overoptimistic noisy branch.',
    'More candidate descendants or correlated errors in unseen-branch evaluation.',
    'Eq.15 takes a maximum without a demonstrated uncertainty correction or calibrated confidence bound.',
    'DEDUCTION: Eq.15, method section; no risk-calibration or search-width bias experiment was identified in reviewed tables.',
    'Insufficiently tested','NavQ avoids explicit rollout through successor semantics; SC²-WM adapts features. Neither directly tests or removes the maximum-selection bias in the same planner.',UN)

row('imaginenav','memory','Local mapless imagination can revisit regions and spend path length on unproductive choices.',
    'Long object-goal searches with repeated visually similar regions.',
    'Limited persistent spatial memory and proposal constraints restrict global exploration history.',
    'PAPER EXPERIMENT: Table 1 p.7 HM3D SR/SPL 53/23.8 versus VLFM 52.5/30.4. DEDUCTION: lower efficiency is compatible with, but does not prove, revisitation as its cause.',
    'Methodological','ImagineNav++ arXiv:2512.17435 adds memory/selective foveation. Its project and forthcoming-code status were inspected, but a matched full implementation/experimental audit is outside this dossier.',UN)
row('imaginenav','views','Novel-view prediction errors leave a substantial action-selection gap.',
    'Unseen scenes and viewpoints outside the synthesizer training distribution.',
    'The action selector depends on generated candidate views whose appearance/content can be wrong.',
    'PAPER EXPERIMENT: Table 1 p.7 oracle views raise SR from 53 to 62 on HM3D and from 51 to 59 on HSSD.',
    'Methodological','HNR changes representation and WMNav changes the scoring interface; cross-protocol gains do not resolve the matched ImagineNav oracle gap.',ST)

row('lcvn','closed_loop','Offline endpoint success does not establish collision-free closed-loop instruction following.',
    'Open-loop predicted trajectories, particularly novel scenes with alternate valid routes.',
    'The evaluated endpoint threshold and expert-latent reward omit intervening collision and real-observation correction evidence.',
    'PAPER EXPERIMENT: Tables 1–2 pp.8–9: unified unseen SR 0.27 versus separate 0.19, despite separate model better SSIM/LPIPS; success is endpoint distance below average step size. INSPECTED CODE: actor uses detached returns and REINFORCE rather than a gradient through frozen DFOT.',
    'Insufficiently tested','WorldVLN uses segmented real observations, while WorldMAP/PiJEPA remain current leads. No matched physical closed-loop validation was reviewed.',UN)

row('learning_to_map','overconfidence','Ensemble variance and repeated map fusion do not provide validated calibrated exploration uncertainty.',
    'Shared model bias or repeated correlated observations in active object-goal navigation.',
    'Ensemble members can agree incorrectly, and registered repeated predictions are not independent evidence; UCB score construction lacks demonstrated coverage.',
    'DEDUCTION: ensemble prediction and UCB mechanism; PAPER EXPERIMENT: Tables 1–2 p.7 show SR/SPL 34.3/13.3 and IoU 25.6, not calibration guarantees.',
    'Insufficiently tested','SGM uses masked map completion and WMNav heuristic VLM curiosity; neither resolves posterior coverage or shared-bias detection in this protocol.',UN)
row('learning_to_map','resources','Active semantic mapping requires privileged training maps, accurate pose and multiple predictor evaluations.',
    'Deployment without reliable pose or training scenes with annotated semantic occupancy.',
    'Map registration relies on known pose; supervised targets and four-model ensembles introduce annotation and inference costs.',
    'AUTHOR METHOD: map construction/training and ensemble details in dossier; no pose-error ablation establishes robustness.',
    'Methodological','SGM introduces masked-map pretraining and priors, partially changing supervision, but retains semantic-map/perception dependencies.',PA)

row('lookbeforeleap','frozen_model','Teacher-heavy model data and frozen deterministic short-horizon prediction poorly cover policy errors.',
    'Actions and scenes outside the environment-model training distribution.',
    'The predictor is trained before policy use and is not corrected online; deterministic features compress multiple possible outcomes.',
    'AUTHOR METHOD and DEDUCTION: two-stage training/rollout sections; Table 1 p.12 unseen SR 24.6 versus model-free 21.5 establishes early utility but not shift robustness.',
    'Methodological','DREAMWALKER/HNR strengthen predictions; NavMorph and SC²-WM add adaptation. SC²-WM specifically supplies delayed real-observation residual updates, but on a different architecture and CE protocol.',PA)

row('monodream','horizon','Longer future prediction supervision can hurt under partial observability.',
    'Monocular VLN with prediction targets beyond immediately visible geometry.',
    'Targets become less identifiable from the current/history view, and auxiliary prediction can emphasize unreliable future details.',
    'PAPER EXPERIMENT: Table 6 p.7 SR 39.1/34.3/29.2 for horizons one/two/three; Table 5 current-depth-only 42.2 versus future-depth-only 39.8. DEDUCTION: target ambiguity is plausible, not experimentally isolated.',
    'Methodological','FantasyVLN also uses training-time imagination; NavForesee keeps future structures in action computation. No matched equal-data horizon intervention demonstrates a resolution.',ST)
row('monodream','training_release','The inspected official release does not provide the full auxiliary training pipeline needed to reproduce all four prediction objectives.',
    'Reproducing training ablations rather than loading published navigation weights.',
    'The released RoboOrchardLab project exposes model, processor, inference and evaluation, but the inspected project listing lacks the complete paper-specific auxiliary trainer.',
    'INSPECTED CODE: HorizonRobotics/RoboOrchardLab commit 48ac43987df9ac1ba40fef72dd7a2b4679c592b4; projects/monodream and models/monodream files listed in the dossier.',
    'Implementation-specific','Official implementation release resolves the earlier no-code search result for inference; a faithful full training reproduction remains unverified.',UN,'High for inspected release scope; no claim about inaccessible future/private code')

row('navforesee','counterfactuals','Predicted target structures are not established as faithful consequences of alternative candidate actions.',
    'Using semantic/depth futures for branch choice under ambiguous or unusual layouts.',
    'Task-conditioned future supervision may encode desired goal templates; it does not alone identify action-specific environment transitions.',
    'DEDUCTION: future prediction/action pipeline; PAPER EXPERIMENT: Table 2 p.7 full SR 66.2 versus 58.6 without long-horizon prediction shows utility, not physical counterfactual fidelity.',
    'Insufficiently tested','WorldVLN and LCVN use other action/observation interfaces; SC²-WM adds observed residual adaptation. None is a matched counterfactual-validity test of NavForesee.',UN)

row('navmorph','correction','Online context adaptation improves scores but does not establish reliable correction of confidently wrong predictions.',
    'Unseen CE scenes and biased stored context.',
    'Contextual memory supplies retrieved features without a calibrated confidence gate; latent alignment can preserve shared prediction bias.',
    'PAPER EXPERIMENT: Table 3 p.8 SR 47.91 versus 45.08 without self-evolution; DEDUCTION: Eq.2–3 memory weights are not correctness probabilities.',
    'Methodological','SC²-WM adds learned feature/state refinement and delayed real-observation TTA, with further CE gains. Residual-to-correct-action reliability is still not isolated.',PA)
row('navmorph','episode_protocol','Cross-episode memory makes evaluation depend on reset, ordering and prior test-scene exposure.',
    'Transductive test evaluation across multiple episodes in a scene.',
    'Memory slots retain scene/context information, and replacement behavior changes available knowledge.',
    'AUTHOR ACKNOWLEDGEMENT: PDF p.8 discussion of test-scene memory/fairness; Figure 4 memory size 1000 performs best and smaller/larger settings can hurt.',
    'Methodological','SC²-WM also adapts online, making episode-reset and matched exposure controls necessary. No reviewed ordering/reset factorial resolves comparability.',ST)

row('navq','aggregation','Successor semantic averaging can erase rare or route-specific evidence.',
    'Long discounted continuations, multiple equal shortest routes, or route-faithful instructions.',
    'A fixed-dimensional weighted semantic target summarizes future captions under a selected continuation rather than representing all branch-specific consequences.',
    'PAPER EXPERIMENT: Table 4 p.8 gamma 0/.3/.5/.7 SR 51.12/51.95/53.22/50.89; Table 3 p.7 predicted full SR 53.22 versus oracle 59.27. INSPECTED CODE: dataset uses one stored shortest path per target.',
    'Methodological','UNeMo adds task-conditioned future features and SC²-WM works in CE, but neither supplies a matched remedy for aggregated-semantic ambiguity.',ST)
row('navq','uniqueness','Shortest continuation is not mathematically unique without a tie-breaking or graph assumption.',
    'Graphs containing equal-length shortest paths to the same destination.',
    'Distinct first actions can share a shortest destination path length; the implementation chooses one stored path operationally.',
    'DEDUCTION/EXECUTED TEST: diamond-graph counterexample in evidence/02/logic_check_results.json; source dataset target construction, lines 180–244 at the dossier commit.',
    'Methodological','The released target construction resolves operational labeling via a stored path, but does not establish the printed general uniqueness claim.',ST,'High: the unqualified uniqueness claim has a counterexample; stored-path labeling remains well defined')

row('nwm','planning_cost','Diffusion-based trajectory evaluation and restricted CEM candidates impose compute and expressivity costs.',
    'Online action planning over multiple imagined time steps.',
    'Each candidate requires generated predictions; the released planner repeats one displacement across the horizon with a final yaw adjustment instead of optimizing fully independent per-step actions.',
    'INSPECTED CODE: planner.py:217–299 at the dossier commit; PAPER EXPERIMENT: Table 2 p.7 is offline ATE/RPE, not closed-loop SR.',
    'Implementation-specific','LCVN trains an actor and unifies language/video/action generation, moving cost into training and changing the evaluation. No matched cost/expressivity resolution was reviewed.',SH)
row('nwm','physical_validity','Offline video/trajectory errors do not establish safe causal action predictions in deployment.',
    'Actions outside logged support, moving obstacles or long physical navigation.',
    'Observational action-conditioned training plus endpoint error metrics does not identify unsupported interventions or validate all intermediate transitions.',
    'PAPER EXPERIMENT: Tables 1–2 pp.6–7 report LPIPS and ATE/RPE; no physical safety claim follows. Access limit: high-resolution SACSoN/HuRoN data are not redistributable in official README.',
    'Insufficiently tested','Vid2World improves causal masking/guidance; LCVN adds language. Different datasets and offline metrics do not resolve this validity gap.',UN)

row('pathdreamer','geometry','Image-generation gains were evaluated with a privileged future navigation graph.',
    'Interpreting navigation results as planning through completely unknown geometry.',
    'The navigation evaluation supplies future graph connectivity while replacing only future visual observations.',
    'PAPER EXPERIMENT: Table 3 p.8 horizon-three SR 50.4 versus repetition 40.6 and oracle 59.3; evaluation explicitly uses ground-truth future graph.',
    'Methodological','DREAMWALKER predicts waypoints and HNR evaluates continuous candidates, partly reducing the oracle-graph assumption while introducing waypoint/geometry prediction error.',PA)
row('pathdreamer','horizon','Generated structure degrades over successive unseen views.',
    'One to six synthesized steps into occluded geometry.',
    'Autoregressive structure hallucinations accumulate; training RGB generation with ground-truth structure adds stage mismatch.',
    'PAPER EXPERIMENT: Table 1 p.7 unseen semantic mIoU 78.3 one-step versus 50.8 across one to six steps.',
    'Methodological','HNR reduces pixel-generation cost and SALI preserves imagined nodes; neither validates the truth of long-horizon occluded structure under equal sensing.',PA)

row('planet','online_search','Receding-horizon latent CEM requires repeated candidate rollouts at each action decision.',
    'Latency-limited control with long horizons or many candidate sequences.',
    'Compute grows with candidate count, horizon and optimizer iterations; one sampled stochastic rollout per candidate can also produce noisy ranking.',
    'AUTHOR METHOD: latent CEM procedure and inspected planner source; Figures 4–5 test performance, not a universal latency bound.',
    'Methodological','Dreamer amortizes search into a learned actor/value model. Online cost is reduced by accepting training cost and approximation error.',SH)
row('planet','model_bias','Selecting high predicted reward can exploit transition and reward-model errors.',
    'Long rollouts or actions poorly supported by real experience.',
    'The same learned model both predicts outcomes and supplies the objective optimized by search.',
    'DEDUCTION: RSSM/reward planning pipeline; World Models supplies direct virtual-versus-real exploitation evidence, while PlaNet tests do not establish general calibration.',
    'Methodological','Dreamer changes policy optimization and CVAML corrects a different value-aware estimator bias; neither is proof of physically correct unsupported rollouts.',ST)

row('sali','persistent_hallucination','Persistent imagined nodes can retain incorrect room/object structure as planning evidence.',
    'Atypical layouts or errors in generated unseen views.',
    'Heuristic pruning and score fusion do not provide calibrated provenance weights separating observed from synthesized content.',
    'DEDUCTION: imagined graph/memory method; qualitative imagery and PSNR are not hidden-world truth tests.',
    'Methodological','HNR/NavMorph/SC²-WM alter representations and adaptation; GenEx improves consistency. None directly demonstrates correct recovery from false persistent SALI memory.',ST)
row('sali','replication','Headline and ablation performance cannot be fully reconciled from the inspected material.',
    'Attributing the large reported gain to imagination rather than a configuration or baseline difference.',
    'Main R2R SPL 78 differs from full-system ablation SPL 70/71, and no verified official code resolves the configuration mapping.',
    'PAPER EXPERIMENT: Table 1 p.6 versus Table 2 p.7; source/access record in dossier. These numbers are author-reported and unexecuted.',
    'Implementation-specific','No reviewed follow-up supplies a matched independent reproduction or configuration reconciliation.',UN)

row('sc2wm','feedback_timing','The broad ambiguity about whether correction uses any real next observation is resolved by the released asynchronous TTA branch.',
    'Interpreting immediate endogenous latent delta versus delayed deployment adaptation.',
    'Main-paper notation mixes current/next targets; immediate state refinement and delayed TTA use different signals.',
    'INSPECTED CODE: ss_trainer_ETP.py:2246–2308 compares prior predicted candidate features with the next real panorama using cosine residual; supplement PDF pp.15–16 describes FRL and Table 6. Immediate Eq.9 delta remains predicted next latent minus current latent.',
    'Implementation-specific','The paper supplement and official release resolve existence/timing for this inspected branch. This does not certify correctness of all formulas or confidence calibration.',RE,'High for inspected branch only')
row('sc2wm','calibration','Residual-triggered adaptation is not yet established to improve decisions under confidently wrong dynamics or residual misalignment.',
    'Domain shifts, ambiguous action selection, prediction/observation mismatch, or entropy confidently favoring a wrong action.',
    'Internal latent delta is not prediction error; delayed feature residual and percentile thresholds are proxies, and the active code updates a broader global encoder than a world-model-only reading suggests.',
    'PAPER EXPERIMENT: Table 3 p.8 full SR 50.90 versus fixed-interval 50.30; supplement Table 6 no adaptation 49.86, entropy 49.48, asynchronous 50.19, FRL 50.90. INSPECTED CODE: TTA_setup_trainable_params:1030–1035 and ss_trainer_ETP.py:2246–2308.',
    'Insufficiently tested','WorldMAP and PiJEPA are current primary-source leads reserved for follow-up audit; no reviewed matched residual-to-action-risk validation resolves this concern.',UN)

row('sgm','target_bias','Predicted semantic targets can attract exploration toward confidently wrong room/object context.',
    'Unusual layouts, segmentation errors or category combinations outside training.',
    'Map completion and argmax target selection use scores without demonstrated probability calibration; observed-map override protects observed cells but not unknown predictions.',
    'PAPER EXPERIMENT: Table 2 p.7 full Gibson SR/SPL 78.0/44.0 versus oracle target 91.7/72.3. DEDUCTION: oracle gap does not uniquely identify calibration as the cause.',
    'Methodological','WMNav uses VLM curiosity and L2M ensembles uncertainty; neither establishes reliable recovery after confidently wrong semantic-map completion.',UN)
row('sgm','iou_gradient','The inspected hard IoU term cannot train the decoder through its detached thresholded mask.',
    'The released main masked-map pretraining loss.',
    'Thresholding and detach remove the differentiable path for IoU, although weighted MSE and BCE still supply gradients.',
    'INSPECTED CODE: main pretrainer imports models_mae_cross (lines 38,172); models_mae_cross.py:254–295 computes detached hard IoU. No training run was executed.',
    'Implementation-specific','No inspected patch or follow-up establishes the empirical effect of replacing the detached term with a differentiable surrogate.',ST,'High for gradient path; unknown for benchmark impact')

row('unemo','future_branch','One provisional predicted state can bias all final action candidates.',
    'The provisional action is wrong or several action-conditioned futures are plausible.',
    'A shared predicted future is fused back into action reasoning instead of maintaining separately validated counterfactual branches.',
    'DEDUCTION: multimodal world-model/action pipeline; Tables 3–4 p.7 give best final-action-only 72.9 versus joint two-action 72.1, complicating a simple auxiliary-action-benefit claim.',
    'Methodological','SC²-WM uses feature refinement and observed residual adaptation in CE; this is not a matched resolution of UNeMo candidate conditioning.',UN)
row('unemo','specification','Missing world-model objective/attention details and unavailable verified training source prevent faithful reproduction.',
    'Implementing the printed CVAE and single-key attention description or reconciling headline versus best ablation.',
    'The supplied specification does not establish all training losses and tensor semantics; no complete official repository was verified.',
    'UNVERIFIED: method equations and Tables 1,3–4, PDF pp.4–7; headline SR 72.1 versus final-action-only 72.9 is not a single interchangeable configuration.',
    'Implementation-specific','No inspected supplementary implementation or follow-up resolves these details.',UN)

row('vid2world','causality','Temporal causal masking and guidance do not establish intervention-valid or calibrated transition probabilities.',
    'Unsupported actions, abrupt dynamics changes or physical policy deployment.',
    'Chronological conditioning prevents future leakage but does not remove confounding in logged action data; guidance changes conditional distributions and linear extrapolation assumes local regularity.',
    'DEDUCTION: transfer/masking/guidance mechanism. PAPER EXPERIMENT: Table 1 p.8 RECON FVD/LPIPS 59.4/.3236 versus NWM 31.2/.295, and the real-to-sim study ranks only three drawer-policy checkpoints.',
    'Insufficiently tested','NWM/LCVN/WorldVLN use video prediction differently; no reviewed cross-task intervention study establishes broad causal validity.',UN)

row('visualimaginations','attribution','Generated landmark prototypes may improve representation learning without faithfully predicting the current hidden scene.',
    'Unusual object appearances or claims that foresight causes the navigation gain.',
    'Images are semantic prototypes derived from instruction phrases, not action-conditioned scene transitions; training losses remain beneficial when prototypes are removed at test time.',
    'PAPER EXPERIMENT: Table 3 p.6 correct/none/wrong imagination SR 67.26/66.92/66.24 versus baseline 66.24; selected attention maps do not establish causality.',
    'Insufficiently tested','MonoDream/FantasyVLN also exploit auxiliary prediction. Separate prototype alignment from actual-state prediction in any comparison; no causal deployment mediation test was reviewed.',UN)

row('visualplanning','verifier','Exact grid-state validators do not transfer automatically to noisy embodied outcomes.',
    'Replacing synthetic grids with real visual navigation or manipulation.',
    'RL rewards and exact-match/progress scoring use a rule-based interpreter and known transition/task structure; a learned real-world validator can be wrong or exploitable.',
    'AUTHOR METHOD and DEDUCTION: interpreter/reward formulation; Table 1 p.6 EM/PR 80.6/84.9 establishes gains within the synthetic domain only.',
    'Methodological','CoT-VLA/FantasyVLN/WorldVLN use richer tasks without inheriting the exact validator. No reviewed real-world counterpart preserves the same correctness guarantees.',UN)

row('vlnsig','causal_use','Auxiliary future-token prediction has not been shown to improve actions specifically by forecasting correct future content.',
    'Distinguishing representation regularization from deployed foresight.',
    'Future-token and semantic objectives train the encoder; dVAE/CLIP targets and vocabulary select what information is retained, while prediction heads are not an explicit deployment simulator.',
    'PAPER EXPERIMENT: Table 2 p.6 baseline SR/SPL 64.4/58.8 versus all 68.1/62.3; supplement p.3 visual reconstructions use 70% ground-truth tokens, so they are not autonomous predicted images.',
    'Insufficiently tested','MonoDream/FantasyVLN share training-time prediction; NavQ inserts predicted semantics during inference. No matched intervention demonstrates causal future-content use in VLN-SIG.',UN)

row('wmnav','recovery','An ordinary cell that receives a falsely low curiosity score cannot recover merely because a later score rises.',
    'Minimum-merge updates to non-goal cells.',
    'The ordinary update takes the minimum of old and new values, creating irreversible pessimism; identified goal-mask cells are an explicit exception reset to 10.',
    'INSPECTED CODE: _merge_evalue and update_curiosity_value:1263–1287. EXECUTED TEST: isolated 10→2→9 merge gives 10→2→2; goal reset produces 10. Not a simulator performance test.',
    'Implementation-specific','Goal-mask reset resolves recovery for those identified cells only. SGM/L2M use different maps, with no matched ordinary-cell recovery intervention reviewed.',PA,'High for update behavior; unknown for aggregate success impact')
row('wmnav','backbone','Heuristic VLM curiosity scores and stopping choices vary substantially with backbone and configuration.',
    'Transferring the recipe to smaller VLMs or comparing different HM3D versions.',
    'Prompt responses are not calibrated probabilities; paper and source stopping thresholds differ and models differ in semantic judgment.',
    'PAPER EXPERIMENT: Table III p.6 Qwen3B SR 29.7 versus Gemini1.5Pro 58.1; INSPECTED CODE uses 0.8m internally, while printed 1.0/0.1m threshold is inconsistent. Evaluation success threshold is a separate setting.',
    'Methodological','ImagineNav++ adds memory and SGM/L2M change score sources; none establishes backbone-invariant calibration under a matched protocol.',UN)

row('womap','reward_exploitation','Optimizing detector-derived reward can favor visible/confident objects without ensuring physically valid goal reaching.',
    'Background/lighting shift or candidate actions outside the learned dynamics support.',
    'Gradient search amplifies errors in learned dynamics and rewards; evaluation success also uses detector confidence/box-size thresholds, potentially sharing reward bias.',
    'PAPER EXPERIMENT: Table 2 p.8 nominal success 63% falls to 50% lighting and 30% background shift. INSPECTED CODE: optional reward/std score exists; joint planner differs from printed smoothing objective.',
    'Methodological','DINO-WM/NWM offer other models and CVAML addresses value-objective bias. None is a matched resolution of detector-based goal/reward validity.',ST)

row('worldmodels','exploitation','A controller can achieve high imagined reward by exploiting a poorly supported world model.',
    'Low-temperature learned Doom dynamics and policy-induced distribution shift.',
    'Random-policy training undercovers controller-induced states, and the controller searches weaknesses of the learned transition/reward process.',
    'PAPER EXPERIMENT: Table 2 p.4 temperature .1 gives virtual/real survival 2086/193; at 1.15, 918/1092. Section 4.2 p.6 explicitly discusses exploitation.',
    'Methodological','PlaNet/Dreamer collect policy experience iteratively and use stronger latent training. This mitigates mismatch but does not prove robustness of unsupported imagined trajectories.',PA)

row('worldvln','feedback_delay','Segmented imagined execution can delay correction until a new real observation arrives.',
    'Aerial navigation with environment changes or prediction errors inside an action segment.',
    'Real-history assimilation is intermittent; within-segment decisions can depend on predicted video rather than fresh measurements.',
    'AUTHOR METHOD and DEDUCTION: segmented rollout/observation assimilation; Table 1 aerial UAVFlow results do not isolate feedback-delay sensitivity.',
    'Methodological','CoT-VLA has a related chunk-feedback issue; LCVN/NWM use different offline protocols. No matched delay-versus-risk intervention was reviewed.',UN)
row('worldvln','reward_scaling','A time-dependent common positive reward scale can cancel under per-group standardized advantages.',
    'Advantages normalized separately for a group sharing the same positive temporal scale, apart from epsilon effects.',
    'Both mean-subtracted rewards and their standard deviation scale by the same factor; this does not hold automatically if groups mix times or the implementation normalizes differently.',
    'DEDUCTION/EXECUTED TEST: evidence/02/logic_check_results.json verifies the algebra numerically for a toy reward vector. The official snapshot is a project site, and the unreleased trainer grouping remains unknown.',
    'Insufficiently tested','No inspected official trainer or correction determines whether the printed temporal factor has the intended effect in the actual training run.',UN,'High for conditional algebra; unknown for actual trainer behavior')

records = json.loads((R/'assignments/02_results.json').read_text(encoding='utf-8'))
by_id = {r['paper_id']:r for r in records}
comparisons = {
 'chasing_ghosts':['hnr','sali'], 'cotvla':['fantasyvln','worldvln'],
 'cvaml':['dinowm','womap','navq'], 'dinowm':['womap','nwm'],
 'dreamer':['navmorph','cvaml'], 'dreamwalker':['hnr','navq','sc2wm'],
 'eoe':[], 'fantasyvln':['monodream','navforesee','worldvln'],
 'genex':['sali','imaginenav','pathdreamer','worldvln'],
 'hnr':['navmorph','sc2wm','navq'], 'imaginenav':['wmnav'],
 'lcvn':['nwm','vid2world','worldvln'], 'learning_to_map':['sgm','wmnav'],
 'lookbeforeleap':['dreamwalker','hnr','navmorph','sc2wm'],
 'monodream':['fantasyvln','navforesee'], 'navforesee':['worldvln','lcvn','sc2wm'],
 'navmorph':['sc2wm','hnr'], 'navq':['unemo','sc2wm'],
 'nwm':['vid2world','lcvn'], 'pathdreamer':['dreamwalker','hnr','sali'],
 'planet':['dreamer','dinowm','navmorph'], 'sali':['hnr','navmorph','sc2wm','genex'],
 'sc2wm':['navmorph','hnr','unemo'], 'sgm':['wmnav','learning_to_map','womap'],
 'unemo':['sc2wm'], 'vid2world':['nwm','lcvn','worldvln'],
 'visualimaginations':['monodream','fantasyvln'],
 'visualplanning':['fantasyvln','cotvla','worldvln'],
 'vlnsig':['monodream','fantasyvln','navq','visualimaginations'],
 'wmnav':['sgm','learning_to_map'], 'womap':['dinowm','nwm','cvaml'],
 'worldmodels':['planet','dreamer','cvaml'], 'worldvln':['nwm','lcvn','cotvla','vid2world']
}
assert set(B) == {r['paper_id'] for r in records}
(R/'evidence/02/bottleneck_records.json').write_text(json.dumps(B,ensure_ascii=False,indent=2),encoding='utf-8')

checks = {
    'cvaml': 'The isolated iid fixed-value finite-sample variance correction check finds expected residual loss 0.0625 for N=4 Rademacher samples; unbiased sample variance correction is approximately zero. This is not a CVAML benchmark run.',
    'navq': 'An isolated diamond graph has two equally short first actions to the same destination, refuting unqualified shortest-continuation uniqueness. The released dataset still defines operational targets through a stored shortest path.',
    'wmnav': 'An isolated ordinary minimum-merge update on values 10,2,9 remains at 2; an explicit goal-mask reset returns 10. This tests update arithmetic, not navigation performance.',
    'worldvln': 'A toy normalized-advantage calculation confirms cancellation of a common positive reward scale within one separately normalized group, up to epsilon. Actual trainer grouping was unavailable and remains unverified.'
}
for r in records:
    pid=r['paper_id']; r['bottlenecks']=B[pid]
    r['followup_sources']=[dict(paper_id=q,dossier_path=by_id[q]['dossier_path'],
        bibliography_sources=by_id[q]['bibliography_sources'],
        role='Reviewed mechanism comparison or follow-up; chronology and strength of resolution are stated in the bottleneck rows') for q in comparisons[pid]]
    if pid=='imaginenav':
        r['followup_sources'].append(dict(title='ImagineNav++',url='https://200203z.github.io/imaginenav-plus/',role='Current follow-up lead; project/release availability inspected, full implementation and matched experiment audit not completed'))
    if pid in ('dinowm','lcvn','sc2wm'):
        r['followup_sources'].append(dict(title='PiJEPA',url='https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/html/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.html',role='Current follow-up lead supplied by coordinator; not counted as a completed paper review in this assignment'))
    r['independent_checks_executed'] = pid in checks
    if pid in checks and not any(k['evidence_type']=='EXECUTED TEST' for k in r['key_findings']):
        r['key_findings'].append(dict(claim=checks[pid],evidence_type='EXECUTED TEST',source='evidence/02/logic_check_results.json',locator='Isolated '+pid+' check; no project code executed'))
    path=R/r['dossier_path']; doc=path.read_text(encoding='utf-8')
    marker='## Structured bottleneck resolution record'
    if marker in doc: doc=doc.split(marker)[0].rstrip()+'\n'
    doc+='\n'+marker+'\n\nThese statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).\n'
    for b in B[pid]:
        doc+=f"\n### {b['id']}\n\n**{b['status']} — {b['kind']}.** {b['description']}\n\nConditions: {b['conditions']} Cause: {b['cause']}\n\nEvidence: {b['evidence']}\n\nFollow-up assessment: {b['followups']} Confidence: {b['confidence']}.\n"
    if pid in checks:
        doc+='\nIndependent isolated check (**EXECUTED TEST**): '+checks[pid]+' Setup and output: `evidence/02/logic_checks.py` and `evidence/02/logic_check_results.json`.\n'
    path.write_text(doc,encoding='utf-8')
(R/'assignments/02_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated',len(records),'papers with',sum(map(len,B.values())),'specific bottleneck rows.')
