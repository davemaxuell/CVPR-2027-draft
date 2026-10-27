from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parents[2]
notes=json.loads((ROOT/'evidence/01/notes.json').read_text('utf-8'))
notes['prevalent']['implementation']=notes['prevalent']['implementation'].replace('868 fb 53d 6b 7978 bbb 10439a 59e 65044c 811 ee 5c 2','868fb53d6b7978bbb10439a59e65044c811ee5c2')
scope={
'duet':('Model, pretraining losses, teacher labels, graph routing, final goal return, data inputs, R2R driver/configuration and graph evaluator','No simulator/checkpoint run; other task branches are not assumed byte-identical. Source inspection and coordinator routing toy check are different evidence types.'),
'scalevln':('R2R and REVERIE DUET branches, teacher loss, feature/annotation input, pretraining launcher and graph evaluator','Full graph generation and GAN image-repair pipeline not reconstructed; recipe/data differences prevent interpreting a benchmark delta as language reliance.'),
'srdf':('Full paper including appendix, official repository tree/README, release-status check','Original repository README-only; later specialist checkpoints/implementations do not establish full flywheel reproduction.'),
'gaussianmap':('Main method, objectives, ablations and official repository tree/README','No model implementation in pinned release; semantic compression and action-probability normalization cannot be resolved from code.'),
'uagm':('Full method and appendix, exact arXiv PDF checksum, official repository tree/README','No model implementation in pinned release; Fisher regularization, VI details and statistical test unit remain unresolved.'),
'bevbert':('BEV projection/pretraining heads, discrete teacher loss/data targets, graph metric functions and configuration','Continuous and discrete branches differ; only the specifically linked source functions were inspected. No depth model was trained.'),
'gridmm':('Grid feature construction/aggregation, discrete action loss, feature targets, graph metric functions and launch script','Continuous policy subset inspected; full simulator integration unexecuted. Commented STOP reweighting is not an active method.'),
'kerm':('Fact database/retrieval scripts, knowledge-fusion model, action loss, pretraining inputs, evaluator and launcher','External features/knowledge data are needed; released facts are not certified observations of the current scene.'),
'ver':('Volumetric detector/head, focal/L1 losses, feature loading, 3D evaluation and configuration','Release inspection did not locate full VLN policy/evaluator. Volumetric perception evaluation does not reproduce navigation claims.'),
'hamt':('Hierarchical/history model, proxy objectives, imitation/RL agent, instruction/features, R2R evaluator and script','No end-to-end ViT training or checkpoint execution. Supplement ablation removes ITM as well as history.'),
'vlnbert':('Recurrent model variants, actor-critic/imitation action path and task-specific code recorded in dossier','Pretrained OSCAR/PREVALENT variants and search/reranking systems are separate models; no execution.'),
'airbert':('Multimodal model, BnB preprocessing/objectives, training driver and reranking output conversion','Complete beams and exploration observations are external inputs. Generative recurrent policy variant is not the same inference protocol as the inspected reranker.'),
'prevalent':('Top-level release links plus separate PREVALENT_R2R pretraining heads, listener loss, train/eval driver and launcher','Application repository pinned separately; HANNA application and original large pretraining run not executed.'),
'speakerfollower':('Speaker/follower model and search code, supervised train driver and evaluator','Augmented instructions and image features required; no independent search-cost measurement.'),
'selfmonitoring':('Co-grounding/progress model, positive loss, rollout, training driver and evaluator','Printed negative MSE sign differs from released positive MSE. No checkpoint verification.'),
'regretful':('Regret gate, progress detach, visit marker, rollout/trainer and evaluator','Detached progress is a static code finding; its performance effect was not measured.'),
'tacticalrewind':('Frontier/ending queues, backtracking, goal reranking and stopping code','Heuristic score calibration remains unproven; mathematical shift counterexample is not benchmark evidence.'),
'rcm':('Main method/objectives and primary author homepage repository search','No official executable implementation located in bounded search; no fetched model-code inspection claim.'),
'r2r':('Dataset/simulator baseline, tokenization/connectivity, train-vs-submission driver and evaluator','Dataset assets not downloaded or simulated; test-submission training intentionally uses validation data and cannot support held-out validation claims.'),
'crossmodalmap':('Semantic map/goal model, position and coverage losses, offline train driver and online stopping/evaluation path','Perception and local controller are external/pretrained components. Timeout and synthetic stop_signal handling require protocol parity.'),
'weakmap':('Map projection/fusion, localization/waypoint losses, DAgger trainer and STOP-aware metrics','Fine/semantic/fused maps have distinct actual tensor sizes; no LMDB collection or simulator execution.'),
'goat':('Confounder extraction/intervention model, DAgger loss, dictionary/data inputs, metrics and script','Implementation attention/gating does not by itself prove causal identification or eliminate all spurious associations.'),
'gavln':('Frozen VGGT/SigLIP geometry, BEV scatter/pooling, supervised text labels, action decoding/evaluation and launcher','Official README supplies evaluation entry point; full training-driver/configuration reproduction remains unverified. External checkpoints and backbones required.'),
'hsgm':('Subtask state/STOP/rollback agent, mapper/path planner, runner metrics and sensor configuration','No external API or simulator call. First STOP is persistent; final-distance metrics differ from an explicit STOP predicate at timeout.'),
'ivln':('Continuous Map-CMA model, iterative DAgger and masks, oracle transit environment, iterative evaluator and config','Separate discrete IVLN implementation not inspected; paper findings and inspected continuous implementation are distinguished.'),
'overnav':('Main method, module supervision, experiment/ablation and primary project/repository search','No official executable implementation located in bounded search; mechanism is paper evidence.'),
'marval':('Full main mechanism, expert-label rules and conference slides; official Marky data link checked','No verified complete executable MARVAL implementation; component/data repositories do not establish model release.'),
'navillm':('Image/graph embedding, navigation classifier, CE training path, separate generation losses, data and evaluator','Paper uniform generation narrative differs from released classifier branch. No full multitask checkpoint execution.'),
'navrag':('Hierarchical generator, shortest-path dataset conversion, DUET loss/model, pretraining driver and evaluator','API model/version configured externally; regeneration not deterministic from code alone.10× STOP loss is active and should be controlled.'),
'roomtour3d':('NaviLLM fork, video-frame candidates, teacher loss, summary generation, data/configuration and train driver','Geometry/data-curation pipeline not rerun; video-action source contains commented DAgger branch, not active sampled video training.'),
'topologicalplanning':('Main planning/controller objectives, BCE STOP alternative, evaluation protocol and project release search','No verified public executable model repository located; pre-exploration is excluded from reported SPL.'),
'reverie':('Dataset/supplement metrics, follower positive CE+MSE and target-ID/navigation evaluator','Object proposals/perception are external; ID match and endpoint visibility are separate evaluator branches.'),
'rxr':('Official data README/schema, dense pose/text alignment interface and full relevant datasheet','No exact RxR baseline model code inspected; related VALAN README and Pangea toolkit lead do not substitute for that inspection.')}
text='# Assignment 01: implementation inspection and reproducibility appendix\n\nCutoff: 2026-10-22. This appendix indexes the actual static source inspection. No downloaded model code, training job, simulator episode, checkpoint validation or API generation call was executed by assignment01. The coordinator’s separate DUET routing toy check is recorded under experiments/duet_routing_check_results.json and is not a navigation benchmark.\n\n'
text+='Files were read at the commit identifiers below. Locators specify focused functions/regions, not an assertion that every line, every task branch, every dependency, or every configuration was audited. Full pipeline reproduction remains narrower than source availability. HTTP403 from the shared GitHub API was treated as rate limiting; immutable raw-source retrieval and git references were used instead.\n\n'
text+='| Paper | Actual phase coverage | Remaining scope or reproducibility limit |\n|---|---|---|\n'
for id,(phases,gap) in scope.items():text+=f'| [{id}](../../dossiers/{id}.md) | {phases} | {gap} |\n'
text+='\n## Pinned source records\n\n'
ledger=[]
for id,n in notes.items():
    p=ROOT/'repositories/01'/id/'source.json';m=json.loads(p.read_text('utf-8')) if p.exists() else {}
    text+=f'### {id}\n\n'
    if m.get('commit'):text+=f"Repository: [{m['repo_url']}]({m['repo_url']}); commit `{m['commit']}`.\n\n"
    else:text+='No verified executable repository in the bounded search record.\n\n'
    if not n.get('code'):text+='No model source inspected; only explicitly described paper, project or release metadata.\n\n'
    for c in n.get('code',[]):
        url=c.get('repo_url',m.get('repo_url',''));sha=c.get('commit',m.get('commit',''))
        text+=f"- [{c['path']}]({url}/blob/{sha}/{c['path']}) ({c['locator']}): {c['finding']}\n"
        rid='prevalent_r2r' if c.get('repo_url','').endswith('/PREVALENT_R2R') else id
        local=ROOT/'repositories/01'/rid/c['path']
        ledger.append(dict(paper_id=id,repo_url=url,commit=sha,path=c['path'],locator=c['locator'],finding=c['finding'],local_path=str(local.relative_to(ROOT)),sha256=hashlib.sha256(local.read_bytes()).hexdigest(),executed=False))
    text+='\n'
(ROOT/'evidence/01/implementation_appendix.md').write_text(text,'utf-8')
(ROOT/'evidence/01/source_inspection_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),'utf-8')
(ROOT/'evidence/01/notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),'utf-8')
from write_dossiers import build
build()
