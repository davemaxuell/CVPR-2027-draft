from pathlib import Path
import json,re,shutil
ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'evidence'/'03'
paths=list(E.glob('notes_*.json'))
notes={n['id']:n for p in paths for n in json.loads(p.read_text(encoding='utf-8'))}
a=json.loads((E/'code_audits.json').read_text(encoding='utf-8'))
fix={
 'Σ_voc ab':'Σ_vocabulary','optimal it y':'optimality','g it ls-remote':'git ls-remote','log it':'logit',
 'Caption er':'Captioner','GP T':'GPT','intrinsic s':'intrinsics','afford an ce':'affordance','afford ances':'affordances',
 'un executable':'unexecutable','substantive ly':'substantively','Aux RN':'AuxRN','de couples':'decouples',
 'sub task':'subtask','Sub tasks':'Subtasks','sub tasks':'subtasks','an not at or':'annotator','Nav -R 1':'Nav-R1',
 'sub instruction':'subinstruction','f rust um':'frustum','super pixel':'superpixel','centro id':'centroid',
 'Rep ViT':'RepViT','Instruct Nav':'InstructNav','A 2 Nav':'A2Nav','time out s':'timeouts','time out':'timeout',
 'un met':'unmet','Context- Nav':'Context-Nav','Chat GPT':'ChatGPT','KER M':'KERM','learn able':'learnable',
 'positiv it y':'positivity','TP AMI':'TPAMI','distract or s':'distractors','canonical ize':'canonicalize','Instance Nav':'InstanceNav',
 'PS L':'PSL','in feasibility':'infeasibility','random iz ation':'randomization','Instruct BLIP':'InstructBLIP','Turtle bot':'TurtleBot',
 '6 D oF':'6 DoF','disc ret i zed':'discretized','Bridge V 2':'Bridge V2','OWL v 2':'OWLv2',
 'pre built':'prebuilt','sub actions':'subactions','Open- Nav':'Open-Nav','Nav RAG':'NavRAG',
 'un constrained':'unconstrained','Deep Seek-R 1':'DeepSeek-R1','non negative':'nonnegative','sr c/':'src/',
 'raster i zed':'rasterized','mis parsed':'misparsed','s alien ce':'salience','mis attributed':'misattributed',
 'ob ser v ability':'observability','ab late d':'ablated','representative ness':'representativeness','pre print':'preprint',
 'ρ_de comp':'ρ_decomp','S_p red':'S_pred','re anchoring':'reanchoring','ST C':'STC','Intern Nav':'InternNav',
 'search-optimal it y':'search optimality','NavGPT2 465/317':'NavGPT 2465/317','NavGPT2 9':'NavGPT 29',
 'disc rim i native':'discriminative','S R38.8':'SR38.8','SigLIP -so 400 M':'SigLIP-so400m','un padding':'unpadding',
 'LLaVA -One Vision':'LLaVA-OneVision','s in/cosφ':'sin/cosφ','re scales':'rescales','trainsets o':'training set, so',
 'con founders':'confounders','evaluationu at or':'evaluator','un solvable':'unsolvable','measure dOOM':'measured out-of-memory event',
 'LLaVA -Video':'LLaVA-Video','NVILA -lite':'NVILA-lite','N 8 turn':'N=8 turn','avg.27 spl us':'average0.27s plus',
 'R 768':'R^768','R 4':'R^4','β∈R 4':'β∈R^4','H 1,7 D':'one-frame history and seven-dimensional',
 'GPU s':'GPUs','A100 s':'A100s','leader board':'leaderboard','over time':'over time',
}
def tidy(x):
 if isinstance(x,str):
  if x.startswith(('http://','https://')):return x
  for old,new in fix.items():x=x.replace(old,new)
  x=re.sub(r'\b(at|by|to|from|in|of|on|with|and|versus|than|over|after|before|uses|reports|Table|Figure|Appendix)(?=\d)',r'\1 ',x)
  return x
 if isinstance(x,list):return [tidy(v) for v in x]
 if isinstance(x,dict):return {k:tidy(v) for k,v in x.items()}
 return x
notes=tidy(notes);a=tidy(a)

a['console']['venue']='IEEE TPAMI 2024 (author publication record); DOI10.1109/TPAMI.2024.3407759'
a['console']['source']='https://arxiv.org/abs/2405.18721'
a['console']['bibliography']='The first author\'s publication page confirms IEEE TPAMI2024: https://expectorlin.github.io/. The supplied paper is the expanded journal treatment, not a CVPR main-conference paper.'
a['console']['limitations']='Principal method and results of the supplied expanded paper and the HAMT/CMT code integration were reviewed. The author publication record confirms TPAMI2024. All alternative integrations and preprocessing assets were not independently reconstructed. No benchmark was executed.'
a['ecot']['venue']='CoRL 2024; proceedings published as PMLR270 in2025'
a['ecot']['source']='https://proceedings.mlr.press/v270/zawalski25a.html'
a['ecot']['bibliography']='The official PMLR270 index identifies the8th Conference on Robot Learning, held6–9 November2024, with proceedings published12 January2025. The supplied text is the March2025 arXiv revision; proceedings byte equivalence was not assumed.'
a['awarevln']['limitations']='The main paper, embedded supplementary annotation/cross-dataset discussion (pp.12–15), alternate proceedings method/results and targeted training/inference code were reviewed. The contradictory RxR training description is preserved as uncertainty. Dataset generation, exact loss masking and all annotation examples were not independently reproduced. No benchmark was executed.'
a['analogical']['limitations']='The supplied six-page paper, released comparative prompts, agent and evaluator were reviewed. No API or benchmark run was executed. Exact caption caches, model-release drift, candidate geometry and additional calls remain comparison controls.'
a['discussnav']['limitations']='The paper and the released single-script method/evaluation path were inspected. No API or robot experiments were executed. The paper\'s terminology of beam search should not be equated with independent API sampling without checking the backend.'
a['ca_nav']['limitations']='The full v4 method and principal tables plus targeted constraint/value-map/evaluation code were reviewed. Remaining qualitative appendix examples were not exhaustively reviewed. Final journal acceptance was not verified. No benchmark was executed.'
a['mapgpt']['pages']='1,3–8,12–13'
a['mapgpt']['limitations']='Main method/results, Appendix A implementation and task prompts (pp.12–13), and the released graph/prompt/action path were inspected. Remaining selected qualitative figures were not exhaustively assessed. No API benchmark was executed.'
a['mapgpt']['results']=notes['mapgpt']['results']+' Appendix B (p.12) identifies inaccurate instruction-detail following and insufficient exploration in ambiguous scenes as failure modes. Figure4 explicitly instructs the REVERIE agent to ignore some interactive actions and relax color adherence; this is a task-specific prompt assumption, not verified instance grounding.'
# Remove the token count rather than propagate an ambiguous OCR-compressed draft number.
notes['mapgpt']['method']=re.sub(r' Text-only system reports.*$','',notes['mapgpt']['method'])

a['navfom']['pages']='1,3–10,19,21–22,26,28–30'
a['navfom']['method']=notes['navfom']['method'].replace('deployment 19.1 GB/218 ms~5 Hz 1600 tokens RTX 4090','deployment costs are version-specific; see Appendix Table11 below')
a['navfom']['method']+='\n\nAppendix A.2 supplies a practical fallback omitted from the core expectation equation: if the minimum sampling probability exceeds the budget at extremely long horizons, remove the oldest frames. Consequently the unresolved issue is evidence loss under truncation and whether every stochastic draw respects the intended budget, rather than an assertion that the entire method has no fallback. Appendix A.1 uses per-dimension99th-percentile trajectory scales. Appendix B.3 caches four-token history features offline while encoding current/QA images online, which is part of the reported training-efficiency recipe.'
a['navfom']['results']=notes['navfom']['results']+'\n\nFigure8 (p.9) compares BATS, uniform sampling and token merging within the four-camera RxR setting: at2048 tokens BATS64.4 SR/56.2 SPL versus token merging63.2/54.9; at1024 tokens BATS62.5/53.9 versus uniform59.7/49.6. These are more informative architectural controls than the headline multi-task numbers. Appendix Table11 reports RTX4090 FP16 at19.8GB/218ms and4bit at10.7GB/248ms;19.1GB/566ms refers to Jetson Thor16bit. Appendix F attributes49% of analyzed failures to model issues such as history alignment, early stopping and insufficient turning, and51% to dataset/simulator issues; the categorization is the authors\' analysis. Six cameras slightly underperform four under the same token budget, exposing coverage/history competition.'
notes['navfom']['bottlenecks'][0].update(description='Budgeted history sampling and oldest-frame truncation can remove task-critical evidence',cause='BATS optimizes an expected recency budget; its practical extreme-horizon fallback removes oldest frames without instruction-conditioned utility',evidence='Main Eqs4–5; Appendix A.2 truncation; Figure8 memory controls; Appendix D camera-count tradeoff')
a['navfom']['limitations']='The main paper, architecture/data/budget appendices, matched history-token ablations, runtime table and failure analysis were reviewed. The official project linked a code placeholder at inspection, so no implementation or checkpoint was inspected. Exact stochastic budget enforcement and metric code remain unverified; Appendix B.2 also prints an nDTW normalization using start-to-goal distance that should be checked against benchmark code before comparison. No experiment was run.'

a['mapnav']['pages']='1,3–8,12–15'
a['mapnav']['method']=notes['mapnav']['method'].replace('Full optimization/loss masks require appendix/code.','Appendix B freezes vision and fine-tunes the projector and Qwen2 for one epoch at1e-6 using bfloat16; SigLIP features have width1152 and Qwen2 hidden width3584. Exact loss masks were not established in the released source audit.')
a['mapnav']['method']+=' Appendix A explicitly adds600K general vision-language samples alongside500K navigation samples for the R2R recipe. Its expert-data prose says300K, while Eq5 says approximately200K; the total counts therefore have an internal ambiguity. Navigation-pair counts must not be treated as total training exposure.'
a['mapnav']['results']=notes['mapnav']['results']+' Appendix Tables6–7 isolate input rendering and segmenter choice: directly adding depth reduces SR27.1→23.1, whereas ASM reaches36.5. YOLOv8/MobileSAM/Mask2Former yield31.5/34.8/36.5 SR, showing sensitivity to perception quality. This supports the representation interface but does not establish the same mechanism under equally accurate semantic inputs.'
a['mapnav']['limitations']='Main method/results and Appendix A–E data/training/sensor/ablation details were read, alongside the mapper/action/metric helper code. Complete DAgger regeneration and VLM loss masking were not independently reconstructed. Memory savings refer to the observation representation; no independent runtime or benchmark was measured.'
a['psgnav']['pages']='1,3–8,13–15,21'
a['psgnav']['results']=notes['psgnav']['results']+'\n\nAppendix A.2/Table5 evaluates a deliberately selected320-episode false-positive subset: EEC intercepts246 wrong stops, recovers97 episodes (30.3% SR), and leaves74 false-positive stops. This supports reducing that failure mode within the selected protocol, not a universal false-positive guarantee. Table6 shows a small memory bank already helps (capacity0/5/10/20 gives63.5/65.2/66.1/66.3 SR). Table7 changes the detector to WeDetect for OVON, so comparisons there cannot isolate graph reasoning alone. Appendix A.3 defines geometric visibility using72 rays; the semantic entropy term still lacks an observation-conditioned expected posterior. The printed overall algorithm returns Success whenever the threshold triggers STOP, whereas actual success still requires environment evaluation.'
a['psgnav']['limitations']='Main method, Appendix A technical/memory/false-positive results and the released world-enumeration/outcome-memory/aggregation code were inspected. Exact episode order and full detector outputs were not independently reconstructed. The selected false-positive subset does not measure calibration on all deployment states. No ObjectNav benchmark was run.'

# Clarify the feedback interface rather than implying the agent sees the reference path.
notes['evonav']['method']=notes['evonav']['method'].replace('H-CoE computes NE/SR/ OSR / SPL from the completed predicted AND ground-truth path (Eq.7), then asks an LLM','An external evaluator computes terminal navigation metrics using benchmark ground truth (Eq.7); the supplement specifies that the agent receives its own rollout and terminal SR/SPL, not a ground-truth trajectory. H-CoE then asks an LLM')
notes['evonav']['bottlenecks'][0]['conditions']='Deployment without reliable terminal outcome feedback; changed scenes or reset-memory evaluation'

# Explicit implementation-specific findings enter the evidence ledger, not the scientific-gap list.
findings={
 'skillnav':('The inspected release assigns self.vln_bert per item while embedding the batch, then applies the last assigned expert head to all items. Batch-one versus mixed-expert batch-eight invariance is an unexecuted baseline diagnostic.','INSPECTED CODE / call-path deduction','skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py:647–664,838; commit b56de12a6af07ce09c9252938b1fb69e55d33103'),
 'mapnav':('The reporting helper adjusts failures within3.2m to success and SPL1.0; it prints both raw and adjusted metrics. This does not establish use in the paper.','INSPECTED CODE','caculate_metrics.py:9–14; commit4a3acc70270688630b4ca8f6df39f26585898e0d'),
 'navgrpo':('The invoked agent_grpo reward branches do not match all printed paper reward components. Exact published-run reproduction is unestablished.','INSPECTED CODE','main_nav.py:25; agent_grpo.py:724–744; commit4668d42506858ea971c62e26fdc6e4b71b75064e'),
 'polarmem':('The inspected release uses a fixed uncertainty margin and weighted database scoring rather than the paper\'s ensemble/dispersion margin and lexicographic ranking.','INSPECTED CODE','otsu_gate.py:25–46; neo4j_client.py:197–217,303–319; commit4d0de0befe3571ddeaccafd55ac19aa764ef9b0d'),
 'context_nav':('Repeated Unknown verification eventually passes, and unsupported/missing-height spatial relations soft-pass.','INSPECTED CODE','_ynu_defer.py:181–196; _spatial_reasoning.py:54–60'),
 'psgnav':('Failed-episode feedback marks recent detections as false positives even though episode failure can arise from control; incidence was not measured.','INSPECTED CODE / deduction','PSG_Nav.py:1497–1554'),
}
for ident,(claim,kind,loc) in findings.items():
 base=notes[ident].get('key_findings',[])
 a[ident]['key_findings']=base+[{'claim':claim,'evidence_type':kind,'source':'Official repository snapshot; linked in dossier','locator':loc}]

# Copy only already-inspected shared source files to give each dossier a self-contained local audit path.
src=ROOT/'repositories'/'03'/'auxthink';dest=ROOT/'repositories'/'03'/'progressthink'
for spec in a['progressthink']['code_files']:
 rel=spec.split(':')[0]; target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
 if (src/rel).exists():shutil.copy2(src/rel,target)

for p in paths:
 ids=[n['id'] for n in json.loads(p.read_text(encoding='utf-8'))]
 p.write_text(json.dumps([notes[i] for i in ids],ensure_ascii=False,indent=2),encoding='utf-8')
(E/'code_audits.json').write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
(E/'bibliography_closeout.json').write_text(json.dumps({'cutoff':'2026-10-22','sources':[{'paper':'ECoT','url':'https://proceedings.mlr.press/v270/zawalski25a.html','supports':'PMLR270 proceedings record, authors, pages3157–3181, publication2025; CoRL2024 event per volume index'},{'paper':'CONSOLE','url':'https://expectorlin.github.io/','supports':'Author confirms TPAMI2024 publication'}]},ensure_ascii=False,indent=2),encoding='utf-8')
print('Final records updated')
