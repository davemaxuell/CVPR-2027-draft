import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'evidence/02'
for f in [E/'notes.json',*sorted(E.glob('notes_extra*.json'))]:
 n=json.loads(f.read_text())
 if 'cvaml' in n:
  n['cvaml']['pages']='1-9,13-14,18-21,23'
  n['cvaml']['authors']='Claas A. Voelcker; Anastasiia Pedan; Arash Ahmadian; Romina Abachi; Igor Gilitschenski; Amir-Massoud Farahmand'
  n['cvaml']['results']=n['cvaml']['results'].replace('20 seeds','30 seeds')+' Figure 4, p. 9 explicitly aggregates 30 seeds per environment. Appendix Table 2, p. 21 specifies latent width 512, batch 128, discount 0.99, actor/critic/model learning rates 3e-4, encoder rate 1e-4, one-step model rollout, and one/four model samples. Appendix C.4, p. 23 shows the BYOL-style auxiliary loss is essential in its humanoid experiments: objective calibration alone is insufficient.'
  n['cvaml']['mechanism']+=' Appendix A.3, pp. 18–20, proves existence by continuity on a compact connected latent image and a measurable selection attaining the expected value. This supplies a measurable value-equivalent mapping for one fixed value function, not a continuous neural realization or a model preserving all language-conditioned values.'
 if 'monodream' in n:
  n['monodream']['access']='An initially inaccessible project page did not establish absence of code. A later primary-source search identified the official MonoDream implementation in HorizonRobotics/RoboOrchardLab, with author/checkpoint links in its project README. The final dossier records inspected inference/model/preprocessing/evaluation files and the absence of the complete auxiliary-training pipeline from that project listing.'
 if 'vid2world' in n:n['vid2world']['results']='Table 1, p. 8: RT1 FVD is 18.5 versus 24.2 for the action-conditioned baseline. Some FID comparisons are flagged because baseline code did not set the Inception network to evaluation mode. On RECON, Vid2World FVD is 59.4 versus NWM 31.2, and LPIPS 0.3236 versus 0.295, although SSIM improves from 0.389 to 0.481. Thus there is no uniform dominance. Table 2, p. 9: extrapolative transfer plus guidance gives FVD 22.4 versus masked transfer plus guidance 25.8. The real-to-sim experiment ranks three RT1 drawer-policy checkpoints; this is a narrow validation setting.'
 if 'visualplanning' in n:n['visualplanning']['results']='Table 1, p. 6: VPRL averages EM 80.6/PR 84.9, versus visual SFT 56.1/65.2 and Qwen SFT 53.6/69.9. EM accepts any shortest valid trajectory by parsing environment states rather than requiring exact pixel equality. This comparison isolates the training change within LVM more clearly than it establishes modality superiority across differently pretrained backbones.'
 if 'wmnav' in n:n['wmnav']['results']='Table I, p. 6: HM3D v0.1 SR 58.1/SPL 31.2; MP3D 45.4/17.2. MP3D SPL is below VLFM 17.5 despite broad superiority wording. Table II, HM3D v0.2: adding the two-stage action proposal raises SR from 70.1 to 72.2 while reducing SPL from 34.7 to 33.3. Table III gives Qwen 3B SR 29.7 versus Gemini 1.5 Pro 58.1, showing substantial backbone dependence. HM3D versions must be compared separately.'
 for x in n.values():
  for k in ['problem','mechanism','novelty','results','bottleneck','followups']:
   s=x.get(k,'')
   for a,b in {'3Dpoint':'3D point','SemanticCE':'Semantic CE','RGB1024':'RGB 1024','retainsIL':'retains IL','omitsMTM':'omits MTM','SectionIII':'Section III','underbackground':'under background','mIoU78':'mIoU 78','unified.27':'unified 0.27','seen.43':'seen 0.43','versus.42':'versus 0.42','BALD30':'BALD 30','IoU25':'IoU 25','specificVLM':'specific VLM','trainingRGB':'training RGB'}.items():s=s.replace(a,b)
   x[k]=s
 f.write_text(json.dumps(n,ensure_ascii=False,indent=2),encoding='utf-8')
# Save an evidence index with immutable links and actual source-extraction records.
codes=json.loads((E/'code_notes.json').read_text());log=json.loads((E/'code_read_log.json').read_text())
lines=['# Assignment 02: implementation and access audit','',
'Cutoff: 2026-10-22. This appendix links exact source snapshots. The narrative identifies substantive inspected lines; the extraction log also includes broader search excerpts and is not a claim that every line of every fetched repository was read. No fetched project code, checkpoint, simulator or training command was executed. Independent arithmetic checks are recorded separately in `logic_check_results.json`.','']
for pid,c in codes.items():
 rid=c.get('snapshot_id',pid);p=R/'repositories/02'/rid/'snapshot.json';meta=json.loads(p.read_text()) if p.exists() else {}
 lines+=['## '+pid,'',c['text'],'', 'Repository status: '+c['status']+'.','']
 if meta.get('commit'):
  lines+=['Snapshot: ['+meta['commit']+']('+meta['url']+'/tree/'+meta['commit']+').','']
  for path in c['files']:
   ranges=[x for x in log if x['repo']==rid and x['file']==path]
   nums=[]
   for x in ranges:
    if x.get('ranges'):nums+=[int(v.split('-')[0]) for v in x['ranges']]
    else:nums+=x.get('selected_lines',[])
   anchor='#L'+str(min(nums)) if nums else ''
   lines+=['- ['+path+']('+meta['url']+'/blob/'+meta['commit']+'/'+path+anchor+')']
  lines+=['']
(R/'assignments/02_repository_audit.md').write_text('\n'.join(lines),encoding='utf-8')
print('Notes and immutable source index updated')
