"""Audit all matched E episodes and report paired scene uncertainty without tuning."""
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from Scorer.scorer import choose_action
from policy.experimental_reference import verify_bundle
from scorer_features.common import FROZEN,POLICY_ID,REFERENCE,TRIALS,RUN as FEATURES_RUN,now,read,sha,write
from tools.run_step9 import load_rows
from tools.run_step10 import RUN,CALIBRATION,verify


def scene_estimates(rows,weights,scenes):
    values=np.zeros((len(scenes),2,3));counts=np.zeros((len(scenes),2))
    for r in rows:
        s=scenes.index(r['scene']);d=('r2r','rxr').index(r['dataset'])
        counts[s,d]+=1;values[s,d]+=[r['success'],r['spl'],r['ndtw']]
    totals=np.einsum('ks,sdm->kdm',weights,values);denom=weights@counts
    assert (denom>0).all()
    return (totals/denom[:,:,None]).mean(axis=1)


def main():
    p=verify();selection=read(RUN/'selection.json')
    for path,digest in selection['dataset_file_sha256'].items():assert sha(path)==digest
    expected={(n,i) for n,ids in selection['episodes'].items() for i in ids}
    rows_by={};systems={};event_count=0;source_hashes={}
    for system in p['systems']:
        name=system['name'];directory=RUN/'evaluation'/name
        job=read(directory/'job.json');summary=read(directory/'summary.json');rows=load_rows(directory)
        assert sha(directory/'job.json')==summary['job_sha256'] and job['selection_sha256']==p['selection_sha256']
        assert job['checkpoint_sha256']==system['checkpoint_sha256'] and job['margin']==system['margin']
        assert {(r['dataset'],r['episode_id']) for r in rows}==expected and len(rows)==224
        for r in rows:
            assert r['group']=='E' and r['policy_identity_sha256']==POLICY_ID and r['STOP_bias']==-.25
            assert r['scene']==selection['episode_scenes'][r['dataset']][r['episode_id']]
            assert len(r['events'])==len(r['executed_actions'])==len(r['collided_after_action'])==r['steps']<=r['original_budget']
            assert len(r['positions'])==len(r['rgb_sha256'])==r['steps']+1
            assert r['forward_attempts']==r['executed_actions'].count(1) and r['collisions']==sum(r['collided_after_action'])
            assert r['executed_actions'][-1]==0 or r['steps']==r['original_budget']
            for t,event in enumerate(r['events']):
                raw=np.array(event['raw_action_scores'],dtype=np.float32);raw[0]-=np.float32(.25)
                assert np.array_equal(raw,np.asarray(event['biased_action_scores'],dtype=np.float32))
                assert int(raw.argmax())==event['base_live'] and event['t']==t
                assert event['selected_live']==r['executed_actions'][t]
                assert (1,2,3,0)[event['selected_canonical']]==event['selected_live']
                if t<3 or system['mode']=='policy':assert event['selected_live']==event['base_live'] and event['world_model_calls']==0
                if system['mode']!='predicted':assert event['world_model_calls']==0
                if event['scores'] is not None:
                    score=torch.tensor([event['scores']['utility']],dtype=torch.float32)
                    chosen=choose_action(score,torch.arange(4)[None],torch.tensor([event['base_canonical']]),system['margin'])
                    assert chosen.actions[0].item()==event['selected_canonical']
                    if system['mode']=='predicted':assert event['world_model_calls']==3
                if event['reason'] in ('prediction_unavailable','nonfinite_scores','insufficient_real_context','disabled','policy_only'):
                    assert event['selected_live']==event['base_live']
                event_count+=1
        memory=read(directory/'gpu_memory.json')
        assert memory['status']=='sampled' and memory['max_process_peak_mib'] is not None
        source_hashes[name]={'summary':sha(directory/'summary.json'),'job':sha(directory/'job.json'),'gpu_memory':sha(directory/'gpu_memory.json')}
        rows_by[name]=rows
        systems[name]={**summary,'sampled_GPU_process_peak_MiB':memory['max_process_peak_mib'],
            'sampled_GPU_workers_sum_peak_MiB':memory['peak_sum_of_worker_processes_mib'],
            'full_episode_seconds_per_action':sum(r['seconds'] for r in rows)/sum(r['steps'] for r in rows)}
    # Shared scene bootstrap preserves pairings and within-scene episode dependence.
    scenes=sorted({r['scene'] for r in rows_by['policy']});assert len(scenes)==11
    rng=np.random.default_rng(10);picks=rng.integers(0,len(scenes),(2000,len(scenes)))
    weights=np.eye(len(scenes),dtype=np.float64)[picks].sum(axis=1)
    draws={name:scene_estimates(rows,weights,scenes) for name,rows in rows_by.items()}
    uncertainty={name:{metric:np.quantile(values[:,i],[.025,.975]).tolist() for i,metric in enumerate(('SR','SPL','nDTW'))} for name,values in draws.items()}
    comparisons={}
    pairs=[(n,'policy') for n in rows_by if n!='policy']+[(n,'null') for n in TRIALS]+[(TRIALS[0],TRIALS[1])]+[('actual',n) for n in TRIALS]
    for left,right in pairs:
        difference=draws[left]-draws[right]
        comparisons[f'{left} minus {right}']={metric:{'difference':systems[left]['macro_'+key]-systems[right]['macro_'+key],
            'scene_bootstrap_95':np.quantile(difference[:,i],[.025,.975]).tolist()}
            for i,(metric,key) in enumerate((('SR','success'),('SPL','spl'),('nDTW','ndtw')))}
    reference=verify_bundle(REFERENCE);assert not reference['selection']['original_stopping_safeguard_passed']
    control=Path(reference['selection']['control_reference'])
    for name,digest in read(control/'sha256.json').items():assert sha(control/name)==digest
    for directory in [FROZEN/'encoder']+[FROZEN/n for n in TRIALS]+[CALIBRATION/'frozen'/n for n in ('pooled','null')]:
        for name,digest in read(directory/'file_sha256.json').items():assert sha(directory/name)==digest
    verify()
    report={'status':'passed','completed_utc':now(),'protocol_sha256':sha(RUN/'protocol.json'),
        'systems':5,'episodes_per_system':224,'total_episode_records':1120,'action_events_checked':event_count,
        'all_episode_and_artifact_hashes_checked':True,'identical_episode_sets_and_per_system_action_rules_verified':True,
        'native_base_decoding_STOP_identity_and_fallback_rules_verified':True,
        'shared_pooled_weights_and_margin_verified':True,'upstream_controls_and_weights_preserved':True,
        'no_E_tuning':True,'source_hashes':source_hashes}
    write(RUN/'verification.json',report)
    root_labels={r['root_id']:r for r in read(FEATURES_RUN/'index/roots.json') if r['group']=='C'}
    branch={}
    for variant in ('pooled','null'):
        bundle=read(CALIBRATION/'frozen'/variant/'bundle.json');step=bundle['selected_update']
        evaluation=read(Path(bundle['source_checkpoint']).parents[1]/'validation'/f'step_{step:06d}.json')
        by_rep={}
        for name in (['null'] if variant=='null' else list(TRIALS)):
            regrets=[];base_regrets=[]
            for row in evaluation['rows'][name]:
                r=root_labels[row['root_id']];scores=np.asarray(row['utility'],dtype=np.float32);base=r['base_action_canonical']
                best=int(scores.argmax());choice=best if scores[best]-scores[base]>np.float32(bundle['margin']) else base
                labels=np.array([r['candidates_canonical'][str(a)]['utility'] for a in range(4)])
                regrets.append(labels.max()-labels[choice]);base_regrets.append(labels.max()-labels[base])
            by_rep[name]={'mean_regret':float(np.mean(regrets)),'base_mean_regret':float(np.mean(base_regrets)),'roots':len(regrets)}
        branch[variant]={'selected_update':step,'margin':bundle['margin'],'representations':by_rep,
            'mean_regret':float(np.mean([x['mean_regret'] for x in by_rep.values()])),
            'scope':'C development branch regret at the frozen margin; no E counterfactual branch labels were fitted'}
    result={'status':'complete','completed_utc':now(),'scope':p['scope'],'policy_identity_sha256':POLICY_ID,'STOP_bias':-.25,
        'protocol_sha256':sha(RUN/'protocol.json'),'verification_sha256':sha(RUN/'verification.json'),
        'systems':systems,'scene_bootstrap_95':uncertainty,'paired_comparisons':comparisons,
        'scorer_selection':branch,'selection_caveat':reference['selection'],'RxR_protocol':p['RxR_protocol'],
        'actual_view_control_is_privileged':True,'no_E_tuning':True,
        'next':'Repeat the recorded procedure for the remaining declared world-model and policy trials under automatic progression authorization'}
    write(RUN/'result.json',result)
    names=['policy','null','actual',*TRIALS]
    columns=['system','episodes','SR','SPL','navigation_error_m','nDTW','SDTW','forward_collision_rate','mean_steps','override_rate','decision_mean_seconds','sampled_process_peak_MiB']
    with (RUN/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns)
        for name in names:
            s=systems[name];o=s['overall'];writer.writerow([name,o['episodes'],s['macro_success'],s['macro_spl'],o['distance_to_goal'],s['macro_ndtw'],s['macro_sdtw'],o['forward_collision_rate'],o['steps'],o['override_rate'],s['decision_latency_seconds']['mean'],s['sampled_GPU_process_peak_MiB']])
    lines=['# Step 10 — controlled unseen-scene pilot','',f"Completed: {result['completed_utc']}",'',
        'All five systems ran the same 224 episodes: 112 R2R and 112 English-guide RxR episodes across all 11 E scenes. This is a scene-balanced pilot, not the complete 5,508-episode unseen benchmark. Checkpoints, margins and preprocessing were frozen on C before any E results.','',
        'Refined SmolVLM2 checkpoint 512 and STOP bias −0.25 remain the reference. Its deliberate selection despite the original stopping-safeguard failure remains recorded. The Step 2 control and upstream frozen artifacts are intact. RxR uses the agreed 15-degree turns, differing from official 30-degree RxR-CE.','',
        '| System | SR % | SPL % | NE m | nDTW % | SDTW % | Forward collision % | Mean actions |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        s=systems[name];o=s['overall'];label=name+(' (privileged)' if name=='actual' else '')
        lines.append(f"| {label} | {100*s['macro_success']:.2f} | {100*s['macro_spl']:.2f} | {o['distance_to_goal']:.2f} | {100*s['macro_ndtw']:.2f} | {100*s['macro_sdtw']:.2f} | {100*o['forward_collision_rate']:.2f} | {o['steps']:.1f} |")
    lines+=['','The actual-view control previews real simulator outcomes and is privileged. It uses the same pooled scorer and margin as the DINO pair. The null control was trained separately with matched architecture, initialization, root order and update budget.','',
        '| System | SR 95% scene interval | ΔSR vs policy, points (95% interval) | Decision mean / p95, ms | Sampled process peak, GiB |',
        '| --- | --- | --- | ---: | ---: |']
    for name in names:
        s=systems[name];ci=uncertainty[name]['SR'];difference='—'
        if name!='policy':
            v=comparisons[f'{name} minus policy']['SR'];interval=v['scene_bootstrap_95']
            difference=f"{100*v['difference']:+.2f} ({100*interval[0]:+.2f}, {100*interval[1]:+.2f})"
        latency=s['decision_latency_seconds']
        lines.append(f"| {name} | {100*ci[0]:.2f}–{100*ci[1]:.2f}% | {difference} | {1000*latency['mean']:.1f} / {1000*latency['p95']:.1f} | {s['sampled_GPU_process_peak_MiB']/1024:.2f} |")
    lines+=['','Intervals resample scenes with pairing preserved (2,000 resamples, seed 10). Eleven scenes limit uncertainty estimates. Decision latency includes contention from four concurrent workers. GPU process peaks include CUDA/Habitat allocations sampled every two seconds; exact PyTorch allocator peaks and sampling records are also retained. These H100 measurements do not establish an 8 GB deployment fit.','',
        'C measured-branch regret is development evidence and is recorded with checkpoint/margin selection; this E pilot reports closed-loop outcomes and does not add counterfactual E branch labels. Full event records include overrides, warm-up and prediction/scoring fallbacks.','',
        '[Machine-readable results](result.json), [all checks](verification.json), [CSV metrics](metrics.csv), [locked protocol](protocol.json), and [episode selection](selection.json).','']
    (RUN/'RESULTS.md').write_text('\n'.join(lines))
    write(RUN/'status.json',{'status':'complete','completed_utc':now(),'result':str(RUN/'result.json')})
    write(ROOT/'runs/automatic_progression.json',{'status':'active','authorization':'User requested automatic progression on 2026-10-27',
        'completed_steps':[7,8,9,10],'completed_trial':'SmolVLM2 refined512, initial DINO pair',
        'next':'Remaining declared world-model/policy trial repeats','updated_utc':now()})
    print(json.dumps({k:v for k,v in result.items() if k not in ('systems','paired_comparisons')},indent=2))

if __name__=='__main__':main()
