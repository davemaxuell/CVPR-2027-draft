"""C-only checkpoint/margin selection, confirmation and frozen deployment bundles."""
import copy
import fcntl
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navigation_four.resources import monitored_run
from navigation_four.jobs import make_job
from navigation_eval.metrics import aggregate
from navigation_four.common import ENCODER_ID,POLICY_ID,TRIALS,TRAIN,PRIOR_CALIBRATION,now,read,sha,torch_save,write

RUN=ROOT/'runs/step9b_smolvlm2_refined512_four_models_v1'


def verify():
    p=read(RUN/'protocol.json')
    assert sha(RUN/'recipe.json')==p['recipe_sha256']
    assert sha(TRAIN/'protocol.json')==p['training_protocol_sha256']
    assert sha(TRAIN/'training_complete.json')==p['training_complete_sha256']
    assert sha(RUN/'check_runtime.json')==p['runtime_check_sha256']
    assert sha(PRIOR_CALIBRATION/'result.json')==p['prior_calibration_result_sha256']
    assert sha(PRIOR_CALIBRATION/'confirmation/policy/summary.json')==p['prior_base_confirmation_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==sha(RUN/'source'/name)==digest,name
    for variant,digest in p['training_result_sha256'].items():assert sha(TRAIN/variant/'result.json')==digest
    for c in p['candidates']:assert sha(c['checkpoint'])==c['checkpoint_sha256']
    for label in ('screen','remaining','full'):assert sha(RUN/f'selection_{label}.json')==p[f'{label}_selection_sha256']
    return p


def load_rows(out):
    out=Path(out);summary=read(out/'summary.json');assert summary['status']=='complete'
    if 'episode_sources' in summary:
        records=[]
        for source in summary['episode_sources'].values():
            assert sha(source['file'])==source['sha256'];records.append(read(source['file']))
        return records
    for name,digest in summary['episode_sha256'].items():assert sha(out/'episodes'/name)==digest
    return [read(out/'episodes'/name) for name in summary['episode_sha256']]


def merge_confirmation(out,parts):
    out=Path(out);rows=[];sources={};signatures=[]
    for part in parts:
        part=Path(part);job=read(part/'job.json')
        signatures.append({k:v for k,v in job.items() if k not in ('output','selection_file','selection_sha256')})
        for row in load_rows(part):
            key=f"{row['dataset']}_{row['episode_id']}.json";assert key not in sources
            file=part/'episodes'/key;sources[key]={'file':str(file),'sha256':sha(file)};rows.append(row)
    assert all(s==signatures[0] for s in signatures)
    selection=read(RUN/'selection_full.json')
    assert {(r['dataset'],r['episode_id']) for r in rows}=={(n,i) for n,ids in selection['episodes'].items() for i in ids}
    result={'status':'complete','completed_utc':now(),'group':'C','episode_sources':sources,
        'component_runs':[str(x) for x in parts],'controller_signature':signatures[0],**aggregate(rows,bootstrap=True)}
    write(out/'summary.json',result);return result


def mean_metrics(reports):
    return {k:float(np.mean([r[k] for r in reports])) for k in ('macro_success','macro_spl','macro_ndtw')}


def worse(value,base):
    return value['macro_success']<base['macro_success']-1e-12 or (
        abs(value['macro_success']-base['macro_success'])<=1e-12 and value['macro_spl']<base['macro_spl']-1e-12)


def freeze(variant,chosen,confirm_paths,confirmation,protocol):
    folder=RUN/'frozen'/variant;folder.mkdir(parents=True,exist_ok=True)
    source=Path(chosen['checkpoint']);state=torch.load(source,map_location='cpu',weights_only=True)
    assert state['variant']==variant and state['stage']=='predicted'
    cfg=copy.deepcopy(state['config']);cfg['selection']['override_margin']=chosen['margin']
    cfg['status']='frozen_after_C_branch_and_closed_loop_selection'
    bundle={'format':'frozen_four_WM_navigation_scorer_v1','variant':variant,'model':state['model'],'config':cfg,
        'policy_identity_sha256':POLICY_ID,'encoder_identity_sha256':ENCODER_ID,'STOP_bias':-.25,
        'world_model_trials':list(TRIALS),'minimum_real_observations':4,'source_checkpoint':str(source),
        'source_checkpoint_sha256':sha(source),'training_protocol_sha256':sha(TRAIN/'protocol.json'),
        'calibration_protocol_sha256':sha(RUN/'protocol.json')}
    path=folder/'scorer.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['source_checkpoint_sha256']==bundle['source_checkpoint_sha256'] and saved['config']==cfg
        for name,value in saved['model'].items():assert torch.equal(value,state['model'][name])
    else:torch_save(path,bundle)
    write(folder/'config.json',cfg)
    metadata={'status':'frozen','variant':variant,'checkpoint':str(path),'checkpoint_sha256':sha(path),
        'source_checkpoint':str(source),'source_checkpoint_sha256':sha(source),'selected_update':state['step'],
        'margin':chosen['margin'],'candidate_id':chosen['id'],'C_confirmation':confirmation,
        'C_confirmation_runs':{k:str(v) for k,v in confirm_paths.items()},
        'policy_identity_sha256':POLICY_ID,'encoder_identity_sha256':ENCODER_ID,'STOP_bias':-.25,
        'selection_caveat':read(TRAIN/'protocol.json')['selection_caveat'],
        'same_weights_and_margin_across_four_models':True,'trainable_parameters':0,
        'calibration_protocol_sha256':sha(RUN/'protocol.json')}
    write(folder/'bundle.json',metadata)
    write(folder/'file_sha256.json',{name:sha(folder/name) for name in ('scorer.pt','config.json','bundle.json')})
    return metadata


def main():
    lock=(RUN/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify()
    if (RUN/'result.json').exists():assert read(RUN/'result.json')['status']=='complete';return
    def execute(output,selection,**kwargs):
        verify();job=make_job(output,RUN/f'selection_{selection}.json',workers=p['workers'],seed=p['sampling_seed'],**kwargs)
        write(RUN/'status.json',{'status':'running','pid':__import__('os').getpid(),'job':str(job),'updated_utc':now()})
        return monitored_run(job)
    try:
        screening=[]
        for candidate in p['candidates']:
            names=list(TRIALS) if candidate['variant']=='pooled' else ['null']
            reports=[];paths={}
            for name in names:
                out=RUN/'screen'/candidate['id']/name
                reports.append(execute(out,'screen',mode='predicted' if candidate['variant']=='pooled' else 'null',
                    checkpoint=candidate['checkpoint'],world_model=name if candidate['variant']=='pooled' else None,margin=candidate['margin']))
                paths[name]=str(out)
            screening.append({**candidate,'screen':mean_metrics(reports),'screen_paths':paths})
            write(RUN/'screening.json',screening)
        selected={}
        for variant in ('pooled','null'):
            choices=[c for c in screening if c['variant']==variant]
            selected[variant]=sorted(choices,key=lambda c:(-c['screen']['macro_success'],-c['screen']['macro_spl'],
                -c['screen']['macro_ndtw'],c['branch_regret'],-c['margin'],c['step']))[0]
        write(RUN/'screen_selected.json',selected)
        base=read(PRIOR_CALIBRATION/'confirmation/policy/summary.json')
        assert base['status']=='complete'
        write(RUN/'confirmation/policy/summary.json',{**base,'reused_from':str(PRIOR_CALIBRATION/'confirmation/policy')})
        # The authoritative selected policy must still reproduce its locked C224 trace.
        source=ROOT/'runs/step3b_smolvlm2_stop/confirmation_A/candidate/episodes'
        for row in load_rows(RUN/'confirmation'/'policy'):
            old=read(source/f"{row['dataset']}_{row['episode_id']}.json")
            for key in ('executed_actions','collided_after_action','positions','success','spl'):assert row[key]==old[key],(row['episode_id'],key)
        frozen={};confirmation_records={}
        for variant,chosen in selected.items():
            names=list(TRIALS) if variant=='pooled' else ['null'];reports=[];paths={}
            for name in names:
                remaining=RUN/'remaining'/chosen['id']/name
                execute(remaining,'remaining',mode='predicted' if variant=='pooled' else 'null',checkpoint=chosen['checkpoint'],
                    world_model=name if variant=='pooled' else None,margin=chosen['margin'])
                merged=RUN/'confirmation'/chosen['id']/name
                reports.append(merge_confirmation(merged,[chosen['screen_paths'][name],remaining]));paths[name]=merged
            confirmation=mean_metrics(reports);fallbacks=[]
            if worse(confirmation,base):
                for conservative in (1.,1000000.):
                    trial_reports=[];trial_paths={};zero=True;exact=True
                    for name in names:
                        out=RUN/'conservative'/variant/f'm{conservative:g}'/name
                        trial_reports.append(execute(out,'full',mode='predicted' if variant=='pooled' else 'null',
                            checkpoint=chosen['checkpoint'],world_model=name if variant=='pooled' else None,margin=conservative))
                        trial_paths[name]=out
                        for row in load_rows(out):
                            old=read(source/f"{row['dataset']}_{row['episode_id']}.json")
                            zero=zero and row['overrides']==0;exact=exact and row['executed_actions']==old['executed_actions']
                    fallbacks.append({'margin':conservative,'zero_overrides':zero,'base_actions_exact':exact,'metrics':mean_metrics(trial_reports)})
                    if zero and exact:
                        chosen={**chosen,'margin':conservative,'id':chosen['id']+f'_conservative_m{conservative:g}'}
                        confirmation=mean_metrics(trial_reports);paths=trial_paths;break
                else:raise RuntimeError('Conservative margin failed to reproduce base C224 behavior')
            confirmation_records[variant]={'screen_selected':selected[variant],'initial_confirmation':mean_metrics(reports),
                'fallbacks':fallbacks,'final_confirmation':confirmation}
            frozen[variant]=freeze(variant,chosen,paths,confirmation,p)
            write(RUN/'confirmation_decisions.json',confirmation_records)
        verify()
        result={'status':'complete','completed_utc':now(),'protocol_sha256':sha(RUN/'protocol.json'),
            'policy_identity_sha256':POLICY_ID,'STOP_bias':-.25,'C_episodes':224,'frozen':frozen,
            'baseline_C':{k:base[k] for k in ('macro_success','macro_spl','macro_ndtw')},
            'confirmation_decisions':confirmation_records,'base_C224_actions_reproduced_exactly':True,
            'E_not_used_for_training_or_selection':True,'next':'Predeclared four-model E pilot repeat, automatically authorized'}
        write(RUN/'result.json',result);write(RUN/'status.json',{'status':'complete','completed_utc':now()})
        training_result={'status':'complete','completed_utc':now(),'protocol_sha256':sha(TRAIN/'protocol.json'),
            'updates_per_variant':2048,'variants':{v:read(TRAIN/v/'result.json') for v in ('pooled','null')},
            'frozen_scorers':frozen,'closed_loop_selection':str(RUN/'result.json'),
            'scope':'One scorer pooled across all four frozen world models, matched learned-null-future control; B fitting, C selection, no E tuning'}
        write(TRAIN/'result.json',training_result);write(TRAIN/'status.json',{'status':'complete','completed_utc':now()})
        write(ROOT/'runs/automatic_progression.json',{'status':'active','authorization':'user requested automatic progression on 2026-10-27',
            'completed_trial':'initial DINO pair','current_trial':'four-model SmolVLM2 repeat','completed_steps':[7,8,9],'current_step':10,'updated_utc':now(),'policy_identity_sha256':POLICY_ID})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:
        write(RUN/'status.json',{'status':'failed','error':repr(exc),'updated_utc':now()});raise

if __name__=='__main__':main()
