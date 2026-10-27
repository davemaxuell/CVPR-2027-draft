"""Lock the matched training recipe after real-data and resumption checks."""
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scorer_training.common import (CACHE,FEATURES_RUN,POLICY_ID,RUN,TRIALS,flags,now,read,sha,verify_inputs,write)


def main():
    manifest=verify_inputs();checks=read(RUN/'check_training.json');orchestration=read(RUN/'check_orchestration.json')
    assert checks['status']==orchestration['status']=='passed'
    for report in (checks,orchestration):
        for name,digest in report['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,info in manifest['files'].items():assert sha(CACHE/name)==info['sha256'],name
    sources=sorted(list((ROOT/'scorer_training').glob('*.py'))+list((ROOT/'tools').glob('*step8*.py')))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    protocol={'format':'scorer_training_protocol_v1','version':'smolvlm2_refined512_dino_pair_v1',
        'feature_manifest_sha256':sha(CACHE/'manifest.json'),'feature_protocol_sha256':manifest['protocol_sha256'],
        'policy_identity_sha256':POLICY_ID,'STOP_bias':-.25,
        'selection_caveat':read(FEATURES_RUN/'result.json')['preserved_selection_caveat'],
        'world_model_trials':list(TRIALS),'world_model_identity_input':False,
        'groups':{'fit':'B','development':'C','final_evaluation_untouched':'E'},
        'seed':8,'epochs':16,'total_updates':2048,'roots_per_epoch':1024,'roots_per_batch':8,
        'stages':[{'name':'actual','start_epoch':0,'end_epoch':4,'updates':512},
                  {'name':'mixed','start_epoch':4,'end_epoch':8,'updates':512},
                  {'name':'predicted','start_epoch':8,'end_epoch':16,'updates':1024}],
        'model_sampling':'Exactly 1/2 each within each predicted epoch; mixed epochs use 512 actual, 256 transfer, 256 fresh roots',
        'optimizer':{'name':'AdamW','learning_rate':1e-4,'weight_decay':.01,'gradient_clip_norm':1.,'schedule':'constant'},
        'bptt':32,'memory':'Recompute entire real prefix for every training batch under current weights; detach before last at most 32 real updates',
        'external_feature_gradients':False,'STOP_future':'current-view identity; no world-model call',
        'variants':{'pooled':'One scorer across both DINO models',
                    'null':'Same parameters, initialization, root order and update budget; zero movement input learns a shared null vector through existing projection biases/type embedding; no actual or predicted movement views enter scoring'},
        'parameters_per_variant':2890115,'checkpoint_every_updates':128,
        'selection':{'branch_metric':'mean C utility regret at zero margin, averaged equally over both DINO models',
            'eligible_stages':['predicted'],'closed_loop_shortlist_per_variant':2,
            'deployment':'C closed-loop checks and margin calibration required before freezing; no E tuning',
            'actual_view_diagnostic':'Report separately; inspect actual-stage fit and C regret before expanding models'},
        'runtime':flags(),'profile_step_seconds':checks['steady_step_seconds'],
        'native_checks_sha256':sha(RUN/'check_training.json'),'orchestration_checks_sha256':sha(RUN/'check_orchestration.json'),
        'source_sha256':hashes}
    if (RUN/'protocol.json').exists():assert read(RUN/'protocol.json')==protocol
    else:
        for name in hashes:
            dest=RUN/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        write(RUN/'protocol.json',protocol)
    print('Locked',sha(RUN/'protocol.json'))

if __name__=='__main__':main()
