"""Verify live four-model features/scores and native-policy simulator parity."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from Scorer.scorer import choose_action
from navigation_four.common import (ADDITIONS, BASE_RUN, CALIBRATION, FEATURES_RUN, TRAIN, TRIALS, now, read, sha, write)
from navigation_four.controller import NavigationController
from navigation_four.jobs import make_job, run_job, source_hashes
from scorer_features.cache import checked_image
from scorer_features.common import LABELS
from scorer_four.data import FrozenScorerDataset


def main():
    assert read(TRAIN / 'training_complete.json')['status'] == 'trained'
    checkpoint = TRAIN / 'pooled/checkpoints/step_000512.pt'
    roots = read(BASE_RUN / 'index/roots.json'); episodes = read(BASE_RUN / 'index/episodes.json')
    images = read(BASE_RUN / 'index/images.json')
    selected = [min((r for r in roots if (r['group'],r['dataset'],r['prefix_source']) == (g,d,m)), key=lambda r:r['t'])
                for g in ('B','C') for d in ('r2r','rxr') for m in ('expert','policy')]
    extension = {r['root_id']:r for r in read(FEATURES_RUN / 'roots.json')}
    checks = []
    for trial in TRIALS:
        controller = NavigationController('predicted', checkpoint, trial, 0.)
        for root in selected:
            data = FrozenScorerDataset(root['group'])
            index = next(i for i,r in enumerate(data.roots) if r['root_id'] == root['root_id'])
            prefix = data.prefix(index); record = data.episode(root['episode_index']); episode = episodes[root['episode_index']]
            controller.reset(episode['instruction']['text'], episode['budget'])
            controller.set_episode_identity(root['group'], root['dataset'], root['episode_id'], root['prefix_source'])
            reference_memory = None
            for t in range(root['t'] + 1):
                item = images[episode['frame_ids'][t]]
                image = checked_image(LABELS / item['path'], item['sha256'], item['raw_rgb_sha256'])
                native = controller.observe(image, episode['actions_live'][:t])
                assert torch.equal(native['context'], record['context'][t])
                assert torch.equal(native['raw_logits'], record['raw_logits_live'][t])
                with torch.inference_mode():
                    reference = controller.scorer.observe(**data.observation(prefix,t), memory=reference_memory)
                    reference_memory = reference.memory
                assert torch.equal(controller.memory, reference_memory)
                if t < root['t']:
                    controller.margin = 0. if t < 3 else None
                    action, event = controller.decide()
                    assert action == native['base_live'] and event['world_model_calls'] == 0
                    controller.commit(episode['actions_live'][t]); controller.margin = 0.
            controller.attempted_predictions = 0
            future = controller.futures(); assert controller.attempted_predictions == 3
            assert torch.equal(future.cpu(), data.futures(index, trial)), (trial,root['root_id'],'live future mismatch')
            assert torch.equal(future[:,3], controller.context.current_patches)
            if trial in ADDITIONS:
                assert controller.last_sample_seeds == {k:extension[root['root_id']][k] for k in ('context_seed','diffusion_seed')}
            with torch.inference_mode():
                actions = torch.arange(4, device='cuda')[None]
                expected = controller.scorer.score(reference, data.futures(index,trial), actions)
                choice = choose_action(expected['utility'], actions, torch.tensor([root['base_action_canonical']], device='cuda'), 0.)
            action,event = controller.decide()
            assert event['selected_canonical'] == int(choice.actions[0]) and event['world_model_calls'] == 3
            for key,value in expected.items(): assert np.array_equal(event['scores'][key],value[0].cpu().numpy())
            before = controller.memory.clone(); original = controller.futures
            def broken(*args): raise RuntimeError('Injected predictor failure')
            controller.futures = broken
            action,event = controller.decide(); assert action == controller.base_live and event['reason'] == 'prediction_unavailable'
            controller.margin = None
            action,event = controller.decide(); assert action == controller.base_live and event['world_model_calls'] == 0
            controller.futures = original; controller.margin = 0.
            if trial in ADDITIONS:
                saved = controller.vae_history.pop(); controller.vae_history.append(None)
                action,event = controller.decide(); assert action == controller.base_live and event['reason'] == 'prediction_unavailable'
                controller.vae_history.pop(); controller.vae_history.append(saved)
            assert torch.equal(before, controller.memory)
            controller.commit(0); assert controller.terminated
            checks.append({'world_model':trial, 'root_id':root['root_id'], 'native_features_memory_future_and_score_exact':True})
            print(checks[-1], flush=True)
        del controller; torch.cuda.empty_cache()
    # The subclass also must preserve a complete native rollout through the simulator.
    full = read(CALIBRATION / 'selection_full.json')
    selection = {**full, 'episodes':{n:ids[:1] for n,ids in full['episodes'].items()}}
    path = CALIBRATION / 'runtime_selection.json'
    if path.exists(): assert read(path) == selection
    else: write(path,selection)
    job = make_job(CALIBRATION / 'runtime_policy',path,mode='policy',workers=2,seed=0)
    run_job(job); parity = []
    source = ROOT / 'runs/step3b_smolvlm2_stop/confirmation_A/candidate/episodes'
    for name,ids in selection['episodes'].items():
        for eid in ids:
            previous=read(source/f'{name}_{eid}.json'); current=read(CALIBRATION/'runtime_policy/episodes'/f'{name}_{eid}.json')
            for key in ('executed_actions','collided_after_action','positions','success','spl','distance_to_goal'):
                assert current[key] == previous[key], (name,eid,key)
            assert [e['raw_action_scores'] for e in current['events']] == previous['action_scores']
            parity.append({'dataset':name,'episode_id':eid,'native_rollout_exact':True})
    sources = source_hashes(); sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    write(CALIBRATION / 'check_runtime.json', {'status':'passed','completed_utc':now(),'source_sha256':sources,
        'checkpoint_sha256':sha(checkpoint),'cache_runtime_checks':checks,'complete_simulator_native_parity':parity,
        'STOP_identity_no_predictions':True,'warmup_disabled_and_predictor_failure_fallback_verified':True,
        'NWM_missing_real_context_fallback_verified':True,'no_E_used':True})


if __name__ == '__main__':
    main()
