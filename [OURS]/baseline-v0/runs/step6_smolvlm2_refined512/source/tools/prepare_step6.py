"""Lock Step 6 policy, evaluator, scene-stratified episode queues and source files."""
import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from habitat.datasets import make_dataset
from omegaconf import OmegaConf
from policy.experimental_reference import verify_bundle
from scorer_data.common import DATA, REFERENCE, RUN, now, read, sha, write
from sim.config import EPISODES, EXPERT, load_manifest, make_ce_config, scene_group
from sim.episodes import select_episodes


def main():
    ref = verify_bundle(REFERENCE)
    manifest = load_manifest()
    assert manifest['simulator'] == ref['identity_fields']['simulator']
    selection, inputs, configs = {}, {}, {}
    for group in ('B', 'C'):
        selection[group] = {}
        for name in ('r2r', 'rxr'):
            cfg = make_ce_config(name, 'train', scenes=scene_group(group), gpu=0)
            full = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
            selection[group][name] = {}
            configs[f'{group}/{name}'] = OmegaConf.to_container(cfg, resolve=True)
            for mode, parity in [('expert', 0), ('policy', 1)]:
                pool = [ep for ep in full.episodes if hashlib.sha256(f'6/{name}/{ep.episode_id}'.encode()).digest()[0] % 2 == parity]
                chosen = select_episodes(pool, 0, 6)
                assert {ep.scene_id.split('/')[-2] for ep in chosen} == set(scene_group(group))
                selection[group][name][mode] = [str(ep.episode_id) for ep in chosen]
            for template in (EPISODES[name], EXPERT[name]):
                path = template.format(split='train', role='guide')
                inputs[path] = sha(path)
    assert not set(scene_group('B')) & set(scene_group('C'))
    for group in ('B', 'C'):
        assert not set(scene_group(group)) & (set(scene_group('A')) | set(scene_group('E')))
    RUN.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    selection_path = RUN / 'selection.json'
    if selection_path.exists():
        assert read(selection_path) == selection
    else:
        write(selection_path, selection)
    write(RUN / 'simulator_configs.json', configs)
    sources = ['scorer_data/__init__.py', 'scorer_data/common.py', 'scorer_data/metrics.py', 'scorer_data/collection.py',
               'tools/prepare_step6.py', 'tools/collect_scorer_labels.py', 'tools/check_step6.py',
               'Scorer/interfaces.py', 'Scorer/labels.py', 'sim/config.py', 'sim/episodes.py', 'sim/rxr_dataset.py',
               'sim/transition_branch.py', 'policy/common.py', 'policy/smolvlm2.py', 'policy/online.py',
               'policy/experimental_reference.py', 'manifest/scene_partition.json',
               'runs/step6_smolvlm2_refined512/assets/vlnce_measures.py',
               'runs/step6_smolvlm2_refined512/assets/evaluator_source.json']
    protocol = {'format': 'scorer_branch_protocol_v1', 'policy_identity_sha256': ref['policy_identity_sha256'],
                'reference': str(REFERENCE), 'decoder': ref['decoder'], 'simulator': manifest['simulator'],
                'data': str(DATA), 'groups': {'fit': 'B', 'development': 'C'},
                'selection': 'seed-6 scene-balanced episode queues; disjoint expert/policy episodes; no policy-success filter',
                'root_sampling': 'up to eight evenly spaced eligible decision indices from t=3 through the final decision; retain short eligible histories',
                'expert_filter': 'actual expert replay succeeds, consumes the full expert schedule, and has forward-location drift below 0.5m',
                'restore': 'reset original episode and RNG seed; replay every prefix action; assert exact pose, RGB hash, collisions, metrics and elapsed count',
                'random_protocol': 'deterministic greedy batch-one native policy; identical per-episode seed restored before every branch',
                'continuation': 'one forced candidate, then frozen selected policy until actual STOP or original task limit',
                'candidates_live': {'STOP': 0, 'FORWARD': 1, 'LEFT': 2, 'RIGHT': 3},
                'canonical_to_live': [1, 2, 3, 0], 'warmup_observations': 4,
                'evaluator': {'success': 'configured Habitat Success, same saved policy evaluator',
                              'ndtw': 'pinned VLN-CE NDTW.update_metric on full actual path, exact DTW Euclidean backend, consecutive duplicates skipped, 3m normalization',
                              'source': read(RUN / 'assets/evaluator_source.json')},
                'utility': {'success': .7, 'ndtw': .3, 'remaining_actions_over_Bmax': -.01, 'remaining_forward_collisions_over_Bmax': -.02},
                'STOP': 'execute actual STOP; current RGB identity; n=1, c=0; evaluator supplies success/nDTW',
                'rgb': 'live 640x480 uint8 RGB, lossless PNG; native saved policy preprocessing; no future pose/collision inputs',
                'budget_counts': 'candidate and executed terminal STOP included; Bmax=500 R2R,700 RxR; original prefix clock retained',
                'selection_caveat': ref['selection'], 'world_models_used_for_collection': False,
                'source_sha256': {p: sha(ROOT / p) for p in sources}, 'input_sha256': inputs,
                'selection_sha256': sha(selection_path), 'simulator_configs_sha256': sha(RUN / 'simulator_configs.json')}
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == protocol, 'Step 6 protocol changed; version the run'
    else:
        write(RUN / 'protocol.json', protocol)
    for name in sources:
        target = RUN / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == protocol['source_sha256'][name]
        else:
            shutil.copy2(ROOT / name, target)
    write(DATA / 'protocol_reference.json', {'protocol': str(RUN / 'protocol.json'), 'sha256': sha(RUN / 'protocol.json')})
    print('Step 6 protocol locked: ' + sha(RUN / 'protocol.json'), flush=True)


if __name__ == '__main__':
    main()
