"""Reuse completed image audits; bind corrective records and full-split source identity."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from context import *

sys.path.insert(0, str(SOURCE))
from stream_policy.config import StreamConfig
from stream_policy.data import TrajectoryWindows


def metadata_identity(root, group, config):
    module_spec = importlib.util.spec_from_file_location('record_alignment', SOURCE / 'tools/audit_stream_data.py')
    audit = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(audit)
    scenes = set(read_json(SOURCE / 'manifest/scene_partition.json')['groups']['FULL_TRAIN']['scenes'])
    signature = sha256(root / 'collection_config.json')
    digest = hashlib.sha256(); counts = Counter()
    for dataset, selected in config['selected_episodes'].items():
        assert len(selected) == len(set(selected))
        paths = sorted(p for p in (root / dataset).glob('*/meta.json') if not p.parent.name.startswith('.'))
        assert {p.parent.name for p in paths} == set(selected)
        for path in paths:
            raw = path.read_bytes(); row = json.loads(raw)
            assert row['episode_id'] == path.parent.name and row['dataset'] == dataset
            assert row['data_kind'] == 'streamvln_dagger' and row['group'] == group and row['split'] == 'train'
            assert row['scene'] in scenes and row['collection_signature_sha256'] == signature
            assert audit.dagger_record_alignment(row)[0]
            expert = read_json(Path(AUTH['expert_root']) / dataset / path.parent.name / 'meta.json')
            assert expert['scene'] == row['scene'] and expert['instruction']['text'] == row['instruction']['text']
            if row['keep']:
                assert (path.parent / 'rgb/0000.jpg').is_file()
                assert (path.parent / 'rgb' / ('%04d.jpg' % (row['num_frames'] - 1))).is_file()
            counts[dataset + '_attempted'] += 1
            counts[dataset + '_kept'] += int(row['keep'])
            digest.update(str(path.relative_to(root)).encode() + b'\0' + raw)
    return dict(meta_sha256=digest.hexdigest(), counts=dict(counts),
        assets={name:sha256(root/name) for name in ['collection_config.json','summary.json','r2r_index.jsonl','rxr_index.jsonl']})


def audit(branch):
    check_authorization()
    fresh = branch == 'J'
    root = Path(AUTH['fresh_dagger_root'] if fresh else AUTH['original_dagger_root'])
    expert = read_json(HERE / 'evidence/expert_data_audit.json')
    identity = read_json(HERE / 'evidence/data_identity.json')
    assert expert['passed'] and identity['status'] == 'verified'
    for path, value in identity['assets'].items():
        assert sha256(path) == value['sha256'], 'Expert identity changed: ' + path
    config = read_json(root / 'collection_config.json')
    summary = read_json(root / 'summary.json')
    assert summary['status'] == 'complete' and config['beta'] == 0 and config['stop_bias'] == 0
    assert config['datasets'] == ['r2r','rxr']
    if fresh:
        policy = f_checkpoint()
        assert config['weights'] == policy['checkpoint'] and config['group'] == 'FULL_TRAIN'
        assert config['weight_sha256']['model.safetensors'] == policy['checkpoint_sha256']['model.safetensors']
        expected = read_json(HERE / 'collection/plan.json')
        assert config['selected_episodes'] == expected['selected_episodes']
        complete = read_json(Path(policy['checkpoint'])/'complete.json')
        for name, digest in config['weight_sha256'].items():assert complete['files'][name] == digest
        module_spec = importlib.util.spec_from_file_location('fresh_dagger_auditor', SOURCE / 'tools/audit_stream_data.py')
        auditor = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(auditor)
        with ProcessPoolExecutor(4) as pool:
            image_audit = auditor.audit_dagger(root, StreamConfig(), auditor.simulator_settings(), pool)
        assert image_audit['passed']
        for path in (root/'workers').glob('*.identity.json'):
            worker = read_json(path)
            assert worker['weights'] == policy['checkpoint'] and worker['weight_sha256'] == config['weight_sha256']
            assert worker['stop_logit_bias'] == 0
        # Token lengths depend on action/instruction layout, not on training root.
        from stream_policy.qwen35 import Qwen35Inputs
        _, token_audit = auditor.audit_windows([str(root)], StreamConfig(), Qwen35Inputs(StreamConfig()), .5, 'fresh J')
        assert token_audit['prompt_tokens']['over_max_length'] == 0
        prior = dict(frame_audit=image_audit, token_audit=token_audit)
    else:
        assert config['weights'] == AUTH['stage_one_checkpoint'] and config['group'] == 'A'
        assert config['weight_sha256'] == {'model.safetensors':AUTH['stage_one_sha256']}
        old = read_json(root/'data_audit.json')
        assert old['passed'] and old['dagger'][str(root)]['passed']
        assert old['dagger'][str(root)]['windows']['prompt_tokens']['over_max_length'] == 0
        prior = dict(reused_audit=str(root/'data_audit.json'), reused_audit_sha256=sha256(root/'data_audit.json'),
            note='All retained frames and token windows previously audited; current metadata, endpoints, selection, source instructions and identity rechecked without redundant rendering or full image decoding.')
    metadata = metadata_identity(root, 'FULL_TRAIN' if fresh else 'A', config)
    data = TrajectoryWindows([str(root)], StreamConfig(), None)
    if not fresh:assert len(data) == 46907
    assert len(data) > 0
    windows = 109784 + len(data)
    report = dict(status='complete', passed=True, time=utc(), train_roots=[AUTH['expert_root'],str(root)],
        expert_audit_sha256=sha256(HERE/'evidence/expert_data_audit.json'),
        expert_windows=109784, dagger_windows=len(data), combined_windows=windows,
        updates=windows//128, samples=windows//128*128, tail_windows=windows%128,
        dagger_usable_episodes=len(data.episodes), metadata_identity=metadata, prior_checks=prior,
        full_training_scene_and_instruction_alignment=True, heldout_leakage=False,
        duplicate_note='Episode IDs shared by expert and corrective roots identify separate trajectories; no duplicate IDs within a root.',
        actual_data_scope='Full audited R2R-CE / RxR-CE English-guide experts; corrections '+('from fresh frozen F' if fresh else 'restricted to original group A from frozen 623'))
    atomic_json(HERE/'audits'/('fresh_dagger.json' if fresh else 'existing_mixture.json'), report)
    print(json.dumps({k:report[k] for k in ['status','combined_windows','updates','dagger_usable_episodes']}), flush=True)


if __name__ == '__main__':
    audit(sys.argv[1])
