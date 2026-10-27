"""Publish verified group-B/C catalogs and the Step 6 handoff without training a model."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.experimental_reference import verify_bundle
from scorer_data.common import DATA, REFERENCE, RUN, now, read, sha, verify_protocol, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='full')
    args = ap.parse_args()
    protocol = verify_protocol()
    signature = sha(RUN / 'protocol.json')
    plan = read(RUN / 'plan.json')
    assert plan['protocol_sha256'] == signature
    assert sha(RUN / 'runtime_environment.json') == plan['runtime_environment_sha256']
    assert sha(RUN / 'simulator_configs.json') == protocol['simulator_configs_sha256']
    for name, expected in read(RUN / 'runtime_environment.json')['native_runtime_source_sha256'].items():
        assert sha(name) == expected, ('Native runtime changed', name)
    stage = RUN / args.stage
    status, verification = read(stage / 'status.json'), read(stage / 'verification.json')
    assert status['status'] == 'complete' and verification['status'] == 'passed'
    assert status['groups'] == ['B', 'C']
    assert status['roots_B'] == plan['roots_B'] and status['roots_C'] == plan['roots_C']
    assert verification['protocol_sha256'] == signature
    assert verification['index_sha256'] == sha(stage / 'index.json')
    for name, expected in read(stage / 'audit_source_sha256.json').items():
        assert sha(ROOT / name) == sha(stage / 'source' / name) == expected
    for name, expected in protocol['source_sha256'].items():
        assert sha(RUN / 'source' / name) == expected
    ref = verify_bundle(REFERENCE)
    assert not ref['selection']['original_stopping_safeguard_passed']
    assert sha(Path(ref['source_checkpoint']) / 'adapter_model.safetensors') == ref['identity_fields']['adapter_sha256']
    control = Path(ref['selection']['control_reference'])
    for name, expected in read(control / 'sha256.json').items():
        assert sha(control / name) == expected
    compatible = {}
    for name in ('dinowm_transfer', 'dinov2_fresh_dynamics'):
        bundle = ROOT / 'runs/step5_smolvlm2_refined512/frozen' / name
        cfg = read(bundle / 'config.json')
        assert cfg['selected_policy_identity_sha256'] == protocol['policy_identity_sha256']
        for filename, expected in read(bundle / 'file_sha256.json').items():
            assert sha(bundle / filename) == expected
        compatible[name] = {'bundle': str(bundle), 'config_sha256': sha(bundle / 'config.json'),
                            'encoder_identity_sha256': cfg['encoder_identity_sha256']}
    root_paths = read(stage / 'index.json')['roots']
    catalogs = {'B': [], 'C': []}
    for relative in root_paths:
        path = DATA / relative
        row = read(path)
        branch_rows = {str(read(path.parent / filename)['candidate_canonical']): read(path.parent / filename)
                       for filename in row['branch_files'].values()}
        item = {'root_id': str(Path(relative).parent), 'group': row['group'], 'dataset': row['dataset'],
                'prefix_source': row['mode'], 'scene': row['scene'], 'episode_id': row['episode_id'], 't': row['root_t'],
                'labels': relative, 'labels_sha256': sha(path),
                'carrier': str((path.parent / row['carrier']).resolve().relative_to(DATA)), 'carrier_sha256': row['carrier_sha256'],
                'base_action_canonical': row['base_action_canonical'],
                'policy_identity_sha256': row['policy_identity_sha256'],
                'candidates_canonical': {a: {'action_name': b['candidate_name'], 'live_action': b['candidate_live'],
                     'utility': b['utility'], 'success': b['success'], 'ndtw': b['ndtw'],
                     'remaining_actions': b['remaining_actions'], 'remaining_collisions': b['remaining_collisions'],
                     'immediate_forward_collision': b['immediate_forward_collision'],
                     'first_next_rgb': str((path.parent / b['first_next_file']).relative_to(DATA)),
                     'first_next_sha256': b['first_next_sha256']} for a, b in sorted(branch_rows.items())}}
        catalogs[row['group']].append(item)
    catalog_info = {}
    for group, rows in catalogs.items():
        path = DATA / f'catalog_{group}.jsonl'
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
        tmp.replace(path)
        catalog_info[group] = {'file': str(path), 'roots': len(rows), 'sha256': sha(path),
                               'purpose': 'fitting' if group == 'B' else 'development only'}
    selection = read(RUN / 'selection.json')
    exclusions, attempted, carrier_steps = [], 0, 0
    for group in ('B', 'C'):
        for rank, (name, mode) in enumerate([('r2r', 'expert'), ('r2r', 'policy'), ('rxr', 'expert'), ('rxr', 'policy')]):
            count = read(stage / f'index_{group}_{rank}.json')['attempted_episodes']
            for eid in selection[group][name][mode][:count]:
                c = read(DATA / group / name / mode / eid / 'carrier/meta.json')
                attempted += 1
                carrier_steps += c['num_actions']
                if not c['eligible']:
                    exclusions.append({'group': group, 'dataset': name, 'mode': mode, 'episode_id': eid,
                                       'reasons': c['exclusion_reasons'], 'num_actions': c['num_actions'],
                                       'max_expert_deviation_m': c['max_expert_deviation_m'], 'terminal_metrics': c['metrics_trace'][-1]})
    write(RUN / 'exclusions.json', exclusions)
    completed = now()
    manifest = {'format': 'verified_scorer_branch_corpus_v1', 'status': 'complete', 'completed_utc': completed,
                'policy_identity_sha256': protocol['policy_identity_sha256'], 'protocol': str(RUN / 'protocol.json'),
                'protocol_sha256': signature, 'catalogs': catalog_info, 'selection_caveat': ref['selection'],
                'verification': str(stage / 'verification.json'), 'verification_sha256': sha(stage / 'verification.json'),
                'groups': {'fitting': 'B', 'development': 'C'}, 'canonical_actions': {'0': 'FORWARD', '1': 'LEFT', '2': 'RIGHT', '3': 'STOP'},
                'branch_count': 4 * len(root_paths), 'prefix_sources': ['expert', 'selected_policy'],
                'world_models_used_in_labels': False, 'compatible_frozen_world_models': compatible}
    write(DATA / 'manifest.json', manifest)
    result = {'status': 'complete', 'completed_utc': completed, 'scope': 'Step 6: measured four-action labels on B and separate C development roots',
              'policy_identity_sha256': protocol['policy_identity_sha256'], 'STOP_bias': -.25,
              'corpus': str(DATA), 'corpus_manifest_sha256': sha(DATA / 'manifest.json'),
              'catalogs': catalog_info, 'branches': 4 * len(root_paths), 'attempted_carrier_episodes': attempted,
              'carrier_actions': carrier_steps, 'excluded_carriers': len(exclusions),
              'by_group': verification['by_group'], 'by_stratum': verification['by_stratum'],
              'verification': str(stage / 'verification.json'), 'runtime_verification': str(RUN / 'check_runtime.json'),
              'all_label_checks_passed': True, 'selected_policy_and_STOP_bias_preserved': True,
              'Step2_control_preserved': True, 'Step5_frozen_models_preserved': True,
              'failed_stopping_safeguard_selection_preserved': True, 'finalizer_source_sha256': sha(__file__),
              'next': 'Step 7 frozen policy/visual feature extraction; then scorer fitting in Step 8'}
    write(RUN / 'result.json', result)
    lines = ['# Step 6 complete — measured scorer labels', '', f'Completed: {completed}', '',
             'Each root has four actual simulator outcomes: forced STOP/FORWARD/LEFT/RIGHT, then the fixed refined SmolVLM2 policy for movement branches. The selected policy is checkpoint 512 with STOP logit bias −0.25, native saved preprocessing, and batch-one greedy decoding.', '',
             '| Group | Purpose | Roots | Branches | Distinct source episodes | Base successful branches | Best-utility successful branches | Mean best-candidate utility gain |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group in ('B', 'C'):
        row = verification['by_group'][group]
        lines.append(f"| {group} | {'Fitting' if group == 'B' else 'Development'} | {row['roots']:,} | {row['branches']:,} | {row['episodes']:,} | {row['base_successes']:,} | {row['best_utility_candidate_successes']:,} | {row['mean_best_candidate_gain']:.6f} |")
    lines += ['', 'The best candidate uses measured outcomes and is a privileged oracle diagnostic. These are branches from mixed expert/policy roots, not end-to-end episode success rates or scorer improvements. Results by dataset and root source, including successes, timeouts, label gaps and distinct outcomes, are in the verification report.', '',
              'Group B and C are scene-disjoint. All four dataset/prefix-source strata have equal root counts within each group. Episode queues and evenly spaced decision indices were fixed independently of branch outcomes. Valid expert prefixes require successful low-drift replay; policy failures are retained. Short eligible histories are retained, and exclusions are listed in [exclusions.json](exclusions.json).', '',
              'Every branch restores the original episode, random seed, actual prefix, metric state and remaining budget by verified replay. nDTW includes the full earlier trajectory. Counts include the candidate and any executed terminal STOP, with only forward collisions charged. STOP executes through the evaluator and uses the current RGB as its next-view identity.', '',
              '[Live runtime checks](check_runtime.json) and [complete label audit](' + args.stage + '/verification.json) passed. The audit independently recomputed every utility and nDTW, checked every continuation decoding decision and budget, and hashed all indexed image files. The saved policy, failed-safeguard selection record, Step 2 control and Step 5 models are preserved.', '',
              'The DINO world models were not used for action selection or label construction. Step 7 will extract frozen features and each world model’s predictions for the same measured labels. The NWM/CDiT pair remains a later repeat.', '',
              f'Corpus: `{DATA}`. `catalog_B.jsonl` is for fitting; `catalog_C.jsonl` is for development. Both record canonical action IDs and immutable image/label references. See [methods](../../docs/Step6_Run.md), [locked protocol](protocol.json), and [result.json](result.json).', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))
    write(RUN / 'status.json', {'status': 'complete', 'stage': args.stage, 'completed_utc': completed, 'result': str(RUN / 'result.json')})
    print(json.dumps({'status': 'complete', 'catalogs': catalog_info, 'branches': 4 * len(root_paths)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
