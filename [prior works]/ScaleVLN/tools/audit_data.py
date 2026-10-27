"""Check every configured training trajectory against local feature and graph keys."""
from collections import Counter
import json
from pathlib import Path

import h5py
import lmdb
import msgpack
import msgpack_numpy

ROOT = Path(__file__).resolve().parents[1]


def audit():
    msgpack_numpy.patch()
    r2r = ROOT / 'VLN-DUET/datasets/R2R'
    rvr = ROOT / 'VLN-DUET-RVR/datasets/REVERIE'
    report = {'passed': True, 'files': {}, 'errors': []}
    feature_keys = []
    for model in ['b16', 'h14']:
        with h5py.File(r2r / 'features' / f'clip_vit-{model}_mp3d_hm3d_gibson.hdf5', 'r') as handle:
            feature_keys.append(set(handle.keys()))
    images = feature_keys[0] & feature_keys[1]
    graph_keys = set()
    for path in (r2r / 'connectivity').glob('*_connectivity.json'):
        scan = path.name[:-len('_connectivity.json')]
        graph_keys.update(scan + '_' + row['image_id'] for row in json.loads(path.read_text()) if row['included'])
    candidates = set(json.loads((r2r / 'annotations/scanvp_candview_relangles_with_hm3d_gibson.json').read_text()))

    def inspect(path, object_db=None):
        counts = Counter()
        missing = []
        with path.open() as stream:
            rows = (json.loads(line) for line in stream) if path.suffix == '.jsonl' else iter(json.load(stream))
            for row in rows:
                counts['trajectories'] += 1
                for vp in set(row['path'] + row.get('pos_vps', [])):
                    key = row['scan'] + '_' + vp
                    for name, inventory in [('image_features', images), ('connectivity', graph_keys), ('candidates', candidates)]:
                        if key not in inventory:
                            counts['missing_' + name] += 1
                            if len(missing) < 8:
                                missing.append(name + ':' + key)
                if object_db is not None:
                    target = row['instr_id'].split('_')[1]
                    # At least one published positive viewpoint must expose the target object.
                    if not any(target in object_db.get(row['scan'] + '_' + vp, set()) for vp in row['pos_vps']):
                        counts['missing_target_objects'] += 1
                        if len(missing) < 8:
                            missing.append('object:' + row['instr_id'])
        result = dict(counts)
        result['examples_of_missing_assets'] = missing
        report['files'][str(path.relative_to(ROOT))] = result
        if missing:
            report['passed'] = False
            report['errors'].append(str(path))
        print(path.name, dict(counts), flush=True)

    for path in sorted((r2r / 'annotations/pretrain_map').glob('*.jsonl')):
        inspect(path)
    inspect(r2r / 'annotations/R2R_scalevln_ft_aug_enc.json')
    for directory, files in [
        ('obj_fts', ['ade20k_pseudo3d_depth2_epoch_94_beam0_zun_3_none.jsonl']),
        ('obj_gtmax_timm_imagenet_vitb16', ['REVERIE_val_seen_enc.jsonl', 'REVERIE_val_unseen_enc.jsonl']),
    ]:
        database = lmdb.open(str(rvr / 'features' / directory), readonly=True, lock=False)
        with database.begin() as transaction:
            objects = {key.decode(): set(msgpack.unpackb(value)['obj_ids']) for key, value in transaction.cursor()}
        database.close()
        for name in files:
            inspect(rvr / 'annotations/pretrain' / name, objects)
        if directory == 'obj_fts':
            inspect(rvr / 'annotations/ade20k_pseudo3d_depth2_epoch_94_beam0_sample10.jsonl', objects)
    report['reverie_pretraining_scope'] = '519342 HM3D trajectories; Gibson object features are absent from the local release'
    return report


if __name__ == '__main__':
    import sys
    result = audit()
    path = ROOT / 'runs/data_audit.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('Report:', path, flush=True)
    sys.exit(0 if result['passed'] else 1)
