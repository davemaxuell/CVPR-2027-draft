#!/usr/bin/env python3
"""Validate the completed R2R, RxR and REVERIE data and write a JSON report."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--datasets', nargs='+', choices=['r2r','rxr','reverie','scenes'],
                        default=['r2r','rxr','reverie','scenes'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.umask(0o007)
    data = args.data_root.resolve()
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'scope': args.datasets,
              'data_root': str(data), 'checks': [], 'datasets': {}}

    def check(name, good, detail):
        report['checks'].append({'name': name, 'passed': bool(good), 'detail': detail})
        print(('PASS ' if good else 'FAIL ') + name + ': ' + str(detail), flush=True)

    graphs = {}
    for path in (data/'connectivity').glob('*_connectivity.json'):
        graphs[path.name.split('_')[0]] = {x['image_id'] for x in json.loads(path.read_text()) if x['included']}

    if 'scenes' in args.datasets:
        scans = (data/'scans.txt').read_text().split()
        missing, bad_jpgs, jpg_count = [], [], 0
        for scan in scans:
            for extension in ['.glb','.navmesh','.house','_semantic.ply']:
                path = data/'scene_datasets/mp3d'/scan/(scan+extension)
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path))
            if scan not in graphs:
                missing.append('connectivity: '+scan)
                continue
            sky = data/'mattersim/v1/scans'/scan/'matterport_skybox_images'
            for viewpoint in graphs[scan]:
                for face in range(6):
                    path = sky/('%s_skybox%d_sami.jpg' % (viewpoint,face))
                    if not path.is_file():
                        missing.append(str(path))
            for path in sky.glob('*.jpg'):
                jpg_count += 1
                with path.open('rb') as stream:
                    head=stream.read(2)
                    stream.seek(-2,2)
                    tail=stream.read(2)
                if head!=b'\xff\xd8' or tail!=b'\xff\xd9':
                    bad_jpgs.append(str(path))
            camera = data/'mattersim/v1/scans'/scan/'undistorted_camera_parameters'
            if not camera.is_dir() or not any(camera.iterdir()):
                missing.append(str(camera))
        check('All 90 Habitat and MatterSim environments', len(scans)==90 and not missing,
              {'scans':len(scans),'missing':missing[:20]})
        check('Skybox JPEG markers',not bad_jpgs,{'jpgs':jpg_count,'invalid':bad_jpgs[:20]})

    if 'r2r' in args.datasets:
        result = {}
        for split in ['train','val_seen','val_unseen','test']:
            path=data/'datasets/R2R'/('R2R_'+split+'.json')
            episodes=json.loads(path.read_text())
            bad=[e['path_id'] for e in episodes if any(v not in graphs.get(e['scan'],set()) for v in e['path'])]
            result[split]={'paths':len(episodes),'instructions':sum(len(e['instructions']) for e in episodes),
                           'scans':len({e['scan'] for e in episodes}), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            expected={'train':4675,'val_seen':340,'val_unseen':783,'test':1391}[split]
            check('Original R2R/'+split,len(episodes)==expected and not bad,{**result[split],'bad_paths':bad[:10]})
        report['datasets']['R2R']=result

    if 'reverie' in args.datasets:
        for version in ['v1','v2']:
            root=data/'datasets/REVERIE'
            bbox_root=root/'bbox' if version=='v1' else root/'v2/bbox'
            boxes={}
            for path in bbox_root.glob('*.json'):
                boxes[path.stem]={vp:set(objects) for vp,objects in json.loads(path.read_text()).items()}
            check('REVERIE '+version+' box files',len(boxes)==10567,{'json_files':len(boxes)})
            result={}
            for split in ['train','val_seen','val_unseen','test']:
                path=root/(split if version=='v1' else 'v2')/(split+'.json')
                episodes=json.loads(path.read_text()); bad_paths=[]; bad_objects=[]
                for e in episodes:
                    if any(v not in graphs.get(e['scan'],set()) for v in e['path']):
                        bad_paths.append(e['id'])
                    vp=e['path'][-1]
                    if 'objId' in e and str(e['objId']) not in boxes.get(e['scan']+'_'+vp,{}).get(vp,set()):
                        bad_objects.append(e['id'])
                result[split]={'paths':len(episodes),'instructions':sum(len(e['instructions']) for e in episodes),
                               'missing_target_boxes':bad_objects,'invalid_paths':bad_paths}
                check('REVERIE '+version+'/'+split,not bad_paths and not bad_objects,result[split])
            report['datasets']['REVERIE_'+version]=result
        archive=data/'datasets/.downloads/REVERIE_BBoxes_v2.zip'
        with zipfile.ZipFile(archive) as z:
            bad=z.testzip()
        check('REVERIE v2 source archive CRC',bad is None,{'bad_member':bad})

    if 'rxr' in args.datasets:
        root=data/'datasets/RxR'
        missing=[]; unreadable=[]; counts=Counter(); sizes=Counter(); examples={}
        inventory=data/'splits/rxr_original_download_manifest.jsonl'
        with inventory.open() as stream:
            for line in stream:
                item=json.loads(line); path=root/item['name']; group=item['name'].rsplit('/',1)[0] if '/' in item['name'] else 'annotations_and_extras'
                counts[group]+=1; sizes[group]+=item['size']
                if not path.is_file() or path.stat().st_size!=item['size']:
                    missing.append(item['name'])
                elif not os.access(path,os.R_OK):
                    unreadable.append(item['name'])
                if path.suffix=='.npz':
                    kind='follower' if '_follower_' in path.name else 'guide' if '_guide_' in path.name else path.stem.split('_')[1]
                    examples.setdefault((group,kind),path)
        check('RxR complete upstream inventory',not missing and not unreadable,
              {'files':sum(counts.values()),'bytes':sum(sizes.values()),'missing_or_wrong_size':missing[:20],'unreadable':unreadable[:20]})
        receipts=list((data/'splits').glob('rxr_original_download_status_[0-9][0-9].json'))
        if receipts:
            rows=[json.loads(path.read_text()) for path in receipts]
        else:
            rows=[json.loads((data/'splits/rxr_original_download_status.json').read_text())]
        verified=sum(row.get('verified_files',0) for row in rows)
        check('RxR published MD5 download receipts',verified==sum(counts.values()) and all(row['phase']=='complete' for row in rows),
              {'verified_files':verified,'receipts':len(rows)})
        features=root/'text_features'; poses=root/'pose_traces'; annotations={}; coverage_missing=[]; bad_paths=[]
        for split in ['train','val_seen','val_unseen','test_standard','test_challenge']:
            name='rxr_'+split+('_public_guide.jsonl.gz' if split.startswith('test_') else '_guide.jsonl.gz')
            total=0; languages=Counter()
            with gzip.open(root/name,'rt') as stream:
                for line in stream:
                    e=json.loads(line); total+=1; languages[e['language']]+=1
                    for language in ['en','hi','te']:
                        path=features/('rxr_'+split)/('%06d_%s_text_features.npz' % (e['instruction_id'],language))
                        if not path.is_file(): coverage_missing.append(str(path))
                    if not split.startswith('test_'):
                        path=poses/('rxr_'+split)/('%06d_guide_pose_trace.npz' % e['instruction_id'])
                        if not path.is_file(): coverage_missing.append(str(path))
                    if any(v not in graphs.get(e['scan'],set()) for v in e.get('path',[])):
                        bad_paths.append(e['instruction_id'])
            annotations[split+'_guide']={'records':total,'languages':dict(languages)}
            if not split.startswith('test_'):
                count=0
                with gzip.open(root/('rxr_'+split+'_follower.jsonl.gz'),'rt') as stream:
                    for line in stream:
                        e=json.loads(line); count+=1
                        path=poses/('rxr_'+split)/('%06d_follower_pose_trace.npz' % e['demonstration_id'])
                        if not path.is_file(): coverage_missing.append(str(path))
                annotations[split+'_follower']={'records':count}
        check('Original RxR annotation, feature and pose coverage',not coverage_missing and not bad_paths,
              {'splits':annotations,'missing_assets':coverage_missing[:20],'invalid_paths':bad_paths[:20]})
        ce_root=data/'datasets/RxR_VLNCE_v0'; ce_missing=[]; ce_counts={}
        check('RxR-CE feature link',(ce_root/'text_features').resolve()==features.resolve(),str(ce_root/'text_features'))
        for split in ['train','val_seen','val_unseen','test_challenge']:
            for path in sorted((ce_root/split).glob('*.json.gz')):
                if path.name.endswith('_gt.json.gz'): continue
                with gzip.open(path,'rt') as stream: episodes=json.load(stream)['episodes']
                ce_counts[path.stem]=len(episodes)
                for e in episodes:
                    inst=e['instruction']; language=inst['language'].split('-')[0]
                    feature=ce_root/'text_features'/('rxr_'+split)/('%06d_%s_text_features.npz' % (int(inst['instruction_id']),language))
                    if not feature.is_file(): ce_missing.append(str(feature))
        check('RxR-CE feature coverage',not ce_missing,{'episodes':ce_counts,'missing':ce_missing[:20]})
        bad_samples=[]
        for path in examples.values():
            with zipfile.ZipFile(path) as z:
                if z.testzip(): bad_samples.append(str(path))
                for name in z.namelist():
                    with z.open(name) as array:
                        if array.read(6)!=b'\x93NUMPY':
                            bad_samples.append(str(path)+': '+name+' invalid NPY magic')
                            continue
                        version=array.read(2)
                        length=int.from_bytes(array.read(2 if version[0]==1 else 4),'little')
                        header=ast.literal_eval(array.read(length).decode('utf-8' if version[0]==3 else 'latin1').strip())
                        if name=='features.npy' and (len(header['shape'])!=2 or header['shape'][1]!=768):
                            bad_samples.append(str(path)+': invalid BERT feature dimensions')
        check('RxR NumPy samples across splits, languages and roles',not bad_samples,{'sample_files':len(examples),'bad_files':bad_samples})
        report['datasets']['RxR']={'components':{key:{'files':counts[key],'bytes':sizes[key]} for key in counts},'annotations':annotations,'ce_episodes':ce_counts}
        # Remove abandoned range chunks only after every final file is present
        # and the published MD5 completion receipts cover the entire mirror.
        if not missing and verified==sum(counts.values()):
            stale=root/'.download_chunks'
            if stale.exists(): shutil.rmtree(stale)
    report['passed']=all(item['passed'] for item in report['checks'])
    output=args.output or data/'splits/dataset_completion_report.json'
    temp=output.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2)+'\n'); temp.replace(output)
    print('Report:',output)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()
