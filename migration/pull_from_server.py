#!/usr/bin/env python3
"""Copy everything that is not on GitHub straight from the old server, with rsync over SSH.

On the new machine, from the repo root (see RETRIEVE_FROM_SERVER.md):

    python3 -I migration/pull_from_server.py list                                   # items, sizes, priorities
    python3 -I migration/pull_from_server.py pull  --host USER@OLD_SERVER --dry-run
    python3 -I migration/pull_from_server.py pull  --host USER@OLD_SERVER           # all 'required' items
    python3 -I migration/pull_from_server.py pull  --host USER@OLD_SERVER --priority recommended
    python3 -I migration/pull_from_server.py pull  --host USER@OLD_SERVER --item explorer_meshes our_checkpoints
    python3 -I migration/pull_from_server.py check --host USER@OLD_SERVER           # anything left to copy?

On the old server, to refresh the sizes in migration/server_manifest.json:

    python3 -I migration/pull_from_server.py manifest

`pull` includes every item at or above the given priority (required < recommended < optional; `skip` items only
with --item). rsync resumes interrupted copies (--partial) and skips files that are already identical, so
re-running is safe. Destinations are relative to this repo; items from the server's model caches go to
--hf-home / --home. The server's host name is not stored here because the repo is public. Standard library + rsync.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'migration/server_manifest.json'
SERVER_ROOT = '/data/team_a/dave-workspace/RoleVLN'
SERVER_HF = '/data/shared/hf_cache'
SERVER_HOME = '/home/bufsgpu'
PRIORITIES = ['required', 'recommended', 'optional', 'skip']

SCANS = 'data/mattersim/v1/scans'
DUET = '[prior works]/ScaleVLN/VLN-DUET/datasets'
RVR = '[prior works]/ScaleVLN/VLN-DUET-RVR/datasets'
ETP = '[prior works]/ETPNav/data'
PROC = 'data/processed/baseline_v0'
CKPT_FILTERS = ['+ */', '+ *.pt', '+ *.pth', '+ *.safetensors', '+ *.ckpt', '+ *.pth.tar', '- *']


def item(id, priority, src, use, size_hint='', filters=(), dest=None, note='', alt=''):
    """src: folder on the server, relative to SERVER_ROOT unless absolute; dest: folder here, relative to the repo
    (or starting with {HF_HOME} / {HOME}); filters: rsync filter rules relative to src."""
    return {'id': id, 'priority': priority, 'src': src, 'dest': dest if dest is not None else src,
            'filters': list(filters), 'use': use, 'note': note, 'alt': alt}


ITEMS = [
    # --- required: the explorer, our training/eval, and testing the prior works --------------------------------
    item('scenes_mp3d', 'required', 'data/scene_datasets/mp3d',
         'Habitat scenes for everything CE: baseline-v0, ETPNav, all CE datasets, build_navmaps.py',
         filters=['- *_semantic.ply'], note='_semantic.ply (15 G) left out: nothing reads it',
         alt='Matterport3D download_mp.py --task habitat'),
    item('mattersim_skyboxes', 'required', SCANS, 'explorer panoramas / front view / "Photo at agent"',
         filters=['+ /*/', '+ /*/matterport_skybox_images/***', '- *'],
         alt='download_mp.py --type matterport_skybox_images'),
    item('mattersim_house_meta', 'required', SCANS, 'explorer floors and room names (.house, panorama_to_region)',
         filters=['+ /*/', '+ /*/house_segmentations/', '- /*/house_segmentations/*.ply',
                  '+ /*/house_segmentations/*', '- *'],
         note='the .ply meshes are the separate optional item mattersim_house_ply',
         alt='download_mp.py --type house_segmentations'),
    item('r2r_ce', 'required', 'data/datasets/R2R_VLNCE_v1-3_preprocessed', 'baseline-v0 R2R-CE episodes + experts',
         alt='VLN-CE README (Google Drive)'),
    item('r2r_ce_minimal', 'required', 'data/datasets/R2R_VLNCE_v1-3', 'R2R-CE minimal release (IR2R-CE links)',
         alt='VLN-CE README'),
    item('rxr_ce', 'required', 'data/datasets/RxR_VLNCE_v0', 'baseline-v0 RxR-CE episodes; ETPNav nDTW ground truth',
         note='its text_features link points at the skipped 159 G RxR text features (item rxr_text_features)',
         alt='VLN-CE README'),
    item('ir2r_ce', 'required', 'data/datasets/IR2R-CE', 'CE dataset (planned iterative-CE experiments)',
         filters=['- /known_maps.zip'], alt='github.com/jacobkrantz/IVLN-CE'),
    item('reverie_ce', 'required', 'data/datasets/REVERIE-CE', 'CE dataset (planned)', alt='HF MrZihanWang/Dynam3D'),
    item('navrag_ce', 'required', 'data/datasets/NavRAG-CE', 'CE dataset (planned; training also needs HM3D, never had)',
         alt='HF MrZihanWang/Dynam3D'),
    item('ha_vln', 'required', 'data/datasets/HA-VLN', 'CE dataset HA-VLN 2.0 (episodes, HAPS 2.0 humans, CMA ckpt)',
         filters=['- /original/huggingface/HAPS2_0/HAPS2_0.zip'], note='the 7.8 G zip is left out; HAPS2_0/ is extracted',
         alt='HF fly1113/HA-VLN'),
    item('etpnav_data', 'required', ETP, 'ETPNav eval: CE episodes, released checkpoints, waypoint predictors',
         filters=['- /scene_datasets', '- /datasets/RxR_VLNCE_v0', '- /logs/explorer/', '- /logs/checkpoints/explorer_*/',
                  '- /logs/tensorboard_dirs/'],
         note='the two dataset symlinks are already in git (relative)', alt='modelscope download --model admagic/ETPNav'),
    item('etpnav_explorer_views', 'required', f'{ETP}/logs/explorer',
         'explorer CE runs: habitat cube faces at every map node (episode logs are already in git)',
         note='Matterport-derived renders, so not in the public repo', alt='rerun tools/log_etpnav_steps.py (~45 min/split)'),
    item('duet_eval', 'required', f'{DUET}/R2R', 'DUET/ScaleVLN eval, explorer what-if and log_duet_steps.py',
         filters=['+ /features/', '+ /features/clip_vit-b16_mp3d_original.hdf5', '+ /features/clip_vit-h14_mp3d_original.hdf5',
                  '+ /trained_models/', '+ /trained_models/finetune/***', '- *'],
         alt='HF OpenGVLab/ScaleVLN'),
    item('explorer_train_steplogs', 'required', f'{DUET}/R2R/exprs_map',
         'explorer: DUET train-split step logs (the two files over GitHub\'s 100 MB limit)',
         filters=['+ /steplogs/', '+ /steplogs/*/', '+ /steplogs/*/steps_train.jsonl', '- *'],
         note='also a release asset'),
    item('our_checkpoints', 'required', '[OURS]/baseline-v0/runs',
         'all our checkpoints + optimizer states (15 are also in git; the rest also on the release)',
         filters=CKPT_FILTERS, note='includes the 3 upstream DINO-WM checkpoints under step5/assets'),
    item('hf_smolvlm2', 'required', f'{SERVER_HF}/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct',
         'policy backbone (revision 7b375e1b)', dest='{HF_HOME}/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct',
         alt='env/download_models.py'),
    item('hf_dinov2_small', 'required', f'{SERVER_HF}/hub/models--facebook--dinov2-small', 'world-model encoder',
         dest='{HF_HOME}/hub/models--facebook--dinov2-small', alt='env/download_models.py'),
    item('hf_nwm', 'required', f'{SERVER_HF}/hub/models--facebook--nwm', 'NWM/CDiT (gated on HF)',
         dest='{HF_HOME}/hub/models--facebook--nwm', alt='HF facebook/nwm (request access)'),
    item('hf_dino_wm', 'required', f'{SERVER_HF}/hub/datasets--gaoyuezhou--dino_wm', 'DINO-WM transfer source',
         dest='{HF_HOME}/hub/datasets--gaoyuezhou--dino_wm', alt='HF gaoyuezhou/dino_wm'),
    item('hf_bert_base_uncased', 'required', f'{SERVER_HOME}/.cache/huggingface/hub/models--bert-base-uncased',
         'ScaleVLN tokenizer (it runs with HF_HUB_OFFLINE=1 and the default cache)',
         dest='{HOME}/.cache/huggingface/hub/models--bert-base-uncased', alt='HF bert-base-uncased'),
    item('clip_cache', 'required', f'{SERVER_HOME}/.cache/clip', 'ETPNav CLIP ViT-B/32', dest='{HOME}/.cache/clip',
         alt='downloaded by clip.load'),
    # --- recommended: saves hours of regeneration ---------------------------------------------------------------
    item('explorer_meshes', 'recommended', '[prior works]/ScaleVLN/analysis/vln_viz/cache',
         'explorer 3D building + per-floor detail meshes', filters=['+ /meshes/***', '+ /meshes_hd/***', '- *'],
         note='Matterport-derived; otherwise rebuild with build_meshes.py --detail from mattersim_house_ply'),
    item('processed_scorer_labels', 'recommended', f'{PROC}/scorer_labels_smolvlm2_refined512_v1',
         'Step 6 labels + branch traces; needed for the relabel-and-retrain scorer test'),
    item('processed_scorer_features', 'recommended', f'{PROC}/scorer_features_smolvlm2_refined512_dino_pair_v1',
         'Step 7 scorer features (also on the release; else tools/cache_step7.py)'),
    item('processed_dagger', 'recommended', PROC, 'DAgger round 1 + STOP round 2 data',
         filters=['+ /dagger_groupA_round1_smolvlm2/***', '+ /dagger_groupA_round2_smolvlm2_stop2m/***', '- *']),
    item('processed_wm_corpus', 'recommended', PROC, 'Step 4 world-model corpus (visits + index)',
         filters=['+ /wm_visits_groupA_smolvlm2_refined512_v1/***', '+ /wm_groupA_smolvlm2_refined512_v1/***', '- *']),
    item('processed_expert_c600', 'recommended', f'{PROC}/expert_groupC_600', 'group-C validation renders (SFT val)'),
    # --- optional ------------------------------------------------------------------------------------------------
    item('mattersim_house_ply', 'optional', SCANS, 'house meshes: only to rebuild the explorer mesh caches',
         filters=['+ /*/', '+ /*/house_segmentations/', '+ /*/house_segmentations/*.ply', '- *']),
    item('mattersim_camera_params', 'optional', SCANS, 'MatterSim camera parameters (nothing here reads them)',
         filters=['+ /*/', '+ /*/undistorted_camera_parameters/***', '- *']),
    item('scenes_mp3d_semantic', 'optional', 'data/scene_datasets/mp3d', 'Habitat semantic meshes (unused)',
         filters=['+ /*/', '+ /*/*_semantic.ply', '- *']),
    item('rxr_pose_traces_val', 'optional', 'data/datasets/RxR/pose_traces', 'only to rebuild the explorer RxR sub-tasks',
         filters=['+ /rxr_val_seen/***', '+ /rxr_val_unseen/***', '- *']),
    item('rxr_annotations', 'optional', 'data/datasets/RxR', 'original RxR annotations (val guide files are in git)',
         filters=['+ /*.jsonl.gz', '+ /*.json', '+ /LICENSE', '+ /README.upstream.md', '- *']),
    item('duet_training', 'optional', f'{DUET}', 'everything else for retraining ScaleVLN/DUET (features, aug, LXMERT)',
         filters=['- /R2R/exprs_map/'], alt='HF OpenGVLab/ScaleVLN'),
    item('reverie_training', 'optional', f'{RVR}/REVERIE', 'ScaleVLN REVERIE features/checkpoints/annotations',
         alt='HF OpenGVLab/ScaleVLN'),
    item('processed_expert_a', 'optional', f'{PROC}/expert_groupA', 'group-A expert renders',
         note='re-render instead: tools/render_expert.py (20 min on the H100)'),
    item('processed_wm_dino_features', 'optional', f'{PROC}/wm_dino_features_refined512_v1', 'cached DINO features',
         note='recompute from the expert renders instead'),
    item('runs_frames_and_indices', 'optional', '[OURS]/baseline-v0/runs', 'simulator frame dumps + Step 5 data index',
         filters=['+ */', '+ *.jpg', '+ /step5_smolvlm2_refined512/data_index/*', '- *']),
    item('discrete_benchmarks', 'optional', 'data/datasets', 'R2R, R4R, REVERIE, SOON, CVDN, NDH, HANNA annotations',
         filters=['+ /R2R/***', '+ /R4R/***', '+ /REVERIE/***', '+ /SOON/***', '+ /CVDN/***', '+ /NDH/***', '+ /HANNA/***',
                  '- *']),
    item('discrete_shared', 'optional', 'data', 'ResNet-152 features + MatterSim graphs for those benchmarks',
         filters=['+ /img_features/***', '+ /connectivity/***', '- *']),
    item('gsa_r2r', 'optional', 'data/datasets/GSA-R2R', 'GSA-R2R annotations, graphs, CLIP features'),
    # --- skip: listed so nothing is forgotten; pulled only with --item ---------------------------------------
    item('rxr_text_features', 'skip', 'data/datasets/RxR/text_features', 'only VLN-CE CMA/Seq2Seq RxR baselines'),
    item('rxr_pose_traces_train', 'skip', 'data/datasets/RxR/pose_traces/rxr_train', 'nothing reads it'),
    item('matterport_zips', 'skip', 'data/.downloads', 'original Matterport/Habitat zips (extracted copies above)'),
    item('dataset_download_cache', 'skip', 'data/datasets/.downloads', 'archives already extracted'),
    item('scalevln_smoke_runs', 'skip', '[prior works]/ScaleVLN/runs', 'setup smoke tests and demo exports'),
    item('legacy', 'skip', 'data', 'PRM800K, NavVEM episodes, hime task data, download logs',
         filters=['+ /datasets/', '+ /datasets/PRM800K/***', '+ /processed/', '+ /processed/navvem_episodes/***',
                  '+ /hime_task_data/***', '+ /logs/***', '- *']),
]


def src_path(it):
    return it['src'] if it['src'].startswith('/') else f'{SERVER_ROOT}/{it["src"]}'


def filter_args(it):
    return [f'--filter={f}' for f in it['filters']] + (['--prune-empty-dirs'] if it['filters'] else [])


def stats(output):
    size = re.search(r'Total file size: ([\d,]+)', output)
    files = re.search(r'Number of files: [\d,]+ \(reg: ([\d,]+)', output) or re.search(r'Number of files: ([\d,]+)', output)
    moved = re.search(r'Number of regular files transferred: ([\d,]+)', output)
    num = lambda m: int(m.group(1).replace(',', '')) if m else 0
    return num(size), num(files), num(moved)


def cmd_manifest(a):
    out = []
    for it in ITEMS:
        src = src_path(it)
        if not os.path.exists(src):
            print(f'  missing on this server: {it["id"]} ({src})')
            size = files = 0
        else:
            r = subprocess.run(['rsync', '-an', '--stats', *filter_args(it), src.rstrip('/') + '/', '/nonexistent-dry-run/'],
                               capture_output=True, text=True)
            size, files, _ = stats(r.stdout)
        print(f'{it["priority"]:12s} {size / 2**30:8.2f} GiB {files:9,d} files  {it["id"]}', flush=True)
        out.append({**it, 'bytes': size, 'files': files})
    MANIFEST.write_text(json.dumps({
        'updated_utc': datetime.now(timezone.utc).isoformat(), 'server_root': SERVER_ROOT,
        'note': 'Everything not on GitHub, as rsync items. migration/pull_from_server.py pull --host USER@OLD_SERVER',
        'items': out}, indent=1) + '\n')
    print(f'wrote {MANIFEST.relative_to(ROOT)}')


def load():
    return json.loads(MANIFEST.read_text())['items']


def selected(items, a):
    if a.item:
        known = {it['id'] for it in items}
        bad = [i for i in a.item if i not in known]
        if bad:
            raise SystemExit(f'unknown item(s): {bad}; see `list`')
        return [it for it in items if it['id'] in a.item]
    cut = PRIORITIES.index(a.priority)
    return [it for it in items if PRIORITIES.index(it['priority']) <= cut and it['priority'] != 'skip']


def dest_path(it, a):
    d = it['dest'].replace('{HF_HOME}', a.hf_home).replace('{HOME}', str(Path.home()))
    return Path(d) if os.path.isabs(d) else ROOT / d


def cmd_list(a):
    items = load()
    for pr in PRIORITIES:
        group = [it for it in items if it['priority'] == pr]
        print(f'\n{pr.upper()}  ({sum(it["bytes"] for it in group) / 2**30:.1f} GiB)')
        for it in group:
            print(f'  {it["id"]:28s} {it["bytes"] / 2**30:8.2f} GiB  {it["use"]}')
            if it['note']:
                print(f'  {"":28s} {"":12s}  note: {it["note"]}')


def rsync_cmd(it, a, dry=False, extra=()):
    dest = dest_path(it, a)
    cmd = ['rsync', '-a', '--partial', '-s', '--info=progress2,stats1', *filter_args(it), *extra]
    if dry:
        cmd += ['-n', '--stats']
    if a.bwlimit:
        cmd.append(f'--bwlimit={a.bwlimit}')
    if a.ssh:
        cmd += ['-e', a.ssh]
    return cmd + [f'{a.host}:{src_path(it).rstrip("/")}/', str(dest) + '/'], dest


def cmd_pull(a):
    items = selected(load(), a)
    total = sum(it['bytes'] for it in items)
    print(f'{len(items)} items, {total / 2**30:.1f} GiB from {a.host}:{SERVER_ROOT}'
          + (f' (~{total / (a.mbps * 2**20) / 3600:.1f} h at {a.mbps} MiB/s)' if a.mbps else ''))
    failed = []
    for i, it in enumerate(items, 1):
        cmd, dest = rsync_cmd(it, a, dry=a.dry_run)
        print(f'\n[{i}/{len(items)}] {it["id"]}  {it["bytes"] / 2**30:.2f} GiB -> {dest}', flush=True)
        if a.dry_run:
            print('  ' + ' '.join(shlex.quote(c) for c in cmd))
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(it['id'])
    if failed:
        raise SystemExit(f'failed (re-run to resume): {failed}')
    print('\ndone' if not a.dry_run else '\n(dry run)')


def cmd_check(a):
    items = selected(load(), a)
    left = 0
    for it in items:
        cmd, dest = rsync_cmd(it, a, dry=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        _, _, moved = stats(r.stdout)
        state = 'ERROR ' + r.stderr.strip().splitlines()[-1] if r.returncode else (f'{moved} files to copy' if moved else 'complete')
        left += bool(moved) or bool(r.returncode)
        print(f'  {it["id"]:28s} {state}')
    return 1 if left else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['list', 'pull', 'check', 'manifest'])
    ap.add_argument('--host', help='SSH destination of the old server, e.g. user@host (or an ~/.ssh/config alias)')
    ap.add_argument('--priority', choices=PRIORITIES[:3], default='required')
    ap.add_argument('--item', nargs='+', help='exact item ids (also allows skip items)')
    ap.add_argument('--hf-home', default=os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    ap.add_argument('--ssh', help='ssh command for rsync -e, e.g. "ssh -p 2222 -i ~/.ssh/key"')
    ap.add_argument('--bwlimit', help='rsync --bwlimit (KiB/s)')
    ap.add_argument('--mbps', type=float, default=6.0, help='for the time estimate only')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.command in ('pull', 'check') and not a.host:
        raise SystemExit('--host USER@OLD_SERVER is required')
    return {'list': cmd_list, 'pull': cmd_pull, 'check': cmd_check, 'manifest': cmd_manifest}[a.command](a)


if __name__ == '__main__':
    sys.exit(main())
