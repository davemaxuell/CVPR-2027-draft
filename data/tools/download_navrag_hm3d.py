#!/usr/bin/env python3
"""One-time, credential-gated HM3D scene download for the saved NavRAG-CE release."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import netrc
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

DATA = Path(__file__).resolve().parents[1]
SCHEDULE = DATA / 'splits/navrag_hm3d_download_schedule.json'
STATUS = DATA / 'splits/navrag_hm3d_download_status.json'


def now():
    return datetime.now(timezone.utc)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def status(config, phase, **fields):
    write(STATUS, dict(schedule_id=config['schedule_id'], scheduled_at=config['scheduled_at'],
                       updated_at=now().isoformat(), phase=phase, **fields))
    print(now().isoformat(), phase, flush=True)


def credentials(config):
    paths = [Path(p).expanduser() for p in config['credential_file_candidates']]
    for path in paths:
        if not path.is_file():
            continue
        if path.stat().st_mode & 0o077:
            continue  # Do not use a credential file readable by other users.
        try:
            entry = netrc.netrc(str(path)).hosts.get('api.matterport.com')
        except (OSError, netrc.NetrcParseError):
            continue
        if entry and entry[0] and entry[2] and not any(
                value.startswith(('YOUR_', 'REPLACE_')) for value in (entry[0], entry[2])):
            return entry[0], entry[2]
    return None


def due(config, instant):
    target = datetime.fromisoformat(config['scheduled_at'])
    # Cron lacks a year field. This also prevents running before the booked time.
    return instant >= target and instant.astimezone(target.tzinfo).date() == target.date()


def remove_cron(config):
    result = subprocess.run(['/usr/bin/crontab', '-l'], capture_output=True, text=True)
    if result.returncode:
        return
    marker = '# ' + config['cron_marker']
    lines = [line for line in result.stdout.splitlines() if not line.rstrip().endswith(marker)]
    if len(lines) != len(result.stdout.splitlines()):
        subprocess.run(['/usr/bin/crontab', '-'], input='\n'.join(lines) + '\n', text=True, check=True)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def download(config, asset, auth):
    path = DATA / asset['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + '.receipt.json')
    if path.is_file() and receipt.is_file():
        saved = read(receipt)
        if saved['bytes'] == path.stat().st_size and saved['sha256'] == digest(path):
            return path
    if path.exists():
        raise RuntimeError('An unverified existing archive requires review: ' + str(path))
    part = path.with_name(path.name + '.part')
    status(config, 'downloading', current_file=asset['path'], current_pid=os.getpid())
    token = auth[0] + ':' + auth[1]
    if '\n' in token or '\r' in token or '\0' in token:
        raise RuntimeError('Credential format is invalid')
    escaped = token.replace('\\', '\\\\').replace('"', '\\"')
    # Credentials go through stdin, never argv, logs, schedule metadata, or shell expansion.
    result = subprocess.run([
        '/usr/bin/curl', '--config', '-', '--fail', '--location', '--silent', '--show-error',
        '--proto', '=https', '--proto-redir', '=https', '--connect-timeout', '30',
        '--speed-limit', '1024', '--speed-time', '180', '--retry', '5', '--retry-delay', '15',
        '--continue-at', '-', '--output', str(part), asset['url']],
        input='user = "' + escaped + '"\n', text=True)
    if result.returncode:
        raise RuntimeError('HM3D transfer failed (curl exit ' + str(result.returncode) + '); partial file retained')
    if not part.is_file() or part.stat().st_size < 512 or not tarfile.is_tarfile(part):
        raise RuntimeError('Downloaded content is not a valid TAR archive')
    status(config, 'checking_archive', current_file=asset['path'])
    sha = digest(part)
    part.replace(path)
    write(receipt, dict(url=asset['url'], bytes=path.stat().st_size, sha256=sha,
                        downloaded_at=now().isoformat(), checksum_scope='Locally computed; upstream checksum not published'))
    return path


def member_path(name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe archive path')
    parts = rel.parts
    if parts[:2] == ('hm3d', 'train'):
        parts = parts[2:]
    elif parts[:1] == ('train',):
        parts = parts[1:]
    return PurePosixPath(*parts)


def extract(config, archive_path, wanted=None):
    destination = DATA / 'scene_datasets/hm3d/train'
    destination.mkdir(parents=True, exist_ok=True)
    status(config, 'extracting', current_file=str(archive_path.relative_to(DATA)))
    found = set()
    with tarfile.open(archive_path, 'r:') as archive:
        for member in archive:
            relative = member_path(member.name)
            if member.isdir():
                continue
            if wanted is not None and str(relative) not in wanted:
                continue
            if not member.isfile():
                raise ValueError('Non-regular file in selected archive contents')
            if wanted is None and member.size > 100 * 1024**2:
                raise ValueError('Unexpectedly large configuration file')
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive path escapes its destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting')
            with archive.extractfile(member) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024**2)
            if temporary.stat().st_size != member.size:
                raise ValueError('Truncated archive member')
            if target.exists():
                if target.stat().st_size != member.size or digest(target) != digest(temporary):
                    raise ValueError('Existing scene asset differs; preserved: ' + str(target))
                temporary.unlink()
            else:
                temporary.replace(target)
            found.add(str(relative))
    if wanted is not None and wanted - found:
        raise ValueError('Archive lacks required NavRAG-CE scene assets: ' + str(len(wanted - found)))
    return len(found)


def update_inventory(config, phase, note):
    # Reuse the earlier annotation audit; scene readiness is set only by a new full audit.
    from complete_ce_expansion import Audit
    audit = Audit(DATA)
    audit.report = read(DATA / 'splits/ce_expansion_completion_report.json')
    audit.report['pending_dependencies'] = [note]
    audit.report['datasets']['NavRAG-CE']['status'] = 'Validation ready; HM3D ' + phase
    audit.publish()
    ce_status = read(DATA / 'splits/ce_expansion_status.json')
    ce_status.update(phase='hm3d_' + phase.replace(' ', '_'), hm3d_schedule=str(SCHEDULE.relative_to(DATA)),
                     scheduled_at=config['scheduled_at'])
    write(DATA / 'splits/ce_expansion_status.json', ce_status)


def execute(config):
    previous = read(STATUS) if STATUS.exists() else {}
    if previous.get('schedule_id') == config['schedule_id'] and previous.get('phase') == 'complete':
        return
    remove_cron(config)
    wanted_scenes = [p for p in read(DATA / 'datasets/NavRAG-CE/required_scenes.json')['required_scenes']
                     if p.startswith('hm3d/train/')]
    wanted = {p.removeprefix('hm3d/train/') for p in wanted_scenes}
    wanted |= {str(PurePosixPath(p).with_suffix('.navmesh')) for p in list(wanted)}
    missing = [p for p in wanted if not (DATA / 'scene_datasets/hm3d/train' / p).is_file()]
    if missing:
        auth = credentials(config)
        if auth is None:
            note = 'NavRAG-CE HM3D download could not start at the scheduled time: approved Matterport credentials are missing.'
            status(config, 'blocked_missing_credentials', missing_scene_asset_files=len(missing), message=note)
            update_inventory(config, 'blocked by missing credentials', note)
            return
        if shutil.disk_usage(DATA).free < 80 * 1024**3:
            raise RuntimeError('At least 80 GiB free space is required for download and extraction')
        for asset in config['assets']:
            archive = download(config, asset, auth)
            extract(config, archive, wanted if asset['kind'] == 'habitat_scenes' else None)
        cfg = DATA / 'scene_datasets/hm3d/train/hm3d_basis.scene_dataset_config.json'
        alias = cfg.parent.parent / cfg.name
        if cfg.is_file() and not alias.exists() and not alias.is_symlink():
            alias.symlink_to('train/' + cfg.name)
    status(config, 'verifying')
    subprocess.run(['/usr/bin/python3', str(DATA / 'tools/complete_ce_expansion.py')], cwd=DATA.parent, check=True)
    report = read(DATA / 'splits/ce_expansion_completion_report.json')
    if not report['datasets']['NavRAG-CE']['scene_dependencies_ready']:
        raise RuntimeError('NavRAG-CE scene verification did not pass')
    readme = DATA / 'datasets/NavRAG-CE/README.local.md'
    text = readme.read_text()
    start = text.find('**All published annotations')
    end = text.find('See [required_scenes.json]', start)
    if start >= 0 and end > start:
        text = text[:start] + '**Annotations, oracle paths, and required training/validation scene files are verified.**\n' + text[end:]
    text = text.replace('No HM3D transfer is running.', 'The scheduled HM3D transfer completed; see the completion report.')
    readme.write_text(text)
    status(config, 'complete', verified_scene_count=len(wanted_scenes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    os.umask(0o007)
    config = read(SCHEDULE)
    if args.preflight:
        print(json.dumps(dict(scheduled_at=config['scheduled_at'], due=due(config, now()),
                              credentials_available=credentials(config) is not None,
                              free_bytes=shutil.disk_usage(DATA).free, assets=config['assets']), indent=2))
        return
    if not due(config, now()):
        print('Outside the one-time scheduled date/time; no transfer started.', flush=True)
        return
    with (DATA / 'splits/navrag_hm3d_download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(config)
        except Exception as exc:
            message = str(exc)
            status(config, 'failed', message=message)
            update_inventory(config, 'download failed', 'NavRAG-CE HM3D job failed: ' + message)
            raise SystemExit(1)


if __name__ == '__main__':
    main()
