#!/usr/bin/env python3
"""Rewrite the old server's absolute paths in code/config (the files in migration/hardcoded_paths.txt).

Only needed if the repo does NOT live at /data/team_a/dave-workspace/RoleVLN on the new machine.

    python3 -I migration/relocate_paths.py --hf-home ~/hf_cache --conda-base ~/miniconda3            # dry run
    python3 -I migration/relocate_paths.py --hf-home ~/hf_cache --conda-base ~/miniconda3 --apply

Run records under runs/ and logs keep the old paths on purpose (they document where things were); resuming one
of the old automatic pipelines from its saved job.json therefore needs the original path (see MIGRATION.md).
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(ROOT), help='new location of the repo (default: this checkout)')
    ap.add_argument('--hf-home', default=os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    ap.add_argument('--conda-base', required=True, help='conda/miniforge install dir (holds envs/ and etc/)')
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    swaps = [  # longest first
        ('/data/team_a/dave-workspace/RoleVLN', a.root.rstrip('/')),
        ('/data/shared/hf_cache', a.hf_home.rstrip('/')),
        ('/home/bufsgpu/yes', a.conda_base.rstrip('/')),
        ('/home/bufsgpu', str(Path.home())),
    ]
    total = 0
    for line in (ROOT / 'migration/hardcoded_paths.txt').read_text().split('\n'):
        if not line:
            continue
        f = ROOT / line
        text = f.read_text()
        new, n = text, 0
        for old, rep in swaps:
            n += new.count(old)
            new = new.replace(old, rep)
        if n:
            total += n
            print(f'{n:4d}  {line}')
            if a.apply:
                f.write_text(new)
    print(f'{total} replacements {"written" if a.apply else "(dry run; add --apply)"}')


if __name__ == '__main__':
    main()
