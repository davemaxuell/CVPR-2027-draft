"""Read-only package audit, except for the saved audit JSON in this review folder."""
from pathlib import Path
import hashlib
import json
import re
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
required = [
    'Research_Report.md', 'Skill_Check.md', 'Sources.md', 'original_specification.md',
    'material_inventory.json', 'claim_register.md', 'assumptions_register.md',
    'candidate_register.md', 'evidence_register.md', 'assignment_ledger.md', 'material_coverage.md',
    'proposal_v1.md', 'proposal_v2.md', 'proposal_v3.md', 'review_reconciliation.md', 'experiments/README.md',
    'assignments/01_problem_baselines.md', 'assignments/02_formulation.md',
    'assignments/03_adversarial.md', 'assignments/04_direct_novelty.md',
    'assignments/05_crossfield.md', 'assignments/06_feasibility.md',
    'assignments/07_experiments.md', 'assignments/08_synthesis.md',
    'assignments/09_correctness.md', 'assignments/10_novelty_viability.md',
    'assignments/02_independent_candidates.md', 'assignments/03_initial_reconstruction.md',
    'assignments/05_independent_candidates.md', 'assignments/06_independent_candidates.md',
    'assignments/09_initial_assessment.md', 'assignments/10_initial_assessment.md',
]
missing = [p for p in required if not (ROOT / p).is_file()]
inventory = json.loads((ROOT / 'material_inventory.json').read_text(encoding='utf-8-sig'))
integrity = []
for item in inventory:
    path = Path(item['path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    integrity.append({'path': str(path), 'matches': digest == item['sha256'].lower()})
versions = {}
for name, expected in {
    'proposal_v1.md': '4cbc57e6781d83bf3c3a88f9030c54efbd39a45b8b98535eb68d5dd163d77f9f',
    'proposal_v2.md': 'b67df5a386c4ffab02672d11a561f7c6025e09a99172e80fee08a52b0cbc6305',
    'proposal_v3.md': '2e4a46c636502d105422951c5afe12dd7731c65d2f3c6733ec69643394000909',
}.items():
    path = ROOT / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    versions[name] = {'sha256': actual, 'matches_reviewed_version': actual == expected}
docs = list(ROOT.glob('*.md')) + list((ROOT / 'assignments').glob('*.md')) + [ROOT / 'experiments/README.md']
broken = []
links_checked = 0
for doc in docs:
    if not doc.is_file():
        continue
    body = doc.read_text(encoding='utf-8-sig')
    for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', body):
        target = match.group(1).strip()
        if target.startswith('<') and '>' in target:
            target = target[1:target.index('>')]
        else:
            target = re.split(r'\s+"', target, maxsplit=1)[0]
        if not target or target.startswith(('#', 'http://', 'https://', 'mailto:', 'skill://', 'plugin://')):
            continue
        target = urllib.parse.unquote(target.split('#', 1)[0])
        target = re.sub(r':\d+$', '', target)
        if not target:
            continue
        links_checked += 1
        dest = Path(target) if re.match(r'^[A-Za-z]:[\\/]', target) else doc.parent / target
        if not dest.exists():
            broken.append({'document': str(doc.relative_to(ROOT)), 'target': target})
evidence_paths = [
    'evidence/02_formal_checks.json', 'evidence/03_adversarial_checks.json',
    'experiments/06/scorer/check_results.json', 'experiments/06/scorer_profile.json',
    'evidence/06/original_integrity.json', 'evidence/skill_structure_check.json',
    'evidence/09_correctness_checks.json',
]
json_status = {}
for rel in evidence_paths:
    try:
        obj = json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))
        json_status[rel] = {'valid_json': True, 'type': type(obj).__name__}
        if isinstance(obj, dict) and 'status' in obj:
            json_status[rel]['recorded_status'] = obj['status']
        if isinstance(obj, dict) and 'checks_passed' in obj:
            json_status[rel]['recorded_check_groups'] = len(obj['checks_passed'])
    except (OSError, ValueError) as exc:
        json_status[rel] = {'valid_json': False, 'error': str(exc)}
result = {
    'missing_required_files': missing,
    'original_files': integrity,
    'proposal_versions': versions,
    'authored_markdown_files_checked': len(docs),
    'local_links_checked': links_checked,
    'broken_local_links': broken,
    'execution_record_json': json_status,
    'limits': 'File/hash/link/JSON checks only; no scientific validity or external-link availability certification.',
}
result['passed'] = (not missing and not broken and all(x['matches'] for x in integrity)
                    and all(x['matches_reviewed_version'] for x in versions.values())
                    and all(x['valid_json'] for x in json_status.values()))
(ROOT / 'evidence/package_verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=True))
raise SystemExit(0 if result['passed'] else 1)
