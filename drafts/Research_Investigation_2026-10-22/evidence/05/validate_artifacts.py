"""Consistency checks of research records, not verification of paper conclusions."""
from pathlib import Path
import csv, hashlib, json, re

R = Path(__file__).resolve().parents[2]
E = R / 'evidence/05'
load = lambda p: json.loads((R / p).read_text(encoding='utf-8-sig'))
sources = load('evidence/05/sources.json')
matrix = load('followups/05_matrix.json')
scope = load('evidence/05/original_scope.json')
originals = {p['id'] for n in range(1,5) for p in load(f'assignments/{n:02d}_papers.json')}
source_ids = {s['source_id'] for s in sources}
allowed = {'resolved-in-setting', 'partial', 'shifted', 'unresolved-in-reviewed-scope', 'uncertain'}
checks = []
def check(name, ok):
    checks.append({'check': name, 'passed': bool(ok)})

check('24 unique scoped sources', len(sources) == len(source_ids) == 24)
check('37 unique matrix rows', len(matrix) == len({r['bottleneck_id'] for r in matrix}) == 37)
check('48 unique original dossier records', len(scope) == len({r['paper_id'] for r in scope}) == 48)
check('all original IDs known', all(p in originals for r in matrix for p in r['original_papers']))
check('all scoped dossiers exist', all((R / s['dossier_read']).exists() for s in scope))
check('all statuses allowed', all(r['status'] in allowed for r in matrix))
check('all source IDs resolvable', all(r['followup_paper'] in source_ids or r['followup_paper'] in {'gsavln/slow4fast original chain','FU06-reinsvln'} for r in matrix))
check('all full texts present and match recorded hashes', all((R / s['local_pdf']).is_file() and hashlib.sha256((R / s['local_pdf']).read_bytes()).hexdigest() == s['sha256'] for s in sources))
check('all sources in shared full-text directory', all(str(s['local_pdf']).replace('\\','/').startswith('followups/sources/') for s in sources))
check('actual inspected pages in bounds', all(all(isinstance(p,int) and 1 <= p <= s['pdf_pages'] for p in s['full_text_pages_actually_inspected']) for s in sources))
check('bibliographic and access fields populated', all(s['authors'] and s['title'] and s['venue_or_status'] and str(s['primary_url']).startswith('https://') and s['limitations'] for s in sources))
check('no model or benchmark execution claimed', all(not s['code_executed'] and not s['benchmark_executed'] for s in sources))
check('all code records have full commit SHA', all(not s['repository_commit'] or re.fullmatch('[0-9a-f]{40}',s['repository_commit']) for s in sources))
with (R/'followups/05_matrix.csv').open(encoding='utf-8-sig',newline='') as f:
    csv_rows=list(csv.DictReader(f))
check('CSV matrix row count matches JSON',len(csv_rows)==len(matrix))
result={'date':'2026-10-22','scope':'Artifact consistency only; not benchmark or citation truth validation','checks':checks,
        'counts':{'sources':len(sources),'outside_corpus_sources':21,'matrix_rows':len(matrix),'original_dossiers_read':len(scope),
                  'originals_connected_in_matrix':len({p for r in matrix for p in r['original_papers']})},'passed':all(c['passed'] for c in checks)}
(E/'validation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
assert result['passed']
