"""Check local references and coverage, not scientific correctness or novelty."""
from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parents[2]
errors = []

def read(rel):
    return json.loads((root / rel).read_text(encoding="utf-8"))

proposals = read("proposals/proposals.json")
coverage = read("evidence/07/coverage_review.json")
claim_ids = {r["claim_id"] for r in read("evidence/claims.json")}
matrix_ids = {r["bottleneck_id"] for n in (5, 6) for r in read(f"followups/{n:02d}_matrix.json")}
source_ids = {r["source_id"] for r in read("evidence/07/sources.json")}
originals = [r for n in range(1, 5) for r in read(f"assignments/{n:02d}_results.json")]
expected_papers = {r["paper_id"] for r in originals}
actual_papers = {r["paper_id"] for r in coverage}
if len(coverage) != 132 or expected_papers != actual_papers:
    errors.append("Coverage differs from the 132 canonical entries.")
if [p["id"] for p in proposals] != ["P1", "P2", "P3", "P4"]:
    errors.append("Unexpected proposal IDs.")
for p in proposals:
    for field, allowed in (("source_claim_ids", claim_ids), ("source_matrix_ids", matrix_ids), ("additional_source_ids", source_ids)):
        for value in p[field]:
            if value not in allowed:
                errors.append(f"{p['id']}: missing {field} reference {value}")

checked_links = 0
for rel in ("proposals/draft_proposals.md", "evidence/07/coverage_review.md", "evidence/07/additional_sources.md", "assignments/07_summary.md"):
    path = root / rel
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if target.startswith(("https://", "http://", "#")):
            continue
        checked_links += 1
        target = target.split("#", 1)[0]
        if not (path.parent / target).exists():
            errors.append(f"{rel}: unresolved local link {target}")

scope = read("evidence/07/inspection_scope.json")
for row in scope["source_code_inspections"]:
    path = root / row["path"]
    if not path.exists():
        errors.append(f"Missing inspected snapshot {row['path']}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        errors.append(f"Snapshot hash changed: {row['path']}")

result = {
    "validation_scope": "Reference integrity and declared coverage only; not scientific validation.",
    "proposals": len(proposals),
    "canonical_entries": len(originals),
    "coverage_entries": len(coverage),
    "local_links_checked": checked_links,
    "pinned_source_hashes_checked": len(scope["source_code_inspections"]),
    "errors": errors,
    "passed": not errors,
}
(root / "evidence/07/package_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
raise SystemExit(bool(errors))
