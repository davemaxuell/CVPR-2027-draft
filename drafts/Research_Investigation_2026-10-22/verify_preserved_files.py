"""Check supplied PDF and original proposal hashes without changing those files."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


with (ROOT / "inventory/all_files.csv").open(encoding="utf-8-sig", newline="") as stream:
    files = list(csv.DictReader(stream))
pdf_checks = []
for row in files:
    path = Path(row["pdf_path"])
    actual = sha256(path) if path.is_file() else None
    pdf_checks.append({"paper_id": row["paper_id"], "file": row["filename"],
                       "expected": row["sha256"], "actual": actual,
                       "matches": actual == row["sha256"]})

expected_drafts = json.loads((ROOT / "evidence/07/reviewed_draft_hashes.json").read_text(encoding="utf-8"))
draft_checks = [{"path": name, "expected": expected, "actual": sha256(ROOT / name)}
                for name, expected in expected_drafts.items()]
for row in draft_checks:
    row["matches"] = row["expected"] == row["actual"]

events = json.loads((ROOT / "assignments/ledger_events.json").read_text(encoding="utf-8"))
max_delegates = max(len(e["active_delegated_assignments_after_event"]) for e in events)
result = {
    "checked_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "File integrity and recorded concurrency only; no scientific or benchmark certification",
    "pdf_count": len(pdf_checks), "pdf_mismatches": sum(not x["matches"] for x in pdf_checks),
    "drafts_unchanged": all(x["matches"] for x in draft_checks),
    "maximum_recorded_concurrent_delegates": max_delegates,
    "maximum_recorded_concurrency_including_coordinator": max_delegates + 1,
    "pdf_checks": pdf_checks, "draft_checks": draft_checks,
}
result["passed"] = result["pdf_mismatches"] == 0 and result["drafts_unchanged"] and max_delegates <= 3
(ROOT / "appendices/preserved_file_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k not in ("pdf_checks", "draft_checks")}, indent=2))
raise SystemExit(not result["passed"])
