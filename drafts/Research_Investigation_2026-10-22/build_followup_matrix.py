"""Combine the two independently owned follow-up audits without changing claims."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = [
    "assignment", "bottleneck_id", "original_papers", "precise_failure",
    "followup_paper", "mechanism", "evidence_locator", "repo_commit_and_files",
    "comparable_setting", "status", "remaining_gap", "confidence",
]


def main():
    inventory = json.loads((ROOT / "inventory/reviewed_papers.json").read_text(encoding="utf-8-sig"))
    def norm(value):
        return re.sub(r"[^a-z0-9+]", "", value.lower())
    names = {}
    for paper in inventory:
        for key in ("paper_id", "short_name", "full_title"):
            names[norm(paper[key])] = paper["paper_id"]
    names["rethinkingtheembodiedgap"] = "embodiedgap"
    rows = []
    missing = []
    for number in ("05", "06"):
        path = ROOT / f"followups/{number}_matrix.json"
        if not path.exists():
            missing.append(number)
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("rows", data.get("bottlenecks", []))
        for record in data:
            canonical = [names[norm(name)] for name in record.get("original_papers", []) if norm(name) in names]
            other = [name for name in record.get("original_papers", []) if norm(name) not in names]
            rows.append({"assignment": number, **record,
                         "original_paper_ids": sorted(set(canonical)),
                         "other_chain_references": other})
    if missing:
        raise SystemExit("Follow-up assignments not yet available: " + ", ".join(missing))
    out = ROOT / "followups/Bottleneck_Matrix.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    extra = sorted({key for row in rows for key in row} - set(FIELDS))
    with out.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + extra)
        writer.writeheader()
        for record in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in record.items()})
    report = {
        "row_count": len(rows),
        "assignments": {n: sum(r["assignment"] == n for r in rows) for n in ("05", "06")},
        "missing_required_fields": [{"row": i + 1, "fields": [f for f in FIELDS if f not in row]} for i, row in enumerate(rows) if any(f not in row for f in FIELDS)],
        "provided_papers_directly_linked": len({pid for row in rows for pid in row["original_paper_ids"]}),
        "other_chain_references": sorted({name for row in rows for name in row["other_chain_references"]}),
        "note": "This concatenation preserves each reviewer judgment. The coordinator's synthesis reconciles overlap and disagreement; no automatic resolution status is inferred from recency or benchmark score.",
    }
    (ROOT / "appendices/followup_matrix_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
