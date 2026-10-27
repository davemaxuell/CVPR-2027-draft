"""Validate coverage and local artifact links; does not certify scientific claims."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def incomplete_status(value):
    return bool(re.search(r"pending|in progress|queued|not reviewed|awaiting", value, flags=re.I))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    errors, flags, counts = [], [], {}
    ledger = load(ROOT / "assignment_ledger.json")
    if sorted(x["assignment"] for x in ledger) != [f"{n:02}" for n in range(1, 11)]:
        errors.append("Ledger must contain exactly assignments 01 through 10")
    if args.require_complete:
        for assignment in ledger:
            if assignment["status"].lower() != "complete":
                errors.append(f"Assignment {assignment['assignment']} not marked complete")
            for output in assignment.get("outputs", []):
                if not (ROOT / output).exists():
                    errors.append(f"Assignment {assignment['assignment']} output missing: {output}")
    inventory = load(ROOT / "inventory/papers.json")
    expected_ids = {p["id"] for p in inventory}
    seen = []
    for n in range(1, 5):
        number = f"{n:02}"
        path = ROOT / f"assignments/{number}_results.json"
        if not path.exists():
            (errors if args.require_complete else flags).append(f"Assignment {number} results pending")
            continue
        rows = load(path)
        assigned = {p["id"] for p in load(ROOT / f"assignments/{number}_papers.json")}
        actual = {r["paper_id"] for r in rows}
        if actual - assigned:
            errors.append(f"Assignment {number} contains unexpected IDs: {sorted(actual-assigned)}")
        if args.require_complete and assigned - actual:
            errors.append(f"Assignment {number} lacks IDs: {sorted(assigned-actual)}")
        for row in rows:
            pid = row["paper_id"]
            seen.append(pid)
            dossier = ROOT / "dossiers" / f"{pid}.md"
            if not dossier.exists():
                errors.append(f"Record {pid} lacks dossier")
            elif len(dossier.read_text(encoding="utf-8").split()) < 250:
                flags.append(f"Short dossier needs substantive review: {pid}")
            if row.get("code_files_inspected") and not re.fullmatch(r"[0-9a-f]{40}", str(row.get("repo_commit", ""))):
                flags.append(f"Inspected code without a full single commit in standard field: {pid}; inspect dossier/version map")
            if not row.get("authors") and pid != "environment_review":
                flags.append(f"Missing author list: {pid}")
            if incomplete_status(row.get("completion_status", "")):
                (errors if args.require_complete else flags).append(f"Review remains incomplete: {pid}: {row.get('completion_status')}")
    if len(seen) != len(set(seen)):
        errors.append("Duplicate IDs in assignment records")
    if args.require_complete and set(seen) != expected_ids:
        errors.append("Record coverage differs from supplied inventory")
    missing_pdfs = [f["pdf_path"] for p in inventory for f in p.get("local_files", []) if not Path(f["pdf_path"]).is_file()]
    errors.extend("Missing supplied PDF: " + p for p in missing_pdfs)

    if args.require_complete:
        required = ["Research_Report.md", "Sources.md", "Research_Index.xlsx",
                    "followups/05_review.md", "followups/06_review.md",
                    "followups/Bottleneck_Matrix.json", "proposals/proposals.json",
                    "proposals/final_proposals.md", "proposals/final_proposals.json",
                    "proposals/Revision_Record.md", "proposals/Final_Recommendation.md",
                    "evaluations/08_logic.md", "evaluations/09_novelty.md",
                    "evaluations/10_feasibility.md", "evaluations/08_revision_check.md",
                    "Assignment_Ledger.md", "appendices/preserved_file_validation.json"]
        errors.extend("Final artifact missing: " + name for name in required if not (ROOT / name).exists())
        report_path = ROOT / "Research_Report.md"
        if report_path.exists() and re.search(r"Status: (investigation|final reconciliation) in progress", report_path.read_text(encoding="utf-8")):
            errors.append("Main research narrative still explicitly marked in progress")
        final_path = ROOT / "proposals/final_proposals.json"
        if final_path.exists():
            final_proposals = load(final_path)
            if [row.get("id") for row in final_proposals] != ["P1", "P2", "P4"]:
                errors.append("Final proposal IDs must match the reconciled retained set P1, P2, P4")
            for row in final_proposals:
                if incomplete_status(row.get("status", "")):
                    errors.append("Final proposal reconciliation is incomplete: " + str(row.get("id")))
        preservation_path = ROOT / "appendices/preserved_file_validation.json"
        if preservation_path.exists() and not load(preservation_path).get("passed"):
            errors.append("Supplied-PDF/original-draft preservation check failed")

    local_links = 0
    # Audit authored reports only. Vendor READMEs and corpus snapshots may contain
    # unresolved upstream relative links by design and are not our artifact index.
    authored = [ROOT / "README.md"]
    for folder in ("dossiers", "inventory", "followups", "proposals", "evaluations", "appendices", "assignments"):
        authored.extend((ROOT / folder).glob("*.md"))
    for folder in ("evidence", "experiments"):
        authored.extend((ROOT / folder).glob("*.md"))
        authored.extend((ROOT / folder).glob("*/*.md"))
    authored.extend(p for p in ROOT.glob("*.md") if p.name != "README.md")
    for document in authored:
        if not document.exists():
            continue
        text = document.read_text(encoding="utf-8")
        for raw in re.findall(r"\]\(([^\n]+?)\)", text):
            target = raw.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            # Simple report links have no quoted tooltip. Ignore formula-like
            # patterns instead of treating arbitrary prose as filesystem paths.
            if not target or any(c in target for c in ["\n", "\r"]):
                continue
            path = Path(target)
            if not path.is_absolute():
                path = document.parent / path
            local_links += 1
            if not path.exists():
                flags.append(f"Unresolved local-link candidate in {document.relative_to(ROOT)}: {raw}")
    counts.update(inventory_entries=len(inventory), pdf_files=sum(len(p.get("local_files", [])) for p in inventory),
                  reviewed_records=len(seen), local_links_checked=local_links, assignment_count=len(ledger))
    report = {"counts": counts, "require_complete": args.require_complete, "errors": errors,
              "review_flags": flags, "scope": "Coverage/schema/filesystem-link checks only; no automatic certification of research validity or benchmark reproduction"}
    (ROOT / "appendices/package_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
