"""Aggregate agent-owned records without silently promoting pending reviews.

Run from any directory. Does not modify paper PDFs or agent-owned dossiers.
The final report remains a manually reconciled research judgment.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: compact(row.get(k)) for k in fields} for row in rows)


def relative_dossier(row, pid):
    value = row.get("dossier_path") or f"dossiers/{pid}.md"
    path = Path(value)
    if path.is_absolute():
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)
    return str(value).replace("\\", "/")


def main():
    inventory = read_json(ROOT / "inventory/papers.json", [])
    records, duplicates, record_assignments = {}, [], {}
    for n in range(1, 5):
        assignment = f"{n:02}"
        rows = read_json(ROOT / f"assignments/{assignment}_results.json", [])
        for row in rows:
            pid = row["paper_id"]
            if pid in records:
                duplicates.append(pid)
            records[pid] = row
            record_assignments[pid] = assignment

    reviewed, claims, bottlenecks, sources, coverage = [], [], [], [], []
    lines = [
        "# Paper dossiers",
        "",
        "The inventory covers 131 papers and one supplied research report. Each entry links to its substantive review, including the exact evidence and access limits. The PDF archive contains 157 files including alternate versions and supplements.",
        "",
        "A record marked pending is not a completed investigation. Repository inspection is distinct from executing a model; see the execution fields and experiment records.",
        "",
        "| Paper | Assignment | Review status | Code inspection | Dossier |",
        "|---|---|---|---|---|",
    ]
    for p in sorted(inventory, key=lambda p: p["title"].casefold()):
        pid = p["id"]
        r = records.get(pid, {})
        dossier = relative_dossier(r, pid)
        exists = (ROOT / dossier).is_file()
        row = {
            "paper_id": pid,
            "short_name": r.get("short_name", p["title"]),
            "full_title": r.get("full_title", p.get("full_title", p["title"])),
            "authors": r.get("authors", []),
            "year": r.get("year"),
            "venue_or_status": r.get("venue_or_status", "Pending primary-source verification"),
            "previous_catalog_venue": p.get("venue"),
            "kind": p.get("kind"),
            "assignment": p["assignment"],
            "bibliography_sources": r.get("bibliography_sources", []),
            "local_files": p.get("local_files", []),
            "main_pdf_pages_read": r.get("main_pdf_pages_read", []),
            "coordinator_additional_review": read_json(ROOT / "evidence/eoe_supplement_review.json") if pid == "eoe" else None,
            "associated_files_reviewed": r.get("associated_files_reviewed", []),
            "repo_url": r.get("repo_url"),
            "repo_commit": r.get("repo_commit"),
            "code_files_inspected": r.get("code_files_inspected", []),
            "code_executed": r.get("code_executed", False),
            "code_execution_field_scope": "The assigned paper reviewer's execution record; coordinator and independent evaluator checks are listed separately in experiments",
            "coordinator_executed_components": ["Isolated FloydGraph class on synthetic four-node graph; experiments/duet_routing_check_results.json"] if pid == "duet" else [],
            "independent_evaluator_executed_components": ([{
                "duet": "AST-extracted native teacher on stipulated synthetic histories; no learned policy",
                "skillnav": "AST-extracted input dispatch and navigation-call statement with fake expert functions; batch/permutation test",
                "psgnav": "AST-extracted write, retrieval, feature and rejection functions with synthetic RGB and explicit stubs; conditional failure-attribution example"
            }[pid] + "; experiments/10/source_check_results.json"] if pid in {"duet", "skillnav", "psgnav"} and (ROOT / "experiments/10/source_check_results.json").exists() else []),
            "repository_status": r.get("repository_status", "Pending"),
            "dossier_path": dossier if exists else None,
            "access_limitations": r.get("access_limitations", []),
            "completion_status": r.get("completion_status", "Pending"),
        }
        reviewed.append(row)
        for i, claim in enumerate(r.get("key_findings", []), 1):
            claims.append({"claim_id": f"{pid}-C{i:02}", "paper_id": pid,
                           "assignment": p["assignment"], **claim})
        for b in r.get("bottlenecks", []):
            bottlenecks.append({"paper_id": pid, "assignment": p["assignment"], **b})
        for src in r.get("bibliography_sources", []):
            sources.append({"paper_id": pid, "type": "paper bibliography", **src})
        if row["repo_url"]:
            sources.append({"paper_id": pid, "type": "repository", "url": row["repo_url"],
                            "commit": row["repo_commit"], "inspected_paths": row["code_files_inspected"],
                            "inspection_status": row["repository_status"]})
        for src in r.get("followup_sources", []):
            sources.append({"paper_id": pid, "type": "follow-up lead; review scope in dossier",
                            **(src if isinstance(src, dict) else {"url": src})})
        for f in p.get("local_files", []):
            sources.append({"paper_id": pid, "type": f.get("role", "local PDF"),
                            "path": f.get("pdf_path"), "sha256": f.get("sha256"),
                            "pages": f.get("pages")})
        status = row["completion_status"]
        code_status = row["repository_status"]
        link = f"[Read](../{quote(dossier, safe='/')})" if exists else "Pending"
        lines.append(f"| {row['short_name']} | {p['assignment']} | {status} | {code_status} | {link} |")
        files = p.get("local_files", [])
        unique_hashes = {f.get("sha256") for f in files if f.get("sha256")}
        coverage.append({
            "paper_id": pid, "assignment": p["assignment"], "has_record": bool(r),
            "has_dossier": exists, "dossier_words": len((ROOT / dossier).read_text(encoding="utf-8").split()) if exists else 0,
            "record_assignment_matches_inventory": record_assignments.get(pid) == p["assignment"],
            "pdf_files": len(files), "distinct_pdf_hashes": len(unique_hashes),
            "associated_review_record_present": bool(row["associated_files_reviewed"]),
            "distinct_supplement_check_needed": len(unique_hashes) > 1 and not row["associated_files_reviewed"],
            "code_files_count": len(row["code_files_inspected"]),
            "status": status, "access_limitations": row["access_limitations"],
        })
    # Coordinator checks remain separately owned and explicitly scoped; they are
    # appended without changing any paper reviewer's execution declaration.
    coordinator_claims = read_json(ROOT / "evidence/coordinator_findings.json", [])
    claims.extend(coordinator_claims)
    claims.extend(read_json(ROOT / "evidence/independent_findings.json", []))
    out = ROOT / "inventory"
    write_json(out / "reviewed_papers.json", reviewed)
    write_csv(out / "reviewed_papers.csv", reviewed, list(reviewed[0]) if reviewed else [])
    (out / "Dossier_Index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bibliography = ["# Sources and implementation references", "",
                    "Paper identity and publication status are recorded as verified by the assigned reviewer. Every dossier gives precise section, equation, table, page, and inspected-code locators. Repository links are discovery or inspection records, not claims of successful reproduction.", "",
                    "For all 157 supplied files, including versions and supplements, see [the file inventory](inventory/all_files.csv). For individual claim locators and code files, see [the structured source ledger](evidence/source_ledger.json) and [claim ledger](evidence/claims.json).", ""]
    for row in reviewed:
        bibliography.extend([f"## {row['short_name']}", "",
                             f"**{row['full_title']}**. {'; '.join(row['authors']) if row['authors'] else 'Author verification pending or not applicable'}. {row['venue_or_status']}.", ""])
        urls = [s.get("url", "") for s in row["bibliography_sources"] if isinstance(s, dict)]
        links = [f"[Primary source {i + 1}]({url})" for i, url in enumerate(urls) if url.startswith(("https://", "http://"))]
        if not links:
            original = next(p for p in inventory if p["id"] == row["paper_id"])
            if original.get("source", "").startswith(("https://", "http://")):
                links = [f"[Supplied catalog source]({original['source']})"]
            else:
                links = [f"[Supplied PDF and version metadata](corpus/{row['paper_id']}/metadata.json)"]
        if row["dossier_path"]:
            links.append(f"[Dossier and precise evidence]({quote(row['dossier_path'], safe='/')})")
        bibliography.extend([" · ".join(links), ""])
        if row["repo_url"]:
            bibliography.extend([f"Repository: [{row['repo_url']}]({row['repo_url']}). Recorded commit: `{row['repo_commit'] or 'not pinned / unavailable'}`. {row['repository_status']}.", ""])
        if row["access_limitations"]:
            bibliography.extend(["Access and scope: " + compact(row["access_limitations"]), ""])
    if (ROOT / "followups/Sources.md").exists():
        bibliography.extend(["## Follow-up literature", "",
                             "The [follow-up source index](followups/Sources.md) records the additional primary papers, versions, inspected pages, repository commits and access scope used in assignments 05–06. The [synthesis source cards](evidence/07/additional_sources.md) add scoped overlap checks and clearly marked leads. Later independent novelty-search references are retained with [evaluation 09](evaluations/09_novelty.md).", ""])
    (ROOT / "Sources.md").write_text("\n".join(bibliography), encoding="utf-8")
    ledger = read_json(ROOT / "assignment_ledger.json", [])
    ledger_lines = ["# Assignment ledger", "", "Exactly ten delegated assignments were used. At most three delegated agents ran alongside the coordinator; retries and follow-up work remained within the same numbered assignments. Completion means the documented evidence/access scope is closed, not that unavailable benchmarks were reproduced.", "", "| Assignment | Scope | Status | Main output |", "|---|---|---|---|"]
    for entry in ledger:
        summary = f"assignments/{entry['assignment']}_summary.md"
        link = f"[Summary]({summary})" if (ROOT / summary).exists() else "Pending"
        ledger_lines.append(f"| {entry['assignment']} | {entry['scope']} | {entry['status']} | {link} |")
    ledger_lines.extend(["", "The [machine-readable ledger](assignment_ledger.json) retains agent names, timestamps and output paths. The [event log](assignments/ledger_events.json) records coordinator status updates. Independent reviews are separate from proposal authorship; the coordinator reconciles their findings in the final proposal and revision artifacts.", ""])
    (ROOT / "Assignment_Ledger.md").write_text("\n".join(ledger_lines), encoding="utf-8")
    write_json(ROOT / "evidence/claims.json", claims)
    write_csv(ROOT / "evidence/claims.csv", claims, ["claim_id", "paper_id", "assignment", "claim", "evidence_type", "source", "locator"])
    write_json(ROOT / "evidence/paper_bottlenecks.json", bottlenecks)
    write_csv(ROOT / "evidence/paper_bottlenecks.csv", bottlenecks, ["id", "paper_id", "assignment", "description", "conditions", "cause", "evidence", "kind", "followups", "status", "confidence"])
    write_json(ROOT / "evidence/source_ledger.json", sources)
    write_csv(ROOT / "evidence/source_ledger.csv", sources, ["paper_id", "type", "url", "locator", "commit", "inspected_paths", "inspection_status", "path", "sha256", "pages"])
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": "2026-10-22", "inventory_entries": len(inventory),
        "pdf_files": sum(len(p.get("local_files", [])) for p in inventory),
        "record_count": len(records), "dossier_count": sum(c["has_dossier"] for c in coverage),
        "duplicate_record_ids": duplicates,
        "unknown_record_ids": sorted(set(records) - {p["id"] for p in inventory}),
        "pending_ids": [c["paper_id"] for c in coverage if not c["has_record"] or re.search(r"pending|in progress|queued|not reviewed|awaiting", c["status"], re.I)],
        "distinct_supplement_check_needed": [c["paper_id"] for c in coverage if c["distinct_supplement_check_needed"]],
        "record_status_counts": dict(Counter(c["status"] for c in coverage)),
        "claim_count": len(claims), "bottleneck_count": len(bottlenecks), "source_records": len(sources),
        "note": "Coverage flags are audit prompts, not automatic quality or scientific-validity judgments. Source availability and precise scope remain in the dossiers.",
        "papers": coverage,
    }
    write_json(ROOT / "appendices/coverage_audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "papers"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
