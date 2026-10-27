"""Index reviewer-owned follow-up source cards without altering their judgments."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path, default):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def normalize_title(value):
    # Preserve plus signs: ImagineNav and ImagineNav++ are distinct works.
    return re.sub(r"[^a-z0-9+]", "", value.lower())


def main():
    original = read(ROOT / "inventory/reviewed_papers.json", [])
    titles = {normalize_title(r["full_title"]): r["paper_id"] for r in original}
    # These identities are independently clear in the primary full titles.
    known_original = {"05-uln": "uln", "05-mtrack": "mtrack", "05-landmark_rxr": "landmark_rxr"}
    rows = []
    for number in ("05", "06"):
        for record in read(ROOT / f"evidence/{number}/sources.json", []):
            rows.append({
                "assignment": number,
                "source_id": record["source_id"],
                "title": record["title"],
                "authors": record.get("authors", []),
                "year": record.get("year"),
                "venue_or_status": record.get("venue_or_status"),
                "version": record.get("version"),
                "primary_url": record.get("primary_url"),
                "provided_paper_id": known_original.get(record["source_id"], titles.get(normalize_title(record["title"]))),
                "local_pdf": (record.get("local_pdf") or "").replace("\\", "/"),
                "sha256": record.get("sha256"),
                "reviewed_pages": record.get("full_text_pages_actually_inspected", record.get("reviewed_pdf_pages")),
                "review_scope": record.get("review_scope", record.get("review_status")),
                "evidence_locators": record.get("evidence_locators", record.get("precise_locators")),
                "repo_url": record.get("repository_url", record.get("repo_url")),
                "repo_commit": record.get("repository_commit", record.get("repo_commit")),
                "code_files_inspected": record.get("code_files_inspected", []),
                "code_executed": record.get("code_executed", False),
                "limitations": record.get("limitations", []),
                "full_source_card": f"evidence/{number}/sources.json",
            })
    out = ROOT / "followups/Source_Index.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    lines = ["# Follow-up sources and inspected versions", "",
             "This index links primary publications, pinned implementations and the reviewers' exact inspection scope. It does not count a downloaded file as a completed review. Some entries revisit supplied papers rather than adding a new paper to the inventory. Full source cards retain the mechanism summaries and limitations without duplicating them here.", "",
             "Literature cutoff: **22 October 2026**. For the substantive synthesis, see [assignment 05](05_review.md), [assignment 06](06_review.md), and [the bottleneck matrix](Bottleneck_Matrix.csv).", ""]
    for r in rows:
        pages = r["reviewed_pages"]
        if isinstance(pages, list):
            pages = ", ".join(str(p) for p in pages)
        authors = "; ".join(r["authors"]) if r["authors"] else "Authors not established in the source card"
        lines.extend([f"## {r['source_id']}: {r['title']}", "", f"{authors}. {r['venue_or_status']}; {r['year']}.", "",
                      f"[Primary source]({r['primary_url']}) · [Detailed source card](../{r['full_source_card']})", "",
                      f"Reviewed PDF pages: {pages}. Precise locators: {r['evidence_locators']}. Scope: {r['review_scope']}.", ""])
        if r["provided_paper_id"]:
            lines.extend([f"Also in the supplied collection: [{r['provided_paper_id']}](../dossiers/{r['provided_paper_id']}.md).", ""])
        if r["repo_url"]:
            lines.extend([f"[Repository]({r['repo_url']}); inspected commit `{r['repo_commit'] or 'not pinned / unavailable'}`. Inspected paths and access limits are in the detailed source card.", ""])
        if r["local_pdf"]:
            lines.extend([f"[Saved PDF](../{r['local_pdf']}); SHA256 `{r['sha256']}`.", ""])
    (ROOT / "followups/Sources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"source_cards": len(rows), "revisited_supplied_papers": sum(r['provided_paper_id'] is not None for r in rows),
                      "with_inspected_code": sum(bool(r['code_files_inspected']) for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
