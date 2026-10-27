"""Record the coordinator's bounded acceptance after the independent recheck.

This one-time editorial reconciliation preserves original reviewed drafts.
It implements only the terminal convention explicitly accepted by reviewer 08.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


review = json.loads((ROOT / "evaluations/08_revision_check.json").read_text(encoding="utf-8"))
for name, expected in review["input_sha256"].items():
    if digest(ROOT / name) != expected:
        raise SystemExit("Reviewed input changed before reconciliation: " + name)
if any(row["remaining_material_logic_blocker"] for row in review["findings"]):
    raise SystemExit("Reviewer reported a remaining material logical blocker")

changes = []


def save(name, text, reason):
    path = ROOT / name
    before = digest(path)
    path.write_text(text, encoding="utf-8")
    changes.append({"path": name, "before_sha256": before, "after_sha256": digest(path), "reason": reason})


def replace(name, pairs, reason):
    text = (ROOT / name).read_text(encoding="utf-8")
    for old, new in pairs:
        if old not in text:
            raise SystemExit("Expected editorial target missing in " + name + ": " + old)
        text = text.replace(old, new)
    save(name, text, reason)


old_terminal = "STOP is **not** a zero-cost self-loop."
new_terminal = ("STOP is **not** a zero-cost self-loop. Use `c(STOP)=0`, "
                "`V*(success terminal)=0`, and `V*(failure terminal)=infinity`; "
                "these terminal states are absorbing and admit no further actions.")
replace("proposals/final_proposals.md", [
    ("Coordinator acceptance and focused recheck remain pending.", "The coordinator accepted this narrowed scope after the [focused logical recheck](../evaluations/08_revision_check.md)."),
    ("| Provisional priority |", "| Priority |"),
    ("All ranks below remain provisional for the coordinator's final reconciliation.", "The [final ranking](Final_Recommendation.md) orders the next diagnostic experiments; it does not certify originality or benefits."),
    (old_terminal, new_terminal),
    ("The coordinator makes the final ranking and acceptance decision after focused recheck.", "The coordinator's [final recommendation](Final_Recommendation.md) accepts these three conditional studies and their stated falsification gates.")
], "Editorial acceptance/rank links plus reviewer-approved explicit STOP terminal convention; no new mechanism or performance claim")

rows = json.loads((ROOT / "proposals/final_proposals.json").read_text(encoding="utf-8"))
for row in rows:
    row["status"] = "Accepted as a conditional diagnostic study after independent review; originality and empirical benefit unestablished"
    row["rank"] = row.pop("provisional_rank")
    row["logical_status"] = "Bounded acceptance in focused review 08; no material logical blocker under stated assumptions, no learned-policy or deployment guarantee"
    if row["id"] == "P1":
        assert old_terminal in row["formal_formulation"]
        row["formal_formulation"] = row["formal_formulation"].replace(old_terminal, new_terminal)
save("proposals/final_proposals.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n", "Mirror Markdown terminal clarification and record final conditional status/rank")

fields = ["research_hypothesis", "target_bottleneck", "proposed_mechanism", "formal_formulation",
          "novelty_argument", "benefits_tradeoffs", "minimal_implementation", "experimental_plan",
          "falsification_criteria", "risks_prerequisites"]
markdown = (ROOT / "proposals/final_proposals.md").read_text(encoding="utf-8")
assert all(row[field] in markdown for row in rows for field in fields)

replace("proposals/Revision_Record.md", [
    ("Revision complete for coordinator and focused reviewer recheck.", "Reconciliation complete; coordinator accepted the narrowed studies after focused review."),
    ("in that provisional order", "in that final diagnostic priority order"),
    ("Coordinator acceptance and focused reviewer recheck remain separate from this completed revision.", "The [focused recheck](../evaluations/08_revision_check.md) found no material logical blocker under the stated assumptions. The coordinator accepted the diagnostic ranking and explicit limits. Its sole subsequent formulation clarification states zero STOP cost, zero accepting-terminal value and infinite failed-terminal value, with absorbing terminals. The [acceptance record](../evidence/coordinator_acceptance.json) preserves before/after hashes and the scope of these edits.")
], "Close acceptance and link the independent recheck and coordinator change record")

replace("Research_Report.md", [
    ("Status: final reconciliation in progress.", "Status: complete within documented evidence and access limits."),
    ("[revision record](proposals/Revision_Record.md).", "[revision record](proposals/Revision_Record.md), [focused final logic check](evaluations/08_revision_check.md).")
], "Mark the completed investigation and expose the final recheck")
replace("README.md", [
    ("Final reconciliation in progress.", "Complete within documented evidence and access limits."),
    ("[revisions](proposals/Revision_Record.md).", "[revisions](proposals/Revision_Record.md), [final logic check](evaluations/08_revision_check.md).")
], "Close the package index and link final review")
replace("proposals/Final_Recommendation.md", [
    ("P2 — Validity and difficulty", "P2 — Route validity"),
    ("P4 — Failure-aware memory", "P4 — Negative memory")
], "Use the same simple project names throughout the final package")

record = {
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "coordinator": "/root",
    "decision": "Accept P1, P2 and P4 as conditional diagnostic studies; reject P3 as an original-method proposal",
    "independent_review": "evaluations/08_revision_check.json",
    "independent_review_sha256": digest(ROOT / "evaluations/08_revision_check.json"),
    "original_reviewed_input_hashes": review["input_sha256"],
    "post_review_changes": changes,
    "substantive_json_markdown_fields_verified": len(rows) * len(fields),
    "additional_experiment_plan_clarification": "Coordinator made current-goal terminal acceptance, node-landmark grammar and action/state ambiguity masks explicit in the companion plan; the same constraints were already in the reviewed final proposal.",
    "original_draft_files_changed": False,
    "new_scientific_experiments": False,
    "does_not_establish": ["Original algorithms", "Natural failure prevalence", "Navigation gains", "Reproduced benchmarks", "Calibrated deployment probabilities"],
}
(ROOT / "evidence/coordinator_acceptance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"decision": record["decision"], "edited_files": len(changes), "substantive_fields_consistent": 30}, indent=2))
