# Academic research suite check

Inspected installed package: `C:/Users/user/.codex/skills/academic-research-suite`. Router and manifest both identify adapter version **3.23.1**. The manifest records upstream commit `4c5449017f46eb64d112a301d4817d22c7c9f3d9`. This is inspected package metadata, not independently authenticated upstream provenance. Required files for this investigation exist; paths and SHA256 hashes are in [the structural check](evidence/skill_structure_check.json).

The skill is usable for this request. Its strongest provisions are evidence verification, separating published claims from local execution, respecting original documents, loading only relevant workflows, explicit worker provenance, and keeping external model review separate from native delegation. The selected entry is `ars/deep-research/WORKFLOW.md`; the requested output is research assessment/proposal development, not manuscript drafting or a PRISMA systematic review.

Your prompt is more specific than the default skill workflow in several places:

| Skill default | Applied behavior in this run | Reason |
|---|---|---|
| Generic 13-role research team and six phases | Your ten substantive work packages; coordinator owns 8 | Exact requested assignment budget and independence rules |
| Phase-1 confirmation and review-form questionnaire | Continue autonomous investigation under your supplied research question and detailed protocol | Your explicit instruction to work autonomously and clarify only materially unresolved choices governs |
| APA report template | CVPR-focused proposal summary, causal/math/novelty audits, linked appendices and compact final claim table | Requested deliverable is specific and already defines structure |
| Model preference for newly configured sessions | Preserve current native runtime; no model/API reconfiguration | Loading a skill cannot change a running model; no external-provider review requested |
| Optional full-runtime planner/hooks | Not installed or run | Native collaboration suffices; hooks require separate opt-in and are not necessary for research |
| Manuscript revision decision gates | Original approach and scorer files remain read-only; changes are review proposals | Review does not require editing your design or declaring author approval |

The package is a workflow aid, not a scientific validator. Correct routing, script checks, or several agreeing workers do not establish novelty or effectiveness. Its internal agent metadata and extensive contract catalog can add process overhead if applied indiscriminately; only the relevant contracts were used. This was a scoped operational check, not an exhaustive security audit or execution of the package's entire validation catalog. No modifications to the skill were needed or made.

Output destination follows the existing local research-document convention next to your approach. No Page, shared external document, email, or remote publication was created.
