# Investigation method and evidence rules

The literature cutoff is **22 October 2026**. The supplied collection has **131 papers and one research report**, represented by **157 PDFs** including alternative versions and supplements. The supplied report is background material, not an independent experimental result. The immutable file inventory records SHA-256 hashes, PDF paths, and page counts. Short filenames are navigation aids; the reviewed bibliography identifies the actual paper.

## Scope and organization

Assignments 01–04 divide the collection into four groups of 33 entries. Assignments 05–06 trace the strongest relevant follow-up chains and test whether their experiments actually address earlier limitations. Assignment 07 formulates candidate contributions. Assignments 08–10 independently challenge their logic, novelty, and empirical feasibility. The coordinator reconciles those reviews and maintains the common inventory and evidence record. At most three delegated agents run alongside the coordinator. No additional unnumbered delegated assignments are used.

The assignment ledger records progress and output paths. A dossier being present does not by itself mean its source audit is finished. The machine-readable coverage audit preserves pending statuses and flags unrecorded distinct supplements; its flags are prompts for review, not automated scientific judgments.

## How sources are used

The supplied full PDFs are the primary evidence for their methods and results. Page-labelled extracted text supports search and reading; equations, tables, and figures must be checked in the PDF when extraction is ambiguous. Distinct extensions and supplements receive separate review records. Byte-identical versions need not be reread. Image-only supplied files may be supplemented by another official full-text version, with the substitution recorded.

Publication labels are checked against the paper or a primary proceedings record. An arXiv posting, a project page, a workshop paper, a Findings paper, and a main-conference paper are not interchangeable. A title or acronym mismatch is recorded explicitly. Earlier local analyses and catalog tags serve as leads, not established evidence.

Current literature searches use method descriptions, citation relationships, competing approaches, and adjacent fields. Search results locate primary sources; a snippet alone does not substantiate a technical conclusion. The search scope is bounded. “Unresolved in the reviewed literature” never means that all possible prior work has been excluded.

Official code snapshots are pinned to commits when accessible. Dossiers identify the actual files and functions inspected. Finding a repository, downloading it, or reading its README does not establish inspection of its model or loss. A repository may contain only placeholders, omit training, or require unreleased data even when the paper claims code availability. GitHub rate limits are recorded as retrieval obstacles, not proof that code is unreleased; git and pinned raw-source retrieval provide fallbacks where available.

## Evidence labels

| Label | What it establishes | What it does not establish |
|---|---|---|
| AUTHOR CLAIM | The authors make the stated assertion | Its independent correctness |
| PAPER EXPERIMENT | The paper reports the result under its stated protocol | Independent reproduction or transfer to other protocols |
| INSPECTED CODE | The inspected version implements the observed behavior | That a released checkpoint was trained using that code, or that the behavior improves performance |
| EXECUTED TEST | The specified local test produced the recorded outcome | A broader theorem or benchmark result beyond the test |
| DEDUCTION | The conclusion follows from the stated evidence and assumptions | An empirical effect size |
| HYPOTHESIS | A falsifiable proposed explanation or research direction | Established performance or novelty |
| UNVERIFIED | The available evidence does not settle the point | A negative finding about unavailable material |

Equations and counterexamples are scoped to their assumptions. A softmax identity is a mathematical property; a finite numerical check illustrates it. A graph-routing unit check shows the function can return a route through visited nodes; it does not measure the learned policy's backtracking frequency. Higher benchmark success alone does not show that a particular causal bottleneck was reduced.

## Comparison requirements

Comparisons must record task and action space, sensor access, data and pretraining, architecture size, inference budget, exploration or reranking protocol, feedback between episodes, and evaluation split. In particular:

- Discrete graph navigation with exact neighbor and pose access differs from continuous control with collision and localization errors.
- Navigation-specific training-free inference still depends on pretrained models; zero-shot transfer does not necessarily mean no navigation training occurred anywhere.
- Persistent memory updates, gradient updates, and privileged outcome feedback are distinct forms of adaptation.
- World models used to construct training examples differ from models queried while choosing test-time actions.
- Success, oracle success, route fidelity, path efficiency, and target-object grounding measure different properties.
- A goal-directed shortest-path teacher or an SPL filter does not certify faithful execution of every clause in an instruction.

## Local execution and resource boundary

The local hardware discovery found an NVIDIA RTX 4070 Laptop GPU with 8,188 MiB reported by `nvidia-smi`. The inspected Python environment has **PyTorch 2.9.0+cpu**, with CUDA unavailable to that installation. Habitat, Habitat-Sim, and MatterSim are absent. These observations are recorded in [environment.json](../experiments/environment.json).

A bounded workspace search located existing R2R training and validation-unseen annotation JSON files: 4,675 training paths across 61 scenes and 783 validation paths across 11 scenes, with no scene overlap between these two local files. Their hashes, structure and exact instruction counts are in [the local data inventory](../experiments/local_data_inventory.json). This is an annotation inspection, not a benchmark execution or upstream byte-equivalence check. The coordinator subsequently downloaded 72 public connectivity graphs at an immutable repository commit for a [static geometry check](../experiments/geometry/results.json). Usable checkpoints, feature arrays, licensed scene assets and simulator access remain absent from the search or unestablished. Availability elsewhere on the computer was not tested.

Local executions are enumerated in the experiment records. No result from a numerical illustration or isolated source-function check is reported as a navigation benchmark. Proposed simulator and training experiments remain untested unless an explicit experiment record gives the setup, execution, and measured result.

## Reproducible artifacts

- [Original file inventory](../inventory/all_files.csv)
- [Reviewed bibliography and review scope](../inventory/reviewed_papers.json)
- [Dossier index](../inventory/Dossier_Index.md)
- [Claim ledger](../evidence/claims.json)
- [Paper-specific bottlenecks](../evidence/paper_bottlenecks.json)
- [Primary source and code ledger](../evidence/source_ledger.json)
- [Coverage audit](coverage_audit.json)
- [Assignment ledger](../assignment_ledger.json)

`build_evidence_package.py` regenerates these aggregate indices from agent-owned records without overwriting the dossiers or promoting their completion statuses. The final research judgments require the coordinator's explicit reconciliation of the independent evaluations.
