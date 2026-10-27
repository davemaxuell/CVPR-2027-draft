# Baselines from the original shortlist

These are comparison roles, not a claim that every model has been executed or that all reported scores are directly comparable. The final proposals specify which controls apply to each experiment. Use the [dossiers](../inventory/Dossier_Index.md) for exact paper versions, code commits and resource limitations.

| Requested baseline | Role in a controlled study | Essential qualification |
|---|---|---|
| [Original DUET](../dossiers/duet.md) | Establish the discrete panoramic graph-navigation baseline and its existing stopping/return behavior. | The release already routes through visited nodes and returns to a cached stop candidate. A new recovery mechanism must improve over that behavior. |
| [DUET + ScaleVLN](../dossiers/scalevln.md) | Control for additional synthetic scenes, trajectories and instructions with a shared architecture. | Match visual features, data and optimization before attributing gains to a new module. This is a system combination, not a separate distinct paper. |
| [SRDF](../dossiers/srdf.md) | Strong data-quality/self-training comparison and a relevant source of language–trajectory consistency filters. | The original pinned repository is README-only. Later SkillNav policy code does not establish release of the full generator/filter flywheel. Compare the same backbone and visual features where possible. |
| [KERM](../dossiers/kerm.md) | Test whether explicit retrieved relational knowledge provides the claimed benefit. | Retrieval adds Visual Genome/CLIP priors and preprocessing; relevance filtering does not certify that a retrieved fact is true in the current scene. |
| [UAGM](../dossiers/uagm.md) | Relevant reported uncertainty/map baseline and an important mathematical comparison. | Its inspected release lacks executable model code. The uncertainty interpretation and printed statistical results need the qualifications in the dossier and independent logic audit. Do not label an unreproduced number a locally verified baseline. |
| [SkillNav](../dossiers/skillnav.md) | Compare skill specialization, routing and temporal-language processing. | Include Temporal DUET and a single model trained on the same union of data. Check mixed-expert batch dispatch before using the pinned release; a release fix is baseline maintenance. |
| [VLN-Trans](../dossiers/vln_trans.md) and a literal rewriting control | Separate latent instruction translation from editing the executor's input text. | VLN-Trans does not emit an actual rewritten instruction at inference. SkillNav rewrites router input while its experts receive the original text. A literal rewrite-to-executor experiment is a separate intervention and must preserve history semantics. |

The earlier literature also supplies inexpensive controls that should not be skipped when applicable: [FAST goal reranking](../dossiers/tacticalrewind.md), [a separate BCE stopping objective](../dossiers/topologicalplanning.md), and [NavRAG's released STOP-loss reweighting](../dossiers/navrag.md). The [comparison guide](Comparison_Guide.md) explains why these controls cannot be replaced by a single leaderboard comparison.

## Novelty references requested by the user

All of the following are included in the supplied-paper review, with their exact mechanisms and limitations in the linked dossiers. They are a reading/citation checklist, not a claim that each is the closest predecessor of every proposal. Cite the work whose mechanism actually overlaps the proposed contribution.

| Reference | Dossier |
|---|---|
| NavNuances | [Read](../dossiers/navnuances.md) |
| CA-Nav | [Read](../dossiers/ca_nav.md) |
| GC-VLN | [Read](../dossiers/gc_vln.md) |
| HAM-VLN | [Read](../dossiers/ham_vln.md) |
| PolarMem | [Read](../dossiers/polarmem.md) |
| EOE | [Read](../dossiers/eoe.md); the user confirmed this OOD-detection paper is the intended reference. |
| Meta-Explore | [Read](../dossiers/metaexplore.md) |
| IEVE | [Read](../dossiers/ieve.md) |
| Context-Nav | [Read](../dossiers/context_nav.md) |
| AIUTA | [Read](../dossiers/aiuta.md) |
| VSI-Bench | [Read](../dossiers/vsi_bench.md) |
| EmbodiedQA and IQA | [EmbodiedQA](../dossiers/embodiedqa.md), [IQA](../dossiers/iqa.md) |
| VLN-Trans | [Read](../dossiers/vln_trans.md) |
| GELA | [Read](../dossiers/gela.md) |

Current follow-ups outside the original shortlist must also be checked. Assignments 05–06 and the independent novelty evaluation record those additional comparisons; the original checklist is not an exhaustive novelty search.
