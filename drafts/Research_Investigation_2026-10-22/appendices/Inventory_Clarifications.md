# Inventory and terminology clarifications

The investigation follows the actual supplied files. Short names and earlier publication labels are useful discovery aids, but they do not override the PDF or primary publication record. The complete bibliography and file associations are in the [reviewed inventory](../inventory/reviewed_papers.json).

| Entry or term | Clarification | Primary-evidence record |
|---|---|---|
| NavRAG | The reviewed publication is in Findings of ACL 2025. It should not be silently described as a main-conference ACL paper. | [NavRAG dossier](../dossiers/navrag.md) |
| NavGRPO | The supplied PDF explicitly identifies CVPR 2026 Findings. | [NavGRPO dossier](../dossiers/navgrpo.md) |
| ECoT | The conference is CoRL 2024; proceedings publication in 2025 does not make it a different conference edition. | [ECoT dossier](../dossiers/ecot.md) |
| CONSOLE | The reviewed journal publication is TPAMI 2024. | [CONSOLE dossier](../dossiers/console.md) |
| HAM-VLN and PolarMem | The review retains the verified preprint status rather than inferring conference acceptance. | [HAM-VLN](../dossiers/ham_vln.md), [PolarMem](../dossiers/polarmem.md) |
| EOE | The supplied file is *Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection* (ICML 2024). The user confirmed on 22 October 2026 that this is the intended shortlist paper. It remains relevant adjacent OOD-detection work, with a different task from navigation. | [EOE dossier](../dossiers/eoe.md) |
| Internal research report | One of the 132 library entries is a supplied research report rather than a primary experimental paper. It is background/hypothesis material, not independent confirmation of the proposals. | [File inventory](../inventory/all_files.csv) |
| DUET+ScaleVLN | This describes a backbone and training-data combination. It is not automatically an additional distinct paper beyond DUET and ScaleVLN. | [DUET](../dossiers/duet.md), [ScaleVLN](../dossiers/scalevln.md) |
| One-shot | AirBERT's one-shot adaptation experiment means fine-tuning on one house. Across the literature, one-shot task execution, few-shot learning and navigation-specific training-free inference are different protocols. | [AirBERT dossier](../dossiers/airbert.md) |
| VLM-based and DUET-based | These categories overlap. SkillNav routes among DUET experts using language/vision models; NaviLLM's released navigation path uses a graph-action classifier with inherited navigation infrastructure. | [SkillNav](../dossiers/skillnav.md), [NaviLLM](../dossiers/navillm.md) |
| CE | VLN-CE refers to continuous environments; CE loss refers to cross-entropy. Keep both axes when comparing papers. | [Comparison guide](Comparison_Guide.md) |

Material source or implementation discrepancies are preserved in the dossiers. A discrepancy in a released snapshot or printed table does not by itself establish how a published checkpoint was trained or how its reported results were computed.
