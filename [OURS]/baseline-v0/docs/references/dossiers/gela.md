# GELA

Grounded Entity-Landmark Adaptive Pre-training for Vision-and-Language Navigation

Yibo Cui; Liang Xie; Yakun Zhang; Meishan Zhang; Ye Yan; Erwei Yin. ICCV 2023. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [GELA.pdf](../../Paper/GELA.pdf). Pages reviewed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].

## Problem and operating conditions

GELA targets the absence of direct phrase-to-landmark supervision in graph VLN. It assumes panoramic RGB features, navigation history, a full instruction/dialogue, and human annotations during training. At inference it predicts next graph node or STOP. Its adaptive pretraining is an offline stage, not test-time adaptation.

## Mechanism, training, and inference

GEL-R2R annotates 71,467 entity phrases and 150,183 panoramic landmark boxes over R2R train/validation splits (Table 1). Training uses only the training annotations; the validation annotations support analysis. Annotators follow free-phrase, coreference, alignment, and unique-landmark rules, with expert revision (§3).

The HAMT/LXMERT-derived transformer has text, current 36-view panorama and history encoders, followed by cross-modal attention. Instruction outputs are token vectors Z, view outputs S have 37 entries including STOP, and hidden width is 768. EPP pools marked landmark views and predicts a distribution over the L+1 text positions. LBP pools marked phrase tokens and regresses a normalized 4D box with smooth-L1 plus generalized-IoU loss. ELSA is symmetric multi-positive contrastive alignment over matching visual/text tokens, with temperature .07 (§4.3, Eqs 1–8). The printed pooling products depend on the masks’ normalization; a binary mask would sum rather than average.

Adaptive pretraining retains MLM, MRC, ITM, SAP and spatial-relation proxy tasks to avoid forgetting; augmented data without boxes contribute only those original tasks. New loss weights and box GIoU weight are 1. Pretraining uses 200k iterations, batch 64 per GPU, lr 5e−5, two 3090s. Fine-tuning uses IL plus RL for 100k iterations at 1e−5, batch 16 R2R/batch 8 CVDN (§5.1). R2R experiments separately include no visual augmentation, style transfer, or semantic-mask image synthesis. Inference uses the pretrained navigation transformer without annotation access or an online phrase-grounding oracle.

## Novelty and closest predecessors

The important addition over FG-R2R and Landmark-RxR is entity-to-region human supervision inside sub-instructions, not just temporal sentence-to-subtrajectory alignment. The objectives adapt phrase-grounding ideas from MDETR/TransVG to panoramic HAMT representations. This is a combination of a new annotation resource and grounded auxiliary pretraining; it is not a new large foundation backbone.

## Results and independent verdict

PAPER EXPERIMENT: Table 2/p6 gives no-augmentation HAMT SR 65.7/SPL 60.9 versus GELA 69.2/63.4 on R2R unseen validation. With semantic-mask visual augmentation GELA reaches 71.1/65.0; attributing that whole result to entity alignment alone would conflate augmentation. Table 5/p8 shows EPP alone 65.8 SR and LBP alone 65.6, ELSA 66.3, EPP+LBP 67.0, all three 69.2. This is evidence of complementary objectives; it does not show every auxiliary is individually beneficial. CVDN test goal progress rises 5.58→5.87m (Table 4), a different metric from success.

Figure 3’s effective-attention score increases on annotated pairs. DEDUCTION: an attention-alignment diagnostic does not establish that the correct grounded entity causally determines a navigation action; a controlled landmark intervention would be stronger. Authors acknowledge costly limited annotation (§5.5). Their object-only/scene-only comparison also changes available data volume, explicitly acknowledged, so it cannot cleanly rank semantic category importance. Verdict: a concrete supervision resource and modest controlled gains; robustness under misleading, missing or repeated landmarks is not established.

## Implementation and reproducibility

INSPECTED CODE: GelDataset selects a trajectory/instruction/entity/box tuple and supplies span labels, landmark boxes and panorama-patch membership (ada_pretrain_src/data/r2r_tasks.py:600–650). forward_bbox explicitly computes span cross-entropy, sigmoid box regression with smooth L1 and generalized IoU, and bidirectional normalized token/patch contrastive loss (model/pretrain_cmt.py:351–424; gelunits.py:6–74). Adaptive pretraining averages task losses, backpropagates and applies scheduled/clipped optimizer updates (main_r2r.py:240–293). HAMT fine-tuning combines actor-critic and entropy terms with imitation loss; greedy inference selects the action-logit maximum (finetune_src/r2r/agent_cmt.py:349–388,487–523). run_r2r.sh specifies AdamW, 1e-5, batch 16, 300,000 iterations, PREVALENT augmentation and EnvEdit features. Thus training-data and augmentation controls matter when attributing improvements to grounding. README links Baidu-hosted annotations/features/checkpoints and GEL-R2R; these assets were not downloaded. The inspected code supports the stated objectives, not a claim of independently reproduced R2R/CVDN performance.

Inspected release reference: [https://github.com/CSir1996/VLN-GELA at 6754fa859807d6504759130c573480e0df34ffc1](https://github.com/CSir1996/VLN-GELA/tree/6754fa859807d6504759130c573480e0df34ffc1). File-level inspection is restricted to the paths listed below.

- ada_pretrain_src/model/pretrain_cmt.py:351–424
- ada_pretrain_src/model/gelunits.py:6–74
- ada_pretrain_src/data/r2r_tasks.py:600–650
- ada_pretrain_src/main_r2r.py:240–293
- finetune_src/r2r/agent_cmt.py:349–388,487–523
- finetune_src/scripts/run_r2r.sh
- README.md

## Specific bottlenecks

**gela-B1: High-quality entity grounding requires costly human boxes and does not certify decision-level language use.**

Conditions: New domains, unseen landmark types, or ambiguous repeated objects beyond the annotated R2R scenes. Cause: Limited human labels and proxy losses align representations without directly testing counterfactual action dependence. Evidence: §5.5 acknowledges limited annotation; Figure 3 measures attention rather than interventional behavior.

Type: methodological. Status: Partially addressed. Confidence: High for annotation cost; medium for the untested decision-level limitation.. Follow-ups: NavNuances tests atomic instruction interventions; synthetic-data methods reduce annotation cost but require correctness controls.

## Sources and limitations

- [Official supplementary or complete author version; reviewed scope listed in associated material](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Cui_Grounded_Entity-Landmark_Adaptive_ICCV_2023_supplemental.pdf)
- The official five-page supplement and objective code were reviewed. No pretrained assets or navigation benchmark were executed.
- Associated material: {"file": "evidence/04/supplements/gela_gela.txt", "pages_read": "1–5", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
