# M3E

M³E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts

Yongliang Jiang; Huaidong Zhang; Xuandi Luo; Shengfeng He. ICLR 2026. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [M3E.pdf](../../Paper/M3E.pdf). Pages reviewed: ['3–10', '14–19'].

## Problem and operating conditions

Replay-free adaptation to sequential scene domains while retaining previous navigation skill. The protocol trains on 80% of each selected val-unseen scene and evaluates held-out episodes and previous domains; this is supervised domain-incremental learning, not unlabeled online TTA.

## Mechanism, training, and inference

NaviLLM/Vicuna-based policy encodes 36-view panoramas and instruction/history into a decoder-only LLM. An online cognitive map supplies node features X in R^(N×d); a GNN propagates topology and instruction attention pools a scene vector. A macro MLP routes from this scene vector, while a micro MLP routes each token. Fused expert weights are beta*w_macro+(1-beta)*w_micro with beta=0.3. Six MoE-LoRA experts, top 2 active, rank 16 and scale 32 modify FFN and attention projections. After supervised learning on each new domain, summed routing weights estimate expert workload. Top-K experts receive momentum 0.4 on old parameters; other experts receive 0.6. Consolidation interpolates old and new weights. This statistic measures frequency of use, not a causal contribution to retained task performance. Batch 4, AdamW lr 2e-5, five epochs per domain, A800 80 GB GPUs; no earlier trajectories are replayed.

## Novelty and closest predecessors

Hierarchical scene/token routing plus expert-dependent consolidation targets continual adaptation on an LLM agent. SAME already uses multimodal state-aware expert routing for joint multi-task training; PerpR/ESR address forgetting with stored trajectories. M3E differs in supervised sequential consolidation without replay, not in inventing MoE or scene memory.

## Results and independent verdict

Table 1 R2R AvgSR/AvgSPL 71.92/66.96 versus Dual-SR 70.25/65.40 and finetuning 63.28/59.08. Reported BWT is 0.04 on R2R but -5.91 on REVERIE (Table 2), so forgetting is not eliminated generally. Table 4 shows dual routing without momentum has BWT -6.05 despite improved SR; consolidation supplies the retention benefit. Appendix Table 7 random 30% edge dropping loses 1.09 SR, but this does not test biased false edges, localization drift, or map aliasing. Table 8 has diminishing returns from six to eight experts only on this eight-domain experiment. Eq. 14 prints FWT as R_(i-1,i)-R_(i,i), which compares pretraining versus posttraining on that same task rather than a fixed untrained baseline; its claimed interpretation as standard forward transfer is therefore unresolved until evaluator code is checked. Appendix A5 compares TTA methods with different supervision access, so its ranking does not isolate architecture alone. Nondecreasing cumulative average SPL does not logically prove each old domain is preserved.

PROTOCOL CHECK: Appendix A.1, PDF p. 14, splits episodes approximately 80/20 within each selected original val-unseen scene. Adaptation therefore receives labeled examples from the scene whose other episodes are tested. This evaluates supervised continual adaptation and within-scene held-out performance, not zero-shot transfer to scenes absent from adaptation. Table 5 has an internally inconsistent printed total and a small-scene row despite the stated exclusion; exact partition reconstruction is not established by the release.

## Implementation and reproducibility

INSPECTED CODE, independently audited by assignment 05 at 35595353e6838b474ea835b9670419fa56e8f615: models/moe_lora.py:91–110 forms macro/micro softmax mixtures but evaluates all experts densely, unlike paper Top-K=2. tools/momentum/experts_stats.py:61–97,163 uses micro-only top-two counts and excludes near-tied pairs, rather than fused routing-probability workload in Eqs. 9–10. momentum_update.py:43–98 uses the intended new/old interpolation direction, with an extra frequency threshold; routers take new values. Visited-node representations are zeroed before macro routing (nav_model.py:171–192,261–267), although visited flags and topology remain. Alternating teacher/sample rollouts optimize supervised action CE, not policy-gradient RL (mp3d_agent.py:507–525,736–756). A compile-only test found import-star inside a function at nav_model.py:95; helper names also do not match definitions, and marco_router.py:28 unpacks a three-dimensional tensor into two shape variables. These are release blockers, not evidence that the authors ran this exact broken source. No cross-stage BWT/FWT aggregator or partition construction was located; FWT interpretation remains unresolved. The statistics launcher uses training data, not held-out labels. Full paths and inspection ranges: evidence/05/m3e_original_audit.md; static check setup/results: evidence/05/m3e_static_check.py and m3e_static_results.json. No fetched source was imported or executed.

Inspected release reference: [https://github.com/YongliangJiang/M3E-Code at 35595353e6838b474ea835b9670419fa56e8f615](https://github.com/YongliangJiang/M3E-Code/tree/35595353e6838b474ea835b9670419fa56e8f615). File-level inspection is restricted to the paths listed below.

- models/nav_model.py:1–135,142–305
- models/moe_lora.py: complete
- models/marco_router.py: complete
- tools/momentum/experts_stats.py: complete
- tools/momentum/momentum_update.py: complete
- train.py:35–115,123–307
- tasks/agents/mp3d_agent.py:500–530,675–778
- tasks/datasets/r2r.py:1–141
- tasks/datasets/mp3d_dataset.py:1–75
- tasks/loaders.py:1–160
- tools/optims.py: complete
- All four scripts, both R2R YAML configurations and README; inspected by assignment 05

## Specific bottlenecks

**m3e-retention: Replay-free consolidation still forgets in object-grounded navigation and uses expert workload as a retention proxy.**

Conditions: REVERIE domain changes or new tasks that reuse experts important for old tasks. Cause: Frequent routing on current data does not reveal old-domain parameter importance; fixed expert capacity and interpolation do not guarantee isolation. Evidence: PAPER EXPERIMENT: Table 2 BWT -5.91; DEDUCTION from Eqs. 9–13.

Type: methodological. Status: Partially addressed. Confidence: High within the evaluated setting. Follow-ups: ['SAME', 'Continual VLN']

## Sources and limitations

- [Author full paper](https://yongliangjiang.top/data/m3e/M3E_ICLR2026.pdf)
- [Official release](https://github.com/YongliangJiang/M3E-Code)
- Unmodified pinned source has independently documented compile/wiring/shape blockers; no project import or execution. Exact data partitions and cross-stage BWT/FWT aggregation were not released in inspected paths; paper-result validity is not adjudicated.
- Associated material: {"file": "evidence/05/m3e_original_audit.md", "pages_read": "All", "scope": "Independent original-source audit from assignment 05; inspected by assignment 04 for integration."}
