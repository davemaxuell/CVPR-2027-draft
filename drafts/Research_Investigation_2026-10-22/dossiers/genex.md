# GenEx

## Bibliography and reviewed material

**GenEx: Generating an Explorable World**. Taiming Lu; Tianmin Shu; Alan Yuille; Daniel Khashabi; Jieneng Chen. ICLR 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://proceedings.iclr.cc/paper_files/paper/2025/hash/8204f54f6cd8f0db1765a274e94ceaac-Abstract-Conference.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-9. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- GenEx.pdf; Main PDF; SHA256 `cf08fa6235c6098344973fede58c50c2db16a12197370e4639801bf5bf967e6e`; 26 pages.

## Problem, task and assumptions

Imagine panoramic exploration from a partial view and use imagined perspectives in embodiedQA, including multi-agent perspective-taking.

## Method, training and inference

Sections 3–4 adapt Stable Video Diffusion to panoramic exploration. The system first rotates the sphere geometrically, then generates a forward video segment. The spherical-consistency objective (Eq. 2) decodes generated latents, rotates predicted and target panoramas together, re-encodes them and penalizes their latent difference alongside ordinary diffusion loss. An LLM proposes orientation/distance and reasons over generated views. GenEx-DB contains four synthetic visual styles; Street View and Behavior Vision Suite provide held-out domains. Equation 4 describes belief updates using imagined observations, but a calibrated posterior-inference algorithm is not established. Generated views are samples from the model conditioned on available evidence; they are not additional camera measurements.

## Novelty and closest predecessors

The contribution combines panoramic video exploration, spherical consistency and an embodied-QA benchmark. It targets coherent imagined perspective-taking rather than a learned navigation action policy.

## Results and independent verdict

Table 1, p. 8: FVD is 69.5 for the full method, 81.9 without spherical consistency and 196.7 for six-view generation. IELC measures Inception-feature error after 1,000 closed imagined loops. Table 2, p. 9 gives 0.105 on street scenes and 0.092 indoors, so a blanket claim that both are below 0.1 is inaccurate. GenEx-EQA contains more than 200 synthetic scenarios and uses GPT-4 o as a judge for logic scoring. These are generated-world/QA evaluations, not R2R success or physical navigation deployment.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/Beckschen/GenEx. Snapshot commit: `eda604588d0a205c57e2b2599eec566d97162c2e`. Retrieval: 2026-10-22.

INSPECTED CODE: `code/train/explorer/train_world_explorer.py:162–169,1233–1301` normalizes panoramic data, computes diffusion MSE, and optionally decodes/rotates/re-encodes frames for spherical consistency. The released mixing schedule ramps its weight and disables the spherical contribution if it exceeds ten times video loss; this is more specific than a fixed two-term objective. `code/inference/explorer.py:122–158` rotates panoramas and recursively feeds generated last frames into further motion generation. `code/evaluate/IECC/iecc_score.py:12–42,73–77` computes Inception-v4 feature MSE after 299×299 normalization. This tests generated loop consistency, not truth about unobserved physical states. The audited code does not establish a calibrated Bayesian belief updater or prove imagined observations are independent evidence.

Inspected files: `code/evaluate/IECC/iecc_score.py`, `code/inference/explorer.py`, `code/train/explorer/train_world_explorer.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

DEDUCTION: conditional on a fixed real history and model, self-generated samples add computation but no independent observational information about the actual hidden state. Counting them as independent likelihood evidence can create unwarranted certainty. A consistently false generated world can satisfy loop closure. Facts about an occluded physical hazard require a real observation or an explicitly justified prior.

Relevant follow-up relationship: SALI, ImagineNav, Pathdreamer and WorldVLN differ in their interfaces between real and imagined evidence. A useful research gap concerns provenance and decision reliability, rather than the mere ability to retain or lengthen generation.

## Access and uncertainty record

Full text is preserved under `corpus/genex/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/genex/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-genex-provenance

**Still unresolved in the reviewed literature — Fundamental under fixed-history conditioning.** Long self-consistent generated trajectories need not describe the actual occluded environment.

Conditions: Using simulated views to answer questions about hidden physical facts or update navigation confidence. Cause: Conditional on a fixed real observation history and fixed model, additional generated samples are computed from that information; loop closure can be satisfied by a consistently false world.

Evidence: DEDUCTION: generative pipeline; PAPER EXPERIMENT: Table 2 p.9 IELC 0.105 streets/0.092 indoors and synthetic GenEx-EQA do not measure truth of hidden real environments.

Follow-up assessment: SALI persists imagined nodes; WorldVLN assimilates later real observations. Real observations add evidence, but no reviewed study establishes reliable confidence for hidden facts from generation alone. Confidence: High for information distinction; medium for system-level practical impact.
