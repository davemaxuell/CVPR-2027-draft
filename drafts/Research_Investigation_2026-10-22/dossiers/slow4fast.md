# slow4fast-VLN

Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning

Yang Li; Aming Wu; Zihao Zhang; Yahong Han. CVPR 2026. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [slow4fast-VLN.pdf](../../Paper/slow4fast-VLN.pdf). Pages reviewed: ['3–8'].

## Problem and operating conditions

GSA-R2R agents must reuse experience across episodes and handle scene/user instruction styles. The fast agent is DUET; a Llama3.2-vision slow component summarizes navigation history into reusable experiences.

## Mechanism, training, and inference

Each step logs viewpoint, topology, instruction, action, visual description and metrics. A CoT prompt turns logs into scene type, spatial context/rules, navigation strategy, historical success rate and frequency. Retrieved experiences are encoded and attended to by visual features: queries are visual features, keys/values are experience vectors. Concatenation and a learned projection fuse attended experience into DUET inputs. An LLM also rewrites styled instructions to Basic style, retaining the original below a confidence threshold. This rewriting occurs during navigation training; the learned fusion is not an entirely training-free method. Library capacity is tuned in the 50–100 range. The main paper delegates exact prompt, retrieval, confidence and training details to supplementary material.

## Novelty and closest predecessors

Adds an LLM-derived experience library and explicit style conversion to GR-DUET-like persistent-scene navigation. GSA-VLN already tested text style translation; this paper’s contribution is the combined experience-to-policy interaction, not the first use of rewriting in this setting.

## Results and independent verdict

Table 1 Basic residential SR 69.3→70.8 versus GR-DUET; non-residential 56.6→58.4 (1.8 points, despite prose saying 2.2). Table 3 Scene SR/SPL 48.1/42.8→50.7/46.6. Table 2 Moira SPL drops 56.2→55.4 despite SR gain. Table 4 ablation rows differ from main full-model values; experimental configuration differences are not fully specified in the main text. Table 5 prints residential K=200 SR 63.1 and SPL 64.0, which is inconsistent with standard SPL≤SR and was confirmed visually on PDF p. 7; those cells are not used as valid performance estimates. Figure 2 logs Success, navigation error and nDTW, raising an unresolved question about whether evaluation-time reflection has privileged feedback. The case study calls a 1.5 m endpoint error a failure despite a 3 m SR threshold; it should be treated as illustrative, not independent benchmark evidence. Main results suggest modest gains but do not establish causal generalization from truthful experiences or compute efficiency including the VLM.

## Implementation and reproducibility

INSPECTED CODE at 75e73db12ff4d37013bb4769b0e081bfc9796bd4: The official documented main_nav.py imports GMapNavAgent, whose _build_model constructs ordinary VLNBert (r2r/agent.py:24–32). No reflection/ExperienceEnhanced integration is present in this agent path. Separate reflection_memory_bank.py:236–252 _call_llm_api returns a fixed mock response; get_experience_embeddings at 441–454 returns random placeholder vectors. experience_enhanced_vlnbert.py:121–127 uses character-code placeholder text encoding. Retrieval unions scene/context/action indices and ranks by stored success rate and confidence (reflection_memory_bank.py:415–439). These source findings prevent verification of the paper’s end-to-end learned reflection mechanism from the documented release. They do not establish that reported experiments used these placeholders. Success labels enter saved experience records, but the incomplete integration prevents determining their exact evaluation-time provenance. No API, training, simulator or navigation benchmark was executed. Line references use raw LF numbering; doubled carriage returns in these files can make universal-newline viewers display twice as many lines.

Inspected release reference: [https://github.com/yl6017339/Slow4Fast-VLN at 75e73db12ff4d37013bb4769b0e081bfc9796bd4](https://github.com/yl6017339/Slow4Fast-VLN/tree/75e73db12ff4d37013bb4769b0e081bfc9796bd4). File-level inspection is restricted to the paths listed below.

- README.md
- r2r/main_nav.py:1–50
- r2r/agent.py:16–36
- r2r/reflection_memory_bank.py:236–280,415–456
- r2r/experience_enhanced_vlnbert.py:23–95,121–127

## Specific bottlenecks

**slow4fast-feedback: Experience quality and evaluation feedback access are insufficiently specified for an autonomous generalization claim.**

Conditions: Reflection on failed routes or noisy captions in unseen scenes, especially when success/nDTW are unavailable to a deployed agent. Cause: LLM summaries may encode unsupported spatial rules; logged metric provenance and retrieval/confidence calibration need implementation verification. Evidence: METHOD: Figure 2 and §§2.3–2.6; uncertainty is an audit limitation rather than proven leakage.

Type: insufficiently tested. Status: Uncertain because evidence is insufficient. Confidence: Medium. Follow-ups: ['GSA-VLN', 'ATENA', 'FeedTTA']

## Sources and limitations

- [Author manuscript and title](https://arxiv.org/abs/2601.09111)
- [Official CVPR 2026 page; main paper and arXiv links, no supplementary link listed at inspection](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Towards_Open_Environments_and_Instructions_General_Vision-Language_Navigation_via_Fast-Slow_CVPR_2026_paper.html)
- Official CVPR page and author arXiv links expose no separate supplement at inspection; the conventional supplemental PDF URL returns 404. Central released modules contain placeholders and are not integrated in the documented runtime, leaving feedback provenance and paper-equivalent configuration unverified.
- Associated material: {"file": "evidence/04/slow4fast_page7.png", "pages_read": "7", "scope": "Rendered original page inspected: Table 5 really prints SPL 64.0 above SR 63.1 for residential K=200."}
