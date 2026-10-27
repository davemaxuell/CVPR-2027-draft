# BabyWalk

BabyWalk: Going Farther in Vision-and-Language Navigation by Taking Baby Steps

Wang Zhu; Hexiang Hu; Jiacheng Chen; Zhiwei Deng; Vihan Jain; Eugene Ie; Fei Sha. ACL 2020. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [BabyWalk.pdf](../../Paper/BabyWalk.pdf). Pages reviewed: ['3–9', '12–18'].

## Problem and operating conditions

Policies learned on short route instructions lose fidelity on longer routes. BabyWalk studies transfer across R2R, R4R, and constructed R6R/R8R sequences in the same Matterport3D navigation formalism. Longer tasks are concatenations of R2R routes, not independently authored real-world missions.

## Mechanism, training, and inference

Heuristics extract landmark phrases and chunk instructions into baby steps. A weakly supervised landmark classifier uses trajectory-level object-word labels: state logits are max-pooled and optimized with binary cross-entropy. Dynamic programming chooses monotonically ordered chunk endpoints from average landmark scores (Appendix A). Each chunk has a language LSTM; past chunks and their executed trajectories are separately encoded, exponentially weighted with recency coefficient 0.5, and fused by an MLP. A recurrent cross-modal policy consumes current panorama, prior action, current chunk and remembered context. Imitation learning initially resets the agent to expert chunk-start context; curriculum policy-gradient training then executes increasingly long suffixes of one to four chunks. Terminal reward is SR plus CLS, with zero intermediate reward (Appendix B.2). Inference needs only heuristic chunks and the agent’s own trajectories, not expert alignment. Features are ResNet-152 2048 plus 128-dimensional tiled orientation; hidden size 512, GloVe words 300. Adam learning rate 1e-4, 50,000 imitation iterations then 10,000 per lecture; RL samples eight trajectories per instruction and discounts by 0.95.

## Novelty and closest predecessors

The combination of weak automatic chunk alignment, hierarchical recency memory, and length curriculum distinguishes BabyWalk from Speaker-Follower, RCM, and progress-monitor agents. Compared with concurrent FGR2R, supervision is automatically aligned rather than human chunk annotations. Memory forgetting and curriculum should be credited separately from additional speaker augmentation.

## Results and independent verdict

PAPER EXPERIMENT: Table 2, PDF p. 7, R4R-trained BabyWalk achieves R4R SR 29.6, CLS 47.8, SDTW 18.1 versus FAST 36.2/34.0/15.5. It prioritizes route fidelity rather than winning endpoint SR. R8R SDTW 11.5 exceeds FAST 6.3, but R8R SR remains 26.3. Table 3, p. 8, removing memory lowers R4R SDTW 17.3→9.9; learned LSTM summary yields 14.4. Table 4 separates curriculum from ordinary RL: SDTW 13.6→17.3 in-domain and 14.1→17.5 on other datasets. Speaker-augmented BabyWalk has lower SR than its unaugmented counterpart in several settings; extra data do not uniformly improve the target metric. The evidence supports hierarchical training for concatenated length transfer, not arbitrary compositional generalization.

## Implementation and reproducibility

INSPECTED CODE: src/follower.py:149–202 reconstructs prior subtask context and combines history embeddings using L1-normalized exponential recency weights, default decay 0.5; negative decay selects learned LSTM memory. This is an explicit forgetting inductive bias, not perfect retention. Sequential inference starts each new subtask at the previous executed endpoint and heading and appends its realized history (486–548). The training driver branches among imitation, reward training and curriculum RL; it advances curriculum and resets optimizers at the configured interval (train_follower.py:50–82,156–194). process_data.py constructs period/landmark chunks and curriculum tasks; no new VLM is involved. Assets and Matterport simulator were not loaded. Inspected source supports the staged mechanism, not independently reproduced long-route scores.

Inspected release reference: [https://github.com/Sha-Lab/babywalk at 4265340d5e521b59030bc50ef71f01b0d3d3ba63](https://github.com/Sha-Lab/babywalk/tree/4265340d5e521b59030bc50ef71f01b0d3d3ba63). File-level inspection is restricted to the paths listed below.

- src/follower.py:49–115,149–202,486–548
- src/train_follower.py:50–82,156–194
- src/process_data.py: make_data and curriculum split dispatch

## Specific bottlenecks

**babywalk-composition: Long-route gains rely on heuristic chunk boundaries and weak landmark alignment, with low final success on the longest routes.**

Conditions: Repeated landmarks, object-poor clauses, incorrect chunk endpoints, or missions whose dependencies extend beyond recent chunks. Cause: Monotonic dynamic programming and exponential forgetting encode a particular compositional prior; training uses artificial concatenations and expert-start curriculum. Evidence: PAPER EXPERIMENT: R8R SR 26.3, Table 2; DEDUCTION from §§4.1–4.4 and Appendix A/B, not a measured repeated-landmark failure rate.

Type: methodological. Status: Partially addressed. Confidence: High for reported low success; medium for inferred failure conditions. Follow-ups: ['FGR2R', 'Landmark-RxR', 'HAMT', 'M-TRACK']

## Sources and limitations

- [Official proceedings](https://aclanthology.org/2020.acl-main.229/)
- [Author implementation](https://github.com/Sha-Lab/babywalk)
- Relevant appendix and core source reviewed; simulator assets and training/checkpoints were not executed. Model-specific memory/curriculum gains do not establish arbitrary-length generalization.
