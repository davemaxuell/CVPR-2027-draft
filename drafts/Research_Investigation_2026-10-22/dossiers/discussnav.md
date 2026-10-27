# DiscussNav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions**. Yuxing Long, Xiaoqi Li, Wenzhe Cai, Hao Dong ICRA 2024 (2024).

Primary record: https://sites.google.com/view/discussnav



Supplied main file: [DiscussNav.pdf](../../Paper/DiscussNav.pdf). Pages actually reviewed: 1-7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Training-free navigation must integrate instruction parsing, perception, progress estimation, and action validation without relying on one undifferentiated LLM response. Graph experiments receive panoramic directions; physical robot uses separate obstacle-aware motion.

## Method, training, inference, and equations

PAPER METHOD (§III): assign eight prompted expert roles across four topics. GPT-4 action decomposition and landmark extraction check ordering; InstructBLIP answers targeted scene questions across 12 directions; RAM provides object tags. ChatGPT summarizes history, GPT-4 estimates executed/in-progress/future actions, and the navigation GPT-4 samples five thought/action responses at temperature 1. Responses with common directions are fused and a decision-testing expert selects among alternatives. Most supporting experts use temperature 0 and one output. The main agent fills a fixed question corpus rather than learning routing. No navigation training loss exists. In simulator the first candidate in the predicted direction is selected; TurtleBot moves.5 m or half obstacle distance, with LiDAR used for collision avoidance.

## Novelty and closest predecessors

Role decomposition, cross-checking and multi-response decision testing extend NavGPT 's ReAct-like loop. The method is a prompted multi-model system rather than learned multi-agent policy. Ablations remove expert categories but also reduce calls and information, leaving compute-matched comparison unresolved.

## Results and independent verdict

PAPER EXPERIMENT: Table I p.6 gives R2R^43 SR/40 SPL versus NavGPT 34/29. These are 9 and 11 percentage-point gains, not 26.47 and 37.93 points (those are relative percentages). Main evaluation uses 783 trajectories in 11 unseen environments; ablations sample one trajectory from each of 72 scenes. Table II reuses the full-row 43/40 although surrounding text describes the sampled set, creating protocol ambiguity. Physical 20-instruction SR 25% versus NavGPT 10% and DUET 0% is small and does not establish general inability of trained navigation. Increased API calls and correlated expert errors remain.

## Implementation audit

INSPECTED CODE. DiscussNav.py contains the released implementation:102–119 decomposes actions/extracts landmarks with GPT-4;121–232 caches RAM and InstructBLIP perception;234–290 summarizes the trajectory and estimates completion;292–347 requests five candidate decisions;363–405 merges reasons by chosen viewpoint and asks a decision-testing expert to select. This is sampled prompting and voting/discussion, not trained expert routing or a search tree with simulator rollouts. The same file provides simulator actions and evaluation at 349–361 and 407–445. Configuration and required external paths are embedded in the script rather than a complete portable experiment package.

No navigation-specific gradient training or loss is introduced. Costs arise from repeated API calls, perception and history growth. Cached descriptions can avoid repeated calls but their version/content must be preserved for reproducibility. The official repository does not supply a full autonomous simulator installation or immutable API backend. The code was read, never executed.

Repository: https://github.com/LYX0501/DiscussNav. Inspected snapshot commit: `1b0c926c25e78122f0459ef50fd87a3889b64580`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [DiscussNav.py:55-92,102-405,407-445,448-455](https://github.com/LYX0501/DiscussNav/blob/1b0c926c25e78122f0459ef50fd87a3889b64580/DiscussNav.py)

## Bottlenecks and follow-up chain

- **03-discussnav-consensus: Multiple language judgments can agree on the same wrong perceptual premise** Conditions: Missing or hallucinated landmark captions and long instructions Cause: Experts share observations/backbones; consensus is not independent evidence Evidence: §III pipeline; Table II perception and completion ablations Type: methodological. Follow-ups:  Analogical descriptions enrich comparative perception; ProFocus requests targeted views; R3/ AdaNav reduce always-on reasoning cost. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

The paper and the released single-script method/evaluation path were inspected. No API or robot experiments were executed. The paper's terminology of beam search should not be equated with independent API sampling without checking the backend.

