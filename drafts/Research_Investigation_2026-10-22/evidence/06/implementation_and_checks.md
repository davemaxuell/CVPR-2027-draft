# Assignment 06: implementation evidence and isolated checks

Cutoff and inspection date: 22 October 2026. All project code below was read, never executed. Downloads do not imply inspection of every file. Immutable commits, file lists, PDF hashes and limitations are also recorded in [sources.json](sources.json). The seven executable releases inspected are distinct from ImagineNav++'s README-only release. Source manifests preserve fetched file hashes.

## Implementation map

| Release and pinned commit | Inspected path | What the source establishes |
|---|---|---|
| [UA-NWM](https://github.com/DurYi/UA-NWM/tree/399960ce5fea5a3aa98084beaf44d92c2def9774) | `uanwm/models/hep.py:111-151`, `uanwm/train_hep.py:35-132`, `uanwm/eval/scoring.py`, post-training losses/config | Ridge projections operate on normalized feature residuals. Training averages per-patch normalized energies. Optional parallel-residual penalties and wrong-action contrastive loss already exist. Default configuration disables those additions. |
| [JanusVLN](https://github.com/MIV-XJTU/JanusVLN/tree/8cfe3129f7af4ca276bd5e38f1fae7e5367a09ec) | `src/evaluation.py:179-198,332-369`; `modeling_qwen2_5_vl.py:1607-1691,1985-2040,2130-2156`; VGGT aggregator | Persistent geometric cache is explicit. The evaluation path resamples RGB from the full accumulated list and starts a new language generation call. Persistent semantic/LLM caches were not found on that path. This is a paper/release discrepancy, not proof about all unpublished experiment branches. |
| [PiJEPA](https://github.com/AmirhoseinCh/PiJEPA/tree/f38a265cbce4fac5aa5fd9b9f7b18c723bccb645) | `scripts/eval_octo_mppi_cast.py:298-404,586-780,872-888`; `scripts/train_cast_wm.py:323-446`; training config | Evaluation selects a future recorded frame as the visual goal. MPPI samples a weighted elite sequence; the released candidate/elite/temperature defaults differ from the paper. Training detaches rollout context/predictions; default rollout length is one. This is not an independently reproduced closed-loop policy. |
| [DART-VLN](https://github.com/Japluto/DART-VLN/tree/a8889cf27723f1ffaac96ee7a802657a0b8e371a) | `map_nav_src/models/vilmodel.py:740-759,817-851`; `map_nav_src/r2r/agent.py:452-501` | Read-side log-priors reweight all retained slots. Next-hop anti-loop penalties subtract from action probabilities before argmax and leave STOP unchanged. This path does not physically prune memory slots. |
| [Memoir](https://github.com/xyz9911/Memoir/tree/ea7fb9066366eb2aff3eabbc87b227a567cc6b90) | `utils/graph_utils.py:419-531`; `nav_src/iter_nav/agent_ir2r.py:998-1082`; `vlnbert/vilmodel.py:1171-1248` | Retrieval uses cumulative compatibility products, while the paper describes a prefix-minimum criterion. The helper receives teacher paths for diagnostic hit rates; its inspected ranking branch does not use them. The `mastermind` flag changes imagined versus observed future matching. Released threshold 0.25 differs from supplementary 0.15. |
| [VLA Calibration](https://github.com/thomaspzollo/vla_calibration/tree/ee1ad08266c803b90bd601cfa6618bdb87a2af79) | `vla_calibration/calibration.py:71-265`; `notebooks/recalibration.ipynb`, cell 2 | Each action dimension receives a separate binary-success fit. The actual experiment requests `combine_method="mean"`; the class's default product is therefore not evidence that this experiment implements the wrong reduction. Temperature fitting is CUDA-specific in the release. No notebooks were run. |
| [ReliableNav](https://github.com/ZhanboShiAI/ReliableNav/tree/5234e9f96292b319463df5ab2d10ab6340cced1e) | `src/reliable_agent.py:90-203`; `src/mapping/map_handler.py:243-291,547-683`; `test.yaml:1-41` | Multiple same-class point clusters trigger numbered visual masks and another subgoal query; nonnavigable decisions trigger warnings/replanning. Configuration defaults to log replay. Only these paths and README were inspected; the full paper and complete control stack were not reviewed. |

ImagineNav++ commit `3803a563af97a1a0a2a6a176573a4a25c60c62a2` contains a coming-soon README, so no executable algorithm was inspected. CPNAV's linked repository was inaccessible. Other source cards specify missing or uninspected releases explicitly. Downloaded files outside the listed scopes must not be promoted to inspection evidence.

## Executed original checks

Setup: local Python with NumPy/SciPy; [math_checks.py](math_checks.py), machine-readable output [math_checks_results.json](math_checks_results.json). Five deterministic, self-contained checks ran successfully. These did not import downloaded project modules, invoke a model, execute a simulator or evaluate any published checkpoint. They test formula interpretations and constructed edge cases only.

### M1: confidence-quantile direction

For independent continuous episode scores, let U be the population coverage at the kth calibration order statistic. The probability-integral transform gives U distributed as Beta(k,n+1-k). With k=n+1-v, a high-confidence lower coverage statement requires a lower-tail quantile to exceed the target. An upper-tail quantile exceeding a target is insufficient.

The executed example uses n=500, v=225 and target 0.55. The 99th percentile is 0.6021825987, so the printed upper-quantile inequality passes. Yet P(U≥0.55)=0.5173256006, far below 0.99; the first percentile is 0.4990167725. This is a finite counterexample to that interpretation of the condition, not a reconstruction of the authors' calibration sample. CPNAV Eq.12 was additionally checked visually in [cpnav_page5.png](cpnav_page5.png). The paper's subsequent `1−δ` wording and calibration-unit ambiguity require resolution before any theorem is inherited. Discrete scores/ties need their appropriate finite-sample treatment; this continuous example is enough to reject the general claimed implication.

### M2: two normalized residual objectives differ

Let e_j be total residual energy and r_j unexplained energy for patch j. `sum(r)/sum(e)` weights patchwise ratios by residual energy; `mean(r/e)` weights patches equally. For e=(0.01,4), r=(0.01,0), the values are 0.0024937656 and 0.5. Thus a reproduction should explicitly select and ablate its reduction. This identifies objective weighting, not which version performs better.

### M3: suppressing a task-relevant direction

Use unit prediction p=(1,0), unit goal g=(0,1) and one unit basis vector b=(g−p)/||g−p||. With ridge λ=0.001, the projection is b(bᵀb+λ)⁻¹bᵀ(g−p). The squared residual after projection is 1.996005992×10⁻⁶ despite cosine distance 1. A basis that spans a task-relevant distinction can therefore suppress it. This does not show that UA-NWM actually learns that harmful basis; the basis was constructed deliberately. Testing real data requires action-ranking labels and controls for the released optional parallel/contrastive objectives.

### M4: equal memory priors cancel

For any shared positive w, softmax(z+log(w)·1)=softmax(z). With z=(0.3,−0.2) and w=0.35, both weight vectors are (0.6224593312,0.3775406688), maximum difference zero. A singleton attention set also always has unit weight. Consequently relative reweighting is not equivalent to absolute deletion or confidence suppression. This is an exact algebraic property illustrated numerically; its frequency in actual DART episodes was not measured.

### M5: path retention is not finite-budget search success

A root has retained branches D and G with scores 0.9 and 0.8. G reaches the goal in one action; D is a decoy requiring an action and a later return. Threshold 0.5 retains both. Highest-score-first search with a one-action budget chooses D and fails despite retaining a successful path. The example refutes the unconditional implication from set inclusion to budgeted success. It does not refute a correctly scoped theorem with additional traversal, budget and execution assumptions.

## Adaptation and conformal evidence inherited with verification limits

The completed paper analyses in assignment 04 were read: GSA-VLN, SAME, M3E, FSTTA, FeedTTA, ATENA, KnowNo and Explore until Confident. Relevant original text sections for FeedTTA's oracle, ATENA's mixture and both conformal methods were reread. Their source audits remain owned by assignment 04; this appendix does not claim to have independently executed or re-inspected all those releases.

FeedTTA's printed gradient multiplier differs from the appendix and its claimed expectation needs that distinction; ATENA's mixture gradient vanishes at λ=1; FSTTA's centered M=3 covariance has rank at most two. These are separately recorded in assignment 04's isolated results, not new assignment 06 tests. Original evaluators must recheck them before relying on a guarantee. Likewise GSA memory reuse, SAME joint training and M3E labelled sequential training are separate protocols; none should be grouped indiscriminately as label-free test-time adaptation.
