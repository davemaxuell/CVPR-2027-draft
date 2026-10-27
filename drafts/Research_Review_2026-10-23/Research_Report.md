# Independent research review: predicted futures for continuous vision-language navigation

Search cutoff: **23 October 2026, Asia/Seoul**. Original design snapshot: ten supplied files in `Research Approach`, individually hashed. Research lead: coordinator/Assignment 8. Evidence types: supplied specification, primary-author reports, inspected code, executed checks, analytical deductions and untested hypotheses remain separate throughout.

## Compact CVPR proposal summary

**Recommendation: defer substantial training pending a decisive predictive-channel test.** The research objective is worthwhile and the original formulation is largely coherent. There is no CE evidence yet that its imagined next views improve action selection, and its generic mechanisms have close precedents. **The requested package of two strong novel contributions is not yet established.**

**Problem:** determine when a small RGB instruction-following policy chooses the wrong primitive action because its decision representation fails to preserve instruction-relevant consequences. The broad navigation problem is established; this specific diagnosis on your selected small policies remains a hypothesis.

**Central hypothesis:** where ordinary visual/geometric prediction disagrees with measured instruction-conditioned action value, adapting the existing predictor to preserve those value differences can improve decisions beyond a matched direct value model, at a worthwhile total cost.

| Possible contribution | Precise claim to establish | Current status |
|---|---|---|
| 1. An instruction-conditioned outcome-ordering finding | Show a consequential mismatch between geometric/deviation ordering and measured continuation utility; independently establish any semantic interpretation and predictive benefit after history, supervision and compute controls | Unmeasured. Utility contains geometry; mismatch alone does not establish semantics or novelty |
| 2. A minimal correction for that failure | Train existing dynamics against measured instruction-conditioned value contrasts through a qualified frozen critic, retaining the native prediction objective; demonstrate an advantage over direct value and established task-weighted alternatives | Specified, unimplemented and substantially related to VAML, MCR and World4Scorer |

These targets form one scientific argument: identify a meaningful failure, then demonstrate a small remedy. They do not become two strong contributions merely by being listed. No third contribution is justified.

**One-paragraph method:** train and freeze one agreed small CE policy. From each real observation/history, predict one next feature for FORWARD, LEFT and RIGHT; STOP uses the current feature. A shared small instruction/history-conditioned critic scores the four actions, using real forced-action continuations as training labels. First establish that predicted features beat a full-history direct critic. Only then adapt the existing deterministic DINO predictor using native prediction loss plus one centered value-residual loss through the frozen critic. Execute a candidate only when it beats the base action by a development-selected margin; update memory from reality only. No extra search, parser, gate or network is needed.

```text
real RGB history + instruction ──> frozen policy ──> base action/context
              │
              └─> existing dynamics(F,L,R) ──> 3 predicted next features
current feature for STOP + real memory/context ──> shared 4-action scorer
                  ──> margin versus base ──> execute once ──> observe reality
```

**First decisive tests:** exact replay and metric/cache integrity → reproducible alternative-action opportunity → actual-next and predicted-next versus a matched no-future critic → one conditional objective → closed-loop and total-device cost. The proposed 64-root/eight-episode pilot is a debugging test, not evidence of a navigation gain. The complete candidate and revision table are in [proposal v3](proposal_v3.md); the preregistrable research plan is [Assignment 7](assignments/07_experiments.md).

## Executive assessment

What survives is the objective, primitive-action setup, frozen-reference continuation labels, real-history-only memory and the insistence on matched controls. A correct full-history value function for the frozen policy can support repeated action improvement; changing the deployment continuation does not by itself invalidate the method. What does not survive is treating imagined futures as additional observations, treating generic future scoring/ranking losses as distinct new contributions, or treating tensor checks as evidence of CE effectiveness or full 8 GB feasibility.

The largest scientific gap is **predictive value beyond direct learning from the same branch outcomes and allowed history**. The largest engineering risks are valid forced-action/cache replay, the actual world-model feature/action contract and full-pipeline resources. One concrete integration mismatch was found: the inspected DINO-WM default uses a 14×14 feature grid, while the supplied scorer assumes 16×16. This must be resolved against the selected checkpoint before actual/predicted feature comparisons.

The installed academic-research suite is operationally usable. Its router/manifest identify version **3.23.1**; relevant workflow files exist. Your exact ten-assignment/autonomous instructions governed over the skill's generic team and confirmation defaults. The skill and original research files were left unchanged. Scope and limitations are in [Skill_Check.md](Skill_Check.md).

## 1. Original idea and material coverage

All ten supplied files were inspected: README, approach, scorer design, two trial registries, scorer configuration, implementation, label helper, check runner and saved check results. [Inventory](material_inventory.json) records versions/hashes; [original specification](original_specification.md) preserves the intake design. Earlier linked investigations were supporting context and search leads; this run does not claim to repeat their entire 131-paper corpus.

Your explicit setup is R2R-CE/RxR-CE with RGB deployment, H=1 and four primitives. Three locked VLM trials—SmolVLM2-500M, LLaVA-OneVision-0.5B and Qwen3-VL-2B—first require navigation training using the adapted StreamVLN procedure, then freezing. They are pretrained VLM initializations, not ready interchangeable navigation policies. The four WM families are transferred/fresh DINO dynamics and transferred/fresh NWM/CDiT with matched components. Exact dynamics checkpoints remain open. Newly added data must use CE training scenes; pretrained external data is still part of provenance.

The feature-only reference scorer uses instruction tokens, actual policy context, current patches, eight 256-wide recurrent real-memory slots and two candidate-attention blocks. Its utility head is accompanied by success/collision auxiliaries. Future features are deliberately detached because the baseline world model is frozen. Imagined states never update persistent memory. The later usefulness-aware dynamics objective was explicitly unfinished, so the extension in this review is a **team proposal**, not an implementation silently attributed to you.

At each restored root, actually execute candidate `a`, then the same frozen one-action policy `pi_ref` to termination. With the original budget and full metric prefix:

`U = .70 success + .30 whole-path nDTW − .01 remaining_actions/B − .02 remaining_collisions/B`.

The target is `Q_ref(I,h,a)=E[U | force a, then pi_ref]`. STOP terminates immediately. The original actual→mixed→predicted curriculum and held-out margin are proposed procedures, not validated choices. No supplied CE trajectories, trained adapters, integrated WM execution or GPU evaluation establish their effectiveness. Missing submission year/track and deadline were not invented. Claim-by-claim assumptions and falsifiers are preserved in [claim register](claim_register.md) and [assumptions register](assumptions_register.md).

## 2. Is the problem real, and what is the actual bottleneck?

Earlier papers show that future representation and evaluation can constrain CE navigation. DREAMWALKER's Table 2 improves SR from 49 to 54 with actual future imagery, while a privileged goal-distance intervention reaches 100; the latter changes evaluation/rollout/STOP behavior and is not evidence for deployable H1 scoring. HNR's Table 3 reports 57.21/58.35/60.74/63.13 for single-view/RGB/HNR/actual-CLIP variants of the same model design; shared frozen weights were not verified. HNR uses RGB-D/pose and waypoint/panoramic assumptions different from this proposal. These establish useful leads, not expected gains for your policies. [DREAMWALKER, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DREAMWALKER_Mental_Planning_for_Continuous_Vision-Language_Navigation_ICCV_2023_paper.pdf); [HNR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Lookahead_Exploration_with_Neural_Radiance_Representation_for_Continuous_Vision-Language_Navigation_CVPR_2024_paper.pdf).

Competing explanations must be tested. StreamVLN's released evaluation generates action chunks and consumes their queue; forcing H1 changes execution even without a world model. SimpleVLN's recent preprint emphasizes training/history choices, with release/evaluation differences that prevent casual score comparisons. FutureNav provides a training-only prediction rival; ReInsVLN provides a history-use lead but its selected intervention examples and reference-derived reminders do not establish a deployable cure. [A1 baseline dossier](assignments/01_problem_baselines.md) records full-paper sections, code paths, pins and these limitations.

| Causal link | Evidence now | Weakest alternative / decisive check |
|---|---|---|
| Policy error → better primitive exists | Adjacent published evidence; unmeasured here | Every primitive may still fail under the frozen continuation; branch opportunity test |
| Error → missing useful consequence representation | Hypothesis | History truncation, weak actor, data, schedule or direct-value learning may explain it |
| Predicted feature → better ranking | No CE result | Critic can ignore futures or exploit action identity; matched full-history direct/copy/tiny-predictor controls |
| Better offline ranking → better episodes | Conditional mathematical result | Occupancy shift and label error; closed-loop evaluation |
| Better episodes → worthwhile lightweight method | Unmeasured | Extra generation/pretraining/latency may be dominated by a direct alternative |

The central weak link is the third row. Additional model-generated features are computation informed by learned priors, not additional sensed evidence about the current hidden world.

## 3. Mathematical assessment: sound core, conditional implications

**Full-history formulation.** Complete-path nDTW is a valid terminal trajectory reward when history and metric prefix are part of state. Remaining-cost labels differ from one globally consistent episode return by a root-common past-cost constant, so action advantages are preserved. Under exact `Q_ref` at every newly visited history, selecting an action with nonnegative reference advantage yields a nonnegative performance-difference sum. With a stochastic reference, compare against its expected value or a coupled sampled base action, not merely its mode. Compressed policy tokens/eight-slot memory need not be sufficient statistics. [A2 derivation](assignments/02_formulation.md), independently challenged in [A3](assignments/03_adversarial.md); standard antecedent: [Sutton–Barto §4.2](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf).

**Margin.** If every candidate score error at a visited history is at most `epsilon`, estimated advantage greater than `delta` implies actual advantage greater than `delta−2epsilon`. Thus `delta≥2epsilon` suffices locally. Development MSE or a tuned scalar margin does not establish that uniform premise. Including the base action alone is not a safety guarantee.

**Actual and imagined futures have different meanings.** An actual next observation can reveal hidden state. With fixed model weights and independent generator noise, an imagined future is conditionally independent of the realized hidden draw given all model inputs. It may help a finite critic through representation, transfer or extra computation. A concrete failure case: action A returns `.8`; B returns `.9` or `.1` equally. Selecting B whenever a perfectly distributed imagined sample is `.9` yields expected return `.65`, worse than `.8`. These are utility-compatible abstract outcomes, not measured CE frequencies. Final predicted-input training and independent model/continuation-seed checks are essential.

**Noisy maxima are not oracle opportunity.** Three equal-value actions with independent `.9/.1` outcomes each have mean `.5`, but the mean maximum of one sample per action is `.8`. Apparent `.3` headroom can be pure selection optimism. Verify deterministic branches or select and evaluate on disjoint continuation-seed panels; report the resulting finite-sample selector's lift without calling it true max-Q regret.

**Proposed objective.** For four scores `s`, measured mean returns `y` and `P=I−11ᵀ/4`, add `lambda ||P(s−y)||²/4` to the native dynamics loss. The all-pairs squared-gap formulation is exactly `4||P(s−y)||²`; weighted gaps are graph-Laplacian regression. This is an established kind of value-sensitive objective, not a second novelty created by notation. A per-root regret bound follows from the centered error when `y` is the true expected target, but does not establish generalization or CE gains. For stochastic independent predicted/actual gaps, squared loss includes model-score variance as well as mean error, as the [CVAML literature](https://arxiv.org/html/2505.22772v2) warns.

**Gradient and failure conditions.** Dynamics adaptation needs a separate differentiable future-input route; the original detach is correct for its frozen-WM baseline. A frozen critic can still be exploited by unrealistic features. Native anchoring limits the tradeoff without certifying it. Identical legal inputs requiring opposite actions cannot be disambiguated by more attention; information lost through compression needs diagnosis. A minimal helpful case is predictable action-dependent features whose native approximation error reverses a correct critic's ordering. A credible no-benefit case is an equally capable direct critic already predicting the values from available history. Detailed assumptions, counterexamples and classification of deductions versus numerical checks are in A2/A3/[A5](assignments/05_crossfield.md).

## 4. Closest work and the narrow remaining novelty hypothesis

| Proposed element | Closest inspected precedent | Already present | Remaining difference and required evidence |
|---|---|---|---|
| Frozen prediction plus candidate critic | VISTAv2 §3; World4Scorer Appendix D.2 | Imagined future scoring; specifically frozen LeWM plus simulator-outcome MLP scoring | CE instruction/real-history conditioning and primitive full-return labels; application difference alone is insufficient |
| Matched real branches for action credit | SeekVLN §3.3, Eqs.9–13; rollout policy-improvement literature | Simulator/context restoration and controlled continuation benefits | Exhaustive primitive Q_ref labels; show why this supervision matters beyond routine data collection |
| Anchored ranking-aware dynamics | MCR §3.3.2, Eqs.6–7 | Pairwise cost ordering with native prediction anchor | Measured instruction utility instead of dead-reckoned deviation; quantify important order reversals and correct them |
| Task-sensitive prediction | VAML/IterVAML/VaGraM/CVAML | Value-based losses, real-feature sensitivity, off-support and stochastic-loss analysis | A CE-specific capability/tradeoff must be demonstrated; no broad theoretical novelty claim |
| Future-free efficient deployment | FutureNav, WorldMAP, privileged-future distillation | Future learning signal without generation at inference | Necessary cheaper rival; a deployment pivot if it wins |
| Controlled mismatch analysis | World4Scorer C.4/D.1 and MCR | Task constraints, interventions and ranking diagnostics | A new reproducible semantic finding, not merely reuse of the evaluation protocol |
| Direct versus predictive decision value | Beyond Next-Screen Prediction §§3.1,3.3,5.3 | Matched direct/predicted ranking with the same candidates, labels and training | Actual CE continuation returns differ from GUI reference-action labels, but the generic diagnostic principle is already explicit |

The strongest overlaps are [World4Scorer](https://arxiv.org/html/2609.36438v1) and [MCR](https://arxiv.org/html/2608.09073v1), both inspected as 2026 preprints; no accepted venue was verified. Primary status and precise locators for all rows are in [Sources](Sources.md), [A4 novelty matrix](assignments/04_direct_novelty.md) and [A5 cross-field audit](assignments/05_crossfield.md). No unsuccessful search is treated as proof of novelty. Search confidence is strong for the identified overlaps, limited for proving a distinct contribution across all literature.

The fresh final review found [Beyond Next-Screen Prediction](https://arxiv.org/html/2610.24372v1), a 17 October 2026 preprint directly testing matched direct/prediction-augmented ranking in GUI tasks. It further narrows the diagnostic claim; its negative predictive result does not establish a CE outcome. [RIWANav](https://arxiv.org/html/2610.08640v1), another inspected preprint, makes geometry/novelty-weighted native dynamics a relevant simple rival. A10 records precise differences and publication-status limits.

## 5. Creative revision and simplicity

The team considered substantively different interventions: measured-gap dynamics, outcome-weighted real-feature contrasts, direct-value residuals, occupancy aggregation, training-only distillation, actual evidence acquisition, longer horizons, adaptive branch repeats, cheap residual prediction, computation gating and a projected task-gradient idea. Independently saved A2/A5/A6 suggestions, operational shortlist details and rejected alternatives are in [A8 synthesis](assignments/08_synthesis.md) and [candidate register](candidate_register.md).

One mechanism was retained conditionally: measured-outcome adaptation through the existing critic. Feature-contrast/per-action weighting is a simpler rival, not an extra component. The projected-gradient idea was rejected: a projected nonlinear gradient need not be the gradient of a scalar objective, and a valid projected-input alternative still hides orthogonal errors without a demonstrated need. Active sensing changes the research mechanism and has direct prior overlap; it is an explicit pivot outside the core. A compute gate and longer horizon must wait for measured gain or a diagnosed H1 failure.

The revision preserves your objective and trial registry. It specifies the previously unfinished objective, proposes a checkpoint-dependent native-grid adapter, adds statistically valid branch evaluation and explicit base fallback, and tests utility-only scorer training before retaining auxiliary losses. Every addition has a removal/replacement comparison. This is a mechanism refinement, not a claim that new losses or engineering fixes themselves supply two novelties. [Versioned proposal](proposal_v3.md) includes the full original→weakness→change→cost→test table.

Independent review narrowed the proposed finding further: nDTW/success are not instruction-clause ground truth. V2 separates strict outcome/proxy reversals from geometric ties, turns and STOP, specifies an immediate geodesic-progress diagnostic and requires independent valid route-stage annotations before a semantic claim. This simple diagnostic is explicitly not an MCR reproduction. B-fitted critic supervision also reaches A-trained dynamics indirectly; v2 discloses this legal reuse and requires equal data access for controls. A7 still prefers the cheaper observed-feature contrast objective; that remains a substantive rival rather than an erased disagreement.

V3 makes one further boundary explicit: `Q_ref` includes the later competence of this particular actor. A better continuation return does not establish that the immediate action is intrinsically more instruction-compliant. Separate blind immediate-compliance annotations and later-failure analysis are required; otherwise retain a policy-relative outcome claim. Even the broad conflict between shortest-goal supervision and instructions has prior precedent in [LAW, EMNLP2021](https://aclanthology.org/2021.emnlp-main.328.pdf), so obvious detour examples alone cannot establish the contribution.

## 6. Implementation, information and cost

| Quantity | Training | Deployment | Consequence |
|---|---|---|---|
| Instruction, real RGB/actions/history, elapsed budget | Available | Available | May condition policy, critic and appropriate model inputs |
| Predicted next features | Computed | Computed | Model output, not observed evidence |
| Actual candidate next views and continuation outcomes | Simulator branch supervision | Unavailable before execution | Privileged diagnostics/labels only |
| Simulator pose, reference route, goal coordinates, whole-path metrics | Label/evaluation construction | Excluded from model inputs | Separate label schema; audit for leakage |
| Future realized displacement/collision | Label/diagnostic | Unavailable | NWM conditioning must map legal command/timing, not use realized future pose |
| Trainable recurrent memory | Replayed with current weights | Updated once per real observation | Cached stale learned states can invalidate training |

Use pinned StreamVLN for policy/evaluation integration, DINO-WM for the first predictor and official VLN-CE metrics. A6 maps concrete files and adapters, including `models/visual_world_model.py`, actor action queues/cache and NWM action/timing construction. The current scorer configuration is not a complete loaded training pipeline. Preserve checkpoint-specific normalization/token ordering; if the inspected DINO contract applies, pooling 14×14 to 7×7 is simpler than modifying pretrained position grids. [A6 implementation dossier](assignments/06_feasibility.md) records pins, file paths and all conditional qualifications.

Measured scorer parameter counts are about 2.87–3.17M, depending on policy width. Model-metadata BF16 weight-only estimates are .945/1.664/3.963 GiB for the three complete VLMs; the OneVision “0.5B” name does not mean .5B total parameters. These estimates exclude KV cache, activations, dynamics, vision and simulator. The inspected NWM default uses 250 diffusion steps; three candidates imply three times that denoising work before decoder/DINO costs. No full inference latency is established.

The probed host reports RTX4070 Laptop 8,188 MiB and 31.7 GiB RAM. The inspected Python3.13 Torch installation is CPU-only; Habitat, trained actor adapters and scoped assets were not demonstrated. This is a scoped environment finding, not a claim that no other working environment or asset location exists. Offline staging can reduce memory residency, but does not prove online 8 GB fit.

## 7. What to implement and measure first

1. Resolve the CUDA/Habitat/assets setup and select the exact DINO checkpoint contract. Train/validate one small CE actor; preserve native-chunk and H1 baselines.
2. Implement immutable branch records, reset-prefix replay, canonical action mapping, full metric prefix/budget and forced-action cache commit. Check STOP, repeated roots, branch order, invalid-score fallback and label-only input exclusion.
3. Collect the 64-root debug bundle. Then estimate opportunity on a separate episode/scene sample with repeated/split seeds if needed; do not train a 3M scorer on the debug bundle and call memorization evidence.
4. Fit learning curves for actual-next and matched full-history no-future critics. Then compare predicted/native dynamics with null, copy, cheap predictor and training-only/direct-value rivals.
5. Only after predictive usefulness, test the one adaptation against native and simpler objective controls. Freeze the final comparison before unseen-scene closed-loop evaluation; measure all generation, encoding, transfer and simulator costs.

[A7](assignments/07_experiments.md) fixes estimands, split policy, seed streams, controls, positive/negative/inconclusive interpretations and planning thresholds. Its proposed defaults are ≥.02 utility candidate lift, ≥.01 utility actual/predicted lift, and a final ≥2-point SR gain with a positive interval and nDTW guardrail. They are **declared design assumptions**, not evidence-derived norms or predicted gains. Confidence intervals cluster scenes/routes/episodes; correlated roots and four actions do not become independent samples. All model selection occurs on training-scene development partitions, with R2R/RxR scene overlap grouped. Small learning-curve tiers and compute caps are planning aids, not power claims.

## 8. Evidence actually executed

| Check | Observed result | Limit |
|---|---|---|
| A2 finite-history arithmetic | 500 depth-four trees, 7,500 history comparisons and 79 margin checks passed | Numerical sanity checks of stated conditional deductions |
| A3 independent checks | 2,150 stochastic-history comparisons, 1,296 margin cases, counterexamples and gradient checks; unchanged A2 replay matched | Not a general proof or measured CE failure frequency |
| A6 original scorer on copied source | All 13 synthetic check groups passed | Tensor/utility/interface behavior, not semantic navigation |
| A6 CPU scorer profile | One synthetic eight-root/32-observation optimizer step: 2.44 s, 1.049 GiB peak process memory; median four-candidate score:16.18 ms | CPU scorer only; excludes full actor/WM/simulator/GPU path |
| A9 independent final checks | Centered/pair identity and gradients agreed numerically;6,561 greedy/margin cases passed; copied scorer passed movement/STOP/frozen-weight gradient checks | Local64-token route only; no adapted world model or49-token integration |
| CE collection/training/unseen evaluation/GPU pipeline | **Not executed** | Missing integrated environment/data/checkpoints; all efficacy claims remain hypotheses |

Reproducible code, logs, seeds, configurations and exact scope are in [experiment register](experiments/README.md). Neither multiple workers agreeing nor passing these checks substitutes for a navigation result.

## 9. Independent review and decision

Assignments 9 and 10 independently reviewed hashed v1 without the synthesis author's preferred verdict or each other's initial assessment, then rechecked the exact v2/v3 revisions. A9 accepts v3 as a coherent conditional proposal, while withholding empirical/semantic validity; A10 recommends revision and a bounded pilot, withholding the two-contribution claim. Their objections, version hashes and reconciliation are in [review reconciliation](review_reconciliation.md). All ten substantive assignments are complete and documented in [assignment ledger](assignment_ledger.md). Native worker independence is procedural; no external-provider or cross-model verification is claimed.

| Judgment | Original design | Recommended conditional revision |
|---|---|---|
| Problem validity | Broad problem well supported; selected-policy bottleneck unverified | Same; narrower semantic-ordering hypothesis explicitly testable |
| Logic/mechanism | Coherent Q_ref reranking; predictive contribution uncertain | Coherent conditional adaptation, with critic exploitation and missing signal risks |
| Mathematics | Valid under history/occupancy/error assumptions; not benchmark guarantees | Centered-loss/gradient contract valid under stated conditions; no new general theorem |
| Novelty | Generic architecture substantially anticipated | Precise CE extension conceivable; two strong contributions not secured |
| Empirical support | Synthetic reference only | No CE improvement or adapted dynamics result |
| Feasibility | Feature scorer works; integrated 8 GB system unproved | Staging improves testability; environment/checkpoint/collection prerequisites remain |
| Scientific value versus cost | Worth a bounded diagnostic, not a full grid on present evidence | Invest first in causal discrimination; stop if a simpler direct model explains the benefit |

**Practical recommendation:** defer substantial model training and the full trial matrix; proceed with the minimal baseline/replay/diagnostic implementation. This is neither confirmation nor wholesale rejection. Evidence changing the recommendation would be reproducible candidate opportunity, predictive gains beyond strong direct/history controls, a consequential semantic mismatch corrected by the proposed adaptation, and retained closed-loop value at acceptable total cost. If fewer than two substantive contributions survive those tests, report that gap rather than adding machinery.

## 10. Sources, limits and final claim table

[Sources](Sources.md) provides publication status and precise primary locators; A1/A4/A5/A9/A10 contain query/access ledgers and source snapshots. [Shared evidence register](evidence_register.md) distinguishes specification, published claims, code, execution, deductions and hypotheses. Important access limits include incomplete released navigation stacks, gated/unselected weights, no demonstrated CE assets, a challenged future-distillation paper endpoint and a timed-out textbook PDF (primary indexed passage checked). These limit reproduction or source coverage as specified; none proves the underlying method invalid. No benchmark gains, citation claims, upstream provenance authentication or acceptance prediction were invented.

| Central claim | Supporting evidence | Strongest objection | Current status | Required change | Next decisive test |
|---|---|---|---|---|---|
| Better primitive choices remain available | Adjacent oracle analyses | Frozen continuation may fail from every candidate | Unverified here | Establish unbiased opportunity | Independent-seed candidate selection lift |
| Predicted futures improve decisions | Plausible finite-capacity/prior benefit | Direct value/history/extra compute may explain everything | Hypothesis | Equalize legal inputs/data and account for full cost | Predicted versus full-history direct/copy/tiny predictor |
| Q_ref supports repeated reranking | Conditional derivation and independent numerical checks | Estimation and occupancy error | Mathematically conditional | Calibrate on relevant roots; no safety claim | Offline-to-closed-loop comparison |
| Measured semantic gaps justify dynamics adaptation | Specific distinction from geometric proxy; operational loss | VAML/MCR/World4Scorer overlap; critic exploitation | Conditional candidate | Demonstrate consequential mismatch and simpler-rival superiority | Native/weighted controls on prespecified mismatch roots |
| Two strong CVPR contributions exist | Two complementary targets articulated | Neither target has CE evidence; diagnostic plus transfer may be insufficient | **Target not yet met** | Establish independent scientific substance | Fresh novelty audit after decisive results |
| Full method is lightweight/8 GB feasible | Small scorer, scoped device metadata | Total actor/dynamics/simulator cost absent | Unproven | Integrate and profile complete path | Long-history p95 latency and total-device peak |
