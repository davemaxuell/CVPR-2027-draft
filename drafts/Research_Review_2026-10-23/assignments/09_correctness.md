# Assignment 9 — Independent final correctness review

2026-10-23 · Search cutoff 2026-10-23 · Native independent worker; not external-model verification

**Disposition: a coherent conditional research candidate after the protocol clarifications below; not an implemented method, demonstrated improvement, or two established CVPR contributions.** No fatal algebraic or information-flow contradiction was found in the deterministic DINO version. The strongest scientific risk is that adaptation learns features that manipulate a frozen critic without improving real decisions. The strongest contribution risk is that a generic outcome/proxy mismatch is presented as a new semantic finding.

## 1. Independence, frozen version, and evidence inspected

I did not design the proposal. I first verified and reviewed:

- `proposal_v1.md`, SHA256 **4CBC57E6781D83BF3C3A88F9030C54EFBD39A45B8B98535EB68D5DD163D77F9F**.
- Original `CE_World_Model_Approach.md`, `Scorer_Design.md`, and the actual `Scorer/scorer.py`, including its future-input detach, STOP replacement, independent candidate processing, real-memory update and exception behavior.
- Primary CVAML v2, World4Scorer v1 and MCR v1 pages, using Sources.md only as a locator.

I saved [09_initial_assessment.md](09_initial_assessment.md), SHA256 `7C63A5CCB3E1185B3BF0A2ADDDF2C1445A489CB5D21684DFC8D76CA92DCFBE6E`, **before reading A1–A7, the synthesis verdict or another final review**. I have not read Assignment 8, the main report or Assignment 10 for this disposition. Later I read the A2/A3 check sources and result JSON, A6 pinned-source/environment/asset evidence and underlying DINO-WM source, and A7/experiment register. The aligned A7 version inspected here has SHA256 `C9A720419FF13C25AA1576FDD084A7A4DA42E52ABC1B692EBF4FBA5E584C1FFE`; the register has SHA256 `4209AC56A6F0F75E69490BEF40E61C62DEE1600F41007A273CB2C9205F136A61`. Earlier A7 E4 text used a different loss/critic schedule; the inspected aligned version replaces it with v1's centered objective and preserves the alternative as a rival.

I directly inspected World4Scorer Appendix C.4 and D.2, MCR §3.3.2 Eqs.6–7 and CVAML §4.2 Eq.3/Proposition3. I independently ran [09_correctness_checks.py](../evidence/09_correctness_checks.py); [results](../evidence/09_correctness_checks.json) passed. No original file or proposal was edited, no model weights were downloaded, and no CE integration/training/evaluation was run.

## 2. Formal target and information flow

**The reference continuation is legitimate.** The label is an action value under a declared frozen one-action policy, with full remaining budget, real prefix and whole-path terminal metric. This is not an estimate of the future reranking policy's value. That distinction does not invalidate policy improvement: on a finite full-history tree, if the new rule chooses an action whose true `Q_ref` is at least the reference value at every visited history, backward induction gives improvement. STOP is an ordinary terminal action. Whole-path nDTW can be represented as a terminal reward on the augmented history; remaining action/collision costs differ from whole-episode costs by a root-common past term.

The guarantee requires exact values and adequate history state everywhere the new rule visits. A compressed learned input is not that state. Approximation errors on new occupancy, aliasing, and stochastic continuation can defeat improvement. A2/A3's synthetic trees support the limited algebraic claim, not CE generalization. Refreshing occupancy while retaining the same reference continuation is therefore coherent; relabeling with a different continuation is not theoretically mandatory.

**Physical prediction need not be instruction-conditioned.** An instruction-independent physical feature can be scored differently by an instruction-conditioned critic. This factorization is valid. It may be capacity-limited: a single feature prediction must work across instructions, and a finite inaccurate critic can induce incompatible gradients. The proposal promises no universal realizability.

**Deterministic imagination adds computation, not new evidence beyond its input.** If `Zhat=F(H,a)`, it carries no additional information conditional on the complete legal `H,a`. A finite direct model may still benefit from the predictor's representation or pretraining. Thus the scientific claim is finite-model usefulness at measured cost. A direct critic receiving only `q_t/M_t` is not information-matched when F receives additional recent raw frames/actions. A7 E2 now explicitly grants the direct path that same history; v1 should state this directly. Count extra prediction pretraining, teacher data and parameters.

Actual-next observations are privileged. In stochastic branches, `E[U|x,a,Z_real]` generally differs from `E[U|x,a,Zhat]`; independent imagined noise does not reveal the real draw. The final predicted-input critic training and separate qualification are appropriate. One deterministic branch is exact for the saved simulator root/protocol, not automatically for the population conditional on an aliased observation representation.

**Legal deployed inputs remain restricted.** The conceptual metric prefix and reference route can define labels; they must not enter the network loader. Removing/shuffling label-only fields should leave predictions unchanged. Real memory is updated only from executed observations/actions. Command encodings must not include realized future displacement or collision. The original DINO-WM code also expects proprioception; transfer must explicitly remove/replace incompatible channels with legally available information.

## 3. Objective, dimensions, normalization and gradient

Let `e=s−y ∈ R^4`, `P=I−11ᵀ/4 ∈ R^(4×4)`. The task loss `||Pe||²/4` is a scalar in squared utility units. `P=Pᵀ=P²`, so its score gradient is `2Pe/4` per root, with an additional batch-average factor. The v1 chain-rule expression is correct up to the stated omitted scale.

The identity

`Σ_(a<b)(e_a−e_b)² = K||Pe||²`

is correct. Therefore v1's task loss equals the **pair sum divided by K²**, or the pair mean multiplied by `(K−1)/(2K)`; for K=4 that multiplier is **3/8**. Uncorrected pair averaging changes the effective lambda. Complete four-action roots are required for this exact implementation; masking or weighting pairs changes the objective. The task is ordinary centered regression and introduces no extra labels. Root-common score offsets do not affect selection or margins.

The native loss's patch/channel/time reduction and any fixed scaling must be recorded. Squared utility units and mean feature-error units do not have a naturally universal conversion. A fixed training-batch normalization plus an equal tuning budget is a reproducible convention, not a new objective principle or a guarantee against poor conditioning.

**The intended gradient is feasible but absent in the original baseline code.** Original scorer line 151 detaches `future_patches`. Freezing critic parameters alone would still allow its input Jacobian; wrapping the critic forward in `no_grad` would not. For adaptation, keep parameters frozen and the forward differentiable, detach targets and real frozen encodings, and retain gradients through predicted features/action encoder/dynamics. STOP is replaced by current real patches, so it has no predictor-input gradient; its residual still contributes to movement gradients through centering.

My in-memory copied scorer check removed only that explicit future detach. All three movement future gradients were finite and nonzero; STOP gradient was zero, and critic/label gradients remained absent. It used the original 64-token contract, not a completed 49-token adapter. This validates a local gradient route, not a trained dynamics update or useful semantic gradient.

**The regret inequality is correct with its stated assumptions.** For true values y, selected greedy action j and best action i, `s_j≥s_i` gives

`y_i−y_j ≤ e_j−e_i = (Pe)_j−(Pe)_i ≤ sqrt(2)||Pe||`.

Finite noisy training residual does not establish the premise or a held-out bound. V1 explicitly states argmax selection; its deployed margin rule is different. The analogous algebraic bound is `regret ≤ margin + sqrt(2)||Pe||`, not the zero-margin expression. Neither creates a calibrated safety certificate.

## 4. Stochasticity, optimization and test validity

For model/outcome gaps independent **conditional on the root**, the stated mean/variance decomposition is correct. Penalizing independent sampled gaps suppresses model score variance. Deterministic DINO removes that model-sampling term but not noisy labels, partial observability, nonlinear-critic mismatch or finite-sample overfitting. CVAML already analyzes the stochastic-loss problem; its §4.2 Eq.3 and Proposition3 rule out presenting a generic variance correction as new. A stochastic CDiT extension would require a separate expectation target and honest gradient/sampling-cost protocol. [CVAML v2](https://arxiv.org/html/2505.22772v2#S4.SS2)

**Frozen-critic exploitation is the strongest remaining objection.** The native anchor cannot prove features remain physically faithful or that critic values remain valid after adaptation. The task loss can be useful where the critic is locally accurate, but fitting that critic's outputs is not independent validation. Require post-adaptation held-out measured action choice/outcomes, calibration and feature/native-error diagnostics, then closed-loop evaluation. Reject an improvement confined to the optimized surrogate. A7 E4 now says the critic stays fixed and that invalidated critic qualification fails the candidate; this is appropriate. Retuning the critic would test a different, explicitly versioned method.

**Noisy-max opportunity is addressed correctly.** A sample maximum is optimistic even when all true action values tie. The split-seed selection/evaluation statistic estimates the performance of a finite-label selector, not the true oracle maximum. A7 makes this distinction, allows negative estimates, keeps model/continuation noise separate and clusters episode/scene uncertainty. With deterministic verified branches, root-specific max-minus-selected is valid.

The A/B/C scene design is suitable for held-out evaluation when all sharing is declared. However, a B-trained frozen critic passes B supervision indirectly into the A-adapted predictor. This is legitimate training reuse, not held-out leakage, but contradicts a strong claim of complete scorer/dynamics information separation. The aligned A7 correctly declares this dependency and grants controls equal access. V1 should carry the same wording. C remains development, including all margin/checkpoint decisions; unseen outcomes cannot be reused for repair and selection.

A7's 0.02/0.01 utility gates, 2-point SR target and resource caps are explicitly planning assumptions. They do not constitute power calculations. The debug sample cannot support a paper result. Actual-next failure may reflect an inadequate finite critic rather than absent predictive signal. Conversely, permutation sensitivity only proves dependence and cannot establish helpfulness.

## 5. Do the two target claims cohere?

**Yes as a conditional argument; no as two contributions already earned.** C1 should establish a prevalent, consequential instruction-conditioned outcome/proxy mismatch and a finite predictive benefit beyond input/supervision/compute-matched direct learning. C2 should correct those same observed errors through the specified predictor intervention. A generic mismatch alone does not show prediction is the cause or cure.

The geometric comparator is underspecified in v1: freeze its target, sign, horizon, orientation/turn treatment, STOP handling and tie band before outcomes. In H1 navigation, turns often leave position distance unchanged. A tied geometric score with different returns is not a strict ordering reversal. Report strict non-tie disagreements separately from ties/unresolved gaps, and report ordinary-root prevalence rather than only curated examples.

The composite success/nDTW utility is a task proxy, not ground truth for every language clause. Until valid matched instruction/route-stage contrasts establish a semantic role, use **instruction-conditioned outcome/proxy mismatch**, not a semantic understanding claim. Disagreement can arise from terminal-success weighting, delayed effects, actor-continuation quirks, orientation blindness or route geometry. These explanations need explicit controls; otherwise C1 may be an expected application observation.

Primary-source checks materially narrow novelty:

- World4Scorer Appendix D.2 already trains outcome scoring on predicted features/action sequences with a frozen world model. Appendix C.4 treats local constraints on candidate comparisons; its statement is local sensitivity, not global learning or performance. D.1's reassignment control also affects a non-future scorer, so permutation sensitivity is not unique evidence for prediction. V1 correctly acknowledges this; its remaining distinction is the CE instruction/history/primitive/full-continuation setting and the fixed-critic predictor adaptation, not outcome supervision or interventions generally. [World4Scorer v1, C.4/D.1/D.2](https://arxiv.org/html/2609.36438v1)
- MCR §3.3.2 already shapes planning order through a frozen-DINO latent predictor, using dead-reckoned deviation and a prediction anchor. Measured instruction-conditioned continuation returns differ from that proxy, but changing the supervision source is not automatically a paper contribution. The claim needs a consequential failure of the proxy and improvement beyond simple outcome-weighted prediction/direct-value learning. [MCR v1, §3.3.2](https://arxiv.org/html/2608.09073v1#S3.SS3.SSS2)

The original reference itself does not currently implement a geometry-based task loss; its future-usefulness extension was unspecified. Describe the proposal as filling that extension and comparing against geometric-proxy rivals, rather than implying a verified existing project loss is being replaced.

## 6. Independent bounded evidence and implementation boundary

[Own results](../evidence/09_correctness_checks.json): centered/pair loss difference `2.22e−16`; analytic/autograd gradient difference `2.78e−17`; 6,561 four-action score/value cases passed greedy and margin-adjusted bounds; copied differentiable scorer route passed movement/STOP/frozen-target checks. Original scorer SHA256 remained `296c15901df8c806d0cc21f5bf5880895929dfe1c181538ac0f29d44b7336e8a`.

I read A2/A3's underlying scripts rather than adopting their verdicts. Their exact-Q, approximate-occupancy, aliasing, noisy-max and stochastic-sampling examples are logically appropriate bounded checks. They are not prevalence estimates.

The pinned DINO-WM source computes `num_side_patches=image_size//16`, then resizes DINO input to that count times its 14-pixel patch size: at configured 224 this yields 196 pixels and 14×14 features. Original scorer pooling, positions and lengths assume 16×16→8×8. V1's conditional 14×14→7×7 correction is justified, but requires all shape-dependent code and every source to change consistently. The selected checkpoint has not yet been loaded. [Pinned DINO-WM code](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py#L54)

A6's CPU-only interpreter and missing demonstrated CE pipeline bound execution status. Its scorer profile does not certify actor/WM/simulator residency or full 8 GB execution. Original nonfinite handling raises; actual base fallback, cache commitment and replay integrity remain implementation work.

## 7. Required corrections, resolving evidence, and stop decision

Before presenting an executable scientific contract:

1. State the complete legal-input match and indirect B-supervision dependency in the proposal.
2. Freeze one operational geometric comparator and separate strict reversals, ties and uncertainty; narrow semantic wording until instruction/route-stage evidence earns it.
3. State post-adaptation independent outcome qualification explicitly; never rescue a failed fixed critic with an unreported refit.
4. Record all loss reductions/normalization, native-grid adapter and legal action/proprioception inputs; these remain implementation gates.
5. Describe the original objective accurately as an unspecified extension, with geometric surrogate objectives as comparators.

These are scope/attribution/contract corrections, not a mathematical refutation of the candidate. The coordinator has been informed promptly and plans a versioned revision; this review is still of the exact v1 hash above.

Evidence sufficient to advance: reproducible ordinary-root opportunity; a matched direct-Q gap attributable to useful predictable consequences; strict consequential proxy disagreements; adapted-model improvements on independent actual outcomes concentrated in the predeclared failure while retained overall; closed-loop benefit and measured total cost. C1 and C2 must survive their strongest simpler rivals.

Stop this mechanism if opportunity is precisely too small, adequate predictive-channel controls tie, simpler equal-data/cost objectives match the remedy, critic-only improvements fail real outcomes, or total cost loses to direct alternatives. Stop a paper-level two-contribution claim if either target remains an expected application observation or an inherited loss with no distinctive correction. Wide intervals or underfit critics remain inconclusive.

**Final v1 disposition: proceed only to staged qualification after the listed clarifications. Do not claim CE success, semantic understanding, novel ranking theory, two established contributions, or full-device feasibility.**


## 8. Same-assignment recheck of frozen v2

I directly verified and read `proposal_v2.md`, SHA256 **B67DF5A386C4FFAB02672D11A561F7C6025E09A99172E80FEE08A52B0CBC6305**, after completing the independent v1 review. I reread aligned A7 E4/E5; its hash remains **C9A720419FF13C25AA1576FDD084A7A4DA42E52ABC1B692EBF4FBA5E584C1FFE**. This is a recheck within Assignment 9, not an eleventh assignment or a new independent reviewer. The initial assessment and v1 disposition above remain preserved.

**V2 resolves the material protocol ambiguities raised in this review.**

- It explicitly declares the `B labels → frozen critic → adapted phi` dependency and equal permissible A/B supervision for rivals.
- It grants direct controls the union of legal predictor/scorer history inputs.
- It requires post-adaptation qualification using independent measured outcomes and closed-loop behavior, with no silent critic refit.
- Section 7 defines a concrete privileged immediate geodesic-progress diagnostic, its sign, movement-only domain, heading blindness, separate STOP stratum and meter-based tie tolerance. The normalization/tolerance units agree.
- It separates strict reversals from geometry ties and uncertain utility gaps. It explicitly prevents the obvious H1 turn/STOP blind spots from earning C1.
- It narrows outcome/proxy disagreement versus independently established semantics, supplies valid blind annotation/matched-stage requirements, and says the immediate-progress comparator is not a faithful MCR reproduction.
- It preserves the cheaper feature-contrast rival and A7's substantive preference instead of declaring the chosen objective superior in advance.

These additions improve falsifiability without changing the mathematical mechanism, so rerunning the bounded algebra/gradient tests would add no new evidence. Those tests continue to apply to the unchanged centered objective; they remain unrelated to CE efficacy.

Two wording cautions remain, neither a fatal logic error. The compact C2 phrase “replace a geometry-derived task surrogate” must be read as a comparison to a geometric-surrogate alternative; the supplied original project had no implemented such extension. The main report should say **specify the previously undefined extension using measured outcome contrasts and compare it against geometric-surrogate rivals**. Also, the older short table heading “C1 Semantic ordering” and uses of “semantic/geometric disagreement” are conditional on Section 7's independent semantic qualification; until then the result is only outcome/proxy ordering. The revised explicit scope governs the recommendation.

I also directly opened the two newly cited primary preprints, without a new broad search. Beyond Next-Screen Prediction §3.3 Eq.3 and §5.3 explicitly define and test matched direct-versus-prediction-augmented decision value under reference-action supervision. It further narrows generic protocol novelty, while its own offline comparison is not a universal result about CE/online prediction. [Beyond Next-Screen Prediction v1](https://arxiv.org/html/2610.24372v1#S3.SS3)

RIWANav §III-C uses geometric expert-consistency filtering and policy-induced curriculum for world-model updates; that supports including simple weighted native-objective rivals. Its recursive policy/model training is different from the proposed fixed-actor/fixed-critic measured-return adaptation. I did not reproduce either paper's experiments. [RIWANav v1, §III-C](https://arxiv.org/html/2610.08640v1#S3.SS3)

**Final v2 disposition: technically coherent conditional candidate, suitable for the staged qualification protocol with the explicit implementation gates and narrow wording above. It remains unproven that prediction is needed, that the frozen critic supplies useful adaptation gradients, that the semantic role is distinctive, or that two contributions exist.** The strongest remaining objection is empirical critic exploitation/generalization; the strongest stop reason is a matched simpler control achieving the same real-outcome benefit at lower cost. No current evidence justifies a CVPR-ready or benchmark-improvement claim.

### A9 search and verification ledger

Primary URL opens succeeded for CVAML v2, World4Scorer v1, MCR v1, Beyond Next-Screen Prediction v1 and RIWANav v1. Targeted primary sections were checked as cited above. No broad literature-completeness or absence-of-prior-art claim is made. Local searches used the original code, source snapshots, A2/A3 check sources/results, A6 pins/environment records and final A7/experiment register. The directly executed A9 test and its result JSON have exact proposal-v1 and original-scorer hashes; v2 retains the checked objective and is separately hash-verified here. Original materials, v1 and v2 remain unchanged.


## 9. Final frozen v3 recheck and operative disposition

I verified `proposal_v3.md` SHA256 **2E4A46C636502D105422951C5AFE12DD7731C65D2F3C6733EC69643394000909** and inspected the complete v2→v3 diff. The only substantive changes are C2's comparator attribution and the new §8 claim-boundary clarification. The architecture, centered loss, training schedule and experimental requirements remain unchanged. V1/v2 review history above is retained.

V3 corrects the attribution issue: the geometry-derived objective belongs to a prior-work comparator; the user's original usefulness extension was unspecified. It also states that `Q_ref` is policy-relative. Even valid instruction changes can alter later actor competence, so higher continuation value does not itself establish more compliant immediate action. Separate blind immediate-action/route-stage annotations and localization of later continuation failures are appropriate requirements. Their existence and discriminative quality remain untested.

**Operative final disposition for v3: pass as a coherent, conditional, falsifiable research proposal; do not pass as a validated implementation, demonstrated CE improvement, semantic finding, or two established contributions.** No remaining fatal mathematical or information-flow error is identified in the declared deterministic first-stage contract. The implementation gates and empirical stop conditions in this review still apply. A sophisticated predictor is justified only if it beats the strongest matched direct/simple-loss controls on independent actual outcomes and at measured total cost. If policy-relative opportunity is real but the immediate semantic attribution is not established, narrow the finding; that narrowing can leave the two-contribution target unmet.

No additional numerical run was warranted by these wording-only changes. The coordinator received the review concerns before revision; this recheck is performed by the same A9 worker and makes no claim of external-model independence.



