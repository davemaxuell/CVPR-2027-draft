# Audit of Earlier VLN Idea Premises

This appendix checks claims in the earlier local idea notes before using them as premises for new proposals. The notes are hypotheses, not established research results. These checks do not establish a new navigation method or benchmark improvement.

## DUET can revisit previously visited nodes

**Inspected code.** DUET's global target selection can mask visited targets while its movement routine still routes through visited nodes. The released R2R agent also caches stop probabilities and, on episode termination, appends a graph path to the node with the highest cached stop score. The claim that an overshot goal is permanently unreachable is therefore false for this implementation. The coordinator independently read the source snapshot inspected by assignment 01. [DUET commit 93e8b233, agent.py lines 287–300 and 394–402](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/agent.py#L287), [termination and return, lines 452–478](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/agent.py#L452).

**Deduction.** Frontier selection, physical/graph traversal, and final endpoint selection are different operations. A proposed repair to visited-node masking needs to show which operation fails and should compare against the existing return mechanism.

**Executed source-function check.** The reviewed `FloydGraph` class was extracted from the pinned source with Python's AST and executed on a four-node tree. From a visited dead end, its route to an unvisited frontier passed through the visited junction; its route back to the visited start was also returned correctly. See [test code](../experiments/duet_routing_check.py) and [recorded output](../experiments/duet_routing_check_results.json). This tests the routing function, not the learned action policy.

## An oracle success gap does not identify a failure mechanism

Let V indicate that a trajectory visits a state satisfying the success criterion and E indicate that its final state satisfies the same criterion. With identical episodes and success conditions, E implies V, so

\[
\mathrm{OSR}-\mathrm{SR}=P(V)-P(E)=P(V\cap \neg E).
\]

**Mathematical deduction.** This identifies trajectories that reach a successful state but finish elsewhere. It does not establish that competition between stop and movement logits caused those outcomes. The gap can motivate an endpoint-selection investigation; attribution requires controlled intervention on the stopping mechanism, with the navigation trajectory and return procedure accounted for.

## Separating a stop variable is not by itself a new probability model

For a categorical policy with stop logit z_s and movement logits z_1 through z_K, define L=log(sum_i exp(z_i)). Then

\[
P(\mathrm{stop})=\sigma(z_s-L),\qquad
P(i\mid\mathrm{continue})=\operatorname{softmax}(z_{1:K})_i.
\]

Multiplying the conditional movement probability by 1-P(stop) exactly recovers the original categorical probability. This is an algebraic identity for finite logits and at least one permitted movement action; the all-movements-masked case requires explicit termination handling.

**Executed numerical check.** Across 1,000 randomly generated finite logit vectors, the maximum difference between the categorical and factored distributions was 2.78e-16. This checks the implementation of the identity, not a navigation hypothesis. A separate head can still change features, supervision, calibration, gradients, or the decision rule; those changes, rather than factorization alone, must carry the scientific contribution.

**Toy illustration.** Holding stop logit 1 and all movement logits 0, adding movement candidates changes stop probability from 0.731 (one candidate) to 0.120 (20 candidates). This illustrates normalization dependence. It does not show that candidate count is the cause of errors in a trained model, whose logits also change with its inputs.

## SPL equal to one does not certify exact route replay

For successful episodes, the usual SPL factor is L*/max(L*,L), where L* is shortest start-to-goal distance and L is executed distance. In a straight-line toy example with L*=10m and a 3m success radius, stopping after 8m is successful and has SPL=1 despite omitting the final 2m of the reference path. Different equal-length paths to the same endpoint provide another counterexample.

**Deduction, not an observed SRDF failure.** Using SPL=1 as a quality filter does not logically certify instruction entailment or exact reference traversal. Whether this matters empirically depends on the retained data. Assignment 01 locates the relevant SRDF filtering description in section 3.2, PDF page 5; the final paper dossier records the experiments that support its data-quality benefits. A proposed replacement filter must demonstrate additional signal beyond those benefits.

**Executed static geometry check.** The coordinator subsequently checked local R2R reference annotations against public connectivity graphs pinned to Matterport3DSimulator commit `589d091b111333f9e9f9d6cfd021b2eb68435925`. Using the graph loader conventions and evaluator formula inspected in the pinned DUET release, 4,051 of 4,675 training paths and 712 of 783 validation-unseen paths have a proper prefix ending within the strict 3m success radius; all those prefixes receive SPL=1. This is expected tolerance behavior, not an evaluator defect. The unit is a reference path, not each language annotation. No model selected the prefixes, no language entailment was labeled, and these are not SRDF-generated data. The result therefore does not measure semantic error prevalence or navigator performance. [Script](../experiments/geometry_audit.py), [setup and results](../experiments/geometry/results.json), [per-path output](../experiments/geometry/path_results.csv).

## A Gaussian instruction window does not guarantee fixed attention shares

The earlier local note specifies a normalized Gaussian with sigma=0.75, while claiming roughly 65% weight at floor(alpha) and zero interference from distant clauses. With seven clauses, the implemented formula gives 53.2% at clause 3 for alpha=3, and 30.1% at clause 3 for alpha=3.8. Every finite-distance Gaussian weight is positive.

**Executed numerical counterexamples.** These disprove the stated universal percentages and literal zero-interference claim. They do not disprove that a soft instruction window could be useful. Its alignment dynamics, revisits, conditional instructions and supervision still need analysis, and sub-instruction alignment has substantial prior art.

## A bounded monotone pointer can still make an irreversible error

The earlier note defines alpha_next=min(K,alpha+delta), with delta constrained to [0,0.5], and claims a single noisy observation cannot cause an irreversible clause jump. For alpha=1.9, delta=0.2 crosses the clause boundary. If that increment was erroneous, the nonnegative-update rule cannot undo it. Bounded increments limit jump magnitude; they do not establish recoverability. Likewise, the proposed target alpha*=1+(K-1)t/T assumes approximately uniform clause progress with time. An instruction containing one long corridor traversal and one short turn violates that assumption. Neither observation establishes that monotone progress is always inappropriate; it identifies unsupported guarantees and supervision assumptions.

The earlier recency penalty also has no universal “near zero after three steps” property without fixing its time constant. Its visit-count penalty does not decay at all. Claims of unchanged latency and numerical benchmark gains in that note are predictions without executed timing or navigation results and are excluded as evidence.

## Language sensitivity is not instruction-following correctness

The earlier “deafness meter” compares action distributions under an original instruction and a distractor. Total variation measures sensitivity, not whether the difference has the correct semantic direction. A model can react strongly to irrelevant wording and score highly; two valid instructions can require the same immediate action and produce a legitimately small difference. Scene, action set, progress, ambiguity, and the validity of the distractor also affect comparisons between on-path and off-path states.

A useful diagnostic therefore needs validated contrasts that require different decisions at the evaluated state, meaning-preserving controls, consistent histories, and labeled acceptable actions or continuations. Any associated training loss must avoid treating every alternative instruction as a reason to suppress a shared correct action. These are deductions about measurement validity, not evidence that language reliance does or does not decrease after data scaling. The observed sequence of scores from different trained systems cannot by itself identify data quantity as the causal factor.

## Established mechanisms must be included in novelty comparisons

The first-round primary-paper reviews found direct predecessors for generic separate stopping and progress/recovery ideas: FAST/Tactical Rewind maintains separate frontier and goal rankings; Topological Planning with Transformers explicitly evaluates a separate BCE stop objective; HSGM maintains ordered subtask states and a recovery rule; IDEAL-VLN supplies a sub-instruction mediator and re-anchoring; AdaNav selectively invokes reasoning; HAM-VLN stores failure-related experience. Exact scope and locators are in their linked [paper dossiers](../inventory/Dossier_Index.md). These precedents do not rule out every more specific contribution, but they rule out presenting the broad mechanism alone as new.

## Evidence and execution scope

[Executable checks](../experiments/premise_checks.py) and [exact outputs](../experiments/premise_checks_results.json) record the seed, Python version, setup and limitations. They also include elementary counterexamples showing non-injectivity of scalar mean pooling and negativity of a positive-definite matrix's log determinant. Those mathematical facts alone do not establish failures of a trained Gaussian-map or uncertainty-map method.

No trained navigation model, simulator, checkpoint or benchmark was executed for these checks. The isolated DUET graph class is the only official navigation-code component executed by the coordinator at this stage. Claims about empirical prevalence, causal effects of data scaling, or proposed performance improvements remain untested.
