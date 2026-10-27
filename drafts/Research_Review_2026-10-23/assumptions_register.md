# Assumptions register

| ID | Origin | Assumption / constraint | Consequence if false |
|---|---|---|---|
| A1 | Explicit supplied design | R2R-CE and RxR-CE; RGB inference; four primitive actions; H=1; STOP identity | Different sensor/action regime requires new comparisons |
| A2 | Explicit supplied design | Three locked small VLM trial families and four WM trial families; CE-only newly collected data; 8 GB target | Cannot silently substitute stronger pretrained navigation agent as claimed reproduction |
| A3 | Team interpretation | One selected backbone and one WM suffice for core pilot; agreed families remain staged comparisons | Full 12-combination sweep would be premature and costly |
| A4 | Explicit supplied design | State, prefix metrics, remaining budget, policy cache and RNG can be restored or replayed | Branch labels invalid without restoration check |
| A5 | Explicit supplied design | Frozen pi_ref continuation is the supervised target; it differs from deployed reranking | No unqualified monotonic improvement guarantee |
| A6 | Team analysis | Generator randomness is independent of hidden current scene state conditional on real history and fixed trained weights | Imagined samples are learned computation/prior, not observed evidence |
| A7 | Missing user context | No particular CVPR year/track or deadline confirmed by current request | Do not invent submission deadlines; prior filenames mentioning 2027 are contextual only |
| A8 | Execution boundary | No large training/model downloads; bounded CPU checks allowed; existing simulator/data availability to inspect | Empirical claim remains untested if CE assets absent |
| A9 | Evidence boundary | Prior investigation is supporting material, not proof of current design validity | Independently inspect primary sources |

Revisions must label any new assumptions and substantial pivots. Original research files remain unchanged.

## Revision assumptions and dependencies

| ID | Origin | Assumption / constraint | Consequence if false |
|---|---|---|---|
| A10 | Team v2 | A frozen critic is informative and calibrated near both native and task-adapted predicted features | Centered score training can exploit it; fail qualification or choose simpler feature objective |
| A11 | Team v2 | Selected DINO checkpoint matches inspected14×14 native path | Verify before implementing49-token adapter; do not universalize the default finding |
| A12 | Team v2 | Deterministic DINO is the first objective experiment | Stochastic extensions need an explicit expectation/variance target and extra cost analysis |
| A13 | Team v2 disclosed relaxation | B-trained critic transfers B-label information into A-trained dynamics | Count union supervision and match controls; do not claim strict A/B statistical isolation |
| A14 | Team v2 | A direct critic receives all legal real history available to WM/scorer | Otherwise predictive gain is confounded by history access |
| A15 | Team v2 | Semantic interpretation is validated beyond geometric U and proxy ties | Without valid annotations/controls, retain outcome/proxy mismatch wording and unmet semantic target |
| A16 | A7 planning defaults | Split ratios, sample tiers, .02/.01 utility gates, +2pp SR, nDTW guardband, compute/latency caps | Version before outcomes; these are not measured gains, universal standards or powered sample claims |
| A17 | Review independence | A9/A10 inspect frozen versions independently and recheck material v2 changes | Do not call an unreviewed subsequent mechanism independently validated |

No additional architecture is assumed necessary. Active sensing, longer-horizon planning and training-only deployment are labeled alternatives/pivots outside the retained core. Third contribution deliberately omitted.
