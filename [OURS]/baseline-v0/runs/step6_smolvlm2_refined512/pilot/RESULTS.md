# Step 6 collection pilot complete

64 group-B roots from eight episodes produced 256 measured action branches. All labels, decoding decisions, costs, budgets, complete-path metrics and 1,368 indexed image files passed the audit.

The base action had successful continuations at 27 roots; the best measured candidate at 41 roots. Mean measured utility gain was 0.231794. This is a privileged branch diagnostic from two scenes, not a scorer or end-to-end navigation result. The four dataset/prefix-source strata each contain 16 roots.

All 256 independent nDTW checks passed. The original SDTW audit used exact equality against an independent floating-point implementation; 27 cases differed by at most 1.11e-16. The corrected audit retains exact stored success × stored nDTW consistency and uses the existing 1e-12 tolerance for independent recomputation. No collected label or collection semantics changed. [Diagnosis](audit_float_diagnosis.json), [audit revision](audit_revision.json), [verification](verification.json).

The locked larger budget is 1,024 fitting roots from B and 256 development roots from C. It reuses these exact 64 pilot roots.
