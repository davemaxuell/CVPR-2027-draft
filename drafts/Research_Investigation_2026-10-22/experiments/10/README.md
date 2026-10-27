# Assignment 10 executed checks

Run `python -X utf8 experiments/10/source_checks.py` from the investigation directory. Dependencies used are the already installed NumPy, OpenCV and CPU PyTorch. See `source_check_results.json` for versions, source hashes and function ranges.

The script parses downloaded source with Python AST and compiles only the reviewed functions. It does **not** import their project modules or start a simulator/model. Saved `extracted_*.py` files show the executed functions. Two methods have zero-argument `.cuda()` removed to use CPU tensors. Fake experts return their expert identity; this tests dispatch rather than learned numerical outputs. The PSG memory methods use real histogram/cosine logic on synthetic RGB, with disk methods stubbed. The terminal metrics are deliberately supplied, and crop truth/control cause are stipulated; no real detection was adjudicated.

Executed results:

1. SkillNav actual dispatch: six of eight selected expert calls differ between batch-one and mixed batch-eight; reversing the batch changes the navigation-head dispatch for all eight items.
2. PSG-Nav actual write/read pathway: a synthetic detection followed by supplied failure is stored as negative and later rejected, including with positive evidence; the actual caller resets the goal and marks 289 cells in the synthetic map. This is a conditional counterexample, not a prevalence or navigation-performance result.
3. DUET actual teacher: different histories at the same reference endpoint both receive STOP. No natural instruction violation was established.

`evidence/10/asset_check.py` independently verifies local annotation hashes/scene disjointness, graph count and package absence, and retrieves a small remote checkpoint listing without downloading weights. No P2 selector, P3 predictor, navigation policy, annotation study or benchmark was executed.

Initial launch emitted no captured output before the tool yielded; the explicit UTF-8 run completed successfully and the results were saved. The rerun was to obtain visible completion, not to search over outcomes.
