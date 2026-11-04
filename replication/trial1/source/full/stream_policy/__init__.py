"""StreamVLN-style streaming navigation policy (trial_1: Qwen3.5-4B).

Two conda environments share this package. Training, the backbone and the policy server
need transformers >= 5 (env `rolevln-qwen35`, Python 3.12). Habitat 0.2.4 exists only for
Python 3.9 (env `rolevln`), so the simulator tools drive the policy through `client.py`.
`config`, `layout`, `protocol`, `controller` and `client` use only the standard library and
run in both; `qwen35`, `data` and `server` need the policy env; `expert` and `rollout` need Habitat.
"""
