# ULN implementation audit shared with assignment 04

Official repository https://github.com/weixi-feng/ULN at `0f251df3ffc6dd9c538d075db8674e6ad54a596b`, inspected from `repositories/04/uln`. Source only; no fetched module imported, training/evaluation executed, or checkpoints loaded. Paths below are relative to `ULN-Rec-VLN-BERT`. Assignment 05 independently read PDF pages 1–9; 04 owns the appendix review.

## Data, objectives and routing

`r2r_src/agent.py:438–511` trains the instruction classifier using full instruction versus goal encoding, with cross entropy. `model.py:796–805` averages word embeddings over sequence positions and applies a two-class linear layer; its pooling has no padding mask. `train.py:190–244` partitions evaluation examples by predicted class; `agent.py:986–996` passes that class to language encoding. `vlnbert/vlnbert_PREVALENT.py:369–445` defines two text encoders and chooses `lalayer_short` for label one.

Uncertainty training follows `train.py:394–429` → `agent.py:1373–1410` → `rollout_exploration:516–657`. It samples full/goal encodings with probabilities 0.3/0.7 (`191–228`) from a preprocessed `train_fg` file. The inspected preprocessing consumes stored goals; it does not establish how every randomly dropped chunk was created.

At `agent.py:620–627`, the target action comes from the stored `result` trajectory, through `_teacher_action(from_env=False)` (`270–294`). Agreement gets class one; class zero triggers exploration. `train.py:267–287` constructs `result` fields from agent output trajectories. The actual released training file was not inspected, so describe these as stored-reference agreement labels, not independently verified semantic or oracle labels. The base navigator is in evaluation mode and only explorer optimizers are stepped in this phase; evaluation mode itself is not gradient freezing.

**Paper/code difference.** Paper Eq. 6 describes concatenated attention/action scores. Active `E2E.forward` (`model.py:865–882`) ignores the two score tensors and feeds concatenated attended visual/language feature vectors to its MLP. `model_PREVALENT.py:38–73` establishes the feature meanings. Commented score-normalization code does not change this path. This discrepancy does not establish which private implementation produced reported tables.

The main navigator retains imitation CE (`agent.py:1099–1101`) and A2C (`1217–1281`); its optimizer receives all navigator parameters (`128`). Dual-encoder selection exists, but a repository-wide source search did not locate a complete explicit routine that resets/trains only the high-level encoder while freezing the rest as described in the paper. Exact reconstruction of that stage remains incomplete.

## Inference and cost

`agent.py:1065–1076` takes the two-class argmax, disables exploration initially, caps it at three events, and optionally retains the previous history state for lookahead. Defaults are three candidates and one future step (`param.py:90–95`).

`make_exploration_batch` uses unnormalized current logits (`682–685`), maximum future logits (`754–755`) and a weighted sum with factor 1.2 (`857–892`). The separate finite additive-offset check in `finite_check_results.json` therefore applies to this scoring formula; it establishes representation dependence, not an observed failure rate.

The implementation synchronizes parallel MatterSim instances (`env.py`, `EnvBatch.setExpSims`/`syncEpisodes`) and records branch views (`agent.py:316–352`). It removes exploratory viewpoints matching the chosen next point, current point or visited history before splicing them into the reported trajectory (`1119–1134`). `eval.py:58–88,119–165` sums shortest-path distances between successive reported viewpoints and computes SR/SPL; it also requires complete expected instruction coverage. The reported trace is thus not simply chronological motion of one physical robot. A matched-cost study should replay and charge complete branch acquisition/return paths. No magnitude or direction of cost bias was measured.

## Release and inspected scope

The root README leaves HAMT implementation/checkpoints and cleanup unchecked. `run/test_agent_final.bash` leaves explorer, state freeze and classifier switches commented out, so it is not a turnkey full-method configuration. MatterSim, features, checkpoints and some paths are external or machine-specific. The source explains recurrent-model components but does not reproduce both paper backbones or all training stages.

Additional paths read: `run/train_explorer.bash`, `run/train_agent_mix.bash`, `run/test_agent.bash`, `run/test_agent_final.bash`, `vlnbert/vlnbert_init.py`. The snapshot `env.py` contains doubled carriage returns; function names and normalized logical lines avoid ambiguous raw line numbering. Exact inspected ranges are recorded in the ULN entry of `sources.json`.
