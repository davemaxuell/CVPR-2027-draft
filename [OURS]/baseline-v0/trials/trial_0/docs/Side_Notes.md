# Side notes

Experiments raised during the runs that are worth testing but are not part of the main pipeline.
Each note records its status, what would decide it, and what it could disturb.

## 1. DINO world model: loss on all history positions

Raised 26 October 2026, after Step 5A. Status: **not started**; queue after Step 6 collection.

Step 5A scores only the final prediction: the predictor receives three real frames and its loss
compares the predicted next frame with the actual one (`wm/training.py:34-36`; objective in
[Step 5](Step5_Run.md)). DINO-WM instead scores every position, so frames 1, 2 and 3 are
predicted from one, two and three frames of history
(`third_party/dino_wm/models/visual_world_model.py:200-238`).

The final-frame loss is not a defect. Deployment reads only the final position, and every step at
t≥3 is already a final target in its own window. The question is whether the extra supervision
lowers final-position error.

| Item | Setting |
| --- | --- |
| Trial | Third matched DINO trial with fresh initialization; keep both Step 5A bundles |
| Held fixed | Data order, seed 5, 37,852 updates, optimizer, schedule, selection rule |
| Change | Mean visual-patch MSE over positions 0–2 against `x[:, 1:4]` |
| Code | `DinoCommandDynamics.forward` (`Scorer/world_models.py:113`) returns only `prediction[:, -1]`; add a training-only path returning all positions. Per-position commands already align (two executed, then the candidate). Frozen inference is unchanged. |
| Decision | Adopt only if final-position group-C macro MSE beats fresh's 1.320767 (transfer 1.347414, copy-current 4.611210) |

Risks and constraints:

- Shorter-history positions have higher error and take part of the gradient, so final-position
  error may get slightly worse at the same budget.
- One pass took about 58 minutes with about 6.5 GB peak at batch 64. Features are cached, so
  Habitat is not needed, but do not share GPU 0 with Step 6 collection. GPU 1 hosts another team's
  services; ask before using it.
- Step 6 labels do not use a world model (`world_models_used_for_collection: false`), but the
  scorer trains on frozen world-model predictions. Decide before scorer training.

Expected payoff is small. Earlier-position targets are already final targets of neighbouring
windows, and validation was still improving slowly at the end of the single pass (fresh 1.3351 to
1.3208 over the last ~8k updates), so a second pass may help more.

Variants:

- Add the 65,913 early DINO-only samples (`dino_early_extra` in `wm/data.py`) as real
  short-history examples.
- A two-frame history is a different experiment. The PointMaze predictor fits only three frames,
  so it can be tested only with fresh initialization.

Related gap: Step 5 validation contains no blocked or collided forward samples, so prediction of
walking into walls is unmeasured. A group-C blocked-forward validation slice would be more
informative than this loss change.
