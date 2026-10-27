# VLN navigation & error visualizer (DUET / ScaleVLN)

Watch an agent's R2R episode walk through the 3D reconstruction of the building, with the agent's front view,
the full 360° panorama at each viewpoint, instruction progress at the selected frame, and logged scores
and attention at every decision. The **Recorded rollout** badge identifies playback of saved predictions;
open it for the model, dataset, run, prediction file, and available evidence. Missing metadata is labelled.

```bash
# from the ScaleVLN checkout; any python3 >= 3.8, standard library only
python3 analysis/vln_viz/visualizer_app.py            # released B/16 + H/14, every R2R split
ssh -L 8765:127.0.0.1:8765 <server>                   # then open http://127.0.0.1:8765 locally
```

In VS Code (Remote-SSH), run the command in the integrated terminal; VS Code forwards the port, or add 8765 in
the Ports panel. Pick the model, the split (val_unseen, val_seen, val_train_seen, test) and an optional model
to compare against in the top bar; the episode list on the left filters every episode of that split.

Drag the dividers between columns to change their widths, or anywhere along the bottom edge of a panel to
change its height. A size readout follows the pointer, and dragging near the edge of a scrolling pane scrolls
it automatically. The divider between the agent view and panorama adjusts their relative widths; they stack
when the navigation pane becomes narrow. Minimum sizes keep the map, camera images, and playback controls
usable. Resizing also preserves a manually chosen 2D map pan/zoom and the selected automatic fit mode. Sizes are saved in
this browser separately for desktop, tablet, and phone layouts. Double-click a grip to reset that dimension,
or use **Reset layout** to restore all sizes. Focus a grip and use arrow keys to resize with the keyboard;
hold Shift for larger steps, press Enter to reset, or Escape to cancel a drag.

In the Environment header, turn **Pre-show nodes** off to hide navigation nodes, paths, and progress
markers in both maps and the camera views. The maps keep the start and goal markers visible. This setting
is saved in the browser. **Jump to goal**, beside **Center agent** in the camera toolbar, pauses playback,
centers the map on the goal, and shows that viewpoint's front view and **Goal preview · 360°** panorama.
Click the panorama or drag the front view to inspect the goal's surroundings. The agent remains at the
selected frame; decision scores, progress, and evaluation results continue to describe the recorded agent.
The goal images omit agent action overlays and start at a neutral heading, rather than an inferred model pose.

**Return to agent** (in the toolbar and beside the goal panorama) restores the agent's viewpoint and previous
look direction, leaving playback paused. Press **Play** to return and resume; seeking also returns to the
agent's view. Goal preview cancels a pending auto-next transition. It temporarily reveals hidden camera panels
and restores that layout on return. Episodes without a published goal, including the R2R test split, disable
goal jumping. Missing goal imagery is reported inside the camera panels.

The route key below the camera toolbar independently shows or hides the selected model, reference, and
comparison routes. Secondary labels are hidden by default; enable **Display → All map labels** to see
them. Hover or focus an action candidate to highlight its viewpoint; click it to keep the highlight until
the frame changes. **Display → 2D analysis view** selects the graph with reduced labels and hides the
camera panels to give the map and timeline more room. Restore them with **Display → Camera panels**.

The active sub-instruction and **Current decision** lead the details pane. **All steps** expands the list
inside a scrollable area; enlarge the panel with its grip when needed. Completion labels only use events
at or before the selected frame, and missed steps are reported at the end. Final outcomes and comparison
metrics explicitly describe the full recorded episode. **Attention** is a logged proxy, not a causal
explanation; **reference-best** is derived from the ground-truth goal, not another model prediction.

The episode summary and event buttons beside the timeline expose closest approach, reference-route
departure, first revisit, and the final frame when available. Event descriptions distinguish leaving the
annotated route from failure. Seek with the frame slider or event buttons. The comparison table reports
both models' final outcomes, navigation errors, path lengths, decision counts, and hops; Δ is selected minus
compared model. **First route divergence** highlights the first different viewpoints (or where one route
ends), showing all floors for that inspection. It compares routes, not synchronized decision numbers.

Use Tab to reach controls. In the episode list, Up/Down or Home/End moves focus; Enter opens an episode.
Map tabs also support arrow keys. Playback shortcuts apply outside focused interactive controls.

The 3D building is shown at full height by default. Enable **Cutaway** beside **Pre-show nodes** to
slice it above the selected floor, then use **Display → Cut height** to adjust the height (initially
1.3 m). Turn Cutaway off to restore the full height. Both settings are saved in the browser.

Enable **Auto-next episode** below the playback controls to continue the demonstration automatically.
After the final movement, a four-second countdown leaves the result visible before loading and playing
the next episode in the current filtered and sorted list. It loops back to the beginning of that list;
if no other episode is available, playback stops. Pause, seek, change the list, or select an episode to
cancel a pending transition. Turning the toggle off during playback lets the current episode finish.
The preference is saved in the browser; press **Play** to begin after reopening the viewer.

## Split analysis, what-if branches and tags

**Split analysis** (top bar, beside **Episode**) replaces the replay panels with statistics over every episode of
the selected run, or only those matching the list filters (checkbox at the top). Click any bar, cell or count to
put that slice on the episode list (shown as a removable chip above the list); click an episode to replay it.
Cards: success rate by reference path length, hops, instruction length, floors or scan (against the split
average); where failed episodes ended (goal room but > 3 m away, same room type in the wrong room, another room,
another floor; Matterport region labels); failure modes; sub-task completion (share completing steps 1…k in
order); the 2×2 outcome grid against the compared model; model confidence vs. agreement with the
reference-best; the first non-best decision of failed episodes (decision index, what went wrong, how sure the
model was); stop probability inside vs. outside the success area; what-if results; your tags; and a table of all
runs.

**Reference-best** (the ★ in the decision card) is the frontier node closest to the goal, or STOP inside the
success area; a move counts as matching it when its distance is within 0.5 m, and inside the success area any
move towards the goal also matches. RxR paths are deliberately not shortest paths, so for RxR runs it follows the
annotated route instead: the option closest to the next reference viewpoint not yet reached. Stops forced by
the step budget are not judged. Decisions that did not match are outlined in red on the timeline; the list can
filter on the first one (**More filters**) and sort by its decision index or confidence.

**What-if branches** force a different option at one recorded decision (the 8 most likely alternatives plus the
reference-best), let the model continue with argmax decoding, and record the outcome. They run on the GPU, so
they are behind the **What-if · GPU** switch in the top bar:

- The switch is **off whenever the server starts**. While it is off nothing runs on the GPU; stored results stay
  visible.
- Switched on, nothing starts by itself. Run branches from the decision card (**Branch this decision**) or the
  split page (the first non-best decision of each listed failure, or every decision of the listed episodes;
  the number of branches is confirmed first). Already stored branches are reused.
- Switching off (or **Cancel**) stops the running job and the GPU worker at once; finished branches are kept.
  A worker left idle for 5 minutes also exits, freeing GPU memory, while the switch stays on.
- `--whatif-gpu N` picks the GPU (default 0); `--no-whatif` removes the switch entirely.

Each decision card then shows "if taken: ✓ success / ✕ fails · NE" per option, and **route** draws that
branch on the map in place of the comparison route. Each job first replays up to two of its episodes without
forcing and reports whether they reproduce the recording (status on the split page). Results are appended to
`exprs_map/whatif/<model>/<split>.jsonl`; the worker log is `cache/whatif_worker.log`. Speed: loading the model
takes about 20 s, then branches run in batches of 16.

**Tags & notes** (card below the decision): toggle tag chips, add your own tags, and write a note per episode;
they save automatically to `episode_notes.json` next to this README, per run. The list marks tagged episodes
with a dot and filters by tag, the split page counts them, and CSV/JSON exports include `tags` and `note`.

## Continuous environments (VLN-CE): ETPNav on R2R-CE and RxR-CE

Runs of [ETPNav](https://github.com/MarSaKi/ETPNav) (TPAMI 2024; DUET's topological planner carried over to
Habitat's continuous environments) appear as model **ETPNav · VLN-CE**, datasets **R2R-CE** / **RxR-CE**. The code
and released checkpoints are in `[prior works]/ETPNav` (conda env `etpnav`: Python 3.8, habitat-sim/lab 0.1.7,
torch 2.4.1+cu121); `tools/log_etpnav_steps.py` there runs ETPNav's own evaluation rollout (argmax,
`back_algo=control`, the released eval settings) and logs every decision:

```bash
cd "[prior works]/ETPNav" && conda activate etpnav
python tools/log_etpnav_steps.py --task r2r --split val_unseen                 # 1,839 episodes, ~45 min on GPU 0
python tools/log_etpnav_steps.py --task rxr --split val_unseen --episodes 200  # English guide episodes
```

It writes `data/logs/explorer/etpnav_<task>/episodes_<split>.jsonl` (appended as episodes finish; a rerun resumes)
and six habitat cube faces per map node (`views_<split>/`, ~1.7 MB per episode). The viewer picks the runs up
automatically, also while they are still being logged (`--no-ce` skips them). Habitat renders on GPU 0 only.

How a continuous episode is shown (`ce_runs.py`): each episode has its own small graph instead of the Matterport
navigation graph: ETPNav's map nodes `n<k>` (where decision k was taken), its ghost waypoints `g<k>` (dashed),
samples of the agent's track every 0.5 m `p<k>`, and the reference path `r<k>` (the R2R/RxR viewpoints, snapped
to the navmesh). Distances are habitat geodesics logged with the episode, so frames, sub-tasks, the decision
review, the split page, tags and exports work as for DUET; the headline metrics (NE, SR, SPL, nDTW) are ETPNav's
own, with the VLN-CE definitions. The decision card lists ETPNav's global-planner scores over map nodes and ghost
waypoints (there is no local branch); "chosen" is the ghost it picked, and the arrival node is where the
low-level controller actually got to. The panorama and agent view come from the cube faces rendered at the map
node the agent decided at, so they show what the habitat camera saw, at its height (1.25 m R2R-CE, 0.88 m
RxR-CE), turned to the agent's logged heading. Between map nodes there are no images: the view stays on the
node the agent decided at, and its caption says so ("view from n2, 1.0 m away"). Note that R2R-CE start
headings are not R2R's: the VLN-CE episodes have their own start rotation (unrelated to the R2R heading, the
same for the three instructions of a path), so a continuous episode can start facing elsewhere than its
discrete counterpart. Episode ids are the discrete ones (R2R `<path_id>_<j>`, RxR instruction ids), so
FGR2R/RxR sub-tasks apply and **Compare with** offers the same instruction in the other setting: a DUET run on
the discrete graph inside a continuous episode, and vice versa. The decision review uses ghost waypoints as the
options, and inside the success area, any move towards the goal counts as matching. What-if branches are DUET-only.

What is specific to the continuous setting and to ETPNav:

- **ETPNav's map as it stood at each decision**, not the final one: map nodes joined by their edges, ghost
  waypoints (dashed, sized by the planner's score), and at the decision point the waypoint predictor's raw
  output as spokes from the agent (a filled end: merged into a visited node). The same in 3D. Panorama markers
  show only the nodes and ghosts ETPNav had then.
- **Navigable area** (2D, Display ▸ Navigable area): the Habitat navmesh of the shown floor, drawn under the map,
  so you can see where an agent could have stood. Built once by `build_navmaps.py` (all 90 scans, 2.3 MB);
  discrete runs get it too.
- **Decision card**: for a move, how many waypoints were predicted (new ghosts, merged into earlier ghosts,
  on visited nodes) and how the controller carried the move out (distance to the aimed waypoint, metres
  walked, how far from the waypoint it ended; "blocked" when the agent got < 0.25 m away). For STOP, the stop
  score each visited node got when the agent stood there; ETPNav stops at the highest, so this shows when a
  node inside the success area was passed with a low score.
- **Diagnostics**: moves that went nowhere, worst/mean distance from the aimed waypoint, low-level actions
  per decision, collisions, stop score of the stop node vs the best node inside the success area.
- **Episode list**: the last column is moves that went nowhere (⊘), with sorts by it, by the worst waypoint
  miss and by collisions.
- **Split analysis**: a *Low-level control* card (moves that went nowhere, worst waypoint miss, collisions; with
  success rate per bucket) and *Passed the goal, stopped elsewhere* (the best stop score among nodes inside
  the success area). On R2R-CE val_unseen, the 95 episodes with ≥ 5 blocked moves succeed 14% of the time,
  and none of the 126 that passed through the goal area scored ≥ 50% to stop at a node there.
- Frames of a continuous run are track points (every 0.5 m), so events and labels say "frame", not "hop".
- In 3D, showing one floor (Follow agent, or Floor N) also hides the floors above it when the cut-away is off,
  so the chase camera is no longer inside the floor above (all runs).

```bash
/home/bufsgpu/yes/envs/etpnav/bin/python analysis/vln_viz/build_navmaps.py   # needs habitat_sim; CPU, ~3 min
```

Other VLN-CE models to add later (Uni-NaVid, InternVLA-N1, StreamVLN, NaVILA), with their weights, simulator
versions and what the explorer would need: `[prior works]/CANDIDATE_MODELS.md`.

## Running new cases

`bash run.sh demo` runs fresh inference, logs every decision, then opens this viewer on those episodes and
starts playback:

```bash
bash run.sh demo --model h14                                   # 3 evenly spaced val_unseen episodes
bash run.sh demo --model h14 --split val_seen --episodes 50    # 50 evenly spaced val_seen episodes
bash run.sh demo --model b16 --split val_unseen --episodes all # the whole split (~5 min on one GPU)
bash run.sh demo --model h14 --ids 601_2 6620_2                # specific instructions
```

A free port is chosen by default. For SSH use `--no-browser --port 8766` and forward that port; open the full
printed URL (including the fragment). `--export-only` saves just the portable offline demo (N val_unseen
episodes) and exits. To open existing predictions with playback, without rerunning inference:
`python3 analysis/vln_viz/visualizer_app.py --open-browser --autoplay`.

## What the screen shows

- **3D building**: the scan's reconstructed mesh (`house_segmentations/<scan>.ply`, ceilings removed),
  shown at full height, with an optional **Cutaway** to reveal the selected floor's interior. The
  whole building loads first at low detail (~0.4M triangles); the floor being shown then switches to its
  high-detail mesh (up to 2M triangles, 10-25 MB, near the scan's own resolution). Turn **High-detail floor**
  off under Display for slow connections. **Photo at agent** (on by default) paints the agent's current 360°
  panorama onto every surface its viewpoint can see. Occlusion comes from a distance cube rendered from the
  viewpoint, so walls block the photo correctly. Floors, rugs and furniture around the agent then show at
  photo resolution (~5 mm per pixel at 3 m); surfaces out of its sight keep the mesh colours. Navigation graph on the floor, ground truth in green (the current sub-task's stretch thicker),
  agent path in blue, remote-frontier jumps and backtracking in orange, the compared model in purple. Frontier
  nodes are wire boxes with DUET's action probabilities; revisited nodes turn amber/red with ×n labels. The
  agent walks each hop smoothly. Cameras: Chase (behind the agent), Orbit (drag; tracks the agent), Overview
  (double-click). The 2D graph tab keeps the earlier top-down view.
- **Views**: the front view at the agent's heading (drag to look around) and the full 360° panorama centred on
  the heading (click to look there). Markers: navigable neighbours with their action probabilities, the next
  hop, the best next hop towards the goal, the remote-jump target and the goal direction.
- **Instruction**: sub-tasks from Fine-Grained R2R (FGR2R), each aligned to a stretch of the ground-truth path:
  ✓ completed at this frame (with the hop), ▶ active, pending steps, and ✗ not completed at episode end.
  DUET does not decompose instructions itself; attention percentages describe the share of logged
  cross-attention at the current decision, and words are shaded by attention. An intermediate sub-task counts as completed when the agent comes within
  1.5 m of its sub-goal; the last one is the episode's success (stopping within 3 m). The timeline shows both
  rows (reference sub-task progress vs. highest attention) and highlights where they disagree.
- **Current decision**: action (local move, remote jump, stop), global-vs-local fusion weight, top candidates
  with probabilities, the chosen node and the reference-best option.

## Files

| File | What it does |
|---|---|
| `visualizer_app.py` | Web app (stdlib HTTP server + `static/`, three.js vendored in `static/vendor/`). Auto-discovers `exprs_map/eval/*/preds/detail_<split>.json` and matching step logs; `--preds name=file --steplog name=file` adds any other run (e.g. original DUET). |
| `parse_trajectories.py` | Aligns predictions with annotations, connectivity and FGR2R sub-tasks; metrics (NE/SR/OSR/SPL/nDTW/SDTW/CLS match the DUET eval log exactly), failure taxonomy, sub-task completion, revisit/backtrack/lost diagnostics, playback frames. CLI: `summary`, `export`. |
| `log_duet_steps.py` | Re-runs a checkpoint and logs each decision: DUET's global map (visited vs frontier), fused/global/local action probabilities, fusion weight, and text cross-attention. Navigation is unchanged: re-runs reproduce the released predictions on every split (e.g. val_unseen 2349/2349). `--episodes N\|all`, `--ids`. |
| `whatif.py` | What-if job queue and the GPU switch (standard library); starts and stops `whatif_worker.py`, scores branches, stores results. |
| `whatif_worker.py` | GPU worker (scalevln env): re-runs DUET with the first decisions forced and returns the trajectories. Started only by the viewer. |
| `static/split.js` | Split analysis page. |
| `ce_runs.py` | Continuous (VLN-CE) runs: reads ETPNav's episode logs and gives each episode a graph with the explorer's graph interface, plus the waypoint predictions and controller execution of each decision (see above). Rows are cached in `cache/ce_rows/`. |
| `build_navmaps.py` | Top-down maps of the navigable area (Habitat navmesh) per scan floor for the 2D view, in `cache/navmaps/` (needs habitat_sim: run with the etpnav env). |
| `rxr_subtasks.py` | RxR sub-instructions aligned to the path by the guide's spoken-word timing and pose trace (needs numpy; writes `cache/rxr_subtasks/rxr_<split>.json`). |
| `build_meshes.py` | Decimates the house meshes for the 3D view: whole buildings into `cache/meshes/<scan>.bin.gz` (320 MB) and, with `--detail`, each floor at up to `--detail-faces` 2M triangles into `cache/meshes_hd/<scan>_f<floor>.bin.gz`. All 90 scans are built; the server builds a missing one on first request (needs numpy; ~15 s per floor). |

Step logs for the released checkpoints (every R2R split: val_unseen, val_seen, val_train_seen, the full
train split with 14,039 instructions, and test) are in `VLN-DUET/datasets/R2R/exprs_map/steplogs/released_{b16,h14}/`; FGR2R annotations are in
`VLN-DUET/datasets/R2R/annotations/FGR2R/`. To log another checkpoint:

```bash
source env.sh
python analysis/vln_viz/log_duet_steps.py --model h14 --checkpoint <ckpt> --split val_unseen --out <dir>
```

**REVERIE and RxR (zero-shot).** The released R2R checkpoints were also run, unchanged, on REVERIE val_seen and
val_unseen, and on the English (en-US + en-IN) val_seen and val_unseen instructions of RxR. They appear as
`REVERIE · val_unseen`, `RxR · val_unseen` and so on. No REVERIE- or RxR-trained ScaleVLN model is
released. REVERIE success means stopping at a viewpoint where the target object is visible (`BBoxes.json`),
and the map's goal zone shows those viewpoints; there is no object grounding. RxR runs use a 25-decision
budget (7% of its English paths have more than 14 hops) and keep up to 512 instruction tokens. RxR
sub-instructions come from `rxr_subtasks.py`. It splits each instruction into steps (sentences, and clauses
starting a new action), then places each step's sub-goal where the guide stood when they began speaking
the next step, using RxR's word timestamps and the guide's pose trace. This covers 98.7% of English val
instructions (cache/rxr_subtasks/); the rest, and REVERIE, fall back to a sentence split with no path
alignment. To repeat:

```bash
python analysis/vln_viz/log_duet_steps.py --model h14 --split reverie_val_unseen --note "zero-shot" \
    --anno VLN-DUET-RVR/datasets/REVERIE/annotations/REVERIE_val_unseen_enc.json --out <steplogs>/released_h14
python analysis/vln_viz/log_duet_steps.py --model h14 --split rxr_val_unseen --max-action-len 25 --max-instr-len 512 \
    --anno ../../data/datasets/RxR/rxr_val_unseen_guide.jsonl.gz --note "zero-shot, English" --out <steplogs>/released_h14
```

Each run's episode rows are cached in `cache/rows/<run>.json` (continuous runs: `cache/ce_rows/`), keyed on the
size and modification time of its predictions, step log, annotations and sub-tasks and of the analysis code,
so the server starts in ~9 s instead of ~45 s; anything that changes rebuilds the cache on the next start.

A run whose `detail_<split>.json` is copied to `exprs_map/eval/<name>/preds/` (with `steps_<split>.jsonl` in
`exprs_map/steplogs/<name>/`) appears in the running viewer on the next page reload; no restart needed.

Failed-episode export (the sidebar exports exactly the filtered list as CSV, JSON or JSONL + steps):

```bash
python3 analysis/vln_viz/parse_trajectories.py export \
    --preds VLN-DUET/datasets/R2R/exprs_map/eval/released_h14/preds/detail_val_unseen.json \
    --failed --min-length 20 --out h14_failed_long.csv
```

## Notes

- Trajectory formats: DUET's per-decision segments (`[[start], [vp], [vp, vp, vp], ...]`) and the R2R standard
  `[[vp, heading, elevation], ...]`. A DUET segment longer than one viewpoint is a remote-frontier jump; a final
  segment ending on an already-visited node is the return to the best stop-score node.
- "Step limit" means the episode used the whole decision budget, `--max-action-len` (DUET evaluates with 15).
  For agents with a different budget, pass that value, e.g. 30 for one-hop-per-step agents.
- Distances are geodesic over the navigation graph, as in R2R evaluation; the live panel also shows the
  straight-line distance. "Lost" = first hop more than 3 m (geodesic) from every ground-truth viewpoint.
- The test split publishes no ground-truth path: episodes show sub-task splits and model attention, but no
  success, distances or sub-task completion.
- Floors and room names come from each scan's `.house` file (panorama → region → level). Without it, floors
  fall back to clustering by floor height (z minus camera height).
- The front view and panorama use MatterSim's projection (same cube-face order, pose rotation and heading
  convention), so they match what the agent's simulator sees. The camera pose is the simulator's: DUET's
  MatterSim uses discretised viewing angles, so an episode starts at the annotated heading rounded to the
  nearest 30° (the caption notes the annotated one), and after each move DUET turns the camera to the centre of
  the 30° view in which the previous node saw the new one, sometimes 30° up or down (stairs).
  `parse_trajectories.view_index` reads that view per viewpoint and neighbour from ScaleVLN's
  `scanvp_candview_relangles_with_hm3d_gibson.json` (identical to MatterSim's, checked on 300 viewpoints) into
  `cache/view_index.json` on first use. Between decisions (remote jumps), the view faces the direction of travel.
  Checked end to end against Habitat renders of the true start pose (R2R, RxR, REVERIE, R2R-CE, RxR-CE).
- REVERIE annotations (object goals: success = stopping where the target object is visible, from
  `BBoxes.json`) and RxR-style files are parsed. Pass `--anno` and `--conn-dir` explicitly for those.
