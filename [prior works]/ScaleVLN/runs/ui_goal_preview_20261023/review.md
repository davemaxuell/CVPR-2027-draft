# Goal panorama preview

Implemented in the interface only. Refresh http://127.0.0.1:8765/.

- Jump to goal pauses playback and opens the reference goal’s front and 360° camera views.
- Return to agent restores the paused frame, previous look direction, and camera-panel preference.
- Goal imagery has no agent action overlays. Loading and missing images are explicitly labelled.
- Playback/seek returns to the agent. A pending auto-next transition is cancelled.

11 checks passed; no browser JavaScript errors. The 98 protected Python files, both sampled raw episode responses, 3D renderer, and panel layout are unchanged.

- Jump loads the six real goal skybox faces, enables goal panorama/free look, preserves frame/decisions/data, and Return restores the exact agent view and look direction.
- Jump pauses active playback without advancing; Play returns and resumes, and seeking returns to the recorded agent view.
- Goal preview temporarily reveals hidden camera panels, preserves display preferences, and never adds the agent’s action overlays to the goal panorama.
- Goal inspection cancels the four-second auto-next countdown and keeps it paused even when the auto-next preference is changed.
- Delayed goal images show a loading state and cannot overwrite the restored agent panorama or its WebGL texture.
- Missing goal images produce a clear message in both camera panels and leave Return to agent working.
- Goal preview, Return, keyboard activation, and resizing remain usable at desktop/tablet/phone widths down to 320 pixels.
- Episode changes exit goal preview; real test episodes without a published goal keep Jump to goal disabled.
- 3D camera targets the actual goal while the recorded agent, frame, URL, and photo projection stay unchanged; Return restores the agent panorama and preserves display settings.
- An auto-next request already in flight cannot replace the episode or goal preview after Jump to goal.
- Play after returning still advances to the next filtered episode after the original four-second pause and renders that agent view.

![Goal surroundings](goal-panorama.png)

![3D goal inspection](3d-goal-preview.png)

![Phone layout](phone-goal-panorama.png)
