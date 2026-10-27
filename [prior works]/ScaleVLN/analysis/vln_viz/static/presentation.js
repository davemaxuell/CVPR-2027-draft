// Read-only summaries for the viewer. These helpers never modify episodes, scores, or evaluation data.
const number = value => value != null && Number.isFinite(value);
const metres = value => number(value) ? `${value.toFixed(2)} m` : 'unavailable';

// `unit` names a frame: a graph hop for discrete runs, a track point for continuous ones.
export function subtaskProgress(reached, index, frame, lastFrame, current, unit = 'hop') {
  if (!reached) return { state: index === current ? 'current' : 'pending', label: 'Not evaluated · no reference' };
  const hop = reached[index];
  if (hop != null && hop <= frame) return { state: 'done', label: `Completed · ${unit} ${hop}` };
  if (frame === lastFrame) return { state: 'missed', label: 'Not completed in this episode' };
  return index === current ? { state: 'current', label: 'Active at this frame' } : { state: 'pending', label: 'Pending at this frame' };
}

export function firstDivergence(path, otherPath) {
  if (!path?.length || !otherPath?.length) return null;
  let index = 0;
  while (index < Math.min(path.length, otherPath.length) && path[index] === otherPath[index]) index++;
  if (index === path.length && index === otherPath.length) return null;
  return { index, frame: Math.min(index, path.length - 1), shared: index ? path[index - 1] : null,
    model: path[index] || null, comparison: otherPath[index] || null };
}

export function episodeEvents(ep, errorMargin) {
  const events = [], d = ep.diag || {}, frames = ep.frames || [];
  const add = (id, label, frame, detail) => {
    if (Number.isInteger(frame) && frame >= 0 && frame < frames.length) events.push({ id, label, frame, detail });
  };
  if (ep.goal) {
    add('closest', 'Closest approach', d.closest_hop, `Route distance to goal: ${metres(d.closest_dist)}. Success requires stopping within ${errorMargin} m.`);
    add('departure', 'Left reference route', d.first_off_gt_hop, 'First recorded viewpoint outside the annotated reference path. An alternate route can still succeed.');
    add('lost', 'Left reference area', d.lost_hop, `First viewpoint more than ${errorMargin} m from every reference-path node.`);
  }
  const revisit = frames.findIndex(f => f.visits >= 2);
  add('revisit', 'First revisit', revisit, `First return to an already visited viewpoint (${frames[revisit]?.visits || 2} visits).`);
  const final = frames.length - 1, stopped = ep.decisions?.[frames[final]?.decision]?.action === 'stop';
  add('final', stopped ? 'Final stop' : 'Episode end', final, ep.goal ? `Recorded final route distance: ${metres(ep.metrics?.ne)}.` : 'The dataset does not publish a reference goal for this episode.');
  return events;
}

export function episodeSummary(ep, errorMargin, unit = 'hop') {
  if (!ep.goal) return 'No reference goal is published for this episode. The recorded route and decisions are available for inspection.';
  const d = ep.diag || {}, end = ep.metrics?.ne;
  if (ep.outcome === 'success') return `Successful final stop: ${metres(end)} from the goal (within ${errorMargin} m).`;
  const ending = ep.outcome === 'step_limit' ? 'The decision limit was reached' : 'The agent stopped';
  if (number(d.closest_dist) && d.closest_dist < errorMargin) {
    return `Within ${errorMargin} m at ${unit} ${d.closest_hop} (${metres(d.closest_dist)}). ${ending} at ${metres(end)} from the goal.`;
  }
  return `Closest approach: ${metres(d.closest_dist)}${d.closest_hop != null ? ` at ${unit} ${d.closest_hop}` : ''}. ${ending} at ${metres(end)}; the success radius is ${errorMargin} m.`;
}

export function metricDifference(value, other, digits = 1, unit = '') {
  if (!number(value) || !number(other)) return '—';
  const delta = value - other;
  return `${delta > 0 ? '+' : ''}${delta.toFixed(digits)}${unit}`;
}
