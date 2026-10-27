"""Episode-level reports and paired scene uncertainty for the STOP study."""
import numpy as np

METRICS = ("success", "spl", "oracle_success", "ndtw", "sdtw", "premature_stop", "reached_but_failed",
           "never_reached_timeout", "distance_to_goal", "steps", "collisions", "stopped")


def summarize(rows):
    def group(rr):
        forward = sum(r["forward_attempts"] for r in rr)
        collided = sum(r["forward_collisions"] for r in rr)
        return {"episodes": len(rr), "successes": int(sum(r["success"] for r in rr)),
                "scenes": len({r["scene"] for r in rr}),
                **{key: float(np.mean([r[key] for r in rr])) for key in METRICS},
                "forward_attempts": forward, "forward_collisions": collided,
                "forward_collision_rate": collided / max(1, forward)}
    by = {name: group([r for r in rows if r["dataset"] == name]) for name in sorted({r["dataset"] for r in rows})}
    return {"overall": group(rows), "by_dataset": by,
            **{"macro_" + key: float(np.mean([v[key] for v in by.values()])) for key in METRICS}}


def paired_comparison(candidate, control, seed=20261025, replicates=50000):
    lookup = {(r["dataset"], r["episode_id"]): r for r in control}
    assert len(candidate) == len(control) == len(lookup)
    assert {(r["dataset"], r["episode_id"]) for r in candidate} == set(lookup)
    scenes = sorted({r["scene"] for r in candidate})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), size=(replicates, len(scenes)))
    report = {}
    for key in ("success", "spl", "oracle_success", "premature_stop", "ndtw", "sdtw"):
        # Confirmation manifests have equal counts in every scene/dataset cell.
        scene_differences = np.array([np.mean([r[key] - lookup[(r["dataset"], r["episode_id"])][key]
                                              for r in candidate if r["scene"] == scene]) for scene in scenes])
        intervals = np.quantile(scene_differences[draws].mean(1), [.025, .975])
        report[key] = {"difference": float(scene_differences.mean()), "scene_bootstrap_95ci": intervals.tolist(),
                       "scene_differences": dict(zip(scenes, scene_differences.tolist()))}
    report["episodes"] = len(candidate)
    report["scenes"] = len(scenes)
    report["replicates"] = replicates
    return report
