"""Reproducible development episode selection, with coverage across scenes."""
import random


def select_episodes(episodes, count, seed=0):
    """Seeded round-robin by scene; selection never uses expert success or actions.

    Prefixes are nested, so a smaller simulator check uses the same population as
    the larger development set. Every scene is represented before taking seconds.
    """
    rng = random.Random(seed)
    by_scene = {}
    for ep in sorted(episodes, key=lambda ep: (ep.scene_id, str(ep.episode_id))):
        by_scene.setdefault(ep.scene_id, []).append(ep)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    for pool in by_scene.values():
        rng.shuffle(pool)
    chosen = []
    limit = min(count, len(episodes)) if count else len(episodes)
    while len(chosen) < limit:
        for scene in scenes:
            if by_scene[scene]:
                chosen.append(by_scene[scene].pop())
                if len(chosen) == limit:
                    break
    return chosen
