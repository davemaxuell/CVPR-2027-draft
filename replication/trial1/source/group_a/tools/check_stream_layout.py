"""CPU checks of the streaming layout and the StreamVLN DAgger controller (any env, stdlib only).

1. Windows reproduce StreamVLN's VLNActionDataset strings: prompt, memory views, one view every
   4 steps, 4 actions per answer, STOP last (reference written from vln_action_dataset.py @ e48f6ff7).
2. The closed-loop context before every action is a prefix of the training window containing it,
   followed by exactly that action's slot.
3. Memory views equal StreamVLN's arange(0, s, s // 8) at every window start.
4. Scripted episodes exercise each controller rule: agreement, accumulated errors, far STOP,
   near STOP veto, beta-driven expert turns, final-waypoint finish and the keep rule.

    python tools/check_stream_layout.py --out logs/check_stream_layout.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stream_policy.config import StreamConfig  # noqa: E402
from stream_policy.controller import STOP, DaggerRule, StreamVLNController  # noqa: E402
from stream_policy.layout import layout, memory_indices, turn_end, windows  # noqa: E402

ARROWS = {0: "STOP", 1: "↑", 2: "←", 3: "→"}


def streamvln_window(instruction, actions, start, conj, num_frames=32, num_history=8, num_future=4):
    """StreamVLN's __getitem__ + prepare_conversation for frames 0..T-1 with actions ending in STOP."""
    prompt = ("You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to "
              "follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE "
              "FORWARD (↑) by 25 centimeters, or STOP.")
    time_ids = list(range(start, min(start + num_frames, len(actions))))
    acts = [actions[t] for t in time_ids]
    views = list(range(time_ids[0], time_ids[-1] + 1, num_future))
    memory = list(range(0, start, max(start // num_history, 1))) if start else []
    first = prompt + (" These are your historical observations: <memory>." if start else "")
    first = first.replace("<instruction>.", instruction)
    turns, i = [], 0
    while i < len(acts):
        human = f"{conj(len(turns))}<image>."
        turns.append(((first + " " + human) if i == 0 else human, "".join(ARROWS[a] for a in acts[i:i + num_future])))
        i += num_future
    return turns, memory, views


def render(pieces, actions):
    """Our pieces in StreamVLN's (human, gpt) string form, plus memory and view frame lists."""
    turns, memory, views, role, text = [], [], [], None, ""
    for piece in pieces:
        if piece[0] == "open":
            role, text = piece[1], ""
        elif piece[0] == "close":
            if role == "user":
                human = text
            elif role == "assistant":
                turns.append((human, text))
        elif piece[0] == "text":
            text += piece[1]
        elif piece[0] == "image":
            (memory if piece[2] == "memory" else views).append(piece[1])
            text += "<memory>" if piece[2] == "memory" and "<memory>" not in text else "" if piece[2] == "memory" else "<image>"
        elif piece[0] == "action":
            text += ARROWS[actions[piece[1]]]
    return turns, memory, views


def check_layout(cfg, trials, rng):
    failures, windows_checked, prefixes_checked = [], 0, 0
    for trial in range(trials):
        n = rng.randint(1, 260)
        actions = [rng.choice([1, 2, 3]) for _ in range(n - 1)] + [STOP]
        instruction = "Walk past the table and stop at the door."
        for start, stop in windows(n, cfg):
            choices = [rng.randrange(len(cfg.conjunctions)) for _ in range(cfg.window_steps)]
            conj = lambda turn: cfg.conjunctions[choices[turn]]  # noqa: E731
            ours = render(layout(cfg, instruction, start, stop, conj), actions)
            theirs = streamvln_window(instruction, actions, start, conj)
            if ours != theirs:
                failures.append({"trial": trial, "steps": n, "window": start, "ours": ours, "streamvln": theirs})
            windows_checked += 1
            full = layout(cfg, instruction, start, stop, conj)
            for t in range(start, stop):
                prefix = layout(cfg, instruction, start, t, conj, open_turn=True)
                if full[:len(prefix)] != prefix or full[len(prefix)] != ("action", t):
                    failures.append({"trial": trial, "prefix_step": t, "window": start})
                assert turn_end(t, cfg) - t >= 1
                prefixes_checked += 1
    return {"windows": windows_checked, "prefixes": prefixes_checked, "failures": failures[:3], "passed": not failures}


class Line:
    """1-D world: FORWARD moves +1; waypoint i sits at position 10*i; the expert is reached within 0."""

    def __init__(self, waypoints):
        self.x, self.points = 0, [10 * i for i in range(waypoints)]

    def label(self, w):
        return STOP if self.x >= self.points[w] else 1

    def step(self, a):
        self.x += a == 1


def run(policy_actions, waypoints=3, reference=20, beta=0.0, goal_distance=lambda x: 5.0, max_steps=200):
    world, plan = Line(waypoints), list(policy_actions)
    c = StreamVLNController(waypoints, reference, actions_per_turn=4, beta=beta, seed=0)
    log = []
    for t in range(max_steps):
        policy = lambda: (plan.pop(0) if plan else 1, 0)  # noqa: E731
        action, source = c.decide(t, lambda: world.label(c.waypoint), lambda: world.label(c.waypoint) == STOP,
                                  policy, goal_distance(world.x))
        log.append((action, source))
        if source == "finish":
            break
        world.step(action)
    return c, log


def check_controller():
    results = {}
    c, log = run([1] * 100)
    results["agreement"] = c.takeovers == [] and log[-1] == (STOP, "finish") and all(s == "policy" for _, s in log[:-1])
    c, log = run([2] * 100, reference=20)            # 10 expert actions per segment: 9 errors > 0.8 * 10
    results["ratio_rule"] = c.takeovers[:1] == [8] and log[8] == (1, "takeover")
    c, log = run([2] * 100, reference=1000)          # per segment 500: the 12-error cap fires first
    results["error_cap"] = c.takeovers[:1] == [12]
    after = [s for _, s in log[13:]]
    results["expert_until_waypoint"] = after[:7] == ["expert"] * 7 and "policy" in after
    c, log = run([STOP] + [1] * 100, goal_distance=lambda x: 5.0)
    results["far_stop_takeover"] = c.takeovers == [0] and log[0] == (1, "takeover")
    c, log = run([STOP] + [1] * 100, goal_distance=lambda x: 1.0)
    results["near_stop_veto"] = c.takeovers == [] and log[0] == (1, "veto")
    c, log = run([2] * 100, beta=1.0)
    results["beta_expert_turns"] = c.takeovers == [] and all(s in ("expert", "finish") for _, s in log)
    rule, k = DaggerRule(), StreamVLNController(2, 10, 4)
    k.needed_help = True
    keep = [k.keep(0.4, 0.9), k.keep(0.6, 0.5), k.keep(0.4, 0.95)]
    k.needed_help = False
    keep += [k.keep(0.4, 0.9), k.keep(0.4, 0.8)]
    results["keep_rule"] = keep == [True, False, False, False, True] and rule.keep_pl_clean == 0.85
    return {"cases": results, "passed": all(results.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = StreamConfig()
    rng = random.Random(args.seed)
    memory = all(memory_indices(s, cfg) == list(range(0, s, s // 8)) for s in range(32, 2048, 32))
    report = {"config": {"window_steps": cfg.window_steps, "actions_per_turn": cfg.actions_per_turn,
                         "memory_frames": cfg.memory_frames},
              "layout": check_layout(cfg, args.trials, rng), "memory_matches_streamvln": memory,
              "controller": check_controller()}
    report["passed"] = report["layout"]["passed"] and memory and report["controller"]["passed"]
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
