"""Dialogue layout shared by training, offline scoring and the policy server (StreamVLN windows).

A trajectory has frames 0..T-1 and actions 0..T-1: frame t is the view before action t, and a
completed episode ends with STOP. Steps form windows of `window_steps` actions. A window that
starts at s > 0 first shows `memory_frames` earlier views, frames i*s//m (StreamVLN's
arange(0, s, s // m) whenever m divides s). Inside a window, a turn shows the view at its first
step, then the actions of up to `actions_per_turn` steps; the next view arrives with the next turn.

A training sample is one whole window with every action supervised, as in StreamVLN's
VLNActionDataset. The policy's distribution over action t is the next-token distribution after
`layout(..., stop=t, open_turn=True)`, which is a prefix of the training window containing step t,
so training, offline scoring and closed-loop decoding share one context by construction.

Pieces: ("open", role) and ("close", role) chat markup, ("text", str), ("image", frame, "memory"
or "view"), ("action", step). Backbones render them; executed actions fill the action slots.
"""
import zlib


def window_start(t, cfg):
    return t - t % cfg.window_steps


def turn_end(t, cfg):
    """First step after the turn that contains step t; turns never cross windows."""
    s, k = window_start(t, cfg), cfg.actions_per_turn
    return min(s + (t - s) // k * k + k, s + cfg.window_steps)


def windows(num_steps, cfg):
    """(start, stop) of every window of a trajectory with `num_steps` actions."""
    return [(s, min(s + cfg.window_steps, num_steps)) for s in range(0, num_steps, cfg.window_steps)]


def memory_indices(start, cfg):
    m = cfg.memory_frames
    return [i * start // m for i in range(m)] if start and m else []


def task_text(cfg, instruction):
    return cfg.task_prompt.format(instruction=instruction.strip(), turn=cfg.turn_angle_deg,
                                  forward_cm=cfg.forward_step_m * 100)


def fixed_conjunction(cfg, start):
    """Closed-loop conjunction for each turn of window `start`: uniform over the list, stable over steps."""
    return lambda turn: cfg.conjunctions[zlib.crc32(f"{start}:{turn}".encode()) % len(cfg.conjunctions)]


def layout(cfg, instruction, start, stop, conjunction, open_turn=False):
    """Pieces of the window that starts at `start`, through the action slot of step stop - 1.

    Every turn is closed. With open_turn the layout instead ends right before action `stop`: the
    turn containing step `stop` is shown (with its view, if it starts there) and left open.
    """
    last = start + cfg.window_steps - (1 if open_turn else 0)
    if start != window_start(start, cfg) or not start <= stop <= last:
        raise ValueError(f"steps {start}..{stop} do not lie in one window")
    k = cfg.actions_per_turn
    pieces = [("open", "system"), ("text", cfg.system_prompt), ("close", "system"),
              ("open", "user"), ("text", task_text(cfg, instruction))]
    memory = memory_indices(start, cfg)
    if memory:
        before, after = cfg.memory_prompt.split("{memory}")
        pieces += [("text", before)] + [("image", i, "memory") for i in memory] + [("text", after)]
    for turn, first in enumerate(range(start, stop + 1 if open_turn else stop, k)):
        if turn:
            pieces.append(("open", "user"))
        pieces += [("text", ("" if turn else " ") + conjunction(turn)), ("image", first, "view"), ("text", "."),
                   ("close", "user"), ("open", "assistant")]
        pieces += [("action", step) for step in range(first, min(first + k, stop))]
        if not (open_turn and first <= stop < first + k):
            pieces.append(("close", "assistant"))
    return pieces
