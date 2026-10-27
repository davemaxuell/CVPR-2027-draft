"""Backbone-independent policy conventions, shared by SFT training and deployment.

Actions are indexed by live habitat task ID (stop, move_forward, turn_left, turn_right).
At step t the prompt holds the full instruction, up to `past_frames` earlier views, the
last `recent_actions` executed actions, and the current view. The target is the expert's
next action word followed by the backbone's end-of-turn token.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np

ACTION_WORDS = ["STOP", "FORWARD", "LEFT", "RIGHT"]  # index = live habitat action ID


@dataclass
class PolicyConfig:
    backbone: str
    repo: str
    revision: str
    past_frames: int = 7
    frame_sampling: str = "uniform"  # "uniform" over the whole past (always includes t-1) or "recent"
    recent_actions: int = 16
    forward_step_m: float = 0.25
    turn_angle_deg: int = 15
    action_words: List[str] = field(default_factory=lambda: list(ACTION_WORDS))
    decoding: str = "argmax over the first token of each action word"
    image_splitting: bool = False
    max_length: int = 4096

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def past_frame_indices(t, cfg: PolicyConfig):
    """Indices (< t) of earlier frames shown at step t, oldest first."""
    if t <= cfg.past_frames:
        return list(range(t))
    if cfg.frame_sampling == "recent":
        return list(range(t - cfg.past_frames, t))
    if cfg.frame_sampling == "uniform":
        return sorted({int(round(i)) for i in np.linspace(0, t - 1, cfg.past_frames)})
    raise ValueError(f"unknown frame_sampling {cfg.frame_sampling!r}")


def prompt_text(cfg: PolicyConfig, instruction, num_past, recent):
    """Text segments around image slots: returns (before_past, between, after_current) parts."""
    header = (f"You are a navigation agent indoors. Follow the instruction one action at a time: "
              f"FORWARD moves {cfg.forward_step_m:g} m, LEFT and RIGHT turn {cfg.turn_angle_deg} degrees, "
              f"STOP ends the episode.\nInstruction: {instruction.strip()}\n")
    past = "Earlier views, oldest first:" if num_past else "Earlier views: none"
    actions = " ".join(ACTION_WORDS[a] for a in recent) if recent else "none"
    return header + past, f"\nRecent actions, oldest first: {actions}\nCurrent view:", "\nWhat is the next action?"


def step_inputs(cfg: PolicyConfig, instruction, executed_actions, t):
    """Earlier-frame indices and text parts for step t, from history available before action t."""
    past = past_frame_indices(t, cfg)
    recent = list(executed_actions[max(0, t - cfg.recent_actions):t])
    return past, prompt_text(cfg, instruction, len(past), recent)
