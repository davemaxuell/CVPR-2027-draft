"""Settings that define a streaming policy: backbone, dialogue window, frame sizes and prompts.

Defaults follow StreamVLN (third_party/streamvln @ e48f6ff7: scripts/streamvln_train_slurm.sh,
streamvln/dataset/vln_action_dataset.py, streamvln/streamvln_eval.py): 32-action dialogue windows,
a user turn with the current view every 4 actions, up to 4 actions per assistant turn, 8 evenly
spaced memory views from before the window, StreamVLN's prompt, conjunctions and arrow actions.
The saved config is part of a checkpoint's identity: training and the policy server both build
prompts from it, so they cannot drift apart.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

MANIFEST = Path(__file__).resolve().parents[1] / "manifest" / "experiment_manifest.yaml"
ACTION_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT"]  # index = live habitat action ID
STREAMVLN_ACTIONS = ["STOP", "↑", "←", "→"]           # StreamVLN idx2actions, same order
CONJUNCTIONS = ["you can see ", "in front of you is ", "there is ", "you can spot ",
                "you are toward the ", "ahead of you is ", "in your sight is "]
TASK_PROMPT = ("You are an autonomous navigation assistant. Your task is to {instruction} Devise an action "
               "sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) "
               "by {turn:g} degrees, MOVE FORWARD (↑) by {forward_cm:g} centimeters, or STOP.")
MEMORY_PROMPT = " These are your historical observations: {memory}."


def simulator_settings():
    """The shared simulator block of the experiment manifest, without importing Habitat (sim.config does)."""
    import yaml
    with open(MANIFEST) as f:
        return yaml.safe_load(f)["simulator"]


@dataclass
class StreamConfig:
    backbone: str = "qwen35_4b"
    repo: str = "Qwen/Qwen3.5-4B"
    revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    window_steps: int = 32         # StreamVLN num_frames: the dialogue restarts every 32 actions
    actions_per_turn: int = 4      # StreamVLN num_future_steps: one view, then up to 4 actions
    memory_frames: int = 8         # StreamVLN num_history: views from before the window, shown at its start
    frame_width: int = 512         # 16 x 12 merged 32-px patches = 192 tokens per view (StreamVLN: 196)
    frame_height: int = 384
    memory_width: int = 512        # StreamVLN encodes memory views like current views
    memory_height: int = 384
    forward_step_m: float = 0.25
    turn_angle_deg: int = 15
    action_tokens: List[str] = field(default_factory=lambda: list(STREAMVLN_ACTIONS))
    system_prompt: str = "You are a helpful assistant."
    task_prompt: str = TASK_PROMPT
    memory_prompt: str = MEMORY_PROMPT
    conjunctions: List[str] = field(default_factory=lambda: list(CONJUNCTIONS))
    supervise_turn_end: bool = True  # also train <|im_end|> after each assistant turn, as StreamVLN does
    max_length: int = 16384

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))
