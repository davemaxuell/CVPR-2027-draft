"""Cache native per-frame image preprocessing for batch-one simulator inference.

Earlier camera frames never change. Reusing their pixel tensors avoids repeatedly
resizing/normalizing them, while preserving the full native vision/model forward.
This does not cache or approximate vision features or language-model KV states.
"""
import torch

from .common import past_frame_indices


class OnlineInputs:
    def __init__(self, inputs, device):
        if inputs.cfg.image_splitting:
            raise ValueError("Online pixel caching currently requires image_splitting=False")
        self.inputs, self.device = inputs, device
        self.frames = []

    def append(self, image):
        vision = self.inputs.processor.image_processor(
            [[image]], return_tensors="pt", return_row_col_info=True)
        rows, cols = vision.pop("rows"), vision.pop("cols")
        pixels = vision["pixel_values"].to(self.device, dtype=torch.bfloat16)
        mask = vision["pixel_attention_mask"].to(self.device)
        if self.frames and pixels.shape != self.frames[0]["pixel_values"].shape:
            raise ValueError("Simulator camera shape changed within an episode")
        self.frames.append({"pixel_values": pixels, "pixel_attention_mask": mask,
                            "rows": rows[0][0], "cols": cols[0][0]})

    def __call__(self, instruction, actions):
        t = len(actions)
        assert len(self.frames) == t + 1
        chosen = [self.frames[i] for i in past_frame_indices(t, self.inputs.cfg) + [t]]
        step = {"instruction": instruction, "executed_actions": actions, "t": t,
                "images": [None] * len(chosen)}
        texts = self.inputs.processor.expand_text_with_image_tokens(
            [self.inputs.prompt(step)], [[f["rows"] for f in chosen]], [[f["cols"] for f in chosen]])
        batch = self.inputs.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch["lengths"] = batch["attention_mask"].sum(1)
        if int(batch["lengths"].max()) > self.inputs.cfg.max_length:
            raise ValueError("Online prompt exceeds the configured token limit")
        for key in ("pixel_values", "pixel_attention_mask"):
            batch[key] = torch.cat([f[key] for f in chosen], dim=1)
        return batch
