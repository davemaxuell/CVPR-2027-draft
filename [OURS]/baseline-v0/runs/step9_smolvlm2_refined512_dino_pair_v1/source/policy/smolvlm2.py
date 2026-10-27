"""SmolVLM2 navigation policy: native processor, decoder LoRA, masked action loss, action scoring.

`SmolVLM2Inputs` turns steps into model inputs on the CPU (it runs inside DataLoader workers);
the same builder serves training (with targets) and deployment (without), so prompt, window
and image handling are identical in both.
"""
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from .common import ACTION_WORDS, PolicyConfig, step_inputs

REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
LORA_TARGETS = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
END_OF_TURN = "<end_of_utterance>"


def default_config(**overrides):
    return PolicyConfig(backbone="smolvlm2", repo=REPO, revision=REVISION, **overrides)


class SmolVLM2Inputs:
    """Prompt + image processing. A step is a dict with instruction, executed_actions (before t), t,
    images (PIL list for past_frame_indices(t) followed by frame t) and, for training, target."""

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.repo, revision=cfg.revision)
        self.processor.image_processor.do_image_splitting = cfg.image_splitting
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"
        self.targets = [f" {w}{END_OF_TURN}" for w in ACTION_WORDS]
        self.target_ids = [self.tokenizer(t, add_special_tokens=False)["input_ids"] for t in self.targets]
        self.first_token = [ids[0] for ids in self.target_ids]
        assert len(set(self.first_token)) == len(ACTION_WORDS), "action words must differ in their first token"

    def prompt(self, step):
        past, (before, between, after) = step_inputs(self.cfg, step["instruction"], step["executed_actions"], step["t"])
        assert len(step["images"]) == len(past) + 1, "images must be the past frames plus the current frame"
        content = [{"type": "text", "text": before}]
        content += [{"type": "image"} for _ in past]
        content += [{"type": "text", "text": between}, {"type": "image"}, {"type": "text", "text": after}]
        return self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)

    def __call__(self, steps, with_targets=True):
        texts = [self.prompt(s) + (self.targets[s["target"]] if with_targets else "") for s in steps]
        batch = self.processor(text=texts, images=[s["images"] for s in steps], return_tensors="pt", padding=True)
        # The frozen vision encoder casts to bf16 before using these pixels anyway.
        # Do it in CPU workers to halve shared-memory and host-to-device traffic.
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        lengths = batch["attention_mask"].sum(1)
        if int(lengths.max()) > self.cfg.max_length:
            raise ValueError(f"prompt of {int(lengths.max())} tokens exceeds max_length {self.cfg.max_length}")
        batch["lengths"] = lengths
        if with_targets:
            labels = torch.full_like(batch["input_ids"], -100)
            for i, (s, n) in enumerate(zip(steps, lengths.tolist())):
                ids = self.target_ids[s["target"]]
                labels[i, n - len(ids):n] = batch["input_ids"][i, n - len(ids):n]
                assert labels[i, n - len(ids):n].tolist() == ids, "target tokenization drift"
            batch["labels"] = labels
            batch["targets"] = torch.tensor([s["target"] for s in steps])
        return batch


class SmolVLM2Policy:
    def __init__(self, cfg: PolicyConfig, adapter=None, lora_rank=16, lora_alpha=32, lora_dropout=0.05,
                 device="cuda", dtype=torch.bfloat16):
        self.cfg = cfg
        self.device = device
        base = AutoModelForImageTextToText.from_pretrained(cfg.repo, revision=cfg.revision, dtype=dtype)
        base.requires_grad_(False)
        if adapter is None:
            lora = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=LORA_TARGETS)
            self.model = get_peft_model(base, lora)
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()  # fp32 adapter weights; forward runs under bf16 autocast
        else:
            self.model = PeftModel.from_pretrained(base, adapter)
        self.model.to(device)
        self.first_token = torch.tensor(SmolVLM2Inputs(cfg).first_token, device=device)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    @staticmethod
    def model_inputs(batch):
        return {k: v for k, v in batch.items() if k not in ("labels", "lengths", "targets")}

    def loss_sum(self, batch):
        """Summed token cross-entropy over target tokens, and the number of target tokens."""
        labels = batch["labels"]
        mask = labels[:, 1:] != -100
        rows, positions = mask.nonzero(as_tuple=True)
        keep, inverse = torch.unique(positions, sorted=True, return_inverse=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**self.model_inputs(batch), logits_to_keep=keep, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[rows, inverse].float(), labels[:, 1:][mask], reduction="sum")
        return loss, int(mask.sum())

    @torch.no_grad()
    def evaluate(self, batch):
        """For a batch with targets: (loss sum, target tokens, first-token action scores at the prompt end)."""
        labels = batch["labels"]
        mask = labels[:, 1:] != -100
        target_rows, positions = mask.nonzero(as_tuple=True)
        keep, inverse = torch.unique(positions, sorted=True, return_inverse=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**self.model_inputs(batch), logits_to_keep=keep, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[target_rows, inverse].float(), labels[:, 1:][mask], reduction="sum")
        prompt_end = batch["lengths"] - (labels != -100).sum(1) - 1
        rows = torch.arange(labels.shape[0], device=labels.device)
        return loss, int(mask.sum()), logits[rows, torch.searchsorted(keep, prompt_end)][:, self.first_token].float()

    @torch.no_grad()
    def action_scores(self, batch, return_context=False):
        """Logits of the four actions' first tokens at the last prompt position (the decoding rule).

        `batch` must be built without targets. With return_context, also returns the final-layer
        state at that position (candidate for the scorer's q_t; verify against the final norm first).
        """
        lengths = batch["lengths"]
        keep, inverse = torch.unique(lengths - 1, sorted=True, return_inverse=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(**self.model_inputs(batch), output_hidden_states=return_context,
                             logits_to_keep=keep, use_cache=False)
        rows = torch.arange(lengths.shape[0], device=lengths.device)
        scores = out.logits[rows, inverse][:, self.first_token].float()
        if return_context:
            return scores, out.hidden_states[-1][rows, lengths - 1].float()
        return scores
