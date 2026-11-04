"""Qwen3.5 streaming policy: token layout, multi-turn action loss, constrained turn decoding.

Prompts are tokenized piece by piece from `layout`, identically for training and deployment, in
Qwen's chat markup without a thinking block (StreamVLN likewise replaces the chat template). A view
is <|vision_start|>, one <|image_pad|> per merged 32-px patch and <|vision_end|>; frames are resized
to the configured size first, so the processor never resizes. Each action is a single token
("STOP", "↑", "←", "→"), and decoding takes the argmax over those four tokens only.

Train modes: "lora" (language-model LoRA, base frozen in bf16), "llm" (language model + merger as
fp32 master weights, vision frozen in bf16), "full" (every weight, StreamVLN's mm_tunable_parts,
with a separate vision learning rate). Compute runs under bf16 autocast. The Gated DeltaNet layers
use flash-linear-attention kernels when `fla` imports (Triton >= 3.7.1 on Hopper; see env/).

Hugging Face's Qwen3.5 reuses linear-attention state only for one-token steps: a multi-token chunk
appended to a cache restarts those layers from zero. So every decision re-encodes the window prefix,
and only the actions of the current turn are decoded with the cache.
"""
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

MODES = ("frozen", "lora", "llm", "full")
LORA_TARGETS = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))")
MODEL_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw")


def is_vision(name):
    return ".visual." in name and ".merger." not in name


class Qwen35Inputs:
    """Layout pieces -> token ids, supervised slots and pixels (CPU; runs in DataLoader workers)."""

    def __init__(self, cfg, processor_dir=None):
        self.cfg = cfg
        if processor_dir:
            self.processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
        else:
            self.processor = AutoProcessor.from_pretrained(cfg.repo, revision=cfg.revision)
        self.image_processor, self.tokenizer = self.processor.image_processor, self.processor.tokenizer
        ids = [self.tokenizer(a, add_special_tokens=False)["input_ids"] for a in cfg.action_tokens]
        if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != len(ids):
            raise ValueError(f"each action must be one distinct token: {cfg.action_tokens} -> {ids}")
        self.action_ids = [i[0] for i in ids]
        names = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
        self.im_start, self.im_end, self.vision_start, self.image_pad, self.vision_end = \
            self.tokenizer.convert_tokens_to_ids(names)
        self.pad_id = self.tokenizer.pad_token_id
        self.patch = self.image_processor.patch_size
        self.cell = self.patch * self.image_processor.merge_size
        for size in (self.size("view"), self.size("memory")):
            if size[0] % self.cell or size[1] % self.cell:
                raise ValueError(f"frame size {size} must be a multiple of {self.cell} px")
        self._text = {}

    def size(self, kind):
        c = self.cfg
        return (c.frame_width, c.frame_height) if kind == "view" else (c.memory_width, c.memory_height)

    def tokens_per_image(self, kind):
        w, h = self.size(kind)
        return (w // self.cell) * (h // self.cell)

    def resize(self, image, kind):
        size = self.size(kind)
        return image if image.size == size else image.resize(size, Image.BICUBIC)

    def text(self, s):
        ids = self._text.get(s)
        if ids is None:
            if len(self._text) > 4096:
                self._text.clear()
            ids = self._text[s] = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        return ids

    def encode(self, pieces, executed, targets=None):
        """Token ids of a layout. executed[step] fills each action slot.

        With targets, also the supervised slots: the logits at label_pos[i] must predict label_tok[i]
        (targets[step] for an action slot, <|im_end|> after an assistant turn when supervise_turn_end;
        label_step is -1 for those). A None target leaves its action unsupervised.
        """
        ids, images, pos, tok, steps = [], [], [], [], []
        for piece in pieces:
            kind = piece[0]
            if kind == "text":
                ids += self.text(piece[1])
            elif kind == "image":
                images.append((piece[1], piece[2]))
                ids += [self.vision_start] + [self.image_pad] * self.tokens_per_image(piece[2]) + [self.vision_end]
            elif kind == "action":
                step = piece[1]
                if targets is not None and targets[step] is not None:
                    pos.append(len(ids) - 1), tok.append(self.action_ids[targets[step]]), steps.append(step)
                ids.append(self.action_ids[executed[step]])
            elif kind == "open":
                ids += [self.im_start] + self.text(piece[1] + "\n")
            elif kind == "close":
                if targets is not None and piece[1] == "assistant" and self.cfg.supervise_turn_end:
                    pos.append(len(ids) - 1), tok.append(self.im_end), steps.append(-1)
                ids += [self.im_end] + self.text("\n")
            else:
                raise ValueError(f"unknown layout piece {piece!r}")
        if len(ids) > self.cfg.max_length:
            raise ValueError(f"layout of {len(ids)} tokens exceeds max_length {self.cfg.max_length}")
        return {"ids": ids, "images": images, "label_pos": pos, "label_tok": tok, "label_step": steps}

    def pixels(self, images):
        """Already-resized PIL images -> (pixel_values in bf16, image_grid_thw)."""
        out = self.image_processor(images=images, return_tensors="pt",
                                   size={"shortest_edge": self.cell ** 2, "longest_edge": 1 << 28})
        expected = [[1, im.size[1] // self.patch, im.size[0] // self.patch] for im in images]
        if out["image_grid_thw"].tolist() != expected:
            raise ValueError("image processor changed the configured frame size")
        return out["pixel_values"].to(torch.bfloat16), out["image_grid_thw"]

    def collate(self, samples):
        """Right-padded batch. Each sample is encode() output plus "pixels" (resized PIL images)."""
        lengths = [len(s["ids"]) for s in samples]
        input_ids = torch.full((len(samples), max(lengths)), self.pad_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for i, s in enumerate(samples):
            input_ids[i, :lengths[i]] = torch.tensor(s["ids"])
            attention[i, :lengths[i]] = 1
        batch = {"input_ids": input_ids, "attention_mask": attention,
                 "mm_token_type_ids": (input_ids == self.image_pad).long() * attention,
                 "lengths": torch.tensor(lengths)}
        images = [im for s in samples for im in s["pixels"]]
        if images:
            batch["pixel_values"], batch["image_grid_thw"] = self.pixels(images)
        if "label_pos" in samples[0]:
            action = {t: a for a, t in enumerate(self.action_ids)}
            batch["label_rows"] = torch.tensor([i for i, s in enumerate(samples) for _ in s["label_pos"]], dtype=torch.long)
            batch["label_pos"] = torch.tensor([p for s in samples for p in s["label_pos"]], dtype=torch.long)
            batch["label_tok"] = torch.tensor([t for s in samples for t in s["label_tok"]], dtype=torch.long)
            batch["label_action"] = torch.tensor([action.get(t, -1) for s in samples for t in s["label_tok"]], dtype=torch.long)
            batch["label_step"] = torch.tensor([t for s in samples for t in s["label_step"]], dtype=torch.long)
        return batch


class Qwen35Policy:
    """Qwen3.5 with the four action tokens. `weights` is a LoRA adapter or a full model directory."""

    def __init__(self, cfg, action_ids, mode="frozen", weights=None, lora_rank=64, lora_alpha=128,
                 lora_dropout=0.05, device="cuda", gradient_checkpointing=False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.cfg, self.mode, self.device = cfg, mode, device
        adapter = weights is not None and os.path.exists(os.path.join(weights, "adapter_config.json"))
        if adapter and mode in ("llm", "full"):
            raise ValueError("an adapter can only be trained further in lora mode")
        source = weights if weights and not adapter else cfg.repo
        dtype = torch.float32 if mode in ("llm", "full") else torch.bfloat16
        kwargs = {"revision": cfg.revision} if source == cfg.repo else {"local_files_only": True}
        model = Qwen3_5ForConditionalGeneration.from_pretrained(source, dtype=dtype, attn_implementation="sdpa", **kwargs)
        model.requires_grad_(False)
        if mode == "lora" or adapter:
            from peft import LoraConfig, PeftModel, get_peft_model
            if adapter:
                model = PeftModel.from_pretrained(model, weights, is_trainable=mode == "lora")
            else:
                model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                                         target_modules=LORA_TARGETS))
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()  # fp32 adapters; forward runs under bf16 autocast
        elif mode in ("llm", "full"):
            for name, p in model.named_parameters():
                p.requires_grad = mode == "full" or not is_vision(name)
                if not p.requires_grad:
                    p.data = p.data.to(torch.bfloat16)
        self.model = model.to(device)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if mode == "lora":
                self.model.enable_input_require_grads()
        self.core = next(m for m in self.model.modules() if isinstance(m, Qwen3_5Model))
        self.action_ids = torch.tensor(action_ids, device=device)
        if mode == "frozen":
            self.model.eval()

    def parameter_groups(self, lr, vision_lr, weight_decay=0.0):
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in named if not is_vision(n)], "lr": lr, "weight_decay": weight_decay, "name": "main"},
                  {"params": [p for n, p in named if is_vision(n)], "lr": vision_lr, "weight_decay": weight_decay, "name": "vision"}]
        return [g for g in groups if g["params"]]

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    @staticmethod
    def model_inputs(batch):
        return {k: batch[k] for k in MODEL_KEYS if k in batch}

    def supervised_logits(self, batch):
        """Full-vocabulary logits at the supervised positions only (never all positions).

        The autocast cast cache is off: with fp32 master weights it would hold a bf16 copy of every
        weight (~8.5 GiB) for the whole forward, while checkpointed layers recast their own anyway.
        """
        keep, inverse = torch.unique(batch["label_pos"], sorted=True, return_inverse=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = self.model(**self.model_inputs(batch), logits_to_keep=keep, use_cache=False).logits
        return logits[batch["label_rows"], inverse].float()

    def loss_sum(self, batch):
        """Summed cross-entropy over the supervised tokens, their count, and the four action logits."""
        logits = self.supervised_logits(batch)
        loss = F.cross_entropy(logits, batch["label_tok"], reduction="sum")
        return loss, int(batch["label_tok"].numel()), logits[:, self.action_ids].detach()

    @torch.no_grad()
    def decode(self, batch, max_actions, stop_bias=0.0, return_hidden=False):
        """Greedy decoding of up to `max_actions` actions after a prompt ending in an open assistant turn.

        Batch of one. Returns the actions (ending early at STOP) and the raw four-action logits before
        each one; the STOP bias is added only for the argmax. The prompt is encoded once; each further
        action is a one-token cached step with explicit M-RoPE positions.
        """
        if batch["input_ids"].shape[0] != 1:
            raise ValueError("decode expects a batch of one")
        bias = torch.zeros(len(self.action_ids), device=self.device)
        bias[0] = stop_bias
        length = int(batch["attention_mask"].sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(**self.model_inputs(batch), use_cache=True, logits_to_keep=1, output_hidden_states=return_hidden)
        hidden = out.hidden_states[-1][0, -1].float() if return_hidden else None
        delta = int(self.core.rope_deltas.reshape(-1)[0])
        cache, actions, scores = out.past_key_values, [], []
        for k in range(max_actions):
            raw = out.logits[0, -1, self.action_ids].float()
            action = int((raw + bias).argmax())
            actions.append(action), scores.append(raw.tolist())
            if action == 0 or k == max_actions - 1:
                break
            position = torch.full((3, 1, 1), length + k + delta, device=self.device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=self.action_ids[action].view(1, 1),
                                 attention_mask=torch.ones((1, length + k + 1), device=self.device, dtype=torch.long),
                                 position_ids=position, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        return (actions, scores, hidden) if return_hidden else (actions, scores)

    def save(self, path):
        """Adapter (lora) or bf16 full weights; the processor and config are saved by the caller."""
        if self.mode == "lora" or hasattr(self.model, "peft_config"):
            self.model.save_pretrained(path, save_embedding_layers=False)  # LoRA never touches the embeddings
        else:  # bf16 copies on the CPU; tied weights (lm_head = embed_tokens) once, as in the original checkpoint
            seen, state = set(), {}
            for name, value in self.model.state_dict().items():
                if value.data_ptr() not in seen:
                    seen.add(value.data_ptr())
                    state[name] = value.detach().to("cpu", torch.bfloat16)
            self.model.save_pretrained(path, state_dict=state, safe_serialization=True)

    def save_master(self, path):
        """fp32 master weights of llm/full modes (checkpoints keep bf16), so a resume is exact."""
        from safetensors.torch import save_file
        save_file({n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}, str(path))

    def load_master(self, path):
        from safetensors.torch import load_file
        state = load_file(str(path), device=str(self.device))
        params = dict(self.model.named_parameters())
        if set(state) != {n for n, p in params.items() if p.requires_grad}:
            raise ValueError("master weights do not match the trainable parameters")
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value)
