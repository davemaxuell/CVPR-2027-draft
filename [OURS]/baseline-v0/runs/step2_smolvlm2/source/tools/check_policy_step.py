"""One masked-loss LoRA step on SmolVLM2 with a synthetic navigation prompt (Step 2 interface check).

Checks: LoRA attaches to decoder projections only; loss covers only the response tokens;
gradients reach only adapters; the prefill context q_t has the documented width; and
token/memory cost per history frame with image splitting disabled.

    CUDA_VISIBLE_DEVICES=1 python tools/check_policy_step.py --frames 8 --out logs/check_policy_step.json
"""
import argparse
import json
import time

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DECODER_TARGETS = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
INSTRUCTION = ("Walk out of the bedroom and turn left. Go down the hallway past the bathroom "
               "and stop next to the kitchen counter.")


def build(processor, frames, past_actions, target):
    content = [{"type": "text", "text": f"Instruction: {INSTRUCTION}\nObservations so far:"}]
    for i in range(frames):
        content.append({"type": "image"})
        if i < frames - 1:
            content.append({"type": "text", "text": f"Action: {past_actions[i % len(past_actions)]}"})
    content.append({"type": "text", "text": "Next action?"})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return prompt, prompt + target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    processor = AutoProcessor.from_pretrained(REPO)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.bfloat16).to(dev)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=DECODER_TARGETS))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n and "text_model" in n for n in trainable), "non-adapter params trainable"

    eot = processor.tokenizer.convert_ids_to_tokens(processor.tokenizer.eos_token_id)
    target = "FORWARD" + "<end_of_utterance>"
    prompt, full = build(processor, args.frames, ["FORWARD", "LEFT", "RIGHT"], target)
    images = [Image.new("RGB", (640, 480), (40 * i % 255, 90, 160)) for i in range(args.frames)]
    batch = processor(text=[full], images=[images], return_tensors="pt").to(dev)
    n_prompt = processor(text=[prompt], images=[images], return_tensors="pt")["input_ids"].shape[1]

    labels = batch["input_ids"].clone()
    labels[:, :n_prompt] = -100
    labels[batch["attention_mask"] == 0] = -100
    supervised = processor.tokenizer.decode(batch["input_ids"][0, n_prompt:])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model(**batch, labels=labels, output_hidden_states=True)
    out.loss.backward()
    torch.cuda.synchronize()
    grads = [n for n, p in model.named_parameters() if p.grad is not None]
    assert grads and all("lora_" in n for n in grads)

    with torch.no_grad():
        pre = processor(text=[prompt], images=[images], return_tensors="pt").to(dev)
        h = model(**pre, output_hidden_states=True).hidden_states[-1]
        last = pre["attention_mask"].sum(1) - 1
        q_t = h[torch.arange(h.shape[0]), last]

    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    report = {
        "repo": REPO,
        "frames": args.frames,
        "lora_rank": args.rank,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "seq_len": batch["input_ids"].shape[1],
        "image_tokens_per_frame": int((batch["input_ids"] == image_token).sum()) // args.frames,
        "supervised_text": supervised,
        "supervised_tokens": int((labels != -100).sum()),
        "eos_token": eot,
        "loss": float(out.loss.detach()),
        "q_t_shape": list(q_t.shape),
        "peak_mem_gb_fwd_bwd": torch.cuda.max_memory_allocated() / 2**30,
        "step_seconds": time.time() - t0,
    }
    print(json.dumps(report, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
