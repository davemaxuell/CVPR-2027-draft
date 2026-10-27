"""Native instruction-only states and real-prefill states, without generated tokens."""
import torch

from policy.experimental_reference import ExperimentalReference
from Scorer.interfaces import PrefillFeatureTap, instruction_states, last_valid_state, validate_policy_config
from .common import REFERENCE


def instruction_batch(inputs, instruction, device):
    text = instruction.strip()  # Same whitespace convention as the navigation prompt.
    assert text, 'Empty instruction'
    def render(s):
        return inputs.processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'text', 'text': s}]}],
            add_generation_prompt=True)
    sentinel = 'SCORER_INSTRUCTION_SPAN_7'
    template = render(sentinel)
    assert template.count(sentinel) == 1
    before, after = template.split(sentinel)
    prompt = render(text)
    assert prompt == before + text + after
    encoded = inputs.tokenizer(prompt, return_tensors='pt', return_offsets_mapping=True,
                               truncation=False, add_special_tokens=True)
    offsets = encoded.pop('offset_mapping')[0].tolist()
    start, end = len(before), len(before) + len(text)
    mask = torch.tensor([[a < end and b > start for a, b in offsets]], dtype=torch.bool)
    assert mask.any() and int(encoded['attention_mask'].sum()) <= inputs.cfg.max_length
    covered = set()
    for (a, b), keep in zip(offsets, mask[0].tolist()):
        if keep:
            covered.update(range(max(start, a), min(end, b)))
    assert all(start + i in covered for i, c in enumerate(text) if not c.isspace()), 'Instruction was truncated'
    selected = encoded['input_ids'][mask].tolist()
    assert inputs.tokenizer.decode(selected, skip_special_tokens=False).strip() == text
    return ({k: v.to(device) for k, v in encoded.items()}, mask.to(device),
            {'text': text, 'prompt': prompt, 'instruction_token_ids': selected,
             'instruction_tokens': len(selected), 'prompt_tokens': encoded['input_ids'].shape[1],
             'character_span': [start, end], 'truncated': False})


class FrozenPolicyFeatures:
    def __init__(self, device='cuda:0'):
        self.policy = ExperimentalReference(REFERENCE, device=device)
        self.device = device
        self.base = self.policy.model.get_base_model()
        validate_policy_config('smolvlm2', self.base.config.to_dict(), 960)
        self.final_norm = self.base.model.text_model.norm
        self.tap = PrefillFeatureTap(self.final_norm, 960)
        self.hook_path = next(n for n, m in self.policy.model.named_modules() if m is self.final_norm)
        assert not any(p.requires_grad for p in self.policy.model.parameters())

    @torch.inference_mode()
    def context(self, batch, *, compare_hidden=False):
        # OnlineInputs has already expanded all image placeholders. The model replaces
        # those token embeddings in-place and keeps this exact sequence/mask length.
        with self.tap.capture():
            if compare_hidden:
                raw, direct = self.policy.raw_action_scores(batch, return_context=True)
            else:
                raw = self.policy.raw_action_scores(batch)
        hidden = self.tap.hidden
        q = last_valid_state(hidden, batch['attention_mask'].bool())
        assert q.shape == (1, 960) and torch.isfinite(q).all() and torch.isfinite(raw).all()
        if compare_hidden:
            assert torch.equal(q.float(), direct), 'Final norm / returned hidden-state mismatch'
        scores = raw.clone(); scores[:, 0] -= .25
        return {'context': q[0].cpu().clone(), 'raw_logits': raw[0].cpu().clone(),
                'biased_logits': scores[0].cpu().clone(), 'base_live': int(scores.argmax(1)),
                'prompt_length': int(batch['lengths'][0])}

    @torch.inference_mode()
    def instruction(self, text, *, compare_hidden=False):
        batch, mask, info = instruction_batch(self.policy.inputs, text, self.device)
        with self.tap.capture(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = self.policy.model(**batch, logits_to_keep=1, use_cache=False,
                                       output_hidden_states=compare_hidden)
        hidden = self.tap.hidden
        if compare_hidden:
            assert torch.equal(hidden, output.hidden_states[-1])
        states, valid = instruction_states(hidden, mask)
        assert states.shape == (1, info['instruction_tokens'], 960) and valid.all()
        assert torch.isfinite(states).all()
        return {'states': states[0].cpu().clone(), 'valid': valid[0].cpu().clone(), **info}
