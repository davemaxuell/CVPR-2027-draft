"""Observational wrapper around the unchanged Stage-A streaming policy.

Records a whole four-action policy call, its logits and optional layer features. Only RGB,
the instruction, and executed action history cross the drive boundary. No maps/targets here.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CODE, atomic_json, read_json, sha256
from memory import inject_saved_places
sys.path.insert(0, str(CODE))
from stream_policy.server import StreamingPolicy


class RecordedPolicy(StreamingPolicy):
    def __init__(self, args):
        self.feature_dir = Path(args.feature_dir) if args.feature_dir else None
        self.capture = bool(args.feature_dir)
        self.captured = {}
        self.saved_records = []
        self.sampling_temperature = 0.
        self.depth_waypoints = None
        super().__init__(args)
        metadata = read_json(Path(args.run_dir)/'queue_policy.json',{})
        self.slot_enabled = bool(metadata.get('saved_places_slot',False) or getattr(args,'force_slot',False))
        self.identity.update(saved_places_slot=self.slot_enabled,queue_policy_metadata=metadata,
                             slot_layout_sha256=sha256(HERE/'memory.py'))
        if self.slot_enabled:
            self.base_layout = self.layout
            self.layout = self.saved_layout
        if self.capture:
            layers = self.policy.core.language_model.layers
            for index in [15, 23]:
                def hook(module, inputs, output, key=str(index + 1)):
                    # Capture the prefill only; token generation must not overwrite the context state.
                    if key not in self.captured:
                        tensor = output[0] if isinstance(output, tuple) else output
                        self.captured[key] = tensor[0, -1].detach().to('cpu', self.torch.float16)
                layers[index].register_forward_hook(hook)

    def reset(self, instruction):
        super().reset(instruction)
        self.episode_key = None
        self.saved_records = []

    def saved_layout(self,*args,**kwargs):
        return inject_saved_places(self.base_layout(*args,**kwargs),self.saved_records)

    def set_records(self,records):
        if len(records)>3:
            raise ValueError('The drive slot is limited to three records.')
        result = []
        for record in records:
            if set(record)!={'views'}:
                raise ValueError('The drive slot accepts past frame IDs only; labels/geometry/text are forbidden.')
            views = list(record['views'])
            if not 1<=len(views)<=2 or any(not isinstance(x,int) or not 0<=x<len(self.frames) for x in views):
                raise ValueError('Saved views must be at most two already observed frames per record.')
            result.append(dict(views=views))
        self.saved_records = result

    def decode(self, t):
        start = self.window_start(t, self.cfg)
        pieces = self.layout(self.cfg, self.instruction, start, t, self.fixed_conjunction(self.cfg, start), open_turn=True)
        sample = self.inputs.encode(pieces, self.executed)
        sample['pixels'] = [self.inputs.resize(self.frames[frame], kind) for frame, kind in sample['images']]
        batch = {k: v.to(self.policy.device) for k, v in self.inputs.collate([sample]).items()}
        self.captured = {}
        if self.sampling_temperature>0:
            with self.torch.no_grad():
                result = self.sample_decode(batch,self.turn_end(t,self.cfg)-t)
        else:
            result = self.policy.decode(batch, self.turn_end(t, self.cfg) - t, self.stop_bias, return_hidden=self.capture)
        actions, scores = result[:2]
        feature_path = None
        if self.capture:
            self.captured['32'] = result[2].to('cpu', self.torch.float16)
            if set(self.captured) != {'16', '24', '32'} or any(tuple(v.shape) != (2560,) for v in self.captured.values()):
                raise RuntimeError('Unexpected hidden-state dimensions or missing layer capture.')
            if not self.episode_key:
                raise RuntimeError('Feature capture requires a dataset/split/episode key.')
            path = self.feature_dir / self.episode_key / f'{t:06d}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.partial')
            with tmp.open('wb') as f:
                self.np.savez(f,**{k:v.numpy() for k,v in self.captured.items()})
            os.replace(tmp, path)
            feature_path = str(path)
        self.plan = dict(start=t, actions=actions, scores=scores, tokens=len(sample['ids']), feature_path=feature_path)
        return self.plan

    def sample_decode(self,batch,max_actions):
        """The native one-token cache path, changing only greedy choice to seeded sampling."""
        torch = self.torch
        policy = self.policy
        length = int(batch['attention_mask'].sum())
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out = policy.model(**policy.model_inputs(batch),use_cache=True,logits_to_keep=1,output_hidden_states=self.capture)
        hidden = out.hidden_states[-1][0,-1].float() if self.capture else None
        delta = int(policy.core.rope_deltas.reshape(-1)[0])
        cache,actions,scores = out.past_key_values,[],[]
        for k in range(max_actions):
            raw = out.logits[0,-1,policy.action_ids].float()
            adjusted = raw.clone()
            adjusted[0] += self.stop_bias
            probabilities = torch.softmax(adjusted/self.sampling_temperature,dim=-1)
            action = int(torch.multinomial(probabilities,1))
            actions.append(action);scores.append(raw.tolist())
            if action==0 or k==max_actions-1:
                break
            position = torch.full((3,1,1),length+k+delta,device=policy.device,dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out = policy.model(input_ids=policy.action_ids[action].view(1,1),
                    attention_mask=torch.ones((1,length+k+1),device=policy.device,dtype=torch.long),
                    position_ids=position,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache = out.past_key_values
        return (actions,scores,hidden) if self.capture else (actions,scores)

    def opening_prediction(self,header,payload):
        if list(header['shape'])!=[12,256,256,1]:
            raise ValueError('Depth prediction accepts only the current-pose sensor panorama.')
        # Loading an auxiliary model must not perturb the policy's sampled action stream.
        cpu_rng = self.torch.get_rng_state()
        gpu_rng = self.torch.cuda.get_rng_state()
        try:
            if self.depth_waypoints is None:
                from depth_waypoint import DepthWaypoints
                self.depth_waypoints = DepthWaypoints(device='cuda:0')
            depth = self.np.frombuffer(payload,dtype=self.np.float32).reshape(12,256,256,1)
            return self.depth_waypoints.predict(depth)
        finally:
            self.torch.set_rng_state(cpu_rng)
            self.torch.cuda.set_rng_state(gpu_rng)

    def act(self, header, payload):
        answer = super().act(header, payload)
        if answer['replanned']:
            logits = self.torch.tensor(self.plan['scores'], dtype=self.torch.float32)
            probabilities = self.torch.softmax(logits, dim=-1).tolist()
            answer.update(action_chunk=self.plan['actions'], action_logits=self.plan['scores'],
                          action_token_probabilities=probabilities + [None] * (4 - len(probabilities)),
                          generated_positions=len(probabilities), feature_path=self.plan.get('feature_path'),
                          distribution_support=['STOP', 'FORWARD', 'LEFT', 'RIGHT'])
            # Sampling state is replay metadata only. It is never encoded into a model prompt.
            answer['sampling_rng_after'] = base64.b64encode(self.torch.cuda.get_rng_state().numpy().tobytes()).decode('ascii')
        return answer

    def restore_sampling_rng(self,value):
        raw = base64.b64decode(value,validate=True)
        expected = self.torch.cuda.get_rng_state().numel()
        if len(raw)!=expected:
            raise ValueError('Sampling replay state has an incompatible CUDA generator format.')
        state = self.torch.tensor(list(raw),dtype=self.torch.uint8)
        self.torch.cuda.set_rng_state(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--stop-bias', type=float, default=0.0)
    ap.add_argument('--feature-dir', default='')
    ap.add_argument('--force-slot',action='store_true')
    args = ap.parse_args()
    out = os.fdopen(os.dup(1), 'wb', buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    from stream_policy.protocol import receive, send
    stdin = sys.stdin.buffer
    try:
        policy = RecordedPolicy(args)
    except Exception:
        send(out, {'error': traceback.format_exc()})
        raise
    while True:
        header, payload = receive(stdin)
        if header is None or header.get('op') == 'close':
            break
        try:
            began = time.monotonic()
            op = header['op']
            if op == 'hello':
                reply = dict(policy.identity, instrumentation='queue_after_stageA/policy_server.py')
            elif op == 'reset':
                # Metadata controls only the local feature filename; never encoded into the model.
                policy.reset(header['instruction'])
                policy.episode_key = header.get('episode_key')
                reply = {}
            elif op == 'act':
                reply = policy.act(header, payload)
            elif op == 'observe':
                policy.observe(header, payload)
                reply = {}
            elif op == 'set_records':
                policy.set_records(header['records'])
                reply = {}
            elif op == 'decoding':
                temperature = float(header['temperature'])
                if temperature not in {0.,1.}:
                    raise ValueError('The queue permits greedy or the prespecified T=1 sampling only.')
                policy.sampling_temperature = temperature
                policy.torch.manual_seed(int(header['seed']))
                policy.plan = None
                reply = dict(temperature=temperature)
            elif op == 'invalidate_plan':
                policy.plan = None
                reply = {}
            elif op == 'restore_sampling_rng':
                policy.restore_sampling_rng(header['state'])
                reply = {}
            elif op == 'openings':
                reply = policy.opening_prediction(header,payload)
            elif op == 'stats':
                reply = dict(peak_allocated_gib=policy.torch.cuda.max_memory_allocated() / 2**30)
            else:
                raise ValueError('Unknown operation: ' + str(op))
            reply['seconds'] = time.monotonic() - began
            send(out, reply)
        except Exception:
            send(out, {'error': traceback.format_exc()})


if __name__ == '__main__':
    main()
