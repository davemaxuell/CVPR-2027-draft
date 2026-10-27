"""GPU worker for the viewer's what-if branches: re-run a DUET/ScaleVLN checkpoint with chosen decisions forced.

visualizer_app.py starts this (through whatif.py) only while the viewer's "What-if · GPU" switch is on, and
stops it when the switch is turned off or after a few idle minutes. It is not meant to be run by hand:

  source env.sh
  CUDA_VISIBLE_DEVICES=0 python analysis/vln_viz/whatif_worker.py --model h14

Protocol: one JSON request per stdin line, JSON replies on stdout (everything DUET prints goes to stderr).
  request  {"anno": file, "languages": "en-US,en-IN" | "", "max_instr_len": 200, "max_action_len": 15,
            "branches": [{"id": ..., "instr_id": ..., "forced": [vp | null, ...]}, ...]}
  replies  {"id": ..., "trajectory": [[vp], [vp, vp], ...], "prefix_agree": true} per branch (or
           {"id": ..., "error": ...}), then {"done": n}

`forced` gives the decision to take at each step from the start: the recorded moves up to the branch point,
then the alternative (a viewpoint, or null for STOP). After the forced decisions the policy decodes with
argmax as in evaluation; the step budget and DUET's return to the best-scored stop node apply as usual.
Forcing replaces the fused logits of that step, but the stop score DUET stores for the node keeps the
model's own value. prefix_agree is false when the model's own argmax differed from a forced recorded move,
i.e. the replay did not reproduce the recorded episode up to the branch point.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_duet_steps import CHECKPOINTS, LazyFeatures, load_rows, setup_duet  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['b16', 'h14'], default='h14')
    ap.add_argument('--checkpoint', type=Path, help='default: released ScaleVLN fine-tuned checkpoint')
    ap.add_argument('--batch-size', type=int, default=16)
    options = ap.parse_args()
    checkpoint = (options.checkpoint or CHECKPOINTS / f'duet_vit-{options.model}_ft_best_val_unseen').resolve()

    replies = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)               # DUET and MatterSim print progress to stdout; keep it for replies only
    sys.stdout = sys.stderr
    reply = lambda msg: replies.write(json.dumps(msg) + '\n')

    args, _ = setup_duet(options.model, options.batch_size)
    import torch
    from transformers import AutoTokenizer
    from r2r.agent import GMapNavAgent
    from r2r.env import R2RNavBatch

    class BranchAgent(GMapNavAgent):
        """Argmax rollout whose first decisions are forced, per batch item, by self.forced."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.forced, self.agree, self.errors, self._step, self._stop = [], [], [], 0, {}
            self.vln_bert.register_forward_hook(self._force)

        def rollout(self, *a, **kw):
            self._step = 0
            self.agree, self.errors = [True] * len(self.forced), [None] * len(self.forced)
            return super().rollout(*a, **kw)

        def _force(self, module, inputs, output):
            if inputs[0] != 'navigation':
                return None
            t, self._step, self._stop = self._step, self._step + 1, {}
            logits = output['fused_logits'] = output['fused_logits'].clone()
            for i, forced in enumerate(self.forced):
                if t >= len(forced) or self.errors[i]:
                    continue
                vpids, want = inputs[1]['gmap_vpids'][i], forced[t]
                j = 0 if want is None else vpids.index(want) if want in vpids else None
                if j is None or (j and bool(inputs[1]['gmap_visited_masks'][i, j])):
                    self.errors[i] = f'decision {t}: {want} is not on the frontier'
                    continue
                if t < len(forced) - 1 and int(logits[i].argmax()) != j:
                    self.agree[i] = False
                self._stop[i] = torch.softmax(logits[i], 0)[0].item()
                logits[i] = float('-inf')
                logits[i, j] = 0.0
            return output

        def make_equiv_action(self, a_t, gmaps, obs, traj=None):
            # rollout() stored the forced (one-hot) stop probability; restore the model's own
            for i, p in self._stop.items():
                scores = gmaps[i].node_stop_scores.get(obs[i]['viewpoint'])
                if scores is not None:
                    scores['stop'] = p
            return super().make_equiv_action(a_t, gmaps, obs, traj)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    rows, env, agent = {}, None, None
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        key = (req['anno'], req.get('max_instr_len', 200), req.get('languages') or '')
        if key not in rows:
            languages = {x for x in key[2].split(',') if x}
            rows[key] = {r['instr_id']: r for r in load_rows(Path(key[0]), tokenizer, key[1], languages)}
        items = rows[key]
        if agent is None:
            env = R2RNavBatch(LazyFeatures(args.val_ft_file, args.image_feat_size), [next(iter(items.values()))],
                              args.connectivity_dir, batch_size=options.batch_size, name='whatif')
            agent = BranchAgent(args, env, rank=0)
            agent.load(str(checkpoint))
            agent.vln_bert.eval()
            agent.critic.eval()
            agent.feedback = 'argmax'
        args.max_action_len = req.get('max_action_len', 15)
        branches = [b for b in req['branches'] if b['instr_id'] in items]
        for b in req['branches']:
            if b['instr_id'] not in items:
                reply({'id': b['id'], 'error': f"{b['instr_id']} is not in {key[0]}"})
        for s in range(0, len(branches), options.batch_size):
            chunk = branches[s:s + options.batch_size]
            batch = [dict(items[b['instr_id']], instr_id=b['id']) for b in chunk]
            pad = options.batch_size - len(batch)             # every simulator in the batch needs an episode
            env.batch = batch + [batch[-1]] * pad
            env.env.newEpisodes([x['scan'] for x in env.batch], [x['path'][0] for x in env.batch],
                                [x['heading'] for x in env.batch])
            agent.forced = [b['forced'] for b in chunk] + [[]] * pad
            with torch.no_grad():
                trajs = agent.rollout(reset=False)
            for k, b in enumerate(chunk):
                if agent.errors[k]:
                    reply({'id': b['id'], 'error': agent.errors[k]})
                else:
                    reply({'id': b['id'], 'trajectory': trajs[k]['path'], 'prefix_agree': agent.agree[k]})
        reply({'done': len(req['branches'])})


if __name__ == '__main__':
    main()
