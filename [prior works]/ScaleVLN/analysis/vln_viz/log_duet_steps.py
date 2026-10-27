"""Re-run a DUET/ScaleVLN checkpoint on R2R and log every navigation decision for the visualizer.

  source env.sh
  python analysis/vln_viz/log_duet_steps.py --model h14 --split val_unseen \
      --out analysis/vln_viz/logs/h14_val_unseen
  python analysis/vln_viz/log_duet_steps.py --split val_seen --episodes 20 --out ...   # 20 evenly spaced
  python analysis/vln_viz/log_duet_steps.py --split val_unseen --ids 601_2 6620_2 --out ...

Other datasets (zero-shot with an R2R checkpoint): --anno takes any R2R-format *_enc.json (REVERIE too; its
instruction ids become <path_id>_<objId>_<j>, as the viewer expects) or an RxR guide .jsonl(.gz), which is
tokenised here and filtered by --languages; --split then only names the outputs:
  python analysis/vln_viz/log_duet_steps.py --split reverie_val_unseen \
      --anno VLN-DUET-RVR/datasets/REVERIE/annotations/REVERIE_val_unseen_enc.json --out ...
  python analysis/vln_viz/log_duet_steps.py --split rxr_val_unseen --max-action-len 25 --max-instr-len 512 \
      --anno ../../data/datasets/RxR/rxr_val_unseen_guide.jsonl.gz --out ...

Writes <out>/steps_<split>.jsonl (one line per episode), <out>/detail_<split>.json (predictions in the
usual --detailed_output format, so the run can be checked against the released evaluation) and
<out>/meta_<split>.json (settings; the viewer reads max_action_len and note from it). Each step has:
  vp           viewpoint where the decision is taken
  gmap         global-map viewpoints in DUET's order (visited first, then unvisited frontier)
  visited      1/0 per gmap entry
  fused        fused action probabilities over [STOP] + gmap (what argmax decoding uses)
  global       global-branch probabilities over [STOP] + gmap
  local        {"cands": [...], "probs": [...]} local-branch probabilities over [STOP] + neighbours
  fuse_w       learned weight of the global branch (1 - fuse_w goes to the local branch)
  choice       chosen viewpoint, or null for STOP; forced=true when STOP was imposed by the step budget
  attn         cross-attention of the chosen node over instruction tokens, averaged over heads and over the
               global encoder's cross-modal layers
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / 'VLN-DUET/datasets/R2R/trained_models/finetune'   # duet_vit-<model>_ft_best_val_unseen


def setup_duet(model, batch_size=8, max_action_len=15, max_instr_len=200):
    """DUET's argument namespace for a ScaleVLN checkpoint (h14 or b16 CLIP features), as used for the released
    evaluation. Changes into VLN-DUET/map_nav_src, which DUET's code expects, and makes it importable.
    Returns (args, scratch): DUET's parser creates ckpts/logs/preds dirs in `scratch`; delete it when done."""
    nav = ROOT / 'VLN-DUET/map_nav_src'
    os.chdir(nav)
    sys.path.insert(0, str(nav))
    from r2r.parser import parse_args
    scratch = tempfile.mkdtemp(prefix='duet_steplog_')
    sys.argv = [sys.argv[0], '--root_dir', str(ROOT / 'VLN-DUET/datasets'), '--output_dir', scratch,
                '--features', 'clip.' + model, '--image_feat_size', '512' if model == 'b16' else '1024',
                '--batch_size', str(batch_size), '--fusion', 'dynamic', '--enc_full_graph', '--graph_sprels',
                '--train_alg', 'dagger', '--num_l_layers', '9', '--num_x_layers', '4', '--num_pano_layers', '2',
                '--max_action_len', str(max_action_len), '--max_instr_len', str(max_instr_len), '--detailed_output']
    return parse_args(), scratch


class LazyFeatures:
    """DUET's feature-DB interface, reading one viewpoint's image features from the HDF5 file on request."""

    def __init__(self, path, dim):
        import h5py
        self.stream, self.dim = h5py.File(path, 'r'), dim

    def get_image_feature(self, scan, viewpoint):
        return self.stream[f'{scan}_{viewpoint}'][...][:, :self.dim].astype('float32')


def load_rows(path, tokenizer, max_instr_len, languages):
    """DUET's per-instruction rows from an R2R-format *_enc.json (REVERIE too) or an RxR guide .jsonl(.gz)."""
    if path.name.endswith(('.jsonl', '.jsonl.gz')):     # RxR: one instruction per line, not tokenised
        rows = []
        with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as f:
            for line in f:
                it = json.loads(line)
                if languages and it['language'] not in languages:
                    continue
                rows.append({'instr_id': str(it['instruction_id']), 'path_id': it['path_id'], 'scan': it['scan'],
                             'path': it['path'], 'heading': it['heading'], 'instruction': it['instruction'],
                             'instr_encoding': tokenizer.encode(it['instruction'])[:max_instr_len]})
        return rows
    rows = []
    for item in json.load(open(path)):
        key = item.get('id', item['path_id'])          # REVERIE: <path_id>_<objId>, unique per target object
        for j, instr in enumerate(item['instructions']):
            row = {k: v for k, v in item.items() if k not in ('instructions', 'instr_encodings')}
            row.update(instr_id=f'{key}_{j}', instruction=instr, instr_encoding=item['instr_encodings'][j][:max_instr_len])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['b16', 'h14'], default='h14')
    parser.add_argument('--checkpoint', type=Path, help='default: released ScaleVLN fine-tuned checkpoint')
    parser.add_argument('--split', default='val_unseen')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--episodes', default='all', help="'all' (default) or N evenly spaced instructions")
    parser.add_argument('--ids', nargs='+', help='specific instruction ids, e.g. 601_2 6620_2')
    parser.add_argument('--limit', type=int, help='only the first N instructions (for quick checks)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--anno', type=Path, help='annotation file for another dataset (see above); default: R2R_<split>_enc.json')
    parser.add_argument('--languages', default='en-US,en-IN', help="RxR languages to keep ('' = all)")
    parser.add_argument('--max-action-len', type=int, default=15, help='decision budget (DUET: 15)')
    parser.add_argument('--max-instr-len', type=int, default=200, help='instruction tokens kept (DUET: 200; BERT max 512)')
    parser.add_argument('--note', default='', help='shown in the viewer next to the dataset, e.g. "zero-shot"')
    options = parser.parse_args()
    if options.episodes != 'all' and not (options.episodes.isdigit() and int(options.episodes) > 0):
        parser.error("--episodes must be 'all' or a positive integer")
    out = options.out.resolve()
    if options.anno:
        options.anno = options.anno.resolve()       # before the chdir below
        if not options.anno.is_file():
            parser.error(f'--anno {options.anno}: file not found')
    out.mkdir(parents=True, exist_ok=True)

    args, scratch = setup_duet(options.model, options.batch_size, options.max_action_len, options.max_instr_len)
    import torch
    from transformers import AutoTokenizer
    from utils.misc import set_random_seed
    from r2r.data_utils import construct_instrs
    from r2r.env import R2RNavBatch
    from r2r.agent import GMapNavAgent
    set_random_seed(0)

    class LoggingAgent(GMapNavAgent):
        """GMapNavAgent that records the model's view of every decision; navigation is unchanged."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.records, self._nav, self._attn = {}, None, []
            self.vln_bert.register_forward_hook(self._capture_nav)
            for layer in self.vln_bert.vln_bert.global_encoder.encoder.x_layers:
                layer.visual_attention.att.register_forward_hook(self._capture_attn)

        def _capture_nav(self, module, inputs, output):
            if inputs[0] == 'navigation':
                self._nav = (inputs[1], output, self._attn)
            self._attn = []

        def _capture_attn(self, module, inputs, output):
            # output[1]: masked pre-softmax scores (batch, heads, gmap nodes, tokens)
            self._attn.append(torch.softmax(output[1], -1).mean(1))

        def rollout(self, *a, **kw):
            self._batch = {}
            trajs = super().rollout(*a, **kw)
            for instr_id, rec in self._batch.items():
                self.records.setdefault(instr_id, rec)   # test() wraps around; keep the first run
            return trajs

        def make_equiv_action(self, a_t, gmaps, obs, traj=None):
            inputs, outs, attn = self._nav
            attn = torch.stack(attn).mean(0) if attn else None
            fuse_w = torch.sigmoid(self.vln_bert.vln_bert.sap_fuse_linear(
                torch.cat([outs['gmap_embeds'][:, 0], outs['vp_embeds'][:, 0]], 1))).squeeze(1)
            probs = {k: torch.softmax(outs[k + '_logits'], 1).cpu() for k in ('fused', 'global', 'local')}
            for i, ob in enumerate(obs):
                rec = self._batch.setdefault(ob['instr_id'], {
                    'instr_id': ob['instr_id'], 'scan': ob['scan'],
                    'tokens': tokenizer.convert_ids_to_tokens(ob['instr_encoding']), 'steps': [], 'done': False})
                if rec['done']:
                    continue
                gmap_vpids = inputs['gmap_vpids'][i]
                n, m = len(gmap_vpids), len(inputs['vp_cand_vpids'][i])
                fused = probs['fused'][i, :n]
                choice = a_t[i]
                if choice is None:
                    rec['done'] = True
                row = gmap_vpids.index(choice if choice is not None else ob['viewpoint'])
                step = {
                    't': len(rec['steps']), 'vp': ob['viewpoint'],
                    'gmap': gmap_vpids[1:],
                    'visited': [int(x) for x in inputs['gmap_visited_masks'][i, 1:n].tolist()],
                    'fused': [round(x, 5) for x in fused.tolist()],
                    'global': [round(x, 5) for x in probs['global'][i, :n].tolist()],
                    'local': {'cands': inputs['vp_cand_vpids'][i][1:],
                              'probs': [round(x, 5) for x in probs['local'][i, :m].tolist()]},
                    'fuse_w': round(fuse_w[i].item(), 4),
                    'choice': choice,
                    'forced': choice is None and int(fused.argmax()) != 0,
                }
                if attn is not None:
                    step['attn'] = [round(x, 4) for x in attn[i, row, :len(ob['instr_encoding'])].tolist()]
                rec['steps'].append(step)
            return super().make_equiv_action(a_t, gmaps, obs, traj)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if options.anno:
        languages = {x for x in options.languages.split(',') if x}
        rows = load_rows(options.anno, tokenizer, args.max_instr_len, languages)
    else:
        rows = construct_instrs(args.anno_dir, 'r2r', [options.split], 'bert', max_instr_len=args.max_instr_len)
    if options.ids:
        rows = [r for r in rows if r['instr_id'] in set(options.ids)]
        missing = set(options.ids) - {r['instr_id'] for r in rows}
        if missing:
            sys.exit(f'Not in {options.split}: {" ".join(sorted(missing))}')
    elif options.limit:
        rows = rows[:options.limit]
    elif options.episodes != 'all':
        n = min(int(options.episodes), len(rows))
        rows = [rows[i * len(rows) // n] for i in range(n)]
    args.batch_size = min(args.batch_size, len(rows))
    env = R2RNavBatch(LazyFeatures(args.val_ft_file, args.image_feat_size), rows, args.connectivity_dir,
                      batch_size=args.batch_size, name=options.split)
    agent = LoggingAgent(args, env, rank=0)
    checkpoint = options.checkpoint or CHECKPOINTS / f'duet_vit-{options.model}_ft_best_val_unseen'
    agent.load(str(checkpoint))
    with torch.no_grad():
        agent.test(use_dropout=False, feedback='argmax')
    predictions = agent.get_results(detailed_output=True)

    (out / f'detail_{options.split}.json').write_text(json.dumps(predictions))
    with open(out / f'steps_{options.split}.jsonl', 'w') as f:
        for rec in agent.records.values():
            rec.pop('done')
            f.write(json.dumps(rec) + '\n')
    meta = {'model': options.model, 'checkpoint': str(checkpoint), 'split': options.split,
            'episodes': len(agent.records), 'max_action_len': options.max_action_len,
            'max_instr_len': options.max_instr_len, 'annotations': str(options.anno or ''),
            'languages': options.languages if options.anno and 'jsonl' in options.anno.name else '', 'note': options.note}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    (out / f'meta_{options.split}.json').write_text(json.dumps(meta, indent=1))
    if 'test' not in options.split:
        metrics, _ = env.eval_metrics(predictions)
        print(json.dumps(metrics, indent=1))
    shutil.rmtree(scratch, ignore_errors=True)
    print('step log:', out / f'steps_{options.split}.jsonl')


if __name__ == '__main__':
    main()
