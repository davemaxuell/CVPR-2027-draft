import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from tools.parser import read_args, random_seed
from tasks.loaders import create_dataloaders
from tasks.feature_db import create_feature_db, create_object_feature_db
from models.nav_model import NavModel
from models.moe_lora import MoELoRALayer
from tools.optims import dist_models
from tools.common_utils import all_gather


class RoutingStatsCollector:

    def __init__(self, model, epsilon: float = 0.05):
        self.epsilon = epsilon
        self.stats = {}
        self.hooks = []
        self._register_hooks(model)

    def _find_moe_layers(self, model):
        m = model.module if hasattr(model, 'module') else model
        layers = m.lang_model.model.layers
        module_names = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
        moe_layers = {}
        for layer_idx, layer in enumerate(layers):
            for module_name in module_names:
                if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parent = layer.self_attn
                elif module_name in ("gate_proj", "up_proj", "down_proj"):
                    parent = layer.mlp
                else:
                    continue
                module = getattr(parent, module_name, None)
                if isinstance(module, MoELoRALayer):
                    key = f"layer{layer_idx}.{module_name}"
                    moe_layers[key] = module
        return moe_layers

    def _register_hooks(self, model):
        moe_layers = self._find_moe_layers(model)
        for point_name, module in moe_layers.items():
            self.stats[point_name] = defaultdict(int)
            hook = module.register_forward_hook(
                self._make_hook(point_name)
            )
            self.hooks.append(hook)

    def _make_hook(self, point_name):
        epsilon = self.epsilon
        stats = self.stats

        def hook_fn(module, input, output):
            x = input[0]
            with torch.no_grad():
                router_weights = F.softmax(module.router(x), dim=-1)
                topk_values, topk_indices = torch.topk(router_weights, 2, dim=-1)
                gap = topk_values[..., 0] - topk_values[..., 1]
                invalid_mask = gap < epsilon

                invalid_expanded = invalid_mask.unsqueeze(-1).expand_as(topk_indices)
                topk_indices[invalid_expanded] = -1

                flat_indices = topk_indices.reshape(-1).cpu().tolist()
                local_stats = stats[point_name]
                for idx in flat_indices:
                    local_stats[idx] += 1

        return hook_fn

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def compute_importance(stats: dict, output_path: str):
    importance = {}
    for point in sorted(stats.keys(),
                        key=lambda x: (int(x.split('.')[0].replace('layer', '')),
                                       x.split('.')[1])):
        counts = stats[point]
        total = sum(counts.values())
        if total == 0:
            importance[point] = {str(k): round(1.0 / 7, 4) for k in [-1, 0, 1, 2, 3, 4, 5]}
        else:
            importance[point] = {}
            for expert_id in sorted(counts.keys()):
                importance[point][str(expert_id)] = round(counts[expert_id] / total, 4)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(importance, f, indent=2, ensure_ascii=False)


def merge_dist_results(results):
    outs = []
    for res in results:
        outs.extend(res)
    return outs


@torch.no_grad()
def run_inference(args, global_cfg, model, dataloaders, agents, logger):
    model.eval()
    for name, loader in dataloaders.items():
        dataset = dataloaders[name].get_dataset()
        agent = agents[name]
        preds = agent.validate(
            name, args, global_cfg, model, loader,
        )


def main():
    import sys
    extra_args = {}
    filtered_argv = []
    i = 0
    argv = sys.argv[1:]
    while i < len(argv):
        if argv[i] == '--importance_output':
            extra_args['importance_output'] = argv[i + 1]
            i += 2
        else:
            filtered_argv.append(argv[i])
            i += 1
    sys.argv = [sys.argv[0]] + filtered_argv

    importance_output = extra_args.get('importance_output', '')

    args, global_cfg, logger, device_id = read_args()
    random_seed(args.seed + args.rank)

    if isinstance(args.validation_split, list):
        args.validation_split = args.validation_split[0]

    feat_db = create_feature_db(
        global_cfg.Feature.feature_database,
        global_cfg.Feature.image_feat_size, args
    )
    obj_feat_db = create_object_feature_db(
        global_cfg.Feature.object_database,
        global_cfg.Feature.obj_feat_size, args
    )

    model = NavModel(args, logger, global_cfg.Model)
    model, optimizer, resume_from_epoch, lr_scheduler = dist_models(args, model, logger)

    val_dataloaders, val_agents = create_dataloaders(
        args, global_cfg, logger,
        training=False, device=device_id,
        feat_db=feat_db, obj_feat_db=obj_feat_db, stage="multi"
    )

    collector = RoutingStatsCollector(model, epsilon=0.04)

    run_inference(args, global_cfg, model, val_dataloaders, val_agents, logger)

    if args.rank == 0:
        if importance_output:
            compute_importance(collector.stats, importance_output)

    collector.remove_hooks()


if __name__ == "__main__":
    main()
