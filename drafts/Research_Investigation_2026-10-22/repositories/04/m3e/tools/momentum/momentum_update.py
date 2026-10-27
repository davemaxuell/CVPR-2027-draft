import os
import json
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_ckpt", type=str, required=True)
    parser.add_argument("--new_ckpt", type=str, required=True)
    parser.add_argument("--importance", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def parse_param_name(name: str):
    if 'experts.' not in name:
        return None, None

    parts = name.split('.')
    layer_idx = None
    module_name = None
    expert_idx = None

    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            layer_idx = int(parts[i + 1])
        if p == 'experts' and i + 1 < len(parts):
            expert_idx = int(parts[i + 1])

    target_modules = {'q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'gate_proj', 'up_proj', 'down_proj'}
    for p in parts:
        if p in target_modules:
            module_name = p
            break

    if layer_idx is not None and module_name is not None and expert_idx is not None:
        return "layer%d.%s" % (layer_idx, module_name), expert_idx
    return None, None


def find_task_specific_experts(importance: dict):
    for sample in importance.values():
        num_experts = len([k for k in sample.keys() if k != '-1'])
        if num_experts > 0:
            break
    threshold = 1.0 / max(num_experts, 1)

    result = {}
    for point_name, scores in importance.items():
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top2 = sorted_items[:2]

        task_specific = set()
        for expert_str, score in top2:
            if expert_str == '-1':
                continue
            if score >= threshold:
                task_specific.add(int(expert_str))

        result[point_name] = task_specific

    return result


def finegrained_merge(old_ckpt_path, new_ckpt_path, importance_path, output_path):
    alpha = 0.6
    with open(importance_path) as f:
        importance = json.load(f)

    task_specific = find_task_specific_experts(importance)

    old_ckpt = torch.load(old_ckpt_path, map_location="cpu")
    old_state = old_ckpt["model_state_dict"]

    new_ckpt = torch.load(new_ckpt_path, map_location="cpu")
    new_state = new_ckpt["model_state_dict"]

    merged_state = {}

    for name, new_param in new_state.items():
        point_name, expert_idx = parse_param_name(name)

        if point_name is not None and expert_idx is not None and name in old_state:
            old_param = old_state[name]
            if old_param.shape != new_param.shape:
                merged_state[name] = new_param
                continue

            specific_experts = task_specific.get(point_name, set())

            if expert_idx in specific_experts:
                merged_state[name] = alpha * new_param + (1 - alpha) * old_param
            else:
                merged_state[name] = alpha * old_param + (1 - alpha) * new_param
        else:
            merged_state[name] = new_param

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save({"model_state_dict": merged_state}, output_path)


if __name__ == "__main__":
    args = parse_args()
    finegrained_merge(
        args.old_ckpt, args.new_ckpt, args.importance,
        args.output
    )
