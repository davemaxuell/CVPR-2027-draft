# === Standard library imports ===
import os
import sys

# Add project root to Python path
root_path = os.getcwd()
sys.path.append(root_path)

import json
import time
from collections import defaultdict

# === Third-party library imports ===
import numpy as np
import torch
from transformers import AutoTokenizer

# === Project-specific utility imports ===
from utils.misc import set_random_seed
from utils.logger import (
    write_to_record_file,
    write_results_to_file,
    write_results,
    flops_count,
)
from utils.distributed import (
    init_distributed,
    is_default_gpu,
    all_gather,
    merge_dist_results,
)
from utils.data import ImageFeaturesDB, KMeansPicker
from r2r.data_utils import construct_instrs
from r2r.env import R2RNavBatch
from r2r.parser import parse_args
from r2r.agent import GMapNavAgent
from data_utils import LoadZdict




# === Load a Roberta tokenizer with caching or fallback ===
def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(
            "roberta-base",
            cache_dir=os.path.join(root_path, "datasets", "pretrained", "roberta"),
        )
    except Exception:
        # Local fallback (debugging without Internet)
        return AutoTokenizer.from_pretrained(
            os.path.join(root_path, "datasets", "pretrained", "roberta")
        )

# === Build dataset and auxiliary components ===
def build_dataset(args, rank=0):
    # Determine instruction z-dictionary path
    if args.dataset == "rxr":
        instr_z_file = args.rxr_instr_zdict_roberta_file
    else:
        instr_z_file = args.instr_zdict_file

    instr_zdict_file = (
        args.backdoor_dict_file if len(args.backdoor_dict_file) > 1 else instr_z_file
    )

    # Load precomputed Z dictionaries for interventions
    ZdictReader = LoadZdict(args.img_zdict_file, instr_zdict_file)
    z_dicts = defaultdict(lambda: None)
    if args.do_back_img:
        z_dicts["img_zdict"] = ZdictReader.load_img_tensor()
    if args.do_back_txt:
        z_dicts["instr_zdict"] = ZdictReader.load_instr_tensor()

    # Initialize front-door intervention features if applicable
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        front_feat_file = (
            args.rxr_front_feat_file if args.dataset == "rxr" else args.front_feat_file
        )
        front_feat_loader = KMeansPicker(front_feat_file, n_clusters=args.front_n_clusters)
    else:
        front_feat_loader = None

    # Load primary image features
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    # Optionally include augmented features (e.g., env edits)
    if args.use_aug_env:
        train_feat_db = [feat_db]
        if args.env_edit:
            envedit_feat_db = ImageFeaturesDB(args.aug_img_ft_file_envedit, args.image_feat_size)
            train_feat_db.append(envedit_feat_db)
        if len(train_feat_db) == 1:
            train_feat_db = feat_db
    else:
        train_feat_db = feat_db

    dataset_class = R2RNavBatch
    bert_tok = load_tokenizer()
    instr_tok = bert_tok

    # Choose validation splits
    val_env_names = ["val_seen", "val_unseen"]
    if args.val_unseen:
        val_env_names = ["val_unseen"]

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(args, [split], tok=instr_tok)
        val_env = dataset_class(
            feat_db,
            val_instr_data,
            args.connectivity_dir,
            batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size,
            seed=args.seed + rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size),
            name=split,
            args=args,
            scanvp_cands_file=args.scanvp_cands_file,
        )
        val_envs[split] = val_env

    return val_envs, bert_tok, z_dicts, front_feat_loader

def compute_and_log_gflops(args, agent, score_summary, env_name):
    # 1) model size
    vln_bert_size = sum(p.numel() for p in agent.vln_bert.parameters())

    # 2) average token-lengths
    stats = agent.token_stats
    avg = lambda k: round(sum(stats[k]) / len(stats[k])) if stats[k] else 0
    text_lens      = avg('text_size')
    global_lens    = avg('gmap_size')
    local_lens     = avg('pano_img_size')
    candidate_lens = avg('candidate_size')

    # 3) vocabulary pruning flag
    irrelevant = agent.irrelevant_words_id if 'vocabulary' in args.txt_prune_strat else None

    # 4) flop counts
    lan, pan, nav = flops_count(
        args, agent.vln_bert, args.batch_size,
        text_lens, global_lens, local_lens, candidate_lens,
        h_dim=768, irrelevant_words_id=irrelevant
    )

    # 5) update summary and write
    score_summary.update({
        'lan_gflops':      lan,
        'pan_gflops':      pan,
        'nav_gflops':      nav,
        'vln_bert_size':   vln_bert_size,
    })

    return write_results(env_name, args, score_summary, agent)


# === Run validation ===
def valid(args, val_envs, bert_tok, rank=-1, z_dicts=None, front_feat_loader=None):
    default_gpu = is_default_gpu(args)

    # Initialize navigation agent
    agent = GMapNavAgent(args, val_envs["val_unseen"], bert_tok, rank=rank)
    if args.resume_file is not None:
        print(
            "Loaded the listener model at iter %d from %s"
            % (agent.load(args.resume_file), args.resume_file)
        )

    # Save validation config and initialize log file
    if default_gpu:
        with open(os.path.join(args.log_dir, "validation_args.json"), "w") as outf:
            json.dump(vars(args), outf, indent=4)
        print("Logging to:", os.path.join(args.log_dir, "valid.txt"))
        record_file = os.path.join(args.log_dir, "valid.txt")

    # Load front-door feature dictionary if required
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        if len(args.frontdoor_dict_file) > 0:
            z_front_dict = front_feat_loader.read_tim_tsv(
                args.frontdoor_dict_file, return_dict=True
            )
        else:
            z_front_dict = front_feat_loader.random_pick_front_features()
    else:
        z_front_dict = None

    # Run validation on each split
    for env_name, env in val_envs.items():
        prefix = "submit" if args.detailed_output is False else "detail"
        agent.logs = defaultdict(list)
        agent.env = env

        feedback = "argmax" if args.feedback != "teacher" else "teacher"

        start_time = time.time()
        agent.test(
            use_dropout=False,
            feedback=feedback,
            iters=None,
            z_dicts=z_dicts,
            z_front_dict=z_front_dict,
        )
        print(env_name, "cost time: %.2fs" % (time.time() - start_time))

        # Gather predictions across distributed workers
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if "test" not in env_name:
                # Evaluate and log metrics
                score_summary, _ = env.eval_metrics(preds)
                result_dict = compute_and_log_gflops(args, agent, score_summary, env_name)

                avg_pano_img_size = (
                    sum(agent.token_stats['pano_img_size']) / len(agent.token_stats['pano_img_size'])
                    if len(agent.token_stats['pano_img_size'])>0 else 0
                )

                loss_str = (
                    f"txt_crop: {args.txt_prune_strat} pruning rate: {args.txt_pruning_rate}\n"
                    f"avg pano_img_size: {avg_pano_img_size:.2f}, method: {args.vp_crop_method}, "
                    f"crop_vp_len_pano: {args.crop_vp_len_pano}\nEnv name: {env_name}"
                )
                for metric, val in score_summary.items():
                    loss_str += f", {metric}: {val:.2f}"
                write_to_record_file(loss_str + "\n", record_file)

                # Write CSV results
                write_results_to_file(
                    result_dict,
                    os.path.join(args.log_dir, f"{env_name}_{args.name}_result.csv"),
                )
            else:
                # Just evaluate (no log) for test
                score_summary, _ = env.eval_metrics(preds)

            # Write JSON results if in submission mode
            if args.submit:
                json.dump(
                    preds,
                    open(
                        os.path.join(
                            args.pred_dir,
                            f"{prefix}_{args.name}_{env_name}.json",
                        ),
                        "w",
                    ),
                    sort_keys=True,
                    indent=4,
                    separators=(",", ": "),
                )

# === Main script entry point ===
def main():
    args = parse_args()
    print("Parsed arguments:", vars(args))

    # Initialize distributed training if needed
    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    # Ensure reproducibility
    set_random_seed(args.seed + rank)

    # Build dataset, tokenizer, and feature dictionaries
    val_envs, bert_tok, z_dicts, front_feat_loader = build_dataset(args, rank=rank)

    # Enable gmap pruning based on threshold
    args.gmap_early_pruning = args.gmap_pruning_t_threshold > 0

    # Run validation
    valid(
        args,
        val_envs,
        bert_tok=bert_tok,
        rank=rank,
        z_dicts=z_dicts,
        front_feat_loader=front_feat_loader,
    )

# === Python script entry guard ===
if __name__ == "__main__":
    main()
