# === Standard library imports ===
import os
import sys
import json
import time
from collections import defaultdict

# Add current working directory to Python path
sys.path.append(os.getcwd())

# === Third-party imports ===
import numpy as np
import torch
import nltk
from transformers import AutoTokenizer

# === Project-specific utility imports ===
from utils.misc import set_random_seed
from utils.logger import (
    flops_count,
    write_to_record_file,
    write_results_to_file,
    write_results_reverie,
)
from utils.distributed import (
    init_distributed,
    is_default_gpu,
    all_gather,
    merge_dist_results,
)
from utils.data import ImageFeaturesDB, KMeansPicker
from r2r.data_utils import LoadZdict

# === REVERIE-specific imports ===
from reverie.agent_obj_goat import GMapObjectNavAgent
from reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps
from reverie.env import ReverieObjectNavBatch
from reverie.parser import parse_args


# === Load REVERIE validation datasets, tokenizer, and feature handlers ===
def build_dataset(args, rank=0):
    """
    Initialize tokenizer, feature databases, instruction z-dictionaries, and frontdoor feature pickers.
    Returns a dict of validation environments and utility objects.
    """

    # Select tokenizer config
    tokenizer_map = {"xlm": "xlm-roberta-base", "roberta": "roberta-base"}
    cfg_name = tokenizer_map.get(args.tokenizer, "bert-base-uncased")

    # Load tokenizer (with fallback path if needed)
    try:
        tok = AutoTokenizer.from_pretrained(cfg_name)
    except Exception:
        tok = AutoTokenizer.from_pretrained(os.path.join("..", cfg_name))

    # Load Z-dictionaries for backdoor or baseline
    instr_zdict_file = (
        args.backdoor_dict_file if args.backdoor_dict_file else args.instr_zdict_file
    )
    ZdictReader = LoadZdict(args.img_zdict_file, instr_zdict_file)
    z_dicts = defaultdict(lambda: None)
    if args.do_back_img:
        z_dicts["img_zdict"] = ZdictReader.load_img_tensor()
    if args.do_back_txt:
        z_dicts["instr_zdict"] = ZdictReader.load_instr_tensor()

    # Initialize KMeans frontdoor feature picker (optional)
    front_feat_loader = (
        KMeansPicker(args.front_feat_file, n_clusters=args.front_n_clusters)
        if args.do_front_img or args.do_front_his or args.do_front_txt
        else None
    )

    # Load feature databases for images and object boxes
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(
        args.obj_ft_file, args.obj_feat_size, cat_file=args.cat_file
    )
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, "BBoxes.json"))

    # Determine validation splits to load
    val_envs = {}
    val_splits = ["val_unseen"] if args.mode == "valid" else ["val_seen", "val_unseen"]

    for split in val_splits:
        val_instr_data = construct_instrs(
            args.anno_dir,
            args.dataset,
            [split],
            tokenizer=args.tokenizer,
            max_instr_len=args.max_instr_len,
        )

        val_envs[split] = ReverieObjectNavBatch(
            feat_db,
            obj_db,
            val_instr_data,
            args.connectivity_dir,
            obj2vps,
            batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size,
            seed=args.seed + rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size),
            name=split,
            max_objects=None,
            multi_endpoints=False,
            multi_startpoints=False,
            tok=tok,
            args=args,
            scanvp_cands_file=args.scanvp_cands_file,
        )

    return val_envs, tok, z_dicts, front_feat_loader

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
    object_lens   = avg('obj_size')

    # 3) vocabulary pruning flag
    irrelevant = agent.irrelevant_words_id if 'vocabulary' in args.txt_prune_strat else None

    # 4) flop counts
    lan, pan, nav = flops_count(
        args, agent.vln_bert, args.batch_size,
        text_lens, global_lens, local_lens, candidate_lens,
        h_dim=768, obj_size= object_lens, irrelevant_words_id=irrelevant
    )

    # 5) update summary and write
    score_summary.update({
        'lan_gflops':      lan,
        'pan_gflops':      pan,
        'nav_gflops':      nav,
        'vln_bert_size':   vln_bert_size,
    })
    return write_results_reverie(env_name, args, score_summary, agent)


# === Run validation ===
def valid(args, val_envs, bert_tok, rank=-1, z_dicts=None, front_feat_loader=None):
    """
    Evaluate agent on validation environments and optionally dump predictions for submission.
    """
    z_dicts = z_dicts or {}
    default_gpu = is_default_gpu(args)

    # Initialize the agent and optionally load pretrained weights
    agent = GMapObjectNavAgent(args, val_envs["val_unseen"], bert_tok, rank=rank)
    if args.resume_file:
        print(
            f"Loaded the listener model at iter {agent.load(args.resume_file)} from {args.resume_file}"
        )

    # Record args and prepare log files on default GPU
    if default_gpu:
        with open(os.path.join(args.log_dir, "validation_args.json"), "w") as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, "valid.txt")
        write_to_record_file(str(args) + "\n\n", record_file)

    # Load frontdoor intervention dict (optional)
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        z_front_dict = (
            front_feat_loader.read_tim_tsv(args.frontdoor_dict_file, return_dict=True)
            if args.frontdoor_dict_file
            else front_feat_loader.random_pick_front_features()
        )
    else:
        z_front_dict = None

    # Evaluate on each validation split
    for env_name, env in val_envs.items():
        prefix = "submit" if not args.detailed_output else "detail"
        agent.logs.clear()
        agent.env = env

        print(f"Starting validation on {env_name}...")
        start_time = time.time()
        agent.test(
            use_dropout=False,
            feedback="argmax",
            iters=None,
            z_dicts=z_dicts,
            z_front_dict=z_front_dict,
        )
        print(f"{env_name} cost time: {time.time() - start_time:.2f}s")

        preds = merge_dist_results(all_gather(agent.get_results(args.detailed_output)))

        if default_gpu:
            # Evaluate predictions
            score_summary, _ = env.eval_metrics(preds)
            result_dict = compute_and_log_gflops(args, agent, score_summary, env_name)

            result_dict = write_results_reverie(env_name, args, score_summary, agent)
            write_results_to_file(
                result_dict,
                os.path.join(args.log_dir, f"{args.name}_valid_results_FastV.csv"),
            )

            avg_pano_img_size = (
                sum(agent.token_stats["pano_img_size"])
                / len(agent.token_stats["pano_img_size"])
                if agent.token_stats["pano_img_size"]
                else 0
            )

            # Log pruning and evaluation info
            log_str = (
                f"txt_crop: {args.txt_prune_strat}, pruning rate: {args.txt_pruning_rate}\n"
                f"avg pano_img_size: {avg_pano_img_size:.2f}, method: {args.vp_crop_method}, "
                f"crop_vp_len_pano: {args.crop_vp_len_pano}\n"
                f"Env name: {env_name}"
            )
            for metric, value in score_summary.items():
                log_str += f", {metric}: {value:.2f}"
            write_to_record_file(log_str + "\n", record_file)

            # Write detailed CSV metrics
            write_results_to_file(
                result_dict,
                os.path.join(args.log_dir, f"{env_name}_{args.name}_result.csv"),
            )

            # Dump predictions in JSON format (for competition submission)
            if args.submit:
                with open(
                    os.path.join(
                        args.pred_dir, f"{prefix}_{args.name}_{env_name}.json"
                    ),
                    "w",
                ) as f:
                    json.dump(
                        preds, f, indent=4, sort_keys=True, separators=(",", ": ")
                    )


# === Main entrypoint ===
def main():
    """
    Main script function. Parses args, builds environments, and runs validation.
    """
    args = parse_args()

    # Configure NLTK WordNet
    nltk_cache = "/root/mount/Matterport3DSimulator/VLN-GOAT/cache"
    nltk.data.path.append(nltk_cache)
    nltk.download("wordnet", download_dir=nltk_cache)

    print("Parsed arguments:", vars(args))

    # Initialize distributed backend if needed
    rank = init_distributed(args) if args.world_size > 1 else 0
    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)

    # Set random seed for reproducibility
    set_random_seed(args.seed + rank)

    # Prepare validation dataset and tokenizers
    val_envs, bert_tok, z_dicts, front_feat_loader = build_dataset(args, rank=rank)
    args.gmap_early_pruning = args.gmap_pruning_t_threshold > 0

    # Run validation and optionally write results
    valid(
        args,
        val_envs,
        bert_tok=bert_tok,
        rank=rank,
        z_dicts=z_dicts,
        front_feat_loader=front_feat_loader,
    )


# === Script entry guard ===
if __name__ == "__main__":
    main()
