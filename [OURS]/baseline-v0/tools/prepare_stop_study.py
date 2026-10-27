"""Lock development and confirmation populations before inspecting new outcomes."""
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/step3b_smolvlm2_stop"
DATA = Path("/data/team_a/dave-workspace/RoleVLN/data")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Locked artifact changed: {path}"
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = ROOT / "runs/step3_smolvlm2/evaluations/step_001024/eval_config.json"
    old = json.loads(previous.read_text())["selected_episodes"]
    write_once(OUT / "development.json", {"group": "C", "selected_episodes": old,
               "purpose": "Calibration and refinement selection only; existing 64 episodes", "source_sha256": digest(previous)})
    group = sorted(json.loads((ROOT / "manifest/scene_partition.json").read_text())["groups"]["C"]["scenes"])
    files = {"r2r": DATA / "datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz",
             "rxr": DATA / "datasets/RxR_VLNCE_v0/train/train_guide.json.gz"}
    available, exclusions, source_hashes = {}, {}, {}
    for name, path in files.items():
        source_hashes[name] = digest(path)
        with gzip.open(path, "rt") as stream:
            rows = json.load(stream)["episodes"]
        rows = [r for r in rows if r["scene_id"].split("/")[-2] in group
                and (name == "r2r" or r["instruction"]["language"] in ("en-US", "en-IN"))]
        seen = set(old[name]) | {p.parent.name for p in (DATA / "processed/baseline_v0/expert_groupC_600" / name).glob("*/meta.json")}
        excluded_trajectories = {str(r["trajectory_id"]) for r in rows if str(r["episode_id"]) in seen}
        pools = {s: defaultdict(list) for s in group}
        for row in rows:
            if str(row["trajectory_id"]) not in excluded_trajectories:
                pools[row["scene_id"].split("/")[-2]][str(row["trajectory_id"])].append(row)
        available[name] = pools
        exclusions[name] = {"episode_ids": sorted(seen), "trajectory_ids": sorted(excluded_trajectories),
                            "remaining_distinct_trajectories_by_scene": {s: len(pools[s]) for s in group}}
    # One C building has no untouched R2R trajectories after prior offline validation.
    # Exclude it by this outcome-independent availability rule; retain equal scene/dataset counts.
    eligible = [s for s in group if all(len(available[n][s]) >= 32 for n in files)]
    assert len(eligible) >= 7, eligible
    selected, records = {"A": {n: [] for n in files}, "B": {n: [] for n in files}}, {"A": [], "B": []}
    rng = random.Random(20261025)
    for name in files:
        for scene in eligible:
            trajectories = sorted(available[name][scene])
            rng.shuffle(trajectories)
            for index, trajectory in enumerate(trajectories[:32]):
                pool = sorted(available[name][scene][trajectory], key=lambda r: str(r["episode_id"]))
                episode = rng.choice(pool)
                which = "A" if index < 16 else "B"
                eid = str(episode["episode_id"])
                selected[which][name].append(eid)
                records[which].append({"dataset": name, "episode_id": eid, "trajectory_id": trajectory, "scene": scene})
    for which in ("A", "B"):
        write_once(OUT / f"confirmation_{which}.json", {
            "group": "C", "selected_episodes": selected[which], "records": records[which],
            "purpose": "A: first locked confirmation. B: reserved only if calibration fails A and training is then needed.",
            "seed": 20261025, "scenes": eligible, "excluded_scenes": sorted(set(group) - set(eligible)),
            "exclusion_rule": "Exclude all previously rendered C validation episodes and the old C64, including their trajectory IDs; unique trajectory per dataset; outcome-independent scene availability.",
            "source_sha256": source_hashes, "exclusions": exclusions})
    plan = {
        "scope": "One bounded STOP repair study; no world-model/scorer fitting, no group E",
        "diagnostic_checkpoints": ["step3/1024", "step3/2048"],
        "calibration": {"family": "additive STOP-logit bias; all other logits unchanged; exact replay on actual movement-only carrier continuations",
                        "bias_grid": [i / 8 for i in range(-32, 49)], "development_episodes": 64,
                        "selection": "maximize macro SR, then SPL, then lower premature-stop rate, then lower bias",
                        "sufficient_on_development": {"macro_success_at_least": 0.20, "spl_at_least_step2": True,
                                                      "premature_stop_rate_at_most_step2": True}},
        "conditional_training": {"start": "runs/step3_smolvlm2/train/checkpoints/step_001024",
            "teacher": "goal-first STOP when geodesic task-goal distance <2m, regardless of route progress; ordered-route movement outside",
            "collection_group": "A", "episodes_per_dataset": 64, "collection_seed": 2, "policy_beta": 0.0,
            "original_round1_used_for_training": False, "retain_expert_navigation": True,
            "batch_categories": {"expert_stop": 5, "round2_stop": 1, "hard_nonstop": 10, "expert_navigation": 32, "round2_navigation": 16},
            "all_sources_use_same_2m_labels": True, "dedup": "cap repeated identical pose+rotation+label states; turns with changed view retained",
            "max_policy_stop_examples_per_episode": 8, "max_expert_stop_examples_per_episode": 2,
            "hard_negatives": "outside 2m STOP region; 3-5m states and policy incorrect-STOP states",
            "updates": 2048, "batch_size": 16, "grad_accum": 4, "learning_rate": 2e-5,
            "optimizer": "fresh AdamW", "seed": 3, "loss": "existing masked target-token cross-entropy",
            "checkpoint_selection_updates": [512, 1024, 2048], "selection_decoder_bias_grid_same_as_calibration": True},
        "confirmation": {"episodes": len(records["A"]), "scenes": eligible,
            "controls": ["step2/3072", "step3/1024"], "extra_control_if_calibration_uses_2048": "step3/2048",
            "pass": {"absolute_macro_SR_gain_vs_each_control_at_least": 0.05,
                     "paired_scene_bootstrap_SR_difference_lower95_above_zero_vs_each_control": True,
                     "macro_SPL_at_least_step2": True, "premature_stop_rate_at_most_step2": True,
                     "mean_nDTW_drop_vs_unmodified_step3_1024_at_most": 0.05},
            "bootstrap": "50000 paired scene-resampling replicates, seed 20261025; finite seven-scene pilot uncertainty",
            "freeze_candidate_and_decoding_before_confirmation": True,
            "if_calibration_fails_A": "one predeclared training round, then use untouched B; no training/tuning on A",
            "if_training_fails": "report failure under the locked pass bar; no third round"},
        "protocol": {"native_batch_one_decoding": True, "rgb": "live simulator", "turn_degrees": 15,
                     "rxr_note": "15 degrees differs from official 30-degree RxR-CE", "success_distance_m": 3.0,
                     "ndtw": "exact DTW with official dense GT locations and Euclidean local distance, normalized by 3m"}}
    write_once(OUT / "plan.json", plan)
    print(json.dumps({"development": 64, "confirmation_each": len(records["A"]), "scenes": eligible,
                      "excluded": sorted(set(group) - set(eligible)), "plan": str(OUT / "plan.json")}))


if __name__ == "__main__":
    main()
