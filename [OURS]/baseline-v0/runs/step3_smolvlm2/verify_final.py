"""Audit completed Step 3 artifacts without changing training or selection."""
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", "/data/shared/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, abs_tol=1e-10), (actual, expected)


def main():
    status = read(RUN / "status.json")
    assert status["status"] == "complete", "Wait for all simulator evaluations"
    plan = read(RUN / "plan.json")
    comparisons = read(RUN / "checkpoint_comparison.json")
    assert [c["step"] for c in comparisons] == [0] + plan["checkpoints"]
    partition = read(ROOT / "manifest/scene_partition.json")
    scenes = set(partition["groups"]["C"]["scenes"])
    source_counts = {}
    for name in ("step2_smolvlm2", "step3_smolvlm2"):
        source = read(ROOT / "runs" / name / "source_sha256.json")
        for relative, expected in source.items():
            assert digest(ROOT / relative) == expected, (name, relative)
            assert digest(ROOT / "runs" / name / "source" / relative) == expected, ("snapshot", name, relative)
        source_counts[name] = len(source)

    reference = read(Path(comparisons[0]["evaluation"]) / "eval_config.json")
    expected_ids = {(name, eid) for name, ids in reference["selected_episodes"].items() for eid in ids}
    assert len(expected_ids) == 64
    results = []
    for candidate in comparisons:
        directory = Path(candidate["evaluation"])
        cfg, summary = read(directory / "eval_config.json"), read(directory / "summary.json")
        assert summary == candidate["closed_loop"]
        assert summary["status"] == "complete" and summary["group"] == "C"
        for key in ("controller", "group", "split", "seed", "selected_episodes", "simulator", "policy_config",
                    "evaluator_sha256", "partition_sha256", "selection_sha256", "policy_source_sha256",
                    "online_source_sha256", "decoding_batch_size", "pixel_preprocessing", "rgb_source"):
            assert cfg[key] == reference[key], (candidate["name"], key)
        assert cfg["adapter_sha256"] == digest(Path(candidate["checkpoint"]) / "adapter_model.safetensors")
        rows = [read(p) for p in sorted((directory / "episodes").glob("*.json"))]
        assert len(rows) == 64
        assert {(r["dataset"], r["episode_id"]) for r in rows} == expected_ids
        for row in rows:
            assert row["scene"] in scenes
            count = row["steps"]
            actions = row["executed_actions"]
            assert 0 < count <= cfg["simulator"]["max_episode_steps"][row["dataset"]]
            assert len(actions) == len(row["collided_after_action"]) == count
            assert len(row["positions"]) == len(row["distances_to_goal"]) == count + 1
            assert all(a in (0, 1, 2, 3) for a in actions)
            assert 0 not in actions[:-1]
            assert row["stopped"] == (actions[-1] == 0)
            assert row["collisions"] == sum(row["collided_after_action"])
            close(row["distance_to_goal"], row["distances_to_goal"][-1])
            assert row["success"] == float(row["stopped"] and row["distance_to_goal"] < cfg["simulator"]["success_distance_m"])
            assert row["oracle_success"] == float(min(row["distances_to_goal"]) < cfg["simulator"]["success_distance_m"])
        keys = ("success", "spl", "distance_to_goal", "oracle_success", "steps", "collisions", "stopped")
        for name, subset in [("overall", rows)] + [(n, [r for r in rows if r["dataset"] == n]) for n in ("r2r", "rxr")]:
            stored = summary["overall"] if name == "overall" else summary["by_dataset"][name]
            assert stored["episodes"] == len(subset)
            assert stored["scenes"] == len({r["scene"] for r in subset})
            for key in keys:
                close(stored[key], mean(r[key] for r in subset))
        close(summary["macro_success"], mean(summary["by_dataset"][n]["success"] for n in ("r2r", "rxr")))
        close(summary["macro_spl"], mean(summary["by_dataset"][n]["spl"] for n in ("r2r", "rxr")))
        results.append({"candidate": candidate["name"], "episodes": len(rows),
                        "successes": int(sum(r["success"] for r in rows)),
                        "macro_success": summary["macro_success"], "macro_spl": summary["macro_spl"],
                        "adapter_sha256": cfg["adapter_sha256"]})

    best = max(comparisons, key=lambda c: (c["closed_loop"]["macro_success"], c["closed_loop"]["macro_spl"],
                                         -c["offline"]["val_loss_per_token"], -c["step"]))
    selected = read(RUN / "best_checkpoint.json")
    assert {k: selected[k] for k in best} == best
    assert status["selected"] == best["name"]
    frozen = RUN / "frozen_reference"
    hashes = read(frozen / "sha256.json")
    assert {str(p.relative_to(frozen)) for p in frozen.rglob("*") if p.is_file() and p.name != "sha256.json"} == set(hashes)
    for relative, expected in hashes.items():
        assert digest(frozen / relative) == expected, relative
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        assert digest(frozen / "adapter" / name) == digest(Path(best["checkpoint"]) / name)
    assert digest(frozen / "policy_config.json") == digest(Path(best["run_dir"]) / "policy_config.json")
    processor_files = {str(p.relative_to(frozen / "processor")) for p in (frozen / "processor").rglob("*") if p.is_file()}
    assert processor_files == {str(p.relative_to(Path(best["run_dir"]) / "processor")) for p in (Path(best["run_dir"]) / "processor").rglob("*") if p.is_file()}
    for relative in processor_files:
        assert digest(frozen / "processor" / relative) == digest(Path(best["run_dir"]) / "processor" / relative)
    for path in (frozen / "source").rglob("*.py"):
        assert digest(path) == digest(ROOT / path.relative_to(frozen / "source"))
    assert (RUN / "best_checkpoint").resolve() == frozen / "adapter"
    inference = read(frozen / "inference.json")
    assert inference["status"] == "frozen" and inference["selected"] == best["name"]
    assert inference["source_checkpoint"] == best["checkpoint"]
    close(inference["macro_success"], best["closed_loop"]["macro_success"])
    close(inference["macro_spl"], best["closed_loop"]["macro_spl"])

    log = [json.loads(line) for line in (RUN / "train/log.jsonl").read_text().splitlines()]
    final = [row for row in log if "samples_seen" in row][-1]
    assert final["step"] == plan["max_steps"] == 2048
    assert final["samples_seen"] == 131072 and final["source_samples_each"] == 65536
    assert all(math.isfinite(r[k]) for r in log for k in ("loss_per_token", "grad_norm", "val_loss_per_token") if k in r)
    assert read(RUN / "check_collection_integrity.json")["status"] == "passed"
    collection = read(Path(plan["collection"]) / "summary.json")
    assert collection["completed_episodes"] == 256 and collection["valid_labels"] == collection["actions"] == 125730

    from policy.common import PolicyConfig
    from policy.smolvlm2 import SmolVLM2Policy
    policy = SmolVLM2Policy(PolicyConfig.load(frozen / "policy_config.json"), adapter=str(frozen / "adapter"), device="cuda:0")
    policy.model.eval()
    trainable = sum(p.numel() for p in policy.model.parameters() if p.requires_grad)
    assert trainable == 0 and not any(m.training for m in policy.model.modules())
    versions = {}
    for package in ("torch", "transformers", "peft", "tokenizers", "numpy", "habitat-lab", "habitat-sim"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not provided in distribution metadata"
    report = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": digest(__file__), "source_files_verified": source_counts,
              "evaluations": results, "identical_group_C_episode_ids_and_protocol": True,
              "metrics_recomputed_from_episode_records": True, "selection_rule_verified": True,
              "frozen_files_verified": len(hashes), "processor_and_adapter_match_selected_checkpoint": True,
              "frozen_model_loaded_on_gpu": True, "trainable_parameters": trainable, "evaluation_mode": True,
              "training_updates": final["step"], "training_samples": final["samples_seen"],
              "source_samples_each": final["source_samples_each"], "finite_training_metrics": True,
              "selected": best["name"], "software": versions}
    temporary = RUN / "final_verification.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(RUN / "final_verification.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
