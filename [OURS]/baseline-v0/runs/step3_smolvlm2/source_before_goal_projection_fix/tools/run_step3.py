"""Durable single-round DAgger pilot: collect, refine, select on C, freeze reference.

Step 2 artifacts are read-only inputs. No group-B/E training or evaluation occurs.
All children use physical GPU 0. Restart the same command to resume completed work.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/step3_smolvlm2"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / "pipeline.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "status.json").exists() and read(out / "status.json")["status"] == "complete":
        print((out / "status.json").read_text(), flush=True)
        return
    os.chdir(ROOT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HOME="/data/shared/hf_cache", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    parent = ROOT / "runs/step2_smolvlm2"
    selected = read(parent / "best_checkpoint.json")
    parent_adapter, parent_train = Path(selected["checkpoint"]), parent / "train"
    collection, training = DATA / "dagger_groupA_round1_smolvlm2", out / "train"
    plan = {"rounds": 1, "group": "A", "collection_episodes_per_dataset": 128, "collection_seed": 1,
            "beta": 0.0, "recovery_rule": "ordered_reference_waypoints_v1",
            "waypoint_radius_m": 0.5, "goal_radius_m": 1.0,
            "collection": str(collection), "parent_checkpoint": str(parent_adapter),
            "parent_adapter_sha256": sha256(parent_adapter / "adapter_model.safetensors"),
            "max_steps": 2048, "batch_size": 16, "grad_accum": 4, "samples": 131072,
            "sampling": "50% original expert / 50% DAgger, uniform within each with replacement",
            "learning_rate": 5e-5, "optimizer": "fresh AdamW", "seed": 1,
            "checkpoints": [512, 1024, 1536, 2048], "include_parent_for_selection": True,
            "eval_group": "C", "eval_episodes_per_dataset": 32, "eval_seed": 0,
            "selection": "highest dataset-macro success, then macro SPL, then lowest offline token CE, then earliest update",
            "budget_rationale": "One 256-episode collection round across all available A scenes (R2R 44, RxR 42); 65,536 examples from each source in refinement. Approximately 2 h training at Step 2 throughput, plus collection and evaluation.",
            "scope": "Step 3 SmolVLM2 pilot only; no world-model/scorer work and no group-E evaluation"}
    if (out / "plan.json").exists() and read(out / "plan.json") != plan:
        raise ValueError("Existing Step 3 plan differs")
    write(out / "plan.json", plan)
    assert read(out / "check_dagger.json")["status"] == "passed"
    for check, count in (("check_expert", 88), ("check_recovery", 16), ("check_policy", 8)):
        report = read(out / check / "summary.json")
        assert report["status"] == "complete" and report["completed_episodes"] == count
        if check != "check_policy":
            assert sum(d["successes"] for d in report["by_dataset"].values()) == count
            assert not report["label_errors"]
    assert read(out / "check_training.json")["status"] == "passed"
    files = ["tools/run_step3.py", "tools/collect_dagger.py", "tools/train_policy_dagger.py", "tools/check_dagger.py",
             "tools/train_policy_sft.py", "tools/eval_policy.py", "policy/dagger_data.py", "policy/data.py",
             "policy/common.py", "policy/smolvlm2.py", "policy/online.py", "sim/recovery.py", "sim/config.py",
             "sim/episodes.py", "sim/rxr_dataset.py", "manifest/scene_partition.json", "manifest/model_revisions.json"]
    hashes = {name: sha256(ROOT / name) for name in files}
    if (out / "source_sha256.json").exists() and read(out / "source_sha256.json") != hashes:
        raise ValueError("Step 3 source changed; review before resuming")
    write(out / "source_sha256.json", hashes)
    for name in files + ["manifest/experiment_manifest.yaml", "env/setup_env.sh", "docs/DAgger_Run.md"]:
        target = out / "source" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
    state = {"status": "running", "pid": os.getpid(), "started_or_resumed_utc": now()}
    child = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Step 3 interrupted")
    signal.signal(signal.SIGTERM, interrupt)

    def stage(name, command, logfile):
        nonlocal child
        state.update(stage=name, command=command, updated_utc=now())
        write(out / "status.json", state)
        print(json.dumps(state), flush=True)
        with open(logfile, "a") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            write(out / "status.json", state)
            code = child.wait()
        child = None
        state.pop("child_pid", None)
        if code:
            raise RuntimeError(f"{name} failed with exit {code}; see {logfile}")

    try:
        stage("collecting", [sys.executable, "-u", "tools/collect_dagger.py", "--out", str(collection),
              "--run-dir", str(parent_train), "--checkpoint", str(parent_adapter), "--episodes", "128",
              "--seed", "1", "--workers", "4"], out / "collect.stdout.log")
        corpus = read(collection / "summary.json")
        assert corpus["completed_episodes"] == 256 and corpus["valid_labels"] > 0
        if corpus["valid_labels"] / corpus["actions"] < 0.99:
            raise RuntimeError("More than 1% of correction labels are invalid; review the expert before refinement")
        assert corpus["by_dataset"]["r2r"]["scenes"] == 44 and corpus["by_dataset"]["rxr"]["scenes"] == 42
        checkpoints = sorted(p for p in (training / "checkpoints").glob("step_*") if (p / "trainer_state.pt").exists())
        last = int(checkpoints[-1].name.split("_")[-1]) if checkpoints else 0
        if last < plan["max_steps"]:
            command = [sys.executable, "-u", "tools/train_policy_dagger.py", "--run-dir", str(training),
                       "--dagger-root", str(collection), "--init-adapter", str(parent_adapter), "--init-run", str(parent_train),
                       "--val-root", str(DATA / "expert_groupC_600"), "--max-steps", "2048", "--batch-size", "16",
                       "--grad-accum", "4", "--workers", "24", "--val-workers", "24", "--seed", "1", "--lr", "5e-5",
                       "--log-every", "32", "--eval-every", "512", "--eval-batches", "125", "--eval-initial", "--save-every", "512"]
            if checkpoints:
                command += ["--resume", str(checkpoints[-1])]
            stage("training", command, out / "train.stdout.log")
        validation = {r["step"]: r for r in map(json.loads, (training / "log.jsonl").read_text().splitlines()) if "val_loss_per_token" in r}
        assert abs(validation[0]["val_loss_per_token"] - selected["offline"]["val_loss_per_token"]) < 1e-5
        parent_evaluation = Path(selected["evaluation"])
        parent_eval_config = read(parent_evaluation / "eval_config.json")
        assert parent_eval_config["adapter_sha256"] == plan["parent_adapter_sha256"]
        assert parent_eval_config["evaluator_sha256"] == hashes["tools/eval_policy.py"]
        comparisons = [{"step": 0, "name": "Step 2 parent (3072)", "checkpoint": str(parent_adapter),
                        "run_dir": str(parent_train), "evaluation": str(parent_evaluation),
                        "offline": selected["offline"], "closed_loop": selected["closed_loop"]}]
        for step in plan["checkpoints"]:
            checkpoint = training / "checkpoints" / f"step_{step:06d}"
            evaluated = out / "evaluations" / checkpoint.name
            if not (evaluated / "summary.json").exists():
                stage(f"evaluating_{step}", [sys.executable, "-u", "tools/eval_policy.py", "--run-dir", str(training),
                      "--checkpoint", str(checkpoint), "--episodes", "32", "--seed", "0", "--out", str(evaluated)],
                      out / f"eval_{step:06d}.stdout.log")
            result, config = read(evaluated / "summary.json"), read(evaluated / "eval_config.json")
            assert result["status"] == "complete" and result["overall"]["episodes"] == 64
            assert config["selected_episodes"] == parent_eval_config["selected_episodes"]
            assert config["simulator"] == parent_eval_config["simulator"]
            assert config["policy_config"] == parent_eval_config["policy_config"]
            assert config["adapter_sha256"] == sha256(checkpoint / "adapter_model.safetensors")
            comparisons.append({"step": step, "name": f"DAgger {step}", "checkpoint": str(checkpoint),
                                "run_dir": str(training), "evaluation": str(evaluated),
                                "offline": validation[step], "closed_loop": result})
            write(out / "checkpoint_comparison.json", comparisons)
        best = max(comparisons, key=lambda c: (c["closed_loop"]["macro_success"], c["closed_loop"]["macro_spl"],
                                               -c["offline"]["val_loss_per_token"], -c["step"]))
        write(out / "best_checkpoint.json", {**best, "selection_rule": plan["selection"]})
        frozen = out / "frozen_reference"
        temporary = out / ".frozen_reference.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        (temporary / "adapter").mkdir(parents=True)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(Path(best["checkpoint"]) / name, temporary / "adapter" / name)
        shutil.copy2(Path(best["run_dir"]) / "policy_config.json", temporary / "policy_config.json")
        shutil.copytree(Path(best["run_dir"]) / "processor", temporary / "processor")
        for name in ("policy/common.py", "policy/smolvlm2.py", "policy/online.py"):
            target = temporary / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write(temporary / "inference.json", {"status": "frozen", "frozen_utc": now(), "selected": best["name"],
              "source_checkpoint": best["checkpoint"], "source_evaluation": best["evaluation"],
              "weights": "backbone and all LoRA adapters frozen; evaluation mode",
              "policy_config": read(temporary / "policy_config.json"), "action_ids": ["STOP", "FORWARD", "LEFT", "RIGHT"],
              "schedule": "one primitive action per model forward; deterministic first-token argmax; batch size 1",
              "simulator": parent_eval_config["simulator"], "partition_sha256": hashes["manifest/scene_partition.json"],
              "source_sha256": hashes, "macro_success": best["closed_loop"]["macro_success"],
              "macro_spl": best["closed_loop"]["macro_spl"],
              "versioning": "Later continuation labels and policy features must identify this bundle; replacing it requires recollection."})
        bundle_hashes = {str(p.relative_to(temporary)): sha256(p) for p in sorted(temporary.rglob("*")) if p.is_file()}
        write(temporary / "sha256.json", bundle_hashes)
        if frozen.exists():
            assert sha256(frozen / "adapter/adapter_model.safetensors") == sha256(temporary / "adapter/adapter_model.safetensors")
            for name, expected in read(frozen / "sha256.json").items():
                assert sha256(frozen / name) == expected
            shutil.rmtree(temporary)
        else:
            temporary.rename(frozen)
        link = out / "best_checkpoint"
        if link.is_symlink():
            link.unlink()
        link.symlink_to("frozen_reference/adapter", target_is_directory=True)
        report = ["# Step 3 completed — SmolVLM2 DAgger pilot", "", f"Completed: {now()}", "",
                  f"Collected {corpus['completed_episodes']} group-A episodes across all available A buildings (44 R2R, 42 RxR): {corpus['actions']:,} policy actions, {corpus['valid_labels']:,} usable corrective labels.",
                  "The Step 2 policy drove every action (beta=0). Collection used the validated Step 2 expert episode pool. Corrective labels followed the actual replay path, with monotonic progress, 0.5 m waypoint tolerance and 1 m final STOP tolerance. Invalid labels were excluded; failed policy episodes were retained.",
                  "Refinement: 2,048 updates × 64 examples = 131,072 samples; 65,536 from each of the original expert and DAgger sources. Decoder LoRA only, learning rate 5e-5 with a fresh AdamW optimizer and warmup/cosine schedule.",
                  "Selection: the same 32 R2R + 32 RxR group-C episodes, including the Step 2 checkpoint as a candidate. No group E use.", "",
                  "| Candidate | Offline action accuracy | Offline CE | Macro success | Macro SPL |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for c in comparisons:
            report.append(f"| {c['name']} | {c['offline']['val_action_acc']:.4f} | {c['offline']['val_loss_per_token']:.4f} | {c['closed_loop']['macro_success']:.4f} | {c['closed_loop']['macro_spl']:.4f} |")
        report += ["", f"Frozen reference: **{best['name']}**, in `frozen_reference/` (adapter, processor, policy configuration, prompt/decoding source and file hashes).",
                   "Selection rule: " + plan["selection"] + ".", "",
                   "These are small development-set pilot results, not final benchmark results. RxR uses 15-degree turns, differing from its official 30-degree protocol.",
                   "World-model fitting and scorer work have not started. The next step is group-A transition collection for the world model.", ""]
        (out / "RESULTS.md").write_text("\n".join(report))
        state.update(status="complete", stage="complete", completed_utc=now(), frozen_reference=str(frozen),
                     selected=best["name"], macro_success=best["closed_loop"]["macro_success"], macro_spl=best["closed_loop"]["macro_spl"])
        state.pop("command", None)
        write(out / "status.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state.update(status="failed", error=repr(exc), updated_utc=now())
        write(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
