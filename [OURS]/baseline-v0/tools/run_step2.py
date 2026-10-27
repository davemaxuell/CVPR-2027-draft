"""Durable Step 2 pilot: SFT, fixed group-C rollouts, and checkpoint selection.

Restart the same command to resume from the latest complete training checkpoint
and already-written evaluation episodes. Uses GPU 0 and never touches group E.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/step2_smolvlm2"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / "pipeline.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(ROOT)
    env = dict(os.environ, HF_HOME="/data/shared/hf_cache", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    env.pop("CUDA_VISIBLE_DEVICES", None)  # Habitat uses physical EGL device 0.
    training = out / "train"
    validation = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupC_600")
    plan = {"max_steps": 4096, "batch_size": 16, "grad_accum": 4, "samples": 262144,
            "fraction_of_pass": 262144 / 2218976, "seed": 0, "checkpoints": [1024, 2048, 3072, 4096],
            "eval_episodes_per_dataset": 32, "eval_group": "C",
            "selection": "highest dataset-macro success, then macro SPL, then lowest offline token CE, then earliest step",
            "budget_rationale": "21.11 samples/s measured: full pass 29.2 h. Pilot budget ~3.5 h plus validation and simulator rollouts.",
            "scope": "SmolVLM2 Step 2 only; no DAgger, world-model/scorer fitting, or group-E evaluation"}
    if (out / "plan.json").exists():
        if json.loads((out / "plan.json").read_text()) != plan:
            raise ValueError("Existing plan differs; use a different output directory")
    else:
        write_json(out / "plan.json", plan)
    required_checks = ["check_selective_logits.json", "check_online_inputs.json", "check_eval_expert/summary.json"]
    for check in required_checks:
        if not (out / check).exists():
            raise RuntimeError(f"Missing prerequisite check: {check}")
    render = json.loads((validation / "render_manifest.json").read_text())
    assert render["args"]["group"] == "C"
    assert all(render["rendered"][d]["episodes"] == 300 for d in ["r2r", "rxr"])
    files = ["policy/smolvlm2.py", "policy/common.py", "policy/data.py", "policy/online.py",
             "tools/train_policy_sft.py", "tools/eval_policy.py", "tools/run_step2.py",
             "sim/config.py", "sim/episodes.py", "manifest/scene_partition.json", "manifest/model_revisions.json"]
    provenance = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
    if (out / "source_sha256.json").exists():
        if json.loads((out / "source_sha256.json").read_text()) != provenance:
            raise ValueError("Run source changed; review provenance before resuming")
    else:
        write_json(out / "source_sha256.json", provenance)
    state = {"status": "running", "pid": os.getpid(), "started_or_resumed_utc": now()}

    def stage(name, command, logfile):
        state.update(stage=name, command=command, updated_utc=now())
        write_json(out / "status.json", state)
        print(json.dumps(state), flush=True)
        child_env = dict(env)
        if name == "training":
            # PyTorch-first imports otherwise miscount this host's mixed full/MIG
            # devices. Simulator children keep physical EGL numbering unchanged.
            child_env["CUDA_VISIBLE_DEVICES"] = "0"
        with open(logfile, "a") as log:
            child = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            write_json(out / "status.json", state)
            result = child.wait()
        state.pop("child_pid", None)
        if result:
            raise RuntimeError(f"{name} exited with code {result}; see {logfile}")

    try:
        checkpoints = sorted(p for p in (training / "checkpoints").glob("step_*")
                             if (p / "trainer_state.pt").exists() and (p / "adapter_model.safetensors").exists())
        completed_step = int(checkpoints[-1].name.split("_")[-1]) if checkpoints else 0
        if completed_step < plan["max_steps"]:
            command = [sys.executable, "-u", "tools/train_policy_sft.py", "--run-dir", str(training),
                       "--val-root", str(validation), "--max-steps", "4096", "--batch-size", "16",
                       "--grad-accum", "4", "--workers", "24", "--val-workers", "24", "--seed", "0",
                       "--log-every", "32", "--eval-every", "512", "--eval-batches", "125",
                       "--eval-initial", "--save-every", "1024"]
            if checkpoints:
                command += ["--resume", str(checkpoints[-1])]
            stage("training", command, out / "train.stdout.log")

        validation_rows = {r["step"]: r for r in
                           [json.loads(s) for s in (training / "log.jsonl").read_text().splitlines()]
                           if "val_loss_per_token" in r}
        comparisons, episode_ids = [], None
        for step in plan["checkpoints"]:
            checkpoint = training / "checkpoints" / f"step_{step:06d}"
            assert (checkpoint / "adapter_model.safetensors").exists()
            eval_out = out / "evaluations" / checkpoint.name
            if not (eval_out / "summary.json").exists():
                stage(f"evaluating_{step}", [sys.executable, "-u", "tools/eval_policy.py", "--run-dir", str(training),
                      "--checkpoint", str(checkpoint), "--episodes", str(plan["eval_episodes_per_dataset"]),
                      "--seed", "0", "--out", str(eval_out)], out / f"eval_{step:06d}.stdout.log")
            result = json.loads((eval_out / "summary.json").read_text())
            config = json.loads((eval_out / "eval_config.json").read_text())
            assert result["status"] == "complete" and result["overall"]["episodes"] == 64
            assert config["adapter_sha256"] == hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
            if episode_ids is None:
                episode_ids = config["selected_episodes"]
            assert episode_ids == config["selected_episodes"], "Checkpoint episode sets differ"
            comparisons.append({"step": step, "checkpoint": str(checkpoint), "evaluation": str(eval_out),
                                "offline": validation_rows[step], "closed_loop": result})
            write_json(out / "checkpoint_comparison.json", comparisons)
        best = max(comparisons, key=lambda c: (c["closed_loop"]["macro_success"], c["closed_loop"]["macro_spl"],
                                               -c["offline"]["val_loss_per_token"], -c["step"]))
        best["selection_rule"] = plan["selection"]
        write_json(out / "best_checkpoint.json", best)
        link = training / "best_checkpoint"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(Path(best["checkpoint"]).relative_to(training), target_is_directory=True)
        data = json.loads((training / "data_summary.json").read_text())
        report = ["# Step 2 completed — SmolVLM2 pilot", "",
                  f"Completed: {now()}", "",
                  f"Group A: {data['train_episodes']:,} usable episodes / {data['train_steps']:,} available steps.",
                  f"Group C: 600 rendered episodes; {data['val_episodes']} usable / {data['val_steps']:,} steps for offline validation.",
                  "Training: 4,096 optimizer updates, effective batch 64, 262,144 sampled steps (0.1181 passes); decoder LoRA only.",
                  "Offline metrics use a fixed 2,000-step subset; simulator selection uses the same 32 R2R + 32 RxR episodes at every checkpoint.",
                  "Simulator observations are actual live RGB and executed-action history. Group E remains untouched.", "",
                  "| Checkpoint | Offline accuracy | Offline CE | Macro success | Macro SPL |", "| --- | ---: | ---: | ---: | ---: |"]
        for c in comparisons:
            report.append(f"| {c['step']} | {c['offline']['val_action_acc']:.4f} | {c['offline']['val_loss_per_token']:.4f} | "
                          f"{c['closed_loop']['macro_success']:.4f} | {c['closed_loop']['macro_spl']:.4f} |")
        report += ["", f"Selected checkpoint: `{best['checkpoint']}` (also `train/best_checkpoint`).",
                   "Selection: " + plan["selection"] + ".", "",
                   "These are pilot development results, not final benchmark results. RxR uses 15-degree turns, differing from its official 30-degree protocol.",
                   "Next step is the separately specified DAgger refinement; this checkpoint is the initial SFT policy, not the final frozen reference.", ""]
        (out / "RESULTS.md").write_text("\n".join(report))
        state.update(status="complete", stage="complete", completed_utc=now(), best_checkpoint=best["checkpoint"],
                     macro_success=best["closed_loop"]["macro_success"], macro_spl=best["closed_loop"]["macro_spl"])
        state.pop("command", None)
        write_json(out / "status.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated_utc=now())
        write_json(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
