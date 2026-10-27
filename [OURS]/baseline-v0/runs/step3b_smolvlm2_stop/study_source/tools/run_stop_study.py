"""Durable bounded study: calibration, conditional one-round repair, held-out test.

Only predeclared group-C development results select models/decoders. Confirmation
sets are opened after their candidate is locked. No third refinement is permitted.
"""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.stop_diagnostics import replay_stop
from tools.calibrate_stop import rows_at

OUT = ROOT / "runs/step3b_smolvlm2_stop"
ROUND2 = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/dagger_groupA_round2_smolvlm2_stop2m")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    os.chdir(ROOT)
    lock = open(OUT / "study.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HOME="/data/shared/hf_cache", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", GLOG_minloglevel="2")
    state, current_child = {"status": "running", "pid": os.getpid()}, None
    for name, expected in read(OUT / "study_source_sha256.json").items():
        assert sha(ROOT / name) == expected, f"Study source drift: {name}"

    def update(**values):
        state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        write(OUT / "experiment_status.json", state)
        print(json.dumps(state), flush=True)

    def run(stage, arguments, completed=None):
        nonlocal current_child
        if completed and Path(completed).exists():
            return
        command = [sys.executable, "-u", *map(str, arguments)]
        update(stage=stage, command=command)
        with open(OUT / f"{stage}.log", "a") as log:
            current_child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=current_child.pid)
            code = current_child.wait()
        current_child = None
        state.pop("child_pid", None)
        if code:
            raise RuntimeError(f"Stage {stage} failed with exit {code}; inspect its log")
        if completed:
            assert Path(completed).exists(), completed

    def evaluate(stage, checkpoint, run_dir, selection, out, bias=0., carrier=False, workers=4):
        command = ["tools/eval_policy_scores.py", "--run-dir", run_dir, "--checkpoint", checkpoint,
                   "--selection", selection, "--out", out, "--stop-bias", bias, "--workers", workers]
        if carrier:
            command.append("--movement-only")
        run(stage, command, Path(out) / "summary.json")

    def decoder_check(candidate, tag):
        import numpy as np
        directory = OUT / f"decoder_check_{tag}"
        if (directory / "verified.json").exists():
            return
        carriers = rows_at(candidate["evaluation"])
        chosen = {}
        for name in ("r2r", "rxr"):
            pairs = [(row, replay_stop(row, candidate["stop_bias"])) for row in carriers if row["dataset"] == name]
            # Use the development set only. Exercise a predicted success and a long continuation.
            pairs.sort(key=lambda pair: (-pair[1]["success"], pair[1]["steps"], pair[0]["episode_id"]))
            first = pairs[0]
            rest = [pair for pair in pairs if pair[0]["episode_id"] != first[0]["episode_id"]]
            second = max(rest, key=lambda pair: pair[1]["steps"])
            chosen[name] = [first[0]["episode_id"], second[0]["episode_id"]]
        manifest = directory / "selection.json"
        write(manifest, {"group": "C", "selected_episodes": chosen, "purpose": "development-only online/offline decoding equivalence"})
        run_dir = Path(candidate["checkpoint"]).parent.parent
        evaluate(f"decoder_check_{tag}", candidate["checkpoint"], run_dir, manifest, directory / "evaluation", candidate["stop_bias"], workers=2)
        carrier_lookup = {(r["dataset"], r["episode_id"]): r for r in carriers}
        for actual in rows_at(directory / "evaluation"):
            carrier = carrier_lookup[(actual["dataset"], actual["episode_id"])]
            expected = replay_stop(carrier, candidate["stop_bias"])
            stop = expected.get("stop_step")
            actions = carrier["executed_actions"] if stop is None else carrier["executed_actions"][:stop] + [0]
            positions = carrier["positions"] if stop is None else carrier["positions"][:stop + 1] + [carrier["positions"][stop]]
            assert actual["executed_actions"] == actions
            assert np.allclose(actual["positions"], positions, atol=1e-6, rtol=0)
            for key in ("success", "spl", "oracle_success", "ndtw", "distance_to_goal", "steps"):
                assert abs(actual[key] - expected[key]) < 1e-6, (key, actual[key], expected[key])
        write(directory / "verified.json", {"status": "passed", "episodes": 4, "candidate": candidate,
              "actions_poses_SR_SPL_nDTW_exact": True})

    def confirmation(candidate, letter):
        directory = OUT / f"confirmation_{letter}"
        locked = {key: candidate[key] for key in ("checkpoint", "adapter_sha256", "stop_bias", "evaluation")}
        locked.update(locked_utc=datetime.now(timezone.utc).isoformat(), selection_manifest_sha256=sha(OUT / f"confirmation_{letter}.json"),
                      development_only_selection=True, plan_sha256=sha(OUT / "plan.json"))
        path = directory / "locked_candidate.json"
        if path.exists():
            old = read(path)
            assert {k: v for k, v in old.items() if k != "locked_utc"} == {k: v for k, v in locked.items() if k != "locked_utc"}
        else:
            assert not list(directory.glob("*/episodes/*.json")), "Cannot choose candidate after opening confirmation"
            write(path, locked)
        assert sha(Path(candidate["checkpoint"]) / "adapter_model.safetensors") == candidate["adapter_sha256"]
        selection = OUT / f"confirmation_{letter}.json"
        evaluations = [("candidate", candidate["checkpoint"], str(Path(candidate["checkpoint"]).parent.parent), candidate["stop_bias"]),
                       ("step2_3072", str(ROOT / "runs/step2_smolvlm2/train/checkpoints/step_003072"), str(ROOT / "runs/step2_smolvlm2/train"), 0.),
                       ("step3_1024", str(ROOT / "runs/step3_smolvlm2/train/checkpoints/step_001024"), str(ROOT / "runs/step3_smolvlm2/train"), 0.)]
        if str(Path(candidate["checkpoint"]).resolve()) == str(ROOT / "runs/step3_smolvlm2/train/checkpoints/step_002048"):
            evaluations.append(("step3_2048", candidate["checkpoint"], str(ROOT / "runs/step3_smolvlm2/train"), 0.))
        for name, checkpoint, run_dir, bias in evaluations:
            evaluate(f"confirm_{letter}_{name}", checkpoint, run_dir, selection, directory / name, bias)
        run(f"report_{letter}", ["tools/report_stop_confirmation.py", "--set", letter], directory / "report.json")
        return read(directory / "report.json")

    def train_one_round():
        control = read(OUT / "teacher_control" / "summary.json")
        assert control["status"] == "complete" and not control["label_errors"]
        assert all(v["successes"] == v["episodes"] for v in control["by_dataset"].values())
        assert read(OUT / "check_teacher.json")["status"] == "passed"
        assert read(OUT / "check_sampling.json")["status"] == "passed"
        run("collect_round2", ["tools/collect_dagger_stop.py", "--out", ROUND2], ROUND2 / "summary.json")
        view = OUT / "data/round2_A"
        run("prepare_round2", ["tools/prepare_stop_data.py", "--root", ROUND2, "--out", view, "--kind", "policy", "--group", "A"], view / "summary.json")
        common = ["--train-views", OUT / "data/expert_A", view, "--val-view", OUT / "data/expert_C", "--eval-initial"]
        run("training_smoke", ["tools/train_policy_stop.py", "--run-dir", OUT / "training_smoke", "--max-steps", 2, *common], OUT / "training_smoke/complete.json")
        train = OUT / "train"
        command = ["tools/train_policy_stop.py", "--run-dir", train, *common]
        checkpoints = sorted(train.glob("checkpoints/step_*/trainer_state.pt"))
        if checkpoints:
            command.extend(["--resume", checkpoints[-1].parent])
        run("train_round2", command, train / "complete.json")
        paths = []
        for step in (512, 1024, 2048):
            path = OUT / "refinement_development" / f"step_{step:06d}"
            paths.append(path)
            evaluate(f"refinement_dev_{step}", train / "checkpoints" / f"step_{step:06d}", train,
                     OUT / "development.json", path, carrier=True)
        report = OUT / "refinement_selection/summary.json"
        run("select_refinement", ["tools/calibrate_stop.py", "--evaluations", *paths, "--out", OUT / "refinement_selection"], report)
        return read(report)["selected"]

    def terminate(signum, frame):
        if current_child and current_child.poll() is None:
            current_child.terminate()
        raise KeyboardInterrupt("Study interrupted")
    signal.signal(signal.SIGTERM, terminate)
    try:
        update(stage="awaiting_score_diagnostics")
        while True:
            diagnostic = read(OUT / "status.json")
            if diagnostic["status"] == "failed":
                raise RuntimeError(f"Diagnostics failed: {diagnostic}")
            if diagnostic["status"] == "diagnostic_complete":
                break
            time.sleep(15)
        run("calibrate", ["tools/calibrate_stop.py"], OUT / "calibration/summary.json")
        calibration = read(OUT / "calibration/summary.json")
        calibration_report = None
        if calibration["sufficient_on_development"]:
            candidate = calibration["selected"]
            decoder_check(candidate, "calibration")
            calibration_report = confirmation(candidate, "A")
        if calibration_report and calibration_report["passed"]:
            result, trained = calibration_report, False
        else:
            candidate = train_one_round()
            decoder_check(candidate, "refinement")
            result = confirmation(candidate, "B" if calibration_report else "A")
            trained = True
        write(OUT / "result.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "calibration_sufficient_on_development": calibration["sufficient_on_development"],
              "calibration_confirmation_passed": calibration_report["passed"] if calibration_report else None,
              "one_refinement_round_trained": trained, "final_confirmation": result,
              "improvement_passed_locked_criteria": result["passed"], "no_further_training_rounds": True})
        update(status="complete", stage="experiment_complete", passed=result["passed"], trained=trained)
    except BaseException as error:
        update(status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
