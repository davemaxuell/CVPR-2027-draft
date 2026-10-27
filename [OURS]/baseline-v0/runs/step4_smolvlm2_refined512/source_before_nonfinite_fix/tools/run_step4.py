"""Durable Step 4: selected-policy visits, actual transition catalog, final audit."""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/step4_smolvlm2_refined512"
DATA = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0")
VISITS = DATA / "wm_visits_groupA_smolvlm2_refined512_v1"
CORPUS = DATA / "wm_groupA_smolvlm2_refined512_v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    lock = open(RUN / "pipeline.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / "status.json").exists() and read(RUN / "status.json")["status"] == "complete":
        print((RUN / "status.json").read_text()); return
    runtime = read(RUN / "check_runtime.json")
    assert runtime["status"] == "passed"
    reference = read(RUN / "reference/reference.json")
    assert runtime["policy_identity_sha256"] == reference["policy_identity_sha256"]
    for relative, expected in runtime["source_sha256"].items():
        assert sha(ROOT / relative) == expected
    plan = {"step": 4, "scope": "SmolVLM2 actual group-A transition collection only; no world-model/scorer fitting",
            "reference": str(RUN / "reference"), "selected_policy_identity_sha256": reference["policy_identity_sha256"],
            "checkpoint": reference["source_checkpoint"], "stop_bias": -0.25,
            "selection": "explicit user choice despite failed original stopping safeguard; Step 2 preserved as control",
            "new_policy_episodes_per_dataset": 128, "collection_seed": 4, "workers": 4,
            "branch_roots": "t=3,35,67,... through actual episode termination", "branch_actions": [1, 2, 3],
            "sources": ["validated original group-A experts", "historical DAgger round 1", "historical DAgger round 2", "new selected-policy visits and branches"],
            "visits": str(VISITS), "corpus": str(CORPUS), "matched_history": 4,
            "STOP_dynamics": False, "fresh_rgb_encoding": "lossless PNG; policy observes uncompressed simulator RGB",
            "historical_rgb_encoding": "existing JPEG quality 95, retained without recompression",
            "integrity": "all frame bytes hashed and image headers checked; sampled full decoding; loader rechecks accessed bytes",
            "budget_rationale": "256 new episodes cover all available group-A scenes; all reusable validated expert and recorded DAgger movement transitions are retained"}
    if (RUN / "plan.json").exists():
        assert read(RUN / "plan.json") == plan
    else:
        write(RUN / "plan.json", plan)
    files = ["tools/package_experimental_reference.py", "tools/run_step4.py", "tools/collect_wm_policy.py", "tools/build_wm_corpus.py",
             "tools/check_step4_runtime.py", "tools/check_wm_corpus.py", "policy/experimental_reference.py",
             "policy/common.py", "policy/smolvlm2.py", "policy/online.py", "sim/transition_branch.py", "sim/config.py",
             "sim/episodes.py", "sim/rxr_dataset.py", "wm/data.py", "wm/__init__.py", "manifest/scene_partition.json"]
    hashes = {p: sha(ROOT / p) for p in files}
    if (RUN / "source_sha256.json").exists():
        assert read(RUN / "source_sha256.json") == hashes, "Review source changes before resuming"
    else:
        write(RUN / "source_sha256.json", hashes)
    for relative in files:
        target = RUN / "source" / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HOME="/data/shared/hf_cache", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    state = {"status": "running", "pid": os.getpid(), "started_or_resumed_utc": now()}
    children, logs = [], []
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Step 4 interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    def launch(name, command):
        log = open(RUN / (name + ".log"), "a"); logs.append(log)
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child
    def wait_for(pairs, stage):
        state.update(stage=stage, child_pids={name: child.pid for name, child in pairs}, updated_utc=now())
        write(RUN / "status.json", state)
        while True:
            codes = {name: child.poll() for name, child in pairs}
            if any(c is not None and c != 0 for c in codes.values()):
                raise RuntimeError(f"Stage failed: {codes}; inspect matching run logs")
            if all(c is not None for c in codes.values()):
                break
            time.sleep(5)
    python = sys.executable
    try:
        initial = []
        if not (CORPUS / "base_ready.json").exists() and not (CORPUS / "manifest.json").exists():
            initial.append(("index_existing", launch("index_existing", [python, "-u", "tools/build_wm_corpus.py", "--stage", "base"])))
        if not (VISITS / "summary.json").exists():
            initial.append(("collect_policy", launch("collect_policy", [python, "-u", "tools/collect_wm_policy.py", "--out", str(VISITS)])))
        if initial:
            wait_for(initial, "collecting_and_indexing")
        if not (CORPUS / "manifest.json").exists():
            child = launch("finalize_corpus", [python, "-u", "tools/build_wm_corpus.py", "--stage", "complete"])
            wait_for([("finalize_corpus", child)], "finalizing_corpus")
        child = launch("verify_corpus", [python, "-u", "tools/check_wm_corpus.py"])
        wait_for([("verify_corpus", child)], "verifying")
        assert read(RUN / "final_verification.json")["status"] == "passed"
        corpus, visits = read(CORPUS / "manifest.json"), read(VISITS / "summary.json")
        counts = corpus["counts"]
        lines = ["# Step 4 completed — actual group-A world-model transitions", "", f"Completed: {now()}", "",
                 "Selected experimental policy: refined SmolVLM2 checkpoint 512, with 0.25 subtracted from the STOP logit before four-action argmax. The saved model configuration and native evaluation preprocessing are retained.",
                 "The user explicitly selected this policy despite its failed original stopping safeguard: 149 incorrect stops among 224 group-C confirmation episodes. It achieved 29/224 successes and a 9.6% forward collision rate. This decision does not relabel the failed criterion as passed. The original Step 2 reference is preserved as a control.", "",
                 "Versioned reference: `reference/`; policy identity: `" + reference["policy_identity_sha256"] + "`.",
                 f"New selected-policy collection: {visits['episodes']} episodes, {visits['actions']:,} executed actions, {visits['branch_roots']:,} actual branch roots and {visits['branch_transitions']:,} F/L/R outcomes.",
                 f"Combined corpus: {counts['episodes']:,} episodes, {counts['transitions']:,} movement transitions, {counts['matched']:,} matched four-observation samples and {counts['dino_early_extra']:,} separately indexed early DINO-only samples.", "",
                 "| Source | Episodes | Movement transitions |", "| --- | ---: | ---: |"]
        for source, values in corpus["by_source"].items():
            lines.append(f"| {source} | {values['episodes']:,} | {sum(v['transitions'] for v in values['actions'].values()):,} |")
        lines += ["", f"Corpus: `{CORPUS}` (`manifest.json`, `catalog.sqlite`).", "Loader: `wm.data.TransitionDataset`; DINO and CDiT share matched sample IDs and observed targets, taking their last three or four real frames respectively.",
                  "All data comes from group A. STOP is excluded from dynamics fitting. Blocked forward motions are retained. The actual executed movement at a branch root is indexed once, with alternate movements added from real simulator branches. No imagined targets or future realized displacement are model inputs.",
                  f"Integrity: all {counts['frames']:,} images hashed and headers checked; {counts['decoded_frames_checked']:,} images fully decoded during indexing. Accessed images are hash-checked by the loader.",
                  "New visits/branches use lossless PNG; historical expert/DAgger RGB remains JPEG quality 95. The catalog references those existing files, so all four source corpora must be retained. Historical DAgger sources retain their original collecting-policy identities; they are not represented as visits from the new refinement.",
                  "[Final verification](final_verification.json) passed, including native preprocessing/decoder parity, branch root restoration, unchanged actual trajectories and metrics, left/right signs, blocked motion, and paired-family data loading.",
                  "The corpus is ready for Step 5. World-model training has not started; the transferred predictor checkpoint and training budget remain Step 5 choices.", ""]
        (RUN / "RESULTS.md").write_text("\n".join(lines))
        state.update(status="complete", stage="complete", completed_utc=now(), corpus=str(CORPUS),
                     reference=str(RUN / "reference"), counts=counts)
        state.pop("child_pids", None)
        write(RUN / "status.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated_utc=now())
        write(RUN / "status.json", state)
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
