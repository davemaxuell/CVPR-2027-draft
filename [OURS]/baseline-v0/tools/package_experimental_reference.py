"""Record the user's explicit experimental selection without changing old decisions."""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "runs/step4_smolvlm2_refined512"
STUDY = ROOT / "runs/step3b_smolvlm2_stop"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    locked = read(STUDY / "confirmation_A/locked_candidate.json")
    evaluation = read(STUDY / "confirmation_A/candidate/eval_config.json")
    result = read(STUDY / "result.json")
    checkpoint = Path(locked["checkpoint"])
    assert checkpoint == STUDY / "train/checkpoints/step_000512"
    assert locked["adapter_sha256"] == evaluation["adapter_sha256"] == sha(checkpoint / "adapter_model.safetensors")
    assert locked["stop_bias"] == evaluation["stop_bias"] == -0.25
    assert not evaluation["movement_only"] and evaluation["decoding_batch_size"] == 1
    assert result["status"] == "complete" and not result["improvement_passed_locked_criteria"]
    assert result["final_confirmation"]["candidate"] == locked
    assert not result["final_confirmation"]["checks"]["premature_stops_at_most_step2"]
    assert read(STUDY / "train/policy_config.json") == evaluation["policy_config"]
    audit = read(STUDY / "check_final.json")
    assert audit["status"] == "passed" and audit["result_sha256"] == sha(STUDY / "result.json")
    assert audit["independently_recomputed"]["candidate"]["incorrect_stops"] == 149
    for relative, expected in read(STUDY / "study_source_sha256.json").items():
        assert sha(ROOT / relative) == expected, ("Study source changed", relative)
    control = ROOT / "runs/step3_smolvlm2/frozen_reference"
    for relative, expected in read(control / "sha256.json").items():
        assert sha(control / relative) == expected, ("Control drift", relative)
    runtime = ["policy/common.py", "policy/smolvlm2.py", "policy/online.py", "policy/experimental_reference.py"]
    assert sha(ROOT / "policy/smolvlm2.py") == evaluation["policy_source_sha256"]
    assert sha(ROOT / "policy/online.py") == evaluation["online_source_sha256"]
    decoder = {"family": "add STOP-only bias then argmax over four first-token action logits",
               "stop_logit_bias": -0.25, "action_order": ["STOP", "FORWARD", "LEFT", "RIGHT"],
               "batch_size": 1, "actions_per_forward": 1, "tie_break": "first action in declared order"}
    identity = {"version": "smolvlm2_step3b_000512_stop_m0p25_v1", "adapter_sha256": locked["adapter_sha256"],
                "policy_config_sha256": sha(STUDY / "train/policy_config.json"), "decoder": decoder,
                "processor_sha256": {p.name: sha(p) for p in sorted((STUDY / "train/processor").iterdir()) if p.is_file()},
                "runtime_source_sha256": {p: sha(ROOT / p) for p in runtime},
                "simulator": evaluation["simulator"], "pixel_preprocessing": evaluation["pixel_preprocessing"],
                "rgb_source": evaluation["rgb_source"]}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    final = RUN / "reference"
    if final.exists():
        assert read(final / "reference.json")["policy_identity_sha256"] == identity_sha
        for relative, expected in read(final / "file_sha256.json").items():
            assert sha(final / relative) == expected
        print(str(final))
        return
    RUN.mkdir(parents=True, exist_ok=True)
    temporary = RUN / ".reference.tmp"
    temporary.mkdir(exist_ok=False)
    (temporary / "adapter").mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(checkpoint / name, temporary / "adapter" / name)
    shutil.copy2(STUDY / "train/policy_config.json", temporary / "policy_config.json")
    shutil.copytree(STUDY / "train/processor", temporary / "processor")
    for relative in runtime:
        target = temporary / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    evidence = ["confirmation_A/locked_candidate.json", "confirmation_A/candidate/eval_config.json", "result.json", "check_final.json", "reference_decision.json"]
    for relative in evidence:
        target = temporary / "evidence" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STUDY / relative, target)
    selection = {"authority": "explicit_user_selection_2026-10-26", "decision": "proceed_with_refined_checkpoint_512_and_STOP_bias_minus_0.25",
                 "original_stopping_safeguard_passed": False, "all_original_locked_criteria_passed": False,
                 "failed_criterion": "premature_stops_at_most_step2", "incorrect_stops": 149, "confirmation_episodes": 224,
                 "reason": "User deliberately selected the highest measured-success refinement despite its failed stopping safeguard.",
                 "historical_decision_preserved": str(STUDY / "reference_decision.json"),
                 "control_reference": str(control), "control_bundle_manifest_sha256": sha(control / "sha256.json")}
    metadata = {"format": "smolvlm2_experimental_reference_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "frozen_experimental_selection", "source_checkpoint": str(checkpoint), "selection": selection,
                "policy_identity_sha256": identity_sha, "identity_fields": identity, "decoder": decoder,
                "runtime_source_sha256": identity["runtime_source_sha256"],
                "confirmation_metrics": result["final_confirmation"]["metrics"],
                "loader": "policy.experimental_reference.ExperimentalReference(bundle, device='cuda:0')",
                "versioning": "Every policy-dependent corpus must record this identity; changing weights, bias, preprocessing or prompt requires a new identity and new dependent data."}
    write(temporary / "reference.json", metadata)
    write(temporary / "file_sha256.json", {str(p.relative_to(temporary)): sha(p) for p in sorted(temporary.rglob("*")) if p.is_file()})
    temporary.rename(final)
    write(RUN / "selection_decision.json", {**selection, "selected_reference": str(final), "policy_identity_sha256": identity_sha})
    print(str(final))


if __name__ == "__main__":
    main()
