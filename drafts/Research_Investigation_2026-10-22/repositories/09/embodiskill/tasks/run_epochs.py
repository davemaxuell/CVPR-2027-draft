"""
run_epochs.py - Multi-epoch task execution.

This script wraps run.py to support iterative execution over multiple epochs.

Usage:
    python tasks/run_epochs.py --task alfworld --skill_module embodiskill \
        --model gpt-3.5-turbo --num_epochs 5
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import gc
import random
import re
import sqlite3
import shutil
import tempfile
from datetime import datetime, timezone
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Import from run.py
from run import (
    TaskManager, build_task, build_system, CONFIG,
    get_dataset_system_prompt, get_task_few_shots
)
from envs import get_recorder, get_task
from utils import get_model_dir_name
import json

# Global working directory
WORKING_DIR: str = None
RUN_STATE_FILE = "run_state.json"
CONFIG_DIR = "config"
STATE_DIR = "state"
LOG_DIR = "logs"
EMBODISKILL_DIR = "EmbodiSkill"
SNAPSHOT_DIR = "snapshots"
MIN_TMP_FREE_BYTES = 512 * 1024 * 1024


def normalize_skill_type(skill_type: str | None) -> str:
    return (skill_type or "embodiskill").strip().lower()


def uses_embodiskill_skill(skill_type: str | None) -> bool:
    return normalize_skill_type(skill_type) == "embodiskill"


def config_dir(output_path: str) -> str:
    return os.path.join(output_path, CONFIG_DIR)


def state_dir(output_path: str) -> str:
    return os.path.join(output_path, STATE_DIR)


def run_state_path(output_path: str) -> str:
    return os.path.join(state_dir(output_path), RUN_STATE_FILE)


def logs_dir(output_path: str) -> str:
    return os.path.join(output_path, LOG_DIR)


def active_skill_dir(output_path: str) -> str:
    return os.path.join(output_path, EMBODISKILL_DIR)


def snapshots_dir(output_path: str) -> str:
    return os.path.join(output_path, SNAPSHOT_DIR)


def snapshot_skill_dir(output_path: str, snapshot_tag: str) -> str:
    return os.path.join(snapshots_dir(output_path), snapshot_tag, EMBODISKILL_DIR)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _resolve_temp_dir() -> str:
    preferred_tmpdir = os.environ.get("EMBODISKILL_TMPDIR")
    candidates = [
        preferred_tmpdir,
        os.environ.get("TMPDIR"),
        "/tmp",
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            os.makedirs(candidate, exist_ok=True)
            if not os.access(candidate, os.W_OK | os.X_OK):
                continue
            free_bytes = shutil.disk_usage(candidate).free
            if free_bytes < MIN_TMP_FREE_BYTES:
                print(
                    f"[startup] Skip temp dir {candidate}: only {_format_bytes(free_bytes)} free "
                    f"(need at least {_format_bytes(MIN_TMP_FREE_BYTES)})."
                )
                continue

            os.environ["TMPDIR"] = candidate
            tempfile.tempdir = candidate
            print(
                f"[startup] Using temp dir {candidate} with {_format_bytes(free_bytes)} free."
            )
            return candidate
        except Exception as e:
            print(f"[startup] Failed to prepare temp dir {candidate}: {e}")

    raise RuntimeError(
        "No writable temp directory has enough free space. "
        f"Set EMBODISKILL_TMPDIR to a large disk path with at least {_format_bytes(MIN_TMP_FREE_BYTES)} free."
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def get_task_dir_name(task_name: str) -> str:
    if _env_flag("EMBODISKILL_GEMINI_DISABLE_THINKING", False):
        return f"{task_name}_think_off"
    return task_name


def _warn_if_low_disk_space(path: str, label: str, min_free_bytes: int = MIN_TMP_FREE_BYTES) -> None:
    try:
        free_bytes = shutil.disk_usage(path).free
    except Exception as e:
        print(f"[startup] Failed to inspect free space for {label} ({path}): {e}")
        return

    if free_bytes < min_free_bytes:
        print(
            f"[startup] Warning: {label} {path} is on a filesystem with only "
            f"{_format_bytes(free_bytes)} free. Consider setting EMBODISKILL_TMPDIR or output_dir to a larger disk."
        )


def get_embodiskill_snapshot_tag(epoch_id: int, phase: str = "after") -> str:
    """Return the snapshot tag for a given epoch and phase."""
    if phase == "after":
        return f"epoch_{epoch_id:03d}_after"
    if phase == "before":
        return f"epoch_{epoch_id:03d}_before"
    raise ValueError(f"Unsupported EmbodiSkill snapshot phase: {phase}")


def get_best_embodiskill_snapshot_tag(epoch_state: Dict[str, Any]) -> Optional[str]:
    """Resolve the rollback anchor snapshot tag for the current best epoch."""
    snapshot_tag = epoch_state.get("best_embodiskill_snapshot_tag")
    if snapshot_tag:
        return snapshot_tag

    if not epoch_state.get("completed_epochs"):
        return None

    best_epoch_id = epoch_state.get("best_epoch_id")
    if best_epoch_id is None:
        return None

    try:
        return get_embodiskill_snapshot_tag(int(best_epoch_id), phase="before")
    except (TypeError, ValueError):
        return None


def get_pending_embodiskill_snapshot_tag(epoch_state: Dict[str, Any]) -> Optional[str]:
    """Return the queued candidate snapshot that should be evaluated next."""
    snapshot_tag = epoch_state.get("pending_embodiskill_snapshot_tag")
    if snapshot_tag:
        return snapshot_tag
    return None


def get_next_active_embodiskill_snapshot_tag(epoch_state: Dict[str, Any]) -> Optional[str]:
    """Resolve the snapshot that should be restored before the next full epoch."""
    snapshot_tag = get_pending_embodiskill_snapshot_tag(epoch_state)
    if snapshot_tag:
        return snapshot_tag

    snapshot_tag = epoch_state.get("next_active_embodiskill_snapshot_tag")
    if snapshot_tag:
        return snapshot_tag
    return get_best_embodiskill_snapshot_tag(epoch_state)


def resolve_embodiskill_restore_snapshot(epoch_state: Dict[str, Any]) -> tuple[Optional[str], str]:
    """
    Resolve which snapshot should be restored before the next full epoch.

    Returns `(snapshot_tag, snapshot_role)` where `snapshot_role` is one of:
    - `candidate`: a newly generated snapshot that still needs evaluation
    - `best`: the current best evaluated snapshot
    - `current`: no explicit snapshot is configured; keep the current working tree
    """
    pending_snapshot_tag = get_pending_embodiskill_snapshot_tag(epoch_state)
    if pending_snapshot_tag:
        return pending_snapshot_tag, "candidate"

    best_snapshot_tag = get_best_embodiskill_snapshot_tag(epoch_state)
    next_snapshot_tag = epoch_state.get("next_active_embodiskill_snapshot_tag")
    if next_snapshot_tag:
        if best_snapshot_tag and next_snapshot_tag == best_snapshot_tag:
            return next_snapshot_tag, "best"
        return next_snapshot_tag, "candidate"

    if best_snapshot_tag:
        return best_snapshot_tag, "best"

    return None, "current"



def load_epoch_state(output_path: str) -> Dict[str, Any]:
    """Load epoch state for checkpoint/resume."""
    state_path = run_state_path(output_path)
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            return state
        except Exception as e:
            print(f"[EpochState] Failed to load state: {e}")
    
    # Default initial state
    return {
        "current_epoch": 0,
        "current_task": 0,
        "completed_epochs": [],
        "epoch_progress": {},
        "epoch_results": {},  # {epoch_id: {"success_rate": float, ...}}
        "best_epoch_id": 0,  # Epoch with highest success rate so far
        "best_success_rate": 0.0,  # Highest success rate achieved
        "best_embodiskill_snapshot_tag": None,
        "pending_embodiskill_snapshot_tag": None,
        "next_active_embodiskill_snapshot_tag": None,
        "embodiskill_snapshots": {},
    }


def save_epoch_state(output_path: str, state: Dict[str, Any]) -> None:
    """Save epoch state for checkpoint/resume."""
    state_path = run_state_path(output_path)
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"[EpochState] Saved state: epoch={state['current_epoch']}, task={state.get('current_task', 0)}")
    except Exception as e:
        print(f"[EpochState] Failed to save state: {e}")


def prepare_output_layout(output_path: str) -> None:
    """Create the top-level run layout."""
    for path in [
        config_dir(output_path),
        state_dir(output_path),
        logs_dir(output_path),
        os.path.join(output_path, EMBODISKILL_DIR),
        snapshots_dir(output_path),
    ]:
        os.makedirs(path, exist_ok=True)


def write_run_config(
    output_path: str,
    args: argparse.Namespace,
    *,
    agent_model: str,
    skill_model: str,
    agent_api_type: str,
    skill_api_type: str,
    agent_local_base_url: str,
    skill_local_base_url: str,
) -> None:
    """Persist resolved run configuration for reproducibility."""
    config_path = os.path.join(config_dir(output_path), "run_config.json")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": output_path,
        "args": vars(args),
        "resolved": {
            "agent_model": agent_model,
            "skill_model": skill_model,
            "agent_api_type": agent_api_type,
            "skill_api_type": skill_api_type,
            "agent_local_base_url": agent_local_base_url,
            "skill_local_base_url": skill_local_base_url,
            "active_skill_dir": active_skill_dir(output_path),
            "logs_dir": logs_dir(output_path),
            "state_file": run_state_path(output_path),
            "snapshots_dir": snapshots_dir(output_path),
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[run_epochs] Saved run config: {config_path}")


def flush_embodiskill_state(task_manager: TaskManager) -> None:
    """Best-effort flush before copying the persisted EmbodiSkill directory."""
    meta_skill = getattr(getattr(task_manager, "orchestrator", None), "meta_skill", None)
    if meta_skill is None:
        return

    main_skill = getattr(meta_skill, "main_skill", None)
    persist = getattr(main_skill, "persist", None)
    if callable(persist):
        try:
            persist()
        except Exception as e:
            print(f"[EmbodiSkillSnapshot] Failed to persist vector store before snapshot: {e}")


def release_embodiskill_state(task_manager: Optional[TaskManager]) -> None:
    """Best-effort resource release before restoring or replacing the persisted EmbodiSkill directory."""
    if task_manager is None:
        return

    meta_skill = getattr(getattr(task_manager, "orchestrator", None), "meta_skill", None)
    if meta_skill is None:
        return

    flush_embodiskill_state(task_manager)
    main_skill = getattr(meta_skill, "main_skill", None)

    orchestrator = getattr(task_manager, "orchestrator", None)
    if orchestrator is not None:
        try:
            orchestrator.meta_skill = None
        except Exception:
            pass

    for attr in ["main_skill", "task_layer", "current_task_context", "skill_cache"]:
        try:
            setattr(meta_skill, attr, None)
        except Exception:
            pass

    try:
        from chromadb.api.client import SharedSystemClient

        try:
            main_skill_client = getattr(main_skill, "_client", None)
            system = getattr(main_skill_client, "_system", None)
            stop = getattr(system, "stop", None)
            if callable(stop):
                stop()
        except Exception as e:
            print(f"[EmbodiSkillSnapshot] Failed to stop Chroma system during release: {e}")

        SharedSystemClient.clear_system_cache()
    except Exception as e:
        print(f"[EmbodiSkillSnapshot] Failed to clear Chroma system cache during release: {e}")

    gc.collect()


def ensure_writable_path_tree(root_dir: str) -> None:
    """Recursively make the restored EmbodiSkill tree writable by the current user."""
    if not os.path.exists(root_dir):
        return

    for current_root, dirnames, filenames in os.walk(root_dir):
        try:
            os.chmod(current_root, 0o775)
        except Exception as e:
            print(f"[EmbodiSkillSnapshot] Failed to chmod directory {current_root}: {e}")
        for dirname in dirnames:
            path = os.path.join(current_root, dirname)
            try:
                os.chmod(path, 0o775)
            except Exception as e:
                print(f"[EmbodiSkillSnapshot] Failed to chmod directory {path}: {e}")
        for filename in filenames:
            path = os.path.join(current_root, filename)
            try:
                os.chmod(path, 0o664)
            except Exception as e:
                print(f"[EmbodiSkillSnapshot] Failed to chmod file {path}: {e}")


def verify_embodiskill_writable(working_dir: str, skill_module: str) -> None:
    """Fail fast if the restored EmbodiSkill directory or SQLite store is not writable."""
    if not uses_embodiskill_skill(skill_module):
        return

    target_dir = active_skill_dir(working_dir)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"EmbodiSkill directory not found: {target_dir}")

    probe_path = os.path.join(target_dir, ".write_probe")
    with open(probe_path, "w", encoding="utf-8") as f:
        f.write("ok\n")
    os.remove(probe_path)

    sqlite_path = os.path.join(target_dir, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=rw", uri=True)
        try:
            conn.execute("PRAGMA schema_version;").fetchone()
        finally:
            conn.close()


def snapshot_embodiskill(
    working_dir: str,
    skill_module: str,
    epoch_id: Optional[int] = None,
    snapshot_tag: Optional[str] = None,
) -> Optional[str]:
    """Copy the current EmbodiSkill persist directory as a named snapshot."""
    if not uses_embodiskill_skill(skill_module):
        return None

    if snapshot_tag is None:
        if epoch_id is None:
            raise ValueError("snapshot_embodiskill requires epoch_id or snapshot_tag")
        snapshot_tag = get_embodiskill_snapshot_tag(epoch_id, phase="after")

    source_dir = active_skill_dir(working_dir)
    if not os.path.isdir(source_dir):
        print(f"[EmbodiSkillSnapshot] Source directory not found, skip snapshot: {source_dir}")
        return None

    snapshot_root = snapshots_dir(working_dir)
    final_epoch_dir = os.path.join(snapshot_root, snapshot_tag)
    final_snapshot_dir = os.path.join(final_epoch_dir, EMBODISKILL_DIR)
    tmp_epoch_dir = os.path.join(snapshot_root, f".{snapshot_tag}.tmp")

    try:
        if os.path.exists(tmp_epoch_dir):
            shutil.rmtree(tmp_epoch_dir)
        os.makedirs(snapshot_root, exist_ok=True)

        shutil.copytree(source_dir, os.path.join(tmp_epoch_dir, EMBODISKILL_DIR))
        with open(os.path.join(tmp_epoch_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch_id": epoch_id,
                    "snapshot_tag": snapshot_tag,
                    "source_dir": source_dir,
                    "snapshot_dir": final_snapshot_dir,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if os.path.exists(final_epoch_dir):
            shutil.rmtree(final_epoch_dir)
        os.replace(tmp_epoch_dir, final_epoch_dir)
        print(f"[EmbodiSkillSnapshot] Saved snapshot {snapshot_tag}: {final_snapshot_dir}")
        return final_snapshot_dir
    except Exception as e:
        if os.path.exists(tmp_epoch_dir):
            shutil.rmtree(tmp_epoch_dir)
        print(f"[EmbodiSkillSnapshot] Failed to save snapshot {snapshot_tag}: {e}")
        raise


def restore_embodiskill_snapshot(
    working_dir: str,
    skill_module: str,
    epoch_id: Optional[int] = None,
    snapshot_tag: Optional[str] = None,
) -> Optional[str]:
    """Restore the current EmbodiSkill persist directory from a named snapshot."""
    if not uses_embodiskill_skill(skill_module):
        return None
    if snapshot_tag is None:
        if epoch_id is None:
            raise ValueError("restore_embodiskill_snapshot requires epoch_id or snapshot_tag")
        if epoch_id < 0:
            raise ValueError("--restore_embodiskill_snapshot_epoch must be non-negative")
        snapshot_candidates = [get_embodiskill_snapshot_tag(epoch_id, phase="after")]
    else:
        snapshot_candidates = [snapshot_tag]

    snapshot_dir = None
    resolved_snapshot_tag = None
    for candidate in snapshot_candidates:
        candidate_dir = snapshot_skill_dir(working_dir, candidate)
        if os.path.isdir(candidate_dir):
            snapshot_dir = candidate_dir
            resolved_snapshot_tag = candidate
            break

    target_dir = active_skill_dir(working_dir)
    skill_root = os.path.dirname(target_dir)
    tmp_target_dir = os.path.join(skill_root, ".EmbodiSkill.restore_tmp")
    backup_target_dir = os.path.join(skill_root, ".EmbodiSkill.before_restore_backup")

    if snapshot_dir is None:
        raise FileNotFoundError(f"EmbodiSkill snapshot not found. Tried: {', '.join(snapshot_candidates)}")

    try:
        if os.path.exists(tmp_target_dir):
            shutil.rmtree(tmp_target_dir)
        if os.path.exists(backup_target_dir):
            shutil.rmtree(backup_target_dir)

        shutil.copytree(snapshot_dir, tmp_target_dir)
        if os.path.exists(target_dir):
            os.rename(target_dir, backup_target_dir)
        os.rename(tmp_target_dir, target_dir)
        ensure_writable_path_tree(target_dir)
        verify_embodiskill_writable(working_dir, skill_module)

        if os.path.exists(backup_target_dir):
            shutil.rmtree(backup_target_dir)

        print(f"[EmbodiSkillSnapshot] Restored EmbodiSkill from snapshot {resolved_snapshot_tag}: {snapshot_dir}")
        return target_dir
    except Exception as e:
        if os.path.exists(tmp_target_dir):
            shutil.rmtree(tmp_target_dir)
        if os.path.exists(backup_target_dir) and not os.path.exists(target_dir):
            os.rename(backup_target_dir, target_dir)
        print(f"[EmbodiSkillSnapshot] Failed to restore snapshot {resolved_snapshot_tag}: {e}")
        raise


def load_epoch_progress_from_log(output_path: str, epoch_id: int, upto_task: int) -> Dict[str, int]:
    """Best-effort progress recovery for resuming inside an epoch from old logs."""
    log_path = os.path.join(logs_dir(output_path), "epochs", f"epoch_{epoch_id:03d}.log")
    progress = {"completed_tasks": 0, "successful_tasks": 0}
    if upto_task <= 0 or not os.path.exists(log_path):
        return progress

    current_task_id = None
    task_marker = re.compile(r"-+ Task: (\d+)(?:\s|\||-)")
    done_marker = re.compile(r"done: (True|False)")

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                task_match = task_marker.search(line)
                if task_match:
                    current_task_id = int(task_match.group(1))
                    if current_task_id >= upto_task:
                        break
                    continue

                done_match = done_marker.search(line)
                if done_match and current_task_id is not None and current_task_id < upto_task:
                    progress["completed_tasks"] += 1
                    if done_match.group(1) == "True":
                        progress["successful_tasks"] += 1
                    current_task_id = None
    except Exception as e:
        print(f"[EpochState] Failed to recover progress from {log_path}: {e}")

    return progress


def parse_resume_from(value: str) -> tuple[int, int]:
    """Parse --resume_from as EPOCH:TASK."""
    if ":" not in value:
        raise argparse.ArgumentTypeError("--resume_from must use EPOCH:TASK format, e.g. 5:109")
    epoch_str, task_str = value.split(":", 1)
    try:
        epoch_id = int(epoch_str)
        task_id = int(task_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError("--resume_from epoch and task must be integers") from e
    if epoch_id < 0 or task_id < 0:
        raise argparse.ArgumentTypeError("--resume_from epoch and task must be non-negative")
    return epoch_id, task_id


def get_epoch_progress(epoch_state: Dict[str, Any], epoch_id: int) -> Dict[str, int]:
    progress_by_epoch = epoch_state.setdefault("epoch_progress", {})
    return progress_by_epoch.setdefault(
        str(epoch_id),
        {"completed_tasks": 0, "successful_tasks": 0},
    )


def get_checkpoint_int(epoch_state: Dict[str, Any], key: str, default: int = 0) -> int:
    value = epoch_state.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"[EpochState] Invalid {key}={value!r}; using {default}")
        return default


def prepare_task_manager(
    args: argparse.Namespace,
    task_name: str,
    working_dir: str,
    namespace: str,
    current_epoch_id: int,
    agent_model: str,
    skill_model: str,
    agent_api_type: str,
    skill_api_type: str,
    agent_local_base_url: str,
    skill_local_base_url: str,
    tasks_override: Optional[List[dict]] = None,
) -> TaskManager:
    task_manager = build_task(
        task_name,
        args.skill_module,
        args.max_trials,
        working_dir=logs_dir(working_dir),
    )
    if tasks_override is not None:
        task_manager.tasks = tasks_override

    task_manager.orchestrator_config['successful_topk'] = args.successful_topk
    task_manager.orchestrator_config['failed_topk'] = args.failed_topk
    task_manager.orchestrator_config['skills_topk'] = args.skills_topk
    task_manager.orchestrator_config['threshold'] = args.threshold
    task_manager.skill_config.update(
        working_dir=working_dir,
        persist_dir=active_skill_dir(working_dir),
        hop=args.hop,
        llm_concurrency=args.llm_concurrency,
        task=task_name,
        current_epoch_id=current_epoch_id,
        manual_reflection_max_tokens=args.manual_reflection_max_tokens,
        manual_revision_max_tokens=args.manual_revision_max_tokens,
    )

    task_manager.recorder = get_recorder(
        task=task_name,
        working_dir=logs_dir(working_dir),
        namespace=namespace,
    )

    build_system(
        task_manager,
        args.reasoning,
        args.skill_module,
        args.model,
        args.api_type,
        args.local_base_url,
        agent_llm_type=agent_model,
        agent_api_type=agent_api_type,
        agent_local_base_url=agent_local_base_url,
        skill_llm_type=skill_model,
        skill_api_type=skill_api_type,
        skill_local_base_url=skill_local_base_url,
    )
    return task_manager


def get_shuffled_task_order(total_tasks: int, pass_id: int, seed: int) -> List[int]:
    order = list(range(total_tasks))
    random.Random(seed + pass_id).shuffle(order)
    return order


def allocate_cyclic_task_indices(
    total_tasks: int,
    chunk_size: int,
    cursor: int,
    pass_id: int,
    seed: int,
) -> tuple[List[int], int, int]:
    if total_tasks <= 0:
        raise ValueError("total_tasks must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if cursor < 0 or cursor > total_tasks:
        raise ValueError(f"cursor must be in [0, {total_tasks}], got {cursor}")

    selected_indices: List[int] = []
    next_cursor = cursor
    next_pass_id = pass_id

    while len(selected_indices) < chunk_size:
        if next_cursor == total_tasks:
            next_cursor = 0
            next_pass_id += 1

        current_order = get_shuffled_task_order(total_tasks, next_pass_id, seed)
        remaining_in_pass = total_tasks - next_cursor
        take_count = min(chunk_size - len(selected_indices), remaining_in_pass)
        selected_indices.extend(current_order[next_cursor:next_cursor + take_count])
        next_cursor += take_count

        if next_cursor == total_tasks and len(selected_indices) < chunk_size:
            next_cursor = 0
            next_pass_id += 1

    return selected_indices, next_cursor, next_pass_id


def run_alfworld_train_eval_epochs(
    args: argparse.Namespace,
    epoch_state: Dict[str, Any],
    working_dir: str,
    agent_model: str,
    skill_model: str,
    agent_api_type: str,
    skill_api_type: str,
    agent_local_base_url: str,
    skill_local_base_url: str,
) -> None:
    num_epochs = args.num_skill_epochs if args.num_skill_epochs is not None else args.num_epochs
    train_seed = args.train_shuffle_seed
    train_tasks_all = get_task('alfworld_train')
    test_tasks_all = get_task('alfworld')

    if not train_tasks_all:
        raise ValueError("No Alfworld training tasks found in alfworld_train.")
    if not test_tasks_all:
        raise ValueError("No Alfworld test tasks found in alfworld.")

    train_chunk_size = args.train_chunk_size if args.train_chunk_size is not None else len(test_tasks_all)
    if train_chunk_size <= 0:
        raise ValueError("--train_chunk_size must be positive")
    epoch_state.setdefault("current_phase", None)
    epoch_state.setdefault("train_shuffle_cursor", 0)
    epoch_state.setdefault("train_shuffle_pass_id", 0)
    epoch_state.setdefault("current_train_chunk_indices", None)
    epoch_state.setdefault("current_eval_snapshot_tag", None)
    epoch_state.setdefault("best_epoch_id", None)
    epoch_state.setdefault("best_success_rate", 0.0)
    epoch_state.setdefault("completed_epochs", [])
    epoch_state.setdefault("epoch_results", {})
    epoch_state.setdefault("embodiskill_snapshots", {})

    start_epoch = get_checkpoint_int(epoch_state, "current_epoch", 0) if args.resume else 0
    resume_phase = epoch_state.get("current_phase") if args.resume else None
    if resume_phase not in {"train", "eval"}:
        resume_phase = None

    print(
        f"[run_epochs] Alfworld train/eval mode: {num_epochs} epochs, "
        f"{train_chunk_size} train tasks per epoch, {len(test_tasks_all)} test tasks per eval."
    )

    for epoch_id in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"[run_epochs] Starting skill epoch {epoch_id}/{num_epochs - 1}")
        print(f"{'='*60}")

        epoch_key = str(epoch_id)
        epoch_record = epoch_state["epoch_results"].setdefault(epoch_key, {})
        current_phase = resume_phase if epoch_id == start_epoch else None

        if current_phase != "eval":
            chunk_indices = epoch_state.get("current_train_chunk_indices")
            if current_phase != "train" or not chunk_indices:
                chunk_indices, next_cursor, next_pass_id = allocate_cyclic_task_indices(
                    total_tasks=len(train_tasks_all),
                    chunk_size=train_chunk_size,
                    cursor=get_checkpoint_int(epoch_state, "train_shuffle_cursor", 0),
                    pass_id=get_checkpoint_int(epoch_state, "train_shuffle_pass_id", 0),
                    seed=train_seed,
                )
                epoch_state["train_shuffle_cursor"] = next_cursor
                epoch_state["train_shuffle_pass_id"] = next_pass_id
                epoch_state["current_train_chunk_indices"] = chunk_indices
                epoch_state["current_epoch"] = epoch_id
                epoch_state["current_task"] = 0
                epoch_state["current_phase"] = "train"
                save_epoch_state(working_dir, epoch_state)
            else:
                print(f"[run_epochs] Resuming epoch {epoch_id} from the saved training chunk.")

            train_chunk = [copy.deepcopy(train_tasks_all[idx]) for idx in epoch_state["current_train_chunk_indices"]]
            train_task_manager = prepare_task_manager(
                args=args,
                task_name='alfworld_train',
                working_dir=working_dir,
                namespace=f"epoch_{epoch_id:03d}_train",
                current_epoch_id=epoch_id,
                agent_model=agent_model,
                skill_model=skill_model,
                agent_api_type=agent_api_type,
                skill_api_type=skill_api_type,
                agent_local_base_url=agent_local_base_url,
                skill_local_base_url=skill_local_base_url,
                tasks_override=train_chunk,
            )
            train_result = run_single_epoch(
                task_manager=train_task_manager,
                epoch_id=epoch_id,
            )
            print(
                f"[Epoch {epoch_id}] Train chunk completed: "
                f"{train_result['successful_tasks']}/{train_result['total_tasks']} "
                f"({train_result['success_rate']:.2%})"
            )

            revised_manual = None
            if (
                uses_embodiskill_skill(args.skill_module)
                and getattr(getattr(train_task_manager, "orchestrator", None), "meta_skill", None) is not None
            ):
                revise_manual = getattr(train_task_manager.orchestrator.meta_skill, "revise_manual", None)
                if callable(revise_manual):
                    revised_manual = revise_manual(
                        epoch_id=epoch_id,
                        success_rate=train_result["success_rate"],
                    )
                flush_embodiskill_state(train_task_manager)

            epoch_snapshot_tag = f"epoch_{epoch_id:03d}_after_train"
            epoch_snapshot_dir = None
            if uses_embodiskill_skill(args.skill_module):
                epoch_snapshot_dir = snapshot_embodiskill(
                    working_dir,
                    args.skill_module,
                    snapshot_tag=epoch_snapshot_tag,
                )

            release_embodiskill_state(train_task_manager)
            train_task_manager = None

            epoch_record["train_total_tasks"] = train_result["total_tasks"]
            epoch_record["train_successful_tasks"] = train_result["successful_tasks"]
            epoch_record["train_success_rate"] = train_result["success_rate"]
            epoch_record["train_chunk_indices"] = epoch_state["current_train_chunk_indices"]
            epoch_record["post_train_snapshot_tag"] = epoch_snapshot_tag
            epoch_record["post_train_snapshot_dir"] = epoch_snapshot_dir
            if revised_manual is not None:
                epoch_record["manual_version_after"] = revised_manual.get("version")

            epoch_state["embodiskill_snapshots"].setdefault(epoch_key, {})
            epoch_state["embodiskill_snapshots"][epoch_key]["after_train"] = epoch_snapshot_dir
            epoch_state["current_phase"] = "eval"
            epoch_state["current_task"] = 0
            epoch_state["current_eval_snapshot_tag"] = epoch_snapshot_tag
            save_epoch_state(working_dir, epoch_state)
        else:
            print(f"[run_epochs] Resuming epoch {epoch_id} from eval phase.")

        eval_snapshot_tag = epoch_state.get("current_eval_snapshot_tag")
        if uses_embodiskill_skill(args.skill_module) and eval_snapshot_tag:
            restore_embodiskill_snapshot(
                working_dir,
                args.skill_module,
                snapshot_tag=eval_snapshot_tag,
            )
            verify_embodiskill_writable(working_dir, args.skill_module)

        eval_task_manager = prepare_task_manager(
            args=args,
            task_name='alfworld',
            working_dir=working_dir,
            namespace=f"epoch_{epoch_id:03d}_eval",
            current_epoch_id=epoch_id,
            agent_model=agent_model,
            skill_model=skill_model,
            agent_api_type=agent_api_type,
            skill_api_type=skill_api_type,
            agent_local_base_url=agent_local_base_url,
            skill_local_base_url=skill_local_base_url,
            tasks_override=[copy.deepcopy(task) for task in test_tasks_all],
        )
        eval_result = run_single_epoch(
            task_manager=eval_task_manager,
            epoch_id=epoch_id,
            update_skill=False,
        )
        print(
            f"[Epoch {epoch_id}] Test eval completed: "
            f"{eval_result['successful_tasks']}/{eval_result['total_tasks']} "
            f"({eval_result['success_rate']:.2%})"
        )
        release_embodiskill_state(eval_task_manager)
        eval_task_manager = None

        epoch_record["success_rate"] = eval_result["success_rate"]
        epoch_record["eval_total_tasks"] = eval_result["total_tasks"]
        epoch_record["eval_successful_tasks"] = eval_result["successful_tasks"]
        epoch_record["eval_success_rate"] = eval_result["success_rate"]
        epoch_record["eval_skill_updates"] = False

        if (
            epoch_state.get("best_epoch_id") is None
            or eval_result["success_rate"] > epoch_state.get("best_success_rate", 0.0)
        ):
            epoch_state["best_epoch_id"] = epoch_id
            epoch_state["best_success_rate"] = eval_result["success_rate"]
            epoch_state["best_embodiskill_snapshot_tag"] = eval_snapshot_tag

        if epoch_id not in epoch_state["completed_epochs"]:
            epoch_state["completed_epochs"].append(epoch_id)

        epoch_state["current_epoch"] = epoch_id + 1
        epoch_state["current_task"] = 0
        epoch_state["current_phase"] = None
        epoch_state["current_train_chunk_indices"] = None
        epoch_state["current_eval_snapshot_tag"] = None
        save_epoch_state(working_dir, epoch_state)

    print(f"\n{'='*60}")
    print("[run_epochs] All skill epochs completed!")
    print(
        f"Best eval epoch: {epoch_state['best_epoch_id']} "
        f"with success_rate={epoch_state['best_success_rate']:.2%}"
    )
    print(f"{'='*60}")

def run_single_epoch(
    task_manager: TaskManager,
    epoch_id: int,
    start_task: int = 0,
    initial_successful_tasks: int = 0,
    epoch_state: Dict[str, Any] = None,
    working_dir: str = None,
    update_skill: bool = True,
) -> Dict[str, Any]:
    """
    Run a single epoch over all tasks.
    """
    task_manager.recorder.dataset_begin()
    
    total_tasks = len(task_manager.tasks)
    if start_task < 0 or start_task > total_tasks:
        raise ValueError(f"start_task must be in [0, {total_tasks}], got {start_task}")

    successful_tasks = initial_successful_tasks
    completed_tasks = start_task
    
    for task_id, task_config in tqdm(
        enumerate(task_manager.tasks[start_task:], start=start_task),
        total=total_tasks,
        initial=start_task,
        desc=f"Epoch {epoch_id}"
    ):
        task_manager.recorder.task_begin(task_id, task_config)
        
        # Set up environment and task config
        task_main, task_description = task_manager.orchestrator.env.set_env(task_config)
        few_shots: List[str] = get_task_few_shots(
            dataset=task_manager.task_name,
            task_config=task_config,
            few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
        )
        task_config.update(
            task_main=task_main, 
            task_description=task_description, 
            few_shots=few_shots
        )
        
        # Set task instruction for agents
        task_instruction: str = get_dataset_system_prompt(
            task_manager.task_name, 
            task_config=task_config
        )
        for agent in task_manager.orchestrator.agents_team.values():
            task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
            task_manager.recorder.log(agent.add_task_instruction(task_instruction))
        
        reward, done, trajectory_data = task_manager.orchestrator.schedule(
            task_config,
            skills=[],
            update_skill=update_skill,
        )
        
        task_manager.recorder.task_end(reward, done)
        
        if done:
            successful_tasks += 1
        completed_tasks += 1

        if epoch_state is not None and working_dir is not None:
            epoch_state["current_epoch"] = epoch_id
            epoch_state["current_task"] = task_id + 1
            progress = get_epoch_progress(epoch_state, epoch_id)
            progress["completed_tasks"] = completed_tasks
            progress["successful_tasks"] = successful_tasks
            save_epoch_state(working_dir, epoch_state)
    
    task_manager.recorder.dataset_end()
    
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
    
    return {
        "epoch_id": epoch_id,
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": success_rate,
    }


def main():
    random.seed(42)
    _resolve_temp_dir()
    
    parser = argparse.ArgumentParser(description='Run multi-epoch tasks.')
    parser.add_argument('--task', type=str, required=True, choices=['alfworld', 'alfworld_train'])
    parser.add_argument('--skill_module', type=str, default='embodiskill', help='Specify skill module')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Fallback LLM model type used when agent/skill model is not separately specified')
    parser.add_argument('--agent_model', type=str, default=None, help='LLM model type for agent action generation')
    parser.add_argument('--skill_model', type=str, default=None, help='LLM model type for EmbodiSkill internals')
    parser.add_argument('--max_trials', type=int, default=50, help='Max number of steps per task')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to retrieve')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to retrieve')
    parser.add_argument('--skills_topk', type=int, default=5, help='Number of skills to retrieve')
    parser.add_argument('--threshold', type=float, default=0.0, help='Threshold for traj similarity')
    parser.add_argument('--hop', type=int, default=1, help='Hop for traj similarity')
    parser.add_argument('--llm_concurrency', type=int, default=4, help='Max concurrent LLM calls inside EmbodiSkill')
    parser.add_argument('--manual_sections_refactor_threshold', type=int, default=100, help='Refactor manual sections when total rule count exceeds this threshold')
    parser.add_argument('--execution_notes_refactor_threshold', type=int, default=50, help='Refactor execution notes when note count exceeds this threshold')
    parser.add_argument('--manual_reflection_max_tokens', type=int, default=512, help='Max output tokens for one episode reflection')
    parser.add_argument('--manual_revision_max_tokens', type=int, default=2048, help='Max output tokens for manual revision and refactor phases')
    parser.add_argument('--api_type', type=str, default='openai', choices=['openai', 'azure', 'local', 'gemini'], help='Fallback API type used when agent/skill API type is not separately specified')
    parser.add_argument('--agent_api_type', type=str, default=None, choices=['openai', 'azure', 'local', 'gemini'], help='API type for agent action generation')
    parser.add_argument('--skill_api_type', type=str, default=None, choices=['openai', 'azure', 'local', 'gemini'], help='API type for EmbodiSkill internals')
    parser.add_argument('--local_base_url', type=str, default='http://localhost:8000/v1', help='Fallback base URL for local vLLM server')
    parser.add_argument('--agent_local_base_url', type=str, default=None, help='Base URL for the local agent model server')
    parser.add_argument('--skill_local_base_url', type=str, default=None, help='Base URL for the local skill model server')
    # Epoch-specific arguments
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to run')
    parser.add_argument('--num_skill_epochs', type=int, default=None, help='Number of skill epochs in Alfworld train/eval mode (defaults to --num_epochs)')
    parser.add_argument('--train_shuffle_seed', type=int, default=42, help='Shuffle seed for Alfworld training chunks')
    parser.add_argument('--train_chunk_size', type=int, default=None, help='Number of Alfworld train tasks per skill epoch (defaults to the test task count)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--resume_from', type=parse_resume_from, default=None, help='Resume from EPOCH:TASK, e.g. --resume_from 5:109')
    parser.add_argument('--resume_epoch', type=int, default=None, help='Epoch id to resume from (0-based)')
    parser.add_argument('--resume_task', type=int, default=None, help='Task id to resume from inside the resumed epoch (0-based)')
    parser.add_argument('--restore_embodiskill_snapshot_epoch', type=int, default=None, help='Restore EmbodiSkill from the snapshot saved after this epoch before running')
    rollback_group = parser.add_mutually_exclusive_group()
    rollback_group.add_argument(
        '--rollback_embodiskill',
        dest='rollback_embodiskill',
        action='store_true',
        help='Automatically restore the queued candidate or best EmbodiSkill snapshot before each full epoch.',
    )
    rollback_group.add_argument(
        '--no_rollback_embodiskill',
        dest='rollback_embodiskill',
        action='store_false',
        help='Disable automatic EmbodiSkill rollback before new epochs.',
    )
    parser.set_defaults(rollback_embodiskill=True)
    parser.add_argument('--workflow', type=str, default='team', help=argparse.SUPPRESS)

    args = parser.parse_args()
    args.skill_module = normalize_skill_type(args.skill_module)
    if args.task == "alfworld":
        args.rollback_embodiskill = False
    
    if args.workflow != "team":
        raise ValueError("Only the team workflow is supported in this refactored codebase.")

    global WORKING_DIR
    agent_model = args.agent_model or args.model
    skill_model = args.skill_model or args.model
    agent_api_type = args.agent_api_type or args.api_type
    skill_api_type = args.skill_api_type or args.api_type
    agent_local_base_url = args.agent_local_base_url or args.local_base_url
    skill_local_base_url = args.skill_local_base_url or args.local_base_url
    
    # Set up output directory
    if args.output_dir:
        WORKING_DIR = args.output_dir
    else:
        model_dir_name = get_model_dir_name(agent_model, skill_model)
        if args.rollback_embodiskill and args.task != "alfworld":
            model_dir_name = f"{model_dir_name}_rollback"
        task_dir_name = get_task_dir_name(args.task)
        output_root = os.environ.get(
            "EMBODISKILL_OUTPUT_ROOT",
            ".db_embodiskill_epochs",
        )
        WORKING_DIR = os.path.join(
            output_root,
            model_dir_name,
            task_dir_name,
            'team',
            args.skill_module
        )
    os.makedirs(WORKING_DIR, exist_ok=True)
    prepare_output_layout(WORKING_DIR)
    print(f"[run_epochs] Output directory: {WORKING_DIR}")
    _warn_if_low_disk_space(WORKING_DIR, "output directory")
    write_run_config(
        WORKING_DIR,
        args,
        agent_model=agent_model,
        skill_model=skill_model,
        agent_api_type=agent_api_type,
        skill_api_type=skill_api_type,
        agent_local_base_url=agent_local_base_url,
        skill_local_base_url=skill_local_base_url,
    )
    
    # Load or initialize epoch state
    epoch_state = load_epoch_state(WORKING_DIR)
    epoch_state.setdefault("best_embodiskill_snapshot_tag", None)
    epoch_state.setdefault("pending_embodiskill_snapshot_tag", None)
    epoch_state.setdefault("next_active_embodiskill_snapshot_tag", None)
    epoch_state.setdefault("embodiskill_snapshots", {})
    checkpoint_epoch = get_checkpoint_int(epoch_state, "current_epoch")
    checkpoint_task = get_checkpoint_int(epoch_state, "current_task")
    epoch_state["current_epoch"] = checkpoint_epoch
    epoch_state["current_task"] = checkpoint_task
    
    if args.resume_from is not None:
        start_epoch, first_epoch_start_task = args.resume_from
        print(f"[run_epochs] Resuming from explicit position epoch {start_epoch}, task {first_epoch_start_task}")
    elif args.resume_epoch is not None or args.resume_task is not None:
        start_epoch = args.resume_epoch if args.resume_epoch is not None else checkpoint_epoch
        first_epoch_start_task = args.resume_task if args.resume_task is not None else 0
        if start_epoch < 0 or first_epoch_start_task < 0:
            raise ValueError("--resume_epoch and --resume_task must be non-negative")
        print(f"[run_epochs] Resuming from explicit position epoch {start_epoch}, task {first_epoch_start_task}")
    elif args.resume and (checkpoint_epoch > 0 or checkpoint_task > 0):
        start_epoch = checkpoint_epoch
        first_epoch_start_task = checkpoint_task
        print(f"[run_epochs] Resuming from checkpoint epoch {start_epoch}, task {first_epoch_start_task}")
    else:
        start_epoch = 0
        first_epoch_start_task = 0
        epoch_state = {
            "current_epoch": 0,
            "current_task": 0,
            "completed_epochs": [],
            "epoch_progress": {},
            "epoch_results": {},
            "best_epoch_id": None,
            "best_success_rate": 0.0,
            "best_embodiskill_snapshot_tag": None,
            "pending_embodiskill_snapshot_tag": None,
            "next_active_embodiskill_snapshot_tag": None,
            "embodiskill_snapshots": {},
        }

    manual_restore_applied = False
    manual_restore_snapshot_tag = None
    if args.restore_embodiskill_snapshot_epoch is not None:
        restore_embodiskill_snapshot(WORKING_DIR, args.skill_module, args.restore_embodiskill_snapshot_epoch)
        manual_restore_snapshot_tag = get_embodiskill_snapshot_tag(args.restore_embodiskill_snapshot_epoch, phase="after")
        manual_restore_applied = True

    if uses_embodiskill_skill(args.skill_module) and first_epoch_start_task > 0:
        epoch_state["current_epoch"] = start_epoch
        epoch_state["current_task"] = first_epoch_start_task
        save_epoch_state(WORKING_DIR, epoch_state)
        print(
            f"[run_epochs] Resuming in-place from active EmbodiSkill at epoch {start_epoch}, "
            f"task {first_epoch_start_task}. Task {first_epoch_start_task} will be rerun from scratch."
        )
    
    task_manager: Optional[TaskManager] = None

    if args.task == "alfworld":
        run_alfworld_train_eval_epochs(
            args=args,
            epoch_state=epoch_state,
            working_dir=WORKING_DIR,
            agent_model=agent_model,
            skill_model=skill_model,
            agent_api_type=agent_api_type,
            skill_api_type=skill_api_type,
            agent_local_base_url=agent_local_base_url,
            skill_local_base_url=skill_local_base_url,
        )
        return

    # Run epochs
    for epoch_id in range(start_epoch, args.num_epochs):
        start_task = first_epoch_start_task if epoch_id == start_epoch else 0
        active_snapshot_tag = None
        active_snapshot_role = "current"
        print(f"\n{'='*60}")
        print(f"[run_epochs] Starting Epoch {epoch_id}/{args.num_epochs - 1} from task {start_task}")
        print(f"{'='*60}")

        should_auto_restore_best = (
            uses_embodiskill_skill(args.skill_module)
            and args.rollback_embodiskill
            and start_task == 0
            and not (epoch_id == start_epoch and manual_restore_applied)
        )
        if should_auto_restore_best:
            release_embodiskill_state(task_manager)
            task_manager = None
            active_snapshot_tag, active_snapshot_role = resolve_embodiskill_restore_snapshot(epoch_state)
            if active_snapshot_tag:
                active_snapshot_dir = snapshot_skill_dir(WORKING_DIR, active_snapshot_tag)
                if os.path.isdir(active_snapshot_dir):
                    print(f"[run_epochs] Restoring active EmbodiSkill snapshot before epoch {epoch_id}: {active_snapshot_tag}")
                    restore_embodiskill_snapshot(
                        WORKING_DIR,
                        args.skill_module,
                        snapshot_tag=active_snapshot_tag,
                    )
                    verify_embodiskill_writable(WORKING_DIR, args.skill_module)
                else:
                    print(
                        f"[run_epochs] Active EmbodiSkill snapshot tag {active_snapshot_tag} not found on disk; "
                        "keeping current EmbodiSkill."
                    )
                    active_snapshot_tag = None
                    active_snapshot_role = "current"
        elif uses_embodiskill_skill(args.skill_module) and start_task > 0:
            active_snapshot_tag = epoch_state.get("active_embodiskill_snapshot_tag")
            active_snapshot_role = epoch_state.get("active_embodiskill_snapshot_role", "resume")
        elif epoch_id == start_epoch and manual_restore_applied:
            active_snapshot_tag = manual_restore_snapshot_tag
            active_snapshot_role = "manual"
        
        task_manager = prepare_task_manager(
            args=args,
            task_name=args.task,
            working_dir=WORKING_DIR,
            namespace=f"epochs/epoch_{epoch_id:03d}",
            current_epoch_id=epoch_id,
            agent_model=agent_model,
            skill_model=skill_model,
            agent_api_type=agent_api_type,
            skill_api_type=skill_api_type,
            agent_local_base_url=agent_local_base_url,
            skill_local_base_url=skill_local_base_url,
        )

        before_snapshot_tag = get_embodiskill_snapshot_tag(epoch_id, phase="before")
        before_snapshot_dir = None
        if uses_embodiskill_skill(args.skill_module) and start_task == 0:
            flush_embodiskill_state(task_manager)
            before_snapshot_dir = snapshot_embodiskill(
                WORKING_DIR,
                args.skill_module,
                snapshot_tag=before_snapshot_tag,
            )
            epoch_state["embodiskill_snapshots"].setdefault(str(epoch_id), {})
            epoch_state["embodiskill_snapshots"][str(epoch_id)]["before"] = before_snapshot_dir
            epoch_state["active_embodiskill_snapshot_tag"] = active_snapshot_tag
            epoch_state["active_embodiskill_snapshot_role"] = active_snapshot_role
            save_epoch_state(WORKING_DIR, epoch_state)

        total_tasks = len(task_manager.tasks)
        if start_task < 0 or start_task > total_tasks:
            raise ValueError(f"Cannot resume epoch {epoch_id} from task {start_task}; task range is 0..{total_tasks}")

        progress = get_epoch_progress(epoch_state, epoch_id)
        if start_task > 0 and progress.get("completed_tasks", 0) != start_task:
            recovered = load_epoch_progress_from_log(WORKING_DIR, epoch_id, start_task)
            progress.update(recovered)
            print(
                f"[run_epochs] Recovered epoch {epoch_id} progress from log: "
                f"{progress['successful_tasks']}/{progress['completed_tasks']} successful/completed"
            )

        initial_successful_tasks = progress.get("successful_tasks", 0) if start_task > 0 else 0
        
        # Run single epoch
        epoch_result = run_single_epoch(
            task_manager=task_manager,
            epoch_id=epoch_id,
            start_task=start_task,
            initial_successful_tasks=initial_successful_tasks,
            epoch_state=epoch_state,
            working_dir=WORKING_DIR,
        )
        
        success_rate = epoch_result["success_rate"]
        print(f"[Epoch {epoch_id}] Completed with success_rate={success_rate:.2%}")
        best_snapshot_tag_before_epoch = get_best_embodiskill_snapshot_tag(epoch_state)
        is_candidate_evaluation = active_snapshot_role == "candidate"
        improved = success_rate > epoch_state.get("best_success_rate", 0.0)
        if epoch_id not in epoch_state["completed_epochs"]:
            epoch_state["completed_epochs"].append(epoch_id)
        epoch_state["epoch_results"][str(epoch_id)] = {
            "success_rate": success_rate,
            "evaluated_input_snapshot_tag": active_snapshot_tag,
            "evaluated_input_snapshot_role": active_snapshot_role,
            "evaluated_snapshot_tag": before_snapshot_tag,
        }
        epoch_state["current_epoch"] = epoch_id + 1
        epoch_state["current_task"] = 0
        progress = get_epoch_progress(epoch_state, epoch_id)
        progress["completed_tasks"] = epoch_result["total_tasks"]
        progress["successful_tasks"] = epoch_result["successful_tasks"]
        revised_manual = None
        should_generate_next_candidate = not (is_candidate_evaluation and not improved)
        if (
            uses_embodiskill_skill(args.skill_module)
            and should_generate_next_candidate
            and task_manager is not None
            and getattr(getattr(task_manager, "orchestrator", None), "meta_skill", None) is not None
        ):
            revise_manual = getattr(task_manager.orchestrator.meta_skill, "revise_manual", None)
            if callable(revise_manual):
                revised_manual = revise_manual(epoch_id=epoch_id, success_rate=success_rate)
        if revised_manual is not None:
            epoch_state["epoch_results"][str(epoch_id)]["manual_version_after"] = revised_manual.get("version")
        save_epoch_state(WORKING_DIR, epoch_state)
        flush_embodiskill_state(task_manager)
        after_snapshot_tag = None
        after_snapshot_dir = None
        if uses_embodiskill_skill(args.skill_module) and should_generate_next_candidate:
            after_snapshot_tag = get_embodiskill_snapshot_tag(epoch_id, phase="after")
            after_snapshot_dir = snapshot_embodiskill(
                WORKING_DIR,
                args.skill_module,
                snapshot_tag=after_snapshot_tag,
            )
        epoch_state["embodiskill_snapshots"].setdefault(str(epoch_id), {})
        epoch_state["embodiskill_snapshots"][str(epoch_id)]["after"] = after_snapshot_dir

        if uses_embodiskill_skill(args.skill_module):
            should_seed_best = best_snapshot_tag_before_epoch is None and before_snapshot_dir is not None

            if should_seed_best or improved:
                epoch_state["best_epoch_id"] = epoch_id
                epoch_state["best_success_rate"] = success_rate
                if before_snapshot_dir is not None:
                    epoch_state["best_embodiskill_snapshot_tag"] = before_snapshot_tag

            if is_candidate_evaluation:
                if improved:
                    epoch_state["epoch_results"][str(epoch_id)]["candidate_disposition"] = "accepted"
                    epoch_state["pending_embodiskill_snapshot_tag"] = after_snapshot_tag if after_snapshot_dir is not None else None
                else:
                    epoch_state["epoch_results"][str(epoch_id)]["candidate_disposition"] = "rejected"
                    epoch_state["pending_embodiskill_snapshot_tag"] = None
            else:
                epoch_state["epoch_results"][str(epoch_id)]["candidate_disposition"] = "queued"
                epoch_state["pending_embodiskill_snapshot_tag"] = after_snapshot_tag if after_snapshot_dir is not None else None

            epoch_state["epoch_results"][str(epoch_id)]["generated_candidate_snapshot_tag"] = after_snapshot_tag
            epoch_state["next_active_embodiskill_snapshot_tag"] = get_next_active_embodiskill_snapshot_tag(epoch_state)
        elif improved:
            epoch_state["best_epoch_id"] = epoch_id
            epoch_state["best_success_rate"] = success_rate

        save_epoch_state(WORKING_DIR, epoch_state)
        release_embodiskill_state(task_manager)
        task_manager = None

    final_active_snapshot_tag = get_next_active_embodiskill_snapshot_tag(epoch_state)
    if uses_embodiskill_skill(args.skill_module) and final_active_snapshot_tag:
        release_embodiskill_state(task_manager)
        task_manager = None
        print(f"[run_epochs] Restoring final active EmbodiSkill snapshot: {final_active_snapshot_tag}")
        restore_embodiskill_snapshot(
            WORKING_DIR,
            args.skill_module,
            snapshot_tag=final_active_snapshot_tag,
        )
        verify_embodiskill_writable(WORKING_DIR, args.skill_module)
    
    # Final summary
    print(f"\n{'='*60}")
    print("[run_epochs] All epochs completed!")
    print(f"Best epoch: {epoch_state['best_epoch_id']} with success_rate={epoch_state['best_success_rate']:.2%}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
