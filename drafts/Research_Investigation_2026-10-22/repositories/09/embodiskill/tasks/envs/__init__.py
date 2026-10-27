import json
import os
from copy import deepcopy

from .base_env import BaseEnv, BaseRecorder
from .alfworld_env import AlfworldEnv, AlfworldRecorder, get_env_name_from_gamefile, prefixes

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TASKS_PATH = {
    'alfworld': 'data/alfworld/alfworld_tasks_suffix.json',
    'alfworld_train': 'data/alfworld/alfworld_tasks_train.json',
}


def _resolve_repo_path(path: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(REPO_ROOT, expanded))


def get_alfworld_data_root() -> str:
    data_root = os.environ.get("ALFWORLD_DATA_ROOT", "data/alfworld")
    return _resolve_repo_path(data_root)


def resolve_alfworld_path(path: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(expanded):
        return expanded
    if expanded.startswith(("json_2.1.1/", "logic/", "detectors/")):
        return os.path.abspath(os.path.join(get_alfworld_data_root(), expanded))
    return _resolve_repo_path(expanded)


def resolve_alfworld_env_config(config: dict) -> dict:
    resolved = deepcopy(config)
    if "task_file" in resolved:
        resolved["task_file"] = _resolve_repo_path(resolved["task_file"])

    for key in ("data_path", "eval_id_data_path", "eval_ood_data_path"):
        value = resolved.get("dataset", {}).get(key)
        if value:
            resolved["dataset"][key] = resolve_alfworld_path(value)

    for key in ("domain", "grammar"):
        value = resolved.get("logic", {}).get(key)
        if value:
            resolved["logic"][key] = resolve_alfworld_path(value)

    value = resolved.get("mask_rcnn", {}).get("pretrained_model_path")
    if value:
        resolved["mask_rcnn"]["pretrained_model_path"] = resolve_alfworld_path(value)

    return resolved


def _load_alfworld_tasks(task_file: str) -> list[dict]:
    task_file = _resolve_repo_path(task_file)
    if not os.path.exists(task_file):
        print(f"[WARN] ALFWorld task file not found: {task_file}")
        return []

    with open(task_file, 'r') as f:
        rows = json.load(f)

    tasks: list[dict] = []
    for row in rows:
        gamefile = resolve_alfworld_path(row.get("gamefile", ""))
        env_name = get_env_name_from_gamefile(gamefile)
        if env_name is None:
            continue
        tasks.append({
            'task': f'{row.get("goal", "")}',
            'env_kwargs': {
                'config': 'alfworld',
                'gamefile': gamefile,
            },
            'task_type': prefixes[env_name],
            'env_name': env_name
        })
    return tasks


alfworld_tasks: list[dict] = _load_alfworld_tasks(TASKS_PATH['alfworld'])

alfworld_train_task_file = os.environ.get('ALFWORLD_TRAIN_TASK_FILE', TASKS_PATH['alfworld_train'])
alfworld_train_tasks: list[dict] = _load_alfworld_tasks(alfworld_train_task_file)


TASK_DATA = {
    'alfworld': alfworld_tasks,
    'alfworld_train': alfworld_train_tasks,
}

ENVS = {
    'alfworld': AlfworldEnv,
    'alfworld_train': AlfworldEnv,
}

RECORDERS = {
    'alfworld': AlfworldRecorder,
    'alfworld_train': AlfworldRecorder,
}


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    if ENVS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    return ENVS.get(task)(env_config, max_trials)


def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    if RECORDERS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    return RECORDERS.get(task)(working_dir=working_dir, namespace=namespace)


def get_task(task: str) -> list[dict]:
    if TASK_DATA.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    return TASK_DATA.get(task)
