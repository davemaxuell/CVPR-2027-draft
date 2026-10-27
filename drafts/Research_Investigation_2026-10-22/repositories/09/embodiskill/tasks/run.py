import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import tempfile
import yaml
import json
from dataclasses import dataclass, field
import argparse
import random
from tqdm import tqdm
from datetime import datetime, timezone

import agentkit
from agentkit.agents import Agent
from agentkit.module_map import module_map
from agentkit.reasoning import ReasoningBase
from agentkit.skill import MASSkillBase
from agentkit.llm import LLMCallable, GPTChat, AzureGPTChat, LocalLLM, GeminiChat, get_price
from agentkit.orchestrator import Orchestrator
from agentkit.utils import EmbeddingFunc

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task, resolve_alfworld_env_config
from workflow import get_workflow
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_dir_name


with open('tasks/configs.yaml') as reader:
    CONFIG: dict = yaml.safe_load(reader)

WORKING_DIR: str = None
MIN_TMP_FREE_BYTES = 512 * 1024 * 1024
CONFIG_DIR = "config"
LOG_DIR = "logs"
EMBODISKILL_DIR = "EmbodiSkill"


def normalize_skill_type(skill_type: str | None) -> str:
    return (skill_type or "embodiskill").strip().lower()


def config_dir(output_path: str) -> str:
    return os.path.join(output_path, CONFIG_DIR)


def logs_dir(output_path: str) -> str:
    return os.path.join(output_path, LOG_DIR)


def active_skill_dir(output_path: str) -> str:
    return os.path.join(output_path, EMBODISKILL_DIR)


def prepare_output_layout(output_path: str) -> None:
    for path in [
        config_dir(output_path),
        logs_dir(output_path),
        os.path.join(output_path, EMBODISKILL_DIR),
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
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[run] Saved run config: {config_path}")


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

@dataclass
class TaskManager:
    task_name: str              # task name
    workflow: str               # fixed MAS workflow name
    skill_type: str            # skill module type
    tasks: list[dict]           # all tasks
    env: BaseEnv                # interative datatset environment
    recorder: BaseRecorder      # record experiment results
    orchestrator: Orchestrator                # multi-agent system
    orchestrator_config: dict = field(default_factory=dict)   # mas configs
    skill_config: dict = field(default_factory=dict)   # skill configs


def build_task(task: str, skill_type: str, max_steps: int, working_dir: str = None) -> TaskManager:

    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config = yaml.safe_load(reader)
    if task.startswith("alfworld"):
        config = resolve_alfworld_env_config(config)

    # Use provided working_dir or fall back to module-level WORKING_DIR
    effective_working_dir = working_dir if working_dir is not None else WORKING_DIR
    
    env: BaseEnv = get_env(task, config, max_steps)
    recorder: BaseRecorder = get_recorder(task, working_dir=effective_working_dir, namespace='total_task')
    tasks: list[dict] = get_task(task)
    workflow_instance: Orchestrator = get_workflow()
    orchestrator_config: dict = CONFIG.get('team', {})

    return TaskManager(
        task_name=task,
        workflow='team',
        skill_type=skill_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        orchestrator=workflow_instance,
        orchestrator_config=orchestrator_config
    )   

def build_system(
    task_manager: TaskManager,
    reasoning: str = None,
    skill_module: str = None,
    llm_type: str = None,
    api_type: str = "openai",
    local_base_url: str = "http://localhost:8000/v1",
    agent_llm_type: str = None,
    agent_api_type: str = None,
    agent_local_base_url: str = None,
    skill_llm_type: str = None,
    skill_api_type: str = None,
    skill_local_base_url: str = None,
) -> None:

    def create_llm(model_name: str, model_api_type: str, model_local_base_url: str) -> LLMCallable:
        if model_api_type == "azure":
            return AzureGPTChat(model_name=model_name)
        if model_api_type == "local":
            return LocalLLM(
                model_name=model_name,
                base_url=model_local_base_url,
            )
        if model_api_type == "gemini":
            return GeminiChat(model_name=model_name)
        return GPTChat(model_name=model_name)

    embed_func = EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"))
    reasoning_module_type, skill_component_type = module_map(reasoning, skill_module)

    agent_llm_type = agent_llm_type or llm_type
    agent_api_type = agent_api_type or api_type
    agent_local_base_url = agent_local_base_url or local_base_url
    skill_llm_type = skill_llm_type or llm_type
    skill_api_type = skill_api_type or api_type
    skill_local_base_url = skill_local_base_url or local_base_url

    if agent_llm_type is None or skill_llm_type is None:
        raise ValueError("Both agent_llm_type and skill_llm_type must be specified.")

    agent_llm_model: LLMCallable = create_llm(agent_llm_type, agent_api_type, agent_local_base_url)
    skill_llm_model: LLMCallable = create_llm(skill_llm_type, skill_api_type, skill_local_base_url)

    # 环境交互时生成 action 用
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=agent_llm_model)
    # EmbodiSkill 内部生成 skill / key steps / mistake analysis 用
    skill_component: MASSkillBase = skill_component_type( 
        namespace=skill_module,
        global_config=task_manager.skill_config,
        llm_model=skill_llm_model,
        embedding_func=embed_func
    )

    task_manager.orchestrator.add_observer(task_manager.recorder)
    task_manager.orchestrator.build_system(reasoning_module, skill_component, task_manager.env, task_manager.orchestrator_config)

def run_task(task_manager: TaskManager) -> None:

    task_manager.recorder.dataset_begin()
    
    for task_id, task_config in tqdm(enumerate(task_manager.tasks), total=len(task_manager.tasks), desc="Running Tasks"):
        task_manager.recorder.task_begin(task_id, task_config)  
        
        task_main, task_description = task_manager.orchestrator.env.set_env(task_config)   
        few_shots: list[str] = get_task_few_shots(
            dataset=task_manager.task_name, 
            task_config=task_config,
            few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
        )
        task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
           
        task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
        for agent in task_manager.orchestrator.agents_team.values():    
            task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
            task_manager.recorder.log(agent.add_task_instruction(task_instruction))

        reward, done, _ = task_manager.orchestrator.schedule(task_config) 
    
        task_manager.recorder.task_end(reward, done)             
    
    task_manager.recorder.dataset_end()



if __name__ == '__main__':
    # settings
    random.seed(42)
    _resolve_temp_dir()

    parser = argparse.ArgumentParser(description='Run tasks with specified modules.')
    parser.add_argument('--task', type=str, choices=['alfworld', 'alfworld_train'])
    parser.add_argument('--skill_module', type=str, default='embodiskill', help='Specify skill module')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Fallback LLM model type used when agent/skill model is not separately specified')
    parser.add_argument('--agent_model', type=str, default=None, help='LLM model type for agent action generation')
    parser.add_argument('--skill_model', type=str, default=None, help='LLM model type for EmbodiSkill internals')
    parser.add_argument('--max_trials', type=int, default=50, help='max number of steps')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajectories to retrieve from the skill store.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajectories to retrieve from the skill store.')
    parser.add_argument('--skills_topk', type=int, default=5, help='Number of skills to retrieve from the skill store.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')
    parser.add_argument('--api_type', type=str, default='openai', choices=['openai', 'azure', 'local', 'gemini'], help='Fallback API type used when agent/skill API type is not separately specified')
    parser.add_argument('--agent_api_type', type=str, default=None, choices=['openai', 'azure', 'local', 'gemini'], help='API type for agent action generation')
    parser.add_argument('--skill_api_type', type=str, default=None, choices=['openai', 'azure', 'local', 'gemini'], help='API type for EmbodiSkill internals')
    parser.add_argument('--local_base_url', type=str, default='http://localhost:8000/v1', help='Fallback base URL for local vLLM server')
    parser.add_argument('--agent_local_base_url', type=str, default=None, help='Base URL for the local agent model server')
    parser.add_argument('--skill_local_base_url', type=str, default=None, help='Base URL for the local skill model server')
    parser.add_argument('--workflow', type=str, default='team', help=argparse.SUPPRESS)

    args = parser.parse_args()

    task: str = args.task
    if args.workflow != 'team':
        raise ValueError("Only the team workflow is supported in this refactored codebase.")
    max_trials: int = args.max_trials
    agent_model_type: str = args.agent_model or args.model
    skill_model_type: str = args.skill_model or args.model
    skill_module_name: str = normalize_skill_type(args.skill_module)
    reasoning_type: str = args.reasoning
    agent_api_type: str = args.agent_api_type or args.api_type
    skill_api_type: str = args.skill_api_type or args.api_type
    agent_local_base_url: str = args.agent_local_base_url or args.local_base_url
    skill_local_base_url: str = args.skill_local_base_url or args.local_base_url
    
    # dir
    WORKING_DIR = os.path.join(
        './.db_alfworld',
        get_model_dir_name(agent_model_type, skill_model_type),
        task,
        'team',
        f'{skill_module_name}'
    )
    # if os.path.exists(WORKING_DIR):
    #     shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    prepare_output_layout(WORKING_DIR)
    _warn_if_low_disk_space(WORKING_DIR, "output directory")
    write_run_config(
        WORKING_DIR,
        args,
        agent_model=agent_model_type,
        skill_model=skill_model_type,
        agent_api_type=agent_api_type,
        skill_api_type=skill_api_type,
        agent_local_base_url=agent_local_base_url,
        skill_local_base_url=skill_local_base_url,
    )
    
    # run tasks
    task_configs: TaskManager = build_task(
        task,
        skill_module_name,
        max_trials,
        working_dir=logs_dir(WORKING_DIR),
    )
    task_configs.orchestrator_config['successful_topk'] = args.successful_topk
    task_configs.orchestrator_config['failed_topk'] = args.failed_topk
    task_configs.orchestrator_config['skills_topk'] = args.skills_topk
    task_configs.orchestrator_config['threshold'] = args.threshold
    task_configs.skill_config.update(
        working_dir=WORKING_DIR,
        persist_dir=active_skill_dir(WORKING_DIR),
        hop=args.hop,
        task=task,
    )

    build_system(
        task_configs,
        reasoning_type,
        skill_module_name,
        args.model,
        args.api_type,
        args.local_base_url,
        agent_llm_type=agent_model_type,
        agent_api_type=agent_api_type,
        agent_local_base_url=agent_local_base_url,
        skill_llm_type=skill_model_type,
        skill_api_type=skill_api_type,
        skill_local_base_url=skill_local_base_url,
    )
    run_task(task_configs)

    # postprocess
    completion_tokens, prompt_tokens, _ = get_price()
    task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, price={completion_tokens*15/1000000+prompt_tokens*5/1000000}')
