# EmbodiSkill: Skill-Aware Reflection for Self-Evolving Embodied Agents


## Introduction
This repo is the official implementation of
[***EmbodiSkill: Skill-Aware Reflection for Self-Evolving Embodied Agents***](https://arxiv.org/abs/2605.10332).

EmbodiSkill is a training-free framework for self-evolving embodied agents. It enables agents to improve reusable procedural skills from their own task-execution trajectories through skill-aware reflection. Instead of rewriting the whole skill from coarse trajectory feedback, EmbodiSkill distinguishes whether a trajectory reveals new knowledge, suggests optimization, exposes a skill defect, or simply shows an execution lapse. Based on these targeted signals, it selectively updates the skill body or emphasizes valid guidance in the skill appendix, allowing embodied agents to accumulate more complete, accurate, and executable skills over time.

![EmbodiSkill overview](assets/fig2.png)

## News

- **SkillOpt Integration.** The skill-aware reflection mechanism proposed in EmbodiSkill has been integrated into [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt), a general-domain framework for self-evolving agent skills.


## Setup
```
conda create -n EmbodiSkill python=3.12
conda activate EmbodiSkill
pip install -r requirements.txt
```

Alternatively, you can create the environment from `environment.yml`:
```
conda env create -f environment.yml
conda activate EmbodiSkill
```


## Quick Start

### Environment
Please download and prepare ALFWorld following the official ALFWorld
instructions:
- [ALFWorld](https://github.com/alfworld/alfworld)

EmbodiSkill expects the ALFWorld data root to contain the TextWorld game files
and logic files. The expected structure is:
```
<ALFWORLD_DATA_ROOT>
└── json_2.1.1
    ├── train
    ├── valid_seen
    └── valid_unseen
└── logic
    ├── alfred.pddl
    └── alfred.twl2
```

Set the data root before running:
```
export ALFWORLD_DATA_ROOT=/path/to/alfworld
```

The task lists shipped in this repository are:
```
data
└── alfworld
    ├── alfworld_tasks_suffix.json
    └── alfworld_tasks_train.json
```

These task files store relative game paths such as
`json_2.1.1/valid_unseen/.../game.tw-pddl`; they are resolved against
`ALFWORLD_DATA_ROOT` at runtime.


### API Keys
You can add API keys to a local `.env` file. The `.env` file is ignored by git.
The default setup uses a local `qwen2.5-14b` executor and `gpt-5.2` for
EmbodiSkill evolution, so OpenAI-compatible credentials are required unless you
override the skill backend.

For OpenAI-compatible backends:
```
OPENAI_API_KEY=""
OPENAI_BASE_URL="https://api.openai.com/v1"
```

For Azure OpenAI:
```
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_BASE_URL=""
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
```

For Gemini:
```
GEMINI_API_KEY=""
GOOGLE_GEMINI_BASE_URL=""
```

### How to Run
- Option 1: Configure with environment variables and run the script.
    ```
    export AGENT_MODEL=qwen2.5-14b
    export AGENT_LOCAL_BASE_URL=http://localhost:8044/v1
    export SKILL_API_TYPE=openai
    export SKILL_MODEL=gpt-5.2
    export NUM_EPOCHS=10
    export TRAIN_CHUNK_SIZE=50
    export TRAIN_SHUFFLE_SEED=42
    bash run_epochs_alfworld.sh
    ```

- Option 2: Run with Python command.
    ```
    python3 tasks/run_epochs.py \
      --task alfworld \
      --reasoning io \
      --skill_module embodiskill \
      --agent_model qwen2.5-14b \
      --agent_api_type local \
      --agent_local_base_url http://localhost:8044/v1 \
      --skill_model gpt-5.2 \
      --skill_api_type openai \
      --num_epochs 10 \
      --train_chunk_size 50 \
      --train_shuffle_seed 42 \
      --output_dir .db_embodiskill_epochs/run
    ```


## Citation
If you find this repository helpful, a citation to our paper would be greatly
appreciated:
```
@misc{ju2026embodiskill,
  title={EmbodiSkill: Skill-Aware Reflection for Self-Evolving Embodied Agents},
  author={Ju, Ruofei and Wang, Xinrui and Ding, Xin and Yang, Yifan and Wu, Hao and Jiang, Shiqi and Zhang, Qianxi and Wen, Hao and Li, Xiangyu and Wang, Weijun and Li, Kun and Liu, Yunxin and Dai, Haipeng and Wang, Wei and Cao, Ting},
  year={2026},
  eprint={2605.10332},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
