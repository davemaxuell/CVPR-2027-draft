<div align="center">

<h1><span style="background: linear-gradient(to right,rgb(128,0,128),rgb(210, 153, 153)); -webkit-background-clip: text; color: transparent;font-style: italic;"> VLN-MME</span>: Diagnosing MLLMs as Language-guided Visual Navigation agents</h1>

<div>
    <a href='https://billzhao1030.github.io/' target='_blank'>Xunyi Zhao<sup>*</sup></a>;
    <a href='https://gengzezhou.github.io' target='_blank'>Gengze Zhou<sup>*†</sup></a>;
    <a href='http://www.qi-wu.me' target='_blank'>Qi Wu</a>
</div>
AIML, University of Adelaide <br>
<em>* Equal Contribution † Project Lead</em>

<br>

<div>
    <a href='https://github.com/billzhao1030/VLN-MME' target='_blank'><img alt="Static Badge" src="https://img.shields.io/badge/VLNMME-v1-blue"></a>
    <a href='https://www.arxiv.org/abs/2512.24851' target='_blank'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

</div>

## 🍹 Abstract
Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across a wide range of vision-language tasks. However, their performance as embodied agents, which requires multi-round interaction with spatial reasoning and sequential action prediction, needs further exploration. Our work investigates this potential in the context of Vision-and-Language Navigation (VLN) by introducing a unified and extensible simulation-free evaluation framework to probe MLLMs as zero-shot agents, named VLN-MME. Simplifying the evaluation with a highly modular and accessible design streamlines experiments, enabling structured comparisons and component-level ablations across diverse MLLM architectures, agent designs, and navigation tasks. Crucially, enabled by VLN-MME, we observe that enhancing prevalent agents with Chain-of-Thought (CoT) reasoning and self-reflection leads to an unexpected performance decrease. This suggests MLLMs exhibit poor context awareness in embodied navigation tasks; although they can follow instructions and structure their output, their 3D spatial reasoning fidelity is low. Furthermore, we demonstrate that agent performance could be largely improved with simple failure cases in context learning. VLN-MME lays the groundwork for systematic evaluation of general-purpose MLLMs in embodied navigation settings and reveals limitations in their sequential decision-making capabilities. We believe these findings offer crucial guidance for MLLM post-training as embodied agents. 

<div align="center">
  <img src="assets/VLNMME.png" width="70%" />
</div>


## 🧋 Prerequisites

### 🍭 Installation

**Note: VLN-MME is simulator-free!** You do not need to install Matterport3D simulator or Habitat simulator. The codebase works entirely with pre-computed visual features and connectivity graphs.

Create a conda environment and install all dependencies:

```bash
conda create --name VLNMME python=3.10
conda activate VLN-MME

pip install -r requirements.txt
```

That's it! No simulator installation required.

## 🍬 Data Preparation

Download the required datasets and features from HuggingFace:

During the evaluation process, the program will automatically download all navigation datasets, environmental information and marked observation images from Hugging face [HuggingFace: VLN-MME](https://huggingface.co/VLN-MME), including:
- [5 navigation datasets](https://huggingface.co/datasets/VLN-MME/VLN_annotations) - R2R, REVERIE, RXR-EN, CVDN, OBJNAV_MP3D (downloaded in `src/tasks/base_dataset.py - _download_data from line 92-102`)
- [Marked Observation Images](https://huggingface.co/datasets/VLN-MME/MP3D_marked_obs) for MP3D simulator (downloaded when creating environment in `src/runner/feature_db.py at line 195`)
- [Environmental Information](https://huggingface.co/datasets/VLN-MME/MP3D) - Connectivity graphs, viewpoint caption and summary, object location for OBJNAV-MP3D for MatterSim (downloaded in `src/runner/load.py from line 10-35`)

The whole project directory should be structured as follows:

```
mllm-eval/
├── data/
│   ├── CVDN/                              # Cooperative Vision-and-Dialog Navigation
│   ├── MP3D/                              # Matterport3D dataset files
│   │   ├── graphs/                        # Connectivity and path data
│   │   ├── caption.json                   # Viewpoint captions
│   │   ├── location.json                  # Node coordinates
│   │   ├── navigable.json                 # Action space & graph connectivity
│   │   └── object_location.json           # Object instance locations
│   ├── OBJECTNAV/                         # Object Navigation task data
│   │   ├── objectnav.json                 # Main episode data
│   ├── R2R/                               # Room-to-Room dataset
│   ├── REVERIE/                           # REVERIE dataset
│   ├── RXR/                               # Room-Across-Room dataset (multilingual)
│   └── marked_obs/                        # Processed observation images
├── src/
│   ├── agents/                            # MLLM Agent implementations
│   │   ├── agent_base.py                  # Base class for agents
│   │   ├── baseline_agent.py              # Text summary as memory agent
│   │   ├── mapgpt_agent.py                # Text map as memory agent
│   │   ├── ...                            # Other agents
│   │   └── prompt.py                      # Prompt templates for agents
│   ├── configs/                           # Experiment and model configs
│   ├── models/                            # Model implementations
│   │   ├── base_mllm.py                   # Base class for MLLMs
│   │   ├── qwen2_5_vl.py                  # Qwen2.5-VL model implementation
│   │   ├── gpt5.py                        # GPT-5 model implementation
│   │   ├── ...                            # Other models
│   ├── outputs/                           # Outputs of evaluation
│   ├── runner/                            # Training/Inference loop management
│   │   ├── env.py                         # Simulation Environment
│   │   ├── feature_db.py                  # Feature Database of Viewpoints
│   │   ├── loader.py                      # Environment Loader (Downloads happen here)
│   ├── tasks/                             # Task-specific data loaders & evaluators
│   │   ├── r2r.py                         # R2R task loader
│   │   ├── reverie.py                     # REVERIE task loader
│   │   └── objectnav.py                   # Object Navigation task loader
│   │   ├── rxr.py                         # RXR task loader
│   │   ├── cvdn.py                        # CVDN task loader
│   ├── utils/                             # Common helper functions
│   ├── main.py                            # Main evaluation script
│   └── demo.py                            # Result Visualizer (Gradio)
```

## 🌟 Key Features

### 🎯 Simulator-Free Architecture

VLN-MME is **completely simulator-free**! The codebase works entirely with:
- Pre-built connectivity graphs
- No need to install or run Matterport3D or Habitat simulators

### 🗂️ Multi-Dataset Supporting

VLN-MME currently supports **5 different navigation datasets - and support more** simultaneously:

- **Fine-grained Language-Guided Navigation:**
  - **R2R** (Room-to-Room): Fine-grained instruction following
  - **RXR-EN**: Multilingual Room-to-Room Navigation (English)

- **Coarse-grained Language-Guided Navigation:**
  - **REVERIE**: Remote Embodied Visual Referring Expression in Real Indoor Environments (Navigation only)

- **Zero-grained Language-Guided Navigation:**
  - **ObjectNav-MP3D**: Object Navigation in Matterport3D

- **Interactive Navigation:**
  - **CVDN**: Cooperative Vision-and-Dialog Navigation

- **Other Potential Navigation Tasks:**
  - **SOON**: Semantic Object-Oriented Navigation
  - **R4R/R6R**: Long Horizon Navigation

The nature also support **VLN-CE** (Visual Language Navigation in Continuous Environment), but need to set up with CE related setting and use Habitat Simulator.


## 🎯 Configuration Guide

VLN-MME uses **OmegaConf** for hierarchical configuration management.

### Configuration Hierarchy

1. **`src/configs/default.yaml`**: Base configuration with all default settings
2. **`src/configs/experiment.yaml`**: Main experiment config (overrides defaults)
3. **`src/condfigs/demo.yaml`**: Result visualizer config

### Key Configuration Sections

#### 1. Experiment Settings

```yaml
experiment:
    id: "default"                # Experiment ID
    output_dir: "./outputs"      # Directory for outputs/experiment ID
    data_dir: "../data"          # Root directory for the datasets
    seed: 42                     # Random seed for reproducibility
    model: None                  # MLLM model
    max_step: 15                 # Max inference steps
    valid_from_file: false       # Valid from file or run
    batch_size: 1                # Batch size
  ```

#### 2. Task Settings
```yaml
task:
  val_source:                  # Source datasets for testing
    ['R2R', 'REVERIE']

  eval_splits: {               # Evaluation splits to test
    "R2R": ["val_unseen"],
    "REVERIE": ["val_unseen"],
    "RXR": ["val_unseen_subset", "val_unseen"],
    "CVDN": ["val_unseen_subset", "val_unseen"],
    "OBJECTNAV": ["val_unseen_subset"]
  }
```

Notice that each datasets (except ObjectNav-MP3D) contains the original splits including: **train**, **val_seen**, **val_unseen**, **test** and our curated subset **val_unseen_subset**. The ``val_source`` section defines the task we want to evaluate, and the ``eval_splits`` defines which splits we will used for evaluation. You can put as many task and splits for each as you want. Everything can be override in ``experiment.yaml``.

#### 3. Agent Setting
```yaml
agent:
  type: "baseline_agent"       # Agent type
```

#### 4. More about Agent, Model, Task Setting
VLN-MME support diverse agent design, MLLM and tasks, which means we can modify and add more code to use more agents, model and task.

We now use Model as an example:
In `src/models/__init__.py`, we see
```python
AVAILABLE_MODELS = {
    "qwen2_5_vl": "Qwen2_5_VL",
    "llava_onevision": "Llava_OneVision",
    ...
    "gpt5": "GPT5"
}
```
Here the keys are the model file name listed in ``src/models`` ("qwen2_5_vl" refers to ``qwen2_5_vl.py``), and the values are the Class name in the model file. Hence, when you want to add a new model, follow these steps:
- Add the model file with the class implementation
- Add the key-value pair in the ``__init__.py``

The above process applied for agent and task as well (for task you need to create the dataset too).

For prietary models like GPT-5, Gemini3 etc. Please put your API key in the ``api_key`` argument in corresponding class.

#### 5. About demo
We use Gradio to create a result visualizer that allow us to see the details of how MLLM navigate in the environment and how they success or fail.
To see the evaluate result, we need to modify the ``demo.yaml`` to specify the agent, model, task and the split. Here we show an example (notice the agent name is without "_agent" for simple representation):
```yaml
agent: baseline
model: qwen2_5_vl
task: R2R
split: val_unseen_subset
```

## 🧪 Evaluating

Evaluate model on certain split(s), using certain agent and certain task(s):

Create a basic config file:

```yaml
# configs/experiment.yaml
experiment:
  seed: 42
  model: "qwen2_5_vl"
  valid_from_file: false

agent:
  type: "baseline_agent"

task:
  val_source:
    ['R2R']

  eval_splits: {
    "R2R": ["val_unseen_subset"],
    "REVERIE": ["val_unseen_subset"],
    "RXR": ["val_unseen_subset"],
    "CVDN": ["val_unseen_subset"],
    "OBJECTNAV": ["val_unseen_subset"]
  }
```

Then, in terminal:
```bash
cd src
python run.py --config_dir configs/experiment.yaml
```

### Evaluation Metrics

VLN-MME evaluates on multiple metrics:
- **TL** (Trajectory Length): Length of the trajectory
- **NE** (Navigation Error): Distance to goal at end
- **SR** (Success Rate): Percentage of successful navigations
- **OSR** (Oracle Success Rate): Success rate with oracle stop
- **SPL** (Success weighted by Path Length): Efficiency metric
- **nDTW** (normalized Dynamic Time Warping): Path similarity to ground truth
- **SDTW** (Success Dynamic Time Warping): Path similarity to ground truth scaled by success rate
- **CLS** (Coverage weighted by Length Score): Another metric for path similarity to ground truth

Results are saved in the outputs directory and logged to console.

## 🥂 Acknowledgements
We extend our gratitude to MatterPort 3D for their valuable contributions to the open-source platform and community.

## 🍺 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{zhao2025vln,
  title={VLN-MME: Diagnosing MLLMs as Language-guided Visual Navigation agents},
  author={Zhao, Xunyi and Zhou, Gengze and Wu, Qi},
  journal={arXiv preprint arXiv:2512.24851},
  year={2025}
}
```