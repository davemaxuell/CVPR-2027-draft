# M³E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts

Vision-and-Language Navigation (VLN) agents have shown strong capabilities in following natural language instructions. However, they often struggle to generalize across environments due to catastrophic forgetting, which limits their practical use in real-world settings where agents must continually adapt to new domains. We argue that overcoming forgetting across environments hinges on decoupling global scene reasoning from local perceptual alignment, allowing the agent to adapt to new domains while preserving specialized capabilities.
To this end, we propose M³E, the *Mixture of Macro and Micro Experts*, an environment-aware hierarchical MoE framework for continual VLN. Our method introduces a dual-router architecture that separates navigation into two levels of reasoning. A macro-level, scene-aware router selects strategy experts based on global environmental features (e.g., office vs. residential), while a micro-level, instance-aware router activates perception experts based on local instruction-vision alignment for step-wise decision making. To preserve knowledge across domains, we adopt a dynamic momentum update strategy that identifies expert utility in new environments and selectively updates or freezes their parameters.
We evaluate M³E in a domain-incremental setting on the R2R and REVERIE datasets, where agents learn across unseen scenes without revisiting prior data. Results show that our method consistently outperforms standard fine-tuning and existing continual learning baselines in both adaptability and knowledge retention, offering a parameter-efficient solution for building generalizable embodied agents.

## Methods
<p align="center">
    <img src="assets/m3e.png" width="95%"><br>
</p>

Overall architecture of M³E. The framework decouples macro-level scene reasoning and micro-level token grounding via a dual-router design. The Macro Router (blue) builds a task-aware scene representation using GNN-based propagation over a cognitive map. The Micro Router (purple) computes token-wise expert weights from hidden states. Both signals are fused to route experts in the MoE-LoRA layers for planning.

## Installation
1. Install the [MatterPort 3D simulator](https://github.com/peteanderson80/Matterport3DSimulator). Please add the simulator path to yout python path.
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Create the conda environment and install the requirements.
```
conda create --name M3E python=3.8.16
conda activate M3E
pip install -r requirements.txt
```

## Data Processing
The data directory is structed as follows. 

```
data
├── connectivity
├── R2R
├── REVERIE
├── eva_features
│   └── mp3d_EVA02-CLIP-L-14-336.hdf5
├── obj_features
│   └── reverie_obj_feat
└── models
    └── Vicuna-7B
```

**1. Orinal Datasets**
* R2R & REVERIE & SOON: we use the annotation provided by [DUET](https://github.com/cshizhe/VLN-DUET) and [Fine-Grained-R2R](https://github.com/YicongHong/Fine-Grained-R2R).

**2. Image Features**

The image features are extracted with [EVA-CLIP-02-Large (428M)](https://github.com/baaivision/EVA). To use EVA-CLIP-02, please install the corresponding environment following the instruction of th original reposity.
```
cd scripts/data_tools
sh extract_features_mp3d.sh         # for Matterport3D
```

**3. Object Features**

We leverage the object features extracted from ViT-B16 by [HM3DAutoVLN](https://github.com/cshizhe/HM3DAutoVLN),  and put the processed features of REVERIE at data/obj_features. You could either disable the object features by removing the flag `--enable_og`.

**4. Models**

The LLM is built upon [Vicuna-7B-v1.1](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#how-to-apply-delta-weights-for-weights-v11-and-v0). Please download the pre-trained model in [NaviLLM](https://github.com/zd11024/NaviLLM) and put it at data/models .

## Training & Inference

The continual learning pipeline consists of 4 steps per scene. Below is an example for a single scene:

**Step 1. Train on a new scene:**
```bash
bash scripts/1_train.sh
```

**Step 2. Collect routing statistics and compute importance scores:**
```bash
bash scripts/2_collect_stats.sh
```

**Step 3. Fine-grained momentum update:**
```bash
bash scripts/3_merge.sh
```

**Step 4. Evaluate on all domains:**
```bash
bash scripts/4_eval.sh
```

For sequential continual learning across multiple scenes, repeat Steps 1–3 for each scene in order, using the merged model (`merged.pt`) from the previous scene as the `RESUME_CKPT` for the next. Run Step 4 after each scene to track performance across all domains.

Our training order is:
```
zsNo4HB9uLZ → TbHJrupSAjP → oLBMNvg9in8 → 2azQ1b91cZZ → X7HyMhZNoso → QUCTc6BB5sX → EU6Fwq7SyZv → Z6MFQCViBuw
```

The training and validation data can be downloaded from [OneDrive](https://1drv.ms/f/c/accb7e2e9ba423c7/IgBkFou_oj6YR7ffubm1Mdp-AVIiSSepJuRhAcxwSCBoRWw?e=EP7dO9).

## Acknowledgements
We would like to thank MatterPort 3D for their contributions to the open-sourced platform and community.
Additionally, this work benefits from [NaviLLM](https://github.com/zd11024/NaviLLM), [DUET](https://github.com/cshizhe/VLN-DUET), [HM3DAutoVLN](https://github.com/cshizhe/HM3DAutoVLN), and [VLN-SIG](https://github.com/jialuli-luka/VLN-SIG). Thanks for their awesome works!
