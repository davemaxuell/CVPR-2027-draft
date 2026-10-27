<p align="center">
  <h1 align="center">
    ReliableNav: Uncertainty-Aware Zero-Shot Vision-Language Navigation in Continuous Environments
    <br>
    [T-MM 2026 Submission]
  </h1>

  <p align="center">
  <a href="https://github.com/ZhanboShiAI"><strong>Zhanbo Shi<sup>1</sup></strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=8VOk_S4AAAAJ&hl=en"><strong>Lin Zhang<sup>1</sup>*</strong></a>
  ·
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=UQK70JoAAAAJ"><strong>Shengjie Zhao<sup>1</sup></strong></a>
  ·
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=Fe5Ru58AAAAJ"><strong>Yicong Zhou<sup>2</sup></strong></a><br>
  <sup>1</sup>School of Computer Science and Technology, Tongji University, Shanghai, China<br>
  <sup>2</sup>Department of Computer and Information Science, University of Macau, China
</p>

## Motivation
In recent years, numerous studies have attempted to solve the Vision-Language Navigation in Continuous Environments (VLN-CE) task in a zero-shot manner, leveraging Large Language Models (LLMs) to generate navigation decisions. However, all of these methods overlook the unreliability of LLMs' outputs from semantic uncertainty and sampling uncertainty, which significantly reduces the success rate of the VLN-CE task. Taking the aforementioned uncertainty into consideration, we propose ReliableNav, the first uncertainty-aware zero-shot navigation framework for the VLN-CE task. 

## Environment Installation
This project is developed with Python 3.9 on Ubuntu 22.04. If you are using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), you can create an environment with following instructions. 

```bash
conda create -n renav python=3.9 cmake=3.14.0 -y
conda activate renav
```

This project is built with CUDA 11.7 and torch 1.13.1, which can be installed with following commands.

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

To build this project, you should install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) v0.1.7, [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) v0.1.7, and [VLN-CE](https://github.com/jacobkrantz/VLN-CE) with the instructions in these projects. 

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
python setup.py install --headless --with-cuda

git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all

git clone http://github.com/jacobkrantz/VLN-CE.git
cd VLN-CE
python -m pip install -r requirements.txt
```

Edit "miniconda/envs/renav/lib/python3.9/site-packages/habitat_sim-0.1.7-py3.9-linux-x86_64.egg/habitat_sim/utils/common.py" file, fix the error due to numpy version in line 84.
```
axis = axis.astype(float)
```
Edit "miniconda/envs/renav/lib/python3.9/site-packages/gym/spaces/discrete.py" file, fix the error due to gym version in line 36. 
```
assert n >= 0, "n (counts) have to be positive"
```

ReliableNav depends on an open-vocalbulary detection and segmentation model [GLEE](https://github.com/FoundationVision/GLEE). You can download the pretrained model from [HuggingFace](https://huggingface.co/spaces/Junfeng5/GLEE_demo/blob/789873b404e7910953a978810011946733d540c2/GLEE_SwinL_Scaleup10m.pth) and put this file under "src/thirdparty/GLEE". 


Now you can install the dependencies ReliableNav:
```
cd ReliableNav
pip install -r requirements.txt
```

## Data
To use ReliableNav, you should firstly generate the following data. 

**Scene Dataset from Matterport3D**

Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). After the scene dataset has been downloaded, you should put the following files under the "ReliableNav/data" folder. 

**R2R-CE Dataset**

The R2R-CE dataset is a part of the Room-to-Room (R2R) dataset for the use with the Habitat-Sim. You can download this dataset following the instructions on their [project webpage](https://github.com/jacobkrantz/VLN-CE) and also put it under the "ReliableNav/data" folder. 

**RxR-CE Dataset**
The RxR-CE dataset is a part of the Room-Across-Room (RxR) dataset for use with the Habitat-Sim. You can also download this dataset following the instructions on their [project webpage](https://github.com/google-research-datasets/RxR)

**Data Folder Structure**
```
data|
    ├── datasets                        # task datasets
    │   └── [R2R_VLNCE_VERSION]
    │   │   └── [split]
    │   └── [RxR_VLNCE_VERSION]
    │       └── [split]
    ├── scene_datasets                  # MP3D scene datasets
    │   └── [dataset]
    │       └── [scene]
    │           └── [scene].house (habitat/mesh_sementic.glb)                           
    ├── logs                            # logs for LLM inpus, outputs, action decisions and error code.
    |──connectivity_graphs.pkl
    |──default.physics_config.json
```

## Usage
Below we show some example commands for evaluating ReliableNav for the VLN-CE task. 
```
python run.py --exp-config src/config/r2r_baselines/test.yaml --run-type eval
```
The above command will reproduce the ablation results using the inputs and ouputs of OpenAI GPT-4o for 100 episodes from the validation unseen split of the R2R-CE dataset. 

To retest ReliableNav, you can modify the "test.yaml" file. First, put your OpenAI API key under LLM.OPENAI_API_KEY. Then set the LOG.CLEAN to True. Now you can run the above command to retest ReliableNav. You can also use other versions of GPT by modifying the LLM.GPT_MODEL in config file. In addition, you can use other LLM APIs (e.g., Gemini, Qwen) by creating inherited class of **PromptHandler** in [llm_handler.py](./src/llm/llm_handler.py), as shown in the [gpt_hander.py](./src/llm/gpt_handler.py). 

## License
As a training-free model, the codebase of this porject is CC-BY-4.0 licensed, as found in the [LICENSE](./LICENSE) file. The Matterport3D dataset is under [CC-BY-NC-SA 3.0 US](https://creativecommons.org/licenses/by-nc-sa/3.0/us/) license with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf). 
